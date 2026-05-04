"""Phase 3b: train StyleEmbedder against Z-Image-Base on PD12M-500
with auxiliary CSD-distillation loss.

Difference from train_phase3.py: adds a small auxiliary head that takes the
mean-pooled StyleEmbedder output and predicts the cached CSD style vector.
This adds gradient pressure that says "your output should encode style
information that CSD recognizes" — a constraint that Phase 3a lacked.

Hypothesis: the auxiliary loss will pressure the StyleEmbedder to extract
style-shaped features from SigLIP, replacing the memorization shortcut
that Phase 3a found with style-aware abstraction.

Inference path is unchanged: the auxiliary head is discarded at save time;
the saved .pt contains only the StyleEmbedder's state_dict (same shape as
phase2b_ssl.pt and phase3a_*.pt — drops cleanly into the ComfyUI loader).

Loss: total = diffusion_MSE + LAMBDA_AUX * (1 - cosine_sim(predicted, cached_CSD))
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, StyleAuxHead, style_injection

from diffusers import BitsAndBytesConfig, ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel
import bitsandbytes as bnb


# ── config ─────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE = torch.bfloat16
BASE_MODEL = "Tongyi-MAI/Z-Image"
HEIGHT = 512
WIDTH = 512
SHIFT = 3.0
LR = float(os.environ.get("LR", "1e-4"))
STEPS = int(os.environ.get("STEPS", "3000"))
WARMUP = int(os.environ.get("WARMUP", "500"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "500"))
VAL_SIZE = int(os.environ.get("VAL_SIZE", "50"))
LAMBDA_AUX = float(os.environ.get("LAMBDA_AUX", "10.0"))   # initial guess; smoke-test will calibrate
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase3_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RUN_NAME = os.environ.get("RUN_NAME", "phase3b")
LOSS_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_loss.txt")
CACHE_PATH = os.path.join(OUT_DIR, "phase3_pre_cache.pt")
SEED = 0


# ── caption cleanup ───────────────────────────────────────────────
PREFIX_RE = re.compile(
    r"^\s*(the image (shows|is|depicts|displays|features|contains|portrays)"
    r"|an image (of|showing|depicting)"
    r"|a (photo|picture|painting|drawing) (of|showing))\s+",
    re.IGNORECASE,
)

def clean_caption(c: str) -> str:
    c = c.strip()
    new = PREFIX_RE.sub("", c, count=1)
    if new and new != c:
        new = new[0].upper() + new[1:]
    return new or c


# ── flow-matching helpers ────────────────────────────────────────
def shifted_t_to_sigma(t, shift):
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_training_timestep(batch, shift, device, dtype):
    u = torch.randn(batch, device=device).sigmoid()
    sigma = shifted_t_to_sigma(u, shift).to(dtype)
    timestep = (sigma * 1000.0).to(dtype)
    return sigma, timestep


# ── encoders ────────────────────────────────────────────────────
def encode_siglip(siglip, processor, image, device, dtype):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    spatial = inputs.spatial_shapes[0]
    sig_H, sig_W = int(spatial[0]), int(spatial[1])
    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state
    C = hidden.shape[-1]
    feats = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, C).to(dtype)
    return feats, sig_H, sig_W


def lr_at(step, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def split_train_val(entries, all_files, val_size, seed):
    rng_split = random.Random(seed)
    files_shuffled = list(all_files)
    rng_split.shuffle(files_shuffled)
    val_files = files_shuffled[:val_size]
    train_data = [entries[f] for f in files_shuffled[val_size:] if f in entries]
    val_data = [entries[f] for f in val_files if f in entries]
    return train_data, val_data


def load_cache():
    print(f"Loading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    if "entries" in cache:
        entries = cache["entries"]
    else:
        entries = {}
        for e in cache.get("train", []):
            entries[e["name"]] = e
        for e in cache.get("val", []):
            entries[e["name"]] = e
    n_with_csd = sum(1 for e in entries.values() if "csd_vector" in e)
    print(f"  Loaded: {len(entries)} entries, {n_with_csd} with CSD vectors")
    if n_with_csd < len(entries):
        raise RuntimeError(
            f"Cache has {len(entries) - n_with_csd} entries without CSD vectors. "
            f"Run src/add_csd_to_cache.py first."
        )
    return entries


# ── training ─────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Run: {RUN_NAME}  STEPS={STEPS}  LR={LR}  GRAD_ACCUM={GRAD_ACCUM}  LAMBDA_AUX={LAMBDA_AUX}")

    with open(CAPTION_PATH) as f:
        captions = json.load(f)
    all_files = sorted(captions.keys())

    entries = load_cache()
    train_data, val_data = split_train_val(entries, all_files, VAL_SIZE, SEED)
    print(f"Dataset: {len(all_files)} total -> train={len(train_data)} val={len(val_data)}")

    # ── load training models ─────────────────────────────────────
    print("Loading transformer (NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    ).to(DEVICE)
    transformer.requires_grad_(False)
    transformer.train()
    transformer.enable_gradient_checkpointing()
    print(f"  VRAM after transformer load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("Loading SigLIP-2...")
    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        ).to(DEVICE).eval()
    )
    siglip.requires_grad_(False)
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    siglip_dim = siglip.config.hidden_size
    print(f"  VRAM after SigLIP load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("Initializing StyleEmbedder + StyleAuxHead...")
    torch.manual_seed(SEED)
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        DEVICE, dtype=DTYPE
    )
    style_embedder.train()
    aux_head = StyleAuxHead(in_dim=transformer.config.dim, hidden_dim=256, out_dim=768).to(
        DEVICE, dtype=DTYPE
    )
    aux_head.train()
    n_emb = sum(p.numel() for p in style_embedder.parameters())
    n_aux = sum(p.numel() for p in aux_head.parameters())
    print(f"  StyleEmbedder params: {n_emb/1e6:.2f}M")
    print(f"  StyleAuxHead params:  {n_aux/1e6:.2f}M (training-only, discarded at save)")

    # both modules' params go into the optimizer
    trainable = list(style_embedder.parameters()) + list(aux_head.parameters())
    optim = bnb.optim.AdamW8bit(trainable, lr=LR)

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    # ── training loop ───────────────────────────────────────────
    rng = random.Random(SEED)
    print(f"\nTraining {STEPS} steps (grad_accum={GRAD_ACCUM}, eff batch={GRAD_ACCUM})...")
    diff_losses, aux_losses = [], []
    val_history = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")
    loss_log.write("step\tdiff_loss\taux_loss\tlr\n")
    optim.zero_grad(set_to_none=True)

    for step in range(STEPS):
        cur_lr = lr_at(step, LR, WARMUP)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        entry = train_data[rng.randrange(len(train_data))]

        ref_pil = Image.open(entry["raw_path"]).convert("RGB")
        siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, ref_pil, DEVICE, DTYPE)

        cap_feats = entry["cap_feats"].to(DEVICE, DTYPE)
        cap_len_padded = entry["cap_len_padded"]
        latent = entry["latent"].to(DEVICE, DTYPE)
        csd_target = entry["csd_vector"].to(DEVICE, DTYPE)   # (768,) L2-normalized

        sigma, timestep = sample_training_timestep(1, SHIFT, DEVICE, DTYPE)
        sigma_b = sigma.view(-1, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn_like(latent)
            noisy = sigma_b * noise + (1.0 - sigma_b) * latent
            target = latent - noise

        x_input = noisy.unsqueeze(2)
        x_list = list(x_input.unbind(dim=0))

        # ── forward through StyleEmbedder, capture mean-pooled output ──
        # We can't easily intercept inside style_injection, so we recompute
        # the embedded features here for the auxiliary head. They're cheap
        # (one Linear over ~hundreds of tokens).
        flat_siglip = siglip_feats.reshape(sig_H * sig_W, -1)
        embedded = style_embedder(flat_siglip)        # (sig_H*sig_W, 3840)
        mean_pooled = embedded.mean(dim=0)            # (3840,)
        predicted_csd = aux_head(mean_pooled.unsqueeze(0)).squeeze(0)  # (768,) L2-norm
        # cosine sim of two L2-norm vectors == dot product
        cos_sim = (predicted_csd.float() * csd_target.float()).sum()
        aux_loss_raw = 1.0 - cos_sim                    # in [0, 2], typically [0, 1]

        # ── forward through transformer with style injection ──
        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            pred_list = transformer(x_list, timestep, [cap_feats], return_dict=False)[0]

        pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)
        diff_loss_raw = F.mse_loss(pred.float(), target.float())

        total_loss = (diff_loss_raw + LAMBDA_AUX * aux_loss_raw) / GRAD_ACCUM
        total_loss.backward()

        diff_val = diff_loss_raw.item()
        aux_val = aux_loss_raw.item()

        if (step + 1) % GRAD_ACCUM == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

        diff_losses.append(diff_val)
        aux_losses.append(aux_val)
        loss_log.write(f"{step}\t{diff_val:.6f}\t{aux_val:.6f}\t{cur_lr:.2e}\n")
        loss_log.flush()

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            recent_d = diff_losses[-LOG_EVERY:] if len(diff_losses) >= LOG_EVERY else diff_losses
            recent_a = aux_losses[-LOG_EVERY:] if len(aux_losses) >= LOG_EVERY else aux_losses
            print(
                f"step {step:4d}  diff={diff_val:.4f} avg_d={np.mean(recent_d):.4f}  "
                f"aux={aux_val:.4f} avg_a={np.mean(recent_a):.4f}  "
                f"lr={cur_lr:.2e}  sigma={sigma.item():.3f}  "
                f"t={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )

        # ── periodic validation ─────────────────────────────────
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            style_embedder.eval()
            aux_head.eval()
            val_diff, val_aux = [], []
            val_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
            with torch.no_grad():
                for ventry in val_data:
                    ref_pil_v = Image.open(ventry["raw_path"]).convert("RGB")
                    sf, vh, vw = encode_siglip(siglip, processor, ref_pil_v, DEVICE, DTYPE)
                    u = torch.randn(1, generator=val_gen, device=DEVICE).sigmoid()
                    sigma_v = shifted_t_to_sigma(u, SHIFT).to(DTYPE)
                    ts_v = (sigma_v * 1000.0).to(DTYPE)
                    sigma_vb = sigma_v.view(-1, 1, 1, 1)
                    v_latent = ventry["latent"].to(DEVICE, DTYPE)
                    v_cap = ventry["cap_feats"].to(DEVICE, DTYPE)
                    v_csd = ventry["csd_vector"].to(DEVICE, DTYPE)
                    n_v = torch.randn(v_latent.shape, generator=val_gen, device=DEVICE, dtype=DTYPE)
                    noisy_v = sigma_vb * n_v + (1.0 - sigma_vb) * v_latent
                    target_v = v_latent - n_v

                    flat_v = sf.reshape(vh * vw, -1)
                    emb_v = style_embedder(flat_v)
                    pred_csd_v = aux_head(emb_v.mean(dim=0).unsqueeze(0)).squeeze(0)
                    val_aux.append(1.0 - (pred_csd_v.float() * v_csd.float()).sum().item())

                    x_v = noisy_v.unsqueeze(2)
                    with style_injection(
                        transformer=transformer,
                        style_embedder=style_embedder,
                        siglip_features=sf,
                        image_size=(image_h_patched, image_w_patched),
                        cap_lens=[ventry["cap_len_padded"]],
                    ):
                        pv = transformer(list(x_v.unbind(dim=0)), ts_v,
                                          [v_cap], return_dict=False)[0]
                    pv_t = torch.stack([p for p in pv], dim=0).squeeze(2)
                    val_diff.append(F.mse_loss(pv_t.float(), target_v.float()).item())

            md = float(np.mean(val_diff))
            ma = float(np.mean(val_aux))
            val_history.append((step, md, ma))
            print(f"  -- val @ step {step}: diff={md:.4f}  aux={ma:.4f}  (n={len(val_diff)})")
            with open(os.path.join(OUT_DIR, f"{RUN_NAME}_val.txt"), "w") as fv:
                fv.write("step\tval_diff\tval_aux\n")
                for s, d, a in val_history:
                    fv.write(f"{s}\t{d:.6f}\t{a:.6f}\n")
            style_embedder.train()
            aux_head.train()

        # ── checkpoint ──────────────────────────────────────────
        # Only StyleEmbedder weights are saved (ComfyUI loader-compatible).
        if (step + 1) % CKPT_EVERY == 0 or step == STEPS - 1:
            ckpt_path = os.path.join(OUT_DIR, f"{RUN_NAME}_step{step+1:04d}.pt")
            torch.save(style_embedder.state_dict(), ckpt_path)
            print(f"  -- saved {ckpt_path}")

    loss_log.close()
    final_path = os.path.join(OUT_DIR, f"{RUN_NAME}_final.pt")
    torch.save(style_embedder.state_dict(), final_path)
    print(f"\nDone. Final: {final_path}")
    print(f"Diff loss: initial={diff_losses[0]:.4f} final={diff_losses[-1]:.4f}")
    print(f"Aux loss:  initial={aux_losses[0]:.4f} final={aux_losses[-1]:.4f}")
    if val_history:
        print(f"Val diff: first={val_history[0][1]:.4f} last={val_history[-1][1]:.4f}")
        print(f"Val aux:  first={val_history[0][2]:.4f} last={val_history[-1][2]:.4f}")


if __name__ == "__main__":
    main()

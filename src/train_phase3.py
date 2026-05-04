"""Phase 3 Run A: train StyleEmbedder against Z-Image-Base on PD12M-500.

Recipe:
- Reference path: PIL -> Siglip2ImageProcessorFast (NO PIL augmentation)
- Target path:    PIL -> resize-short-side(512) -> CenterCrop(512) -> VAE
- Frozen NF4 transformer + bf16 trainable StyleEmbedder
- batch=1, grad_accum=4, lr=1e-4, warmup 500, AdamW8bit
- 3000 steps total, checkpoint every 500
- 50-image held-out val split for periodic loss tracking

No CFG dropout: the IP-Adapter dropout recipe is for training cross-attention
layers that need to handle the "no reference" case. We have no such layers —
only a projector. With the ComfyUI node's column-only key-masking for uncond
rows, cap/x output positions in the uncond pass are computed entirely from
cap/x keys (style keys masked out), which is identical to the base model's
native [cap, x] pretraining configuration. No additional training needed for
the absence case to behave correctly under CFG.

VRAM strategy on 12GB:
- Pre-compute phase loads only text encoder (Qwen3) + VAE on GPU; transformer
  stays on CPU. Latents and caption embeddings are computed once, moved to
  CPU, persisted to disk as `{RUN_NAME}_pre_cache.pt`.
- Training phase frees the pre-compute pipeline, then loads the NF4 transformer
  and SigLIP-2. Cached tensors are streamed back to GPU per step.
- On re-run, the disk cache short-circuits pre-compute entirely.
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
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, style_injection

from diffusers import BitsAndBytesConfig, ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel
import bitsandbytes as bnb


# ── config ─────────────────────────────────────────────────────────
DEVICE = "cuda"
DTYPE = torch.bfloat16
BASE_MODEL = "Tongyi-MAI/Z-Image"      # the Base, not Turbo
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
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase3_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RUN_NAME = os.environ.get("RUN_NAME", "phase3a")
LOSS_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_loss.txt")
CACHE_PATH = os.path.join(OUT_DIR, "phase3_pre_cache.pt")  # shared across runs
SEED = 0


# ── caption cleanup ───────────────────────────────────────────────
# PD12M captions are BLIP-style and start with "The image shows ..."
# Z-Image was trained on natural prompts; strip the prefix.
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
def encode_target(vae, image, device, dtype):
    """PIL (any size) -> resize short side 512 -> center-crop 512 -> VAE latent."""
    w, h = image.size
    if w < h:
        new_w, new_h = HEIGHT, int(round(h * HEIGHT / w))
    else:
        new_w, new_h = int(round(w * WIDTH / h)), WIDTH
    img = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - WIDTH) // 2
    top = (new_h - HEIGHT) // 2
    img = img.crop((left, top, left + WIDTH, top + HEIGHT))

    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        encoded = vae.encode(t).latent_dist.sample()
    encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    return encoded


def encode_siglip(siglip, processor, image, device, dtype):
    """PIL (raw, any size) -> SigLIP-2 features (sig_H, sig_W, 1152).

    No PIL augmentation — SigLIP-2 naflex handles arbitrary aspect ratios.
    """
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


# ── pre-compute (text encoder + VAE only) ────────────────────────
def precompute_to_disk(all_files, captions):
    """Encode caption embeds + target latents once, save to disk on CPU.

    Stores ALL entries (no train/val split) — split happens at load time so
    VAL_SIZE can change without regenerating the cache.

    Loads ZImagePipeline but keeps the transformer on CPU (saves ~3GB VRAM).
    Frees the pipeline entirely before returning.
    """
    print(f"Loading ZImagePipeline (transformer stays on CPU during pre-compute)...")
    pipe = ZImagePipeline.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
    # only move text encoder + VAE to GPU; leave transformer on CPU
    pipe.text_encoder.to(DEVICE).eval()
    pipe.vae.to(DEVICE).eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    print(f"  VRAM after partial pipeline load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    entries = {}
    t0 = time.time()

    def cache_one(fname):
        path = os.path.join(DATA_DIR, fname)
        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            return None
        latent = encode_target(pipe.vae, img_pil, DEVICE, DTYPE).cpu()
        clean = clean_caption(captions[fname])
        with torch.no_grad():
            cap_feats_list = pipe._encode_prompt(clean, device=DEVICE)
        cap_feats = cap_feats_list[0].cpu()
        cap_len_raw = cap_feats.shape[0]
        cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
        return {
            "name": fname,
            "raw_path": path,
            "latent": latent,
            "cap_feats": cap_feats,
            "cap_len_padded": cap_len_padded,
            "caption": clean,
        }

    print(f"Pre-computing all {len(all_files)} images...")
    for i, fname in enumerate(all_files):
        e = cache_one(fname)
        if e is not None:
            entries[fname] = e
        if (i + 1) % 25 == 0 or (i + 1) == len(all_files):
            print(
                f"  {i+1}/{len(all_files)}  "
                f"elapsed={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.2f}GB"
            )

    print(f"Saving cache to {CACHE_PATH}...")
    torch.save({"entries": entries}, CACHE_PATH)

    print(f"Freeing pipeline (text encoder + VAE)...")
    del pipe
    torch.cuda.empty_cache()
    print(f"  VRAM after pipeline free: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    return entries


def load_cache():
    print(f"Loading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    if "entries" in cache:
        entries = cache["entries"]
    else:
        # backwards-compat: old cache stored split train/val
        entries = {}
        for e in cache.get("train", []):
            entries[e["name"]] = e
        for e in cache.get("val", []):
            entries[e["name"]] = e
    print(f"  Loaded: {len(entries)} entries")
    return entries


def split_train_val(entries, all_files, val_size, seed):
    """Deterministic split of cached entries into train/val."""
    rng_split = random.Random(seed)
    files_shuffled = list(all_files)
    rng_split.shuffle(files_shuffled)
    val_files = files_shuffled[:val_size]
    train_files_set = set(files_shuffled[val_size:])
    train_data = [entries[f] for f in files_shuffled[val_size:] if f in entries]
    val_data = [entries[f] for f in val_files if f in entries]
    return train_data, val_data


# ── training ─────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Run: {RUN_NAME}  STEPS={STEPS}  LR={LR}  GRAD_ACCUM={GRAD_ACCUM}")

    # load captions
    with open(CAPTION_PATH) as f:
        captions = json.load(f)
    all_files = sorted(captions.keys())

    # ── pre-compute (or load from disk) ──────────────────────────
    if os.path.exists(CACHE_PATH):
        entries = load_cache()
    else:
        entries = precompute_to_disk(all_files, captions)

    # split happens here, NOT in the cache — VAL_SIZE can change between runs
    train_data, val_data = split_train_val(entries, all_files, VAL_SIZE, SEED)
    print(f"Dataset: {len(all_files)} total -> train={len(train_data)} val={len(val_data)}")

    # ── now load training models ─────────────────────────────────
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

    print("Initializing StyleEmbedder...")
    torch.manual_seed(SEED)
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        DEVICE, dtype=DTYPE
    )
    style_embedder.train()
    n_params = sum(p.numel() for p in style_embedder.parameters())
    print(f"  Trainable params: {n_params/1e6:.2f}M")
    optim = bnb.optim.AdamW8bit(style_embedder.parameters(), lr=LR)

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    # ── training loop ───────────────────────────────────────────
    rng = random.Random(SEED)
    print(f"\nTraining {STEPS} steps (grad_accum={GRAD_ACCUM}, eff batch={GRAD_ACCUM})...")
    losses = []
    val_history = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")
    optim.zero_grad(set_to_none=True)

    for step in range(STEPS):
        cur_lr = lr_at(step, LR, WARMUP)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        entry = train_data[rng.randrange(len(train_data))]

        # SigLIP encode the *raw* image (variable resolution, no augmentation)
        ref_pil = Image.open(entry["raw_path"]).convert("RGB")
        siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, ref_pil, DEVICE, DTYPE)

        cap_feats = entry["cap_feats"].to(DEVICE, DTYPE)
        cap_len_padded = entry["cap_len_padded"]
        latent = entry["latent"].to(DEVICE, DTYPE)

        sigma, timestep = sample_training_timestep(1, SHIFT, DEVICE, DTYPE)
        sigma_b = sigma.view(-1, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn_like(latent)
            noisy = sigma_b * noise + (1.0 - sigma_b) * latent
            target = latent - noise

        x_input = noisy.unsqueeze(2)
        x_list = list(x_input.unbind(dim=0))
        cap_feats_list_in = [cap_feats]

        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            pred_list = transformer(x_list, timestep, cap_feats_list_in, return_dict=False)[0]

        pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)
        loss = F.mse_loss(pred.float(), target.float()) / GRAD_ACCUM
        loss.backward()
        loss_value = loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            optim.step()
            optim.zero_grad(set_to_none=True)

        losses.append(loss_value)
        loss_log.write(f"{step}\t{loss_value:.6f}\t{cur_lr:.2e}\n")
        loss_log.flush()

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            recent = losses[-LOG_EVERY:] if len(losses) >= LOG_EVERY else losses
            print(
                f"step {step:4d}  loss={loss_value:.4f}  avg_recent={np.mean(recent):.4f}  "
                f"lr={cur_lr:.2e}  sigma={sigma.item():.3f}  "
                f"t={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )

        # ── periodic validation ─────────────────────────────────
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            style_embedder.eval()
            val_losses = []
            # Use a fixed CUDA generator so (sigma, noise) draws are identical
            # across every val pass — val loss differences are then the model,
            # not RNG.
            val_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
            with torch.no_grad():
                for ventry in val_data:
                    ref_pil = Image.open(ventry["raw_path"]).convert("RGB")
                    sf, _, _ = encode_siglip(siglip, processor, ref_pil, DEVICE, DTYPE)
                    u = torch.randn(1, generator=val_gen, device=DEVICE).sigmoid()
                    sigma_v = shifted_t_to_sigma(u, SHIFT).to(DTYPE)
                    ts_v = (sigma_v * 1000.0).to(DTYPE)
                    sigma_vb = sigma_v.view(-1, 1, 1, 1)
                    v_latent = ventry["latent"].to(DEVICE, DTYPE)
                    v_cap = ventry["cap_feats"].to(DEVICE, DTYPE)
                    n_v = torch.randn(v_latent.shape, generator=val_gen, device=DEVICE, dtype=DTYPE)
                    noisy_v = sigma_vb * n_v + (1.0 - sigma_vb) * v_latent
                    target_v = v_latent - n_v
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
                    val_losses.append(F.mse_loss(pv_t.float(), target_v.float()).item())
            mean_val = float(np.mean(val_losses))
            val_history.append((step, mean_val))
            print(f"  ── val MSE @ step {step}: {mean_val:.4f}  (n={len(val_losses)})")
            with open(os.path.join(OUT_DIR, f"{RUN_NAME}_val.txt"), "w") as fv:
                for s, v in val_history:
                    fv.write(f"{s}\t{v:.6f}\n")
            style_embedder.train()

        # ── checkpoint ──────────────────────────────────────────
        if (step + 1) % CKPT_EVERY == 0 or step == STEPS - 1:
            ckpt_path = os.path.join(OUT_DIR, f"{RUN_NAME}_step{step+1:04d}.pt")
            torch.save(style_embedder.state_dict(), ckpt_path)
            print(f"  ── saved {ckpt_path}")

    loss_log.close()
    final_path = os.path.join(OUT_DIR, f"{RUN_NAME}_final.pt")
    torch.save(style_embedder.state_dict(), final_path)
    print(f"\nDone. Final: {final_path}")
    print(f"Loss: initial={losses[0]:.4f} final={losses[-1]:.4f}")
    if val_history:
        print(f"Val MSE: first={val_history[0][1]:.4f} last={val_history[-1][1]:.4f}")


if __name__ == "__main__":
    main()

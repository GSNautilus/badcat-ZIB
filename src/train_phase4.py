"""Phase 4: Lumina-Accessory recipe applied to Z-Image-Base on 22 paintings.

The architectural test of whether sequence-concat + RoPE has a path on Base.
Recipe (per discussion 2026-05-03):

  Architecture: StyleEmbedder + position offset (h+H_target, w+W_target)
                + dual-time AdaLN (style at t=1)
  Data:         22 hand-curated paintings (data/wikimedia_train), geometric
                augmentation only (crop + h-flip — no photometric)
  Loss:         pure RF + per-token CSD aux (LAMBDA_AUX=50)
  Hyperparams:  3000 steps, LR 1e-4, AdamW8bit, shift=3.0, batch=1,
                grad_clip=100, warmup=500
  Trunk:        Z-Image-Base, NF4-quantized, frozen

Both architectural pieces (offset positions + dual-time AdaLN) live in
src/adapter.py. The training loop just passes `timestep=` to style_injection
to enable dual-time AdaLN. STYLE_ROPE_CONVENTION must be "offset".

CSD discrimination on this dataset was validated 2026-05-03 via
test_encoder_discrimination.py: mean pairwise cos 0.465, range 0.504,
~2x SigLIP's discriminative spread.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, style_injection, STYLE_ROPE_CONVENTION

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
LAMBDA_AUX = float(os.environ.get("LAMBDA_AUX", "50.0"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "100.0"))
BRIDGE_SEED = int(os.environ.get("BRIDGE_SEED", "42"))
REF_BASE_SIZE = int(os.environ.get("REF_BASE_SIZE", "384"))
# Reference encoder size (square). 384 matches phase 2b's REF_BASE_SIZE.
USE_DUAL_TIME = os.environ.get("USE_DUAL_TIME", "1") == "1"
# Whether to enable dual-time AdaLN (Lumina-Accessory recipe: style tokens at
# t=1, image+cap at diffusion t). Set to 0 to test offset positions alone.
# Caveat: t=1 modulation is meaningful on Z-Image-Omni (which was trained with
# this signal for clean conditions) but the Base trunk has no learned
# association for it.
STYLE_DROPOUT_PROB = float(os.environ.get("STYLE_DROPOUT_PROB", "0.10"))
USE_GRADIENT_CHECKPOINTING = os.environ.get("USE_GRADIENT_CHECKPOINTING", "1") == "1"
# Gradient checkpointing trades compute for memory: each layer's forward is
# recomputed during backward instead of storing activations. With it on, peak
# VRAM is ~4 GB at our settings; with it off, peak is ~6-7 GB but ~1.3-1.5x
# faster per step. Disable when you have spare VRAM; re-enable if you OOM.
# Probability per step of training without style (drop_style=True). Standard
# IP-Adapter recipe uses 5-10% to teach the projector to handle the CFG-mask
# differential at inference: at cfg>1, the inference path masks style out of
# uncond → cond-uncond differential carries the style contribution at the
# offset-RoPE privileged position (bottom-right) → CFG amplifies a structured
# bias the projector wasn't trained against. Style dropout exposes the
# projector to "without style" trunk states during training so it can learn
# to keep its differential structure clean. Default 0.10.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wikimedia_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RUN_NAME = os.environ.get("RUN_NAME", "phase4_offset_dualtime")
LOSS_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_loss.txt")
VAL_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_val.txt")
CACHE_PATH = os.path.join(OUT_DIR, "phase4_pre_cache.pt")
SEED = 0
TRANSFORMER_DIM = 3840
CSD_DIM = 768


# ── geometric-only augmentation ───────────────────────────────────
def augment_reference(img: Image.Image, rng: random.Random) -> Image.Image:
    """Geometric augmentation only (crop + h-flip). Drops phase 2b's
    photometric pieces (brightness/contrast/saturation/blur/downscale)
    per discussion: photometric ops change style, geometric ops preserve
    style while changing what content is visible inside the frame.

    Output: REF_BASE_SIZE square.
    """
    work = img.resize((512, 512), Image.LANCZOS)
    crop_frac = rng.uniform(0.6, 0.8)
    crop_size = int(512 * crop_frac)
    x0 = rng.randint(0, 512 - crop_size)
    y0 = rng.randint(0, 512 - crop_size)
    work = work.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    work = work.resize((REF_BASE_SIZE, REF_BASE_SIZE), Image.LANCZOS)
    if rng.random() < 0.5:
        work = work.transpose(Image.FLIP_LEFT_RIGHT)
    return work


# ── flow-matching helpers ────────────────────────────────────────
def shifted_t_to_sigma(t, shift):
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_training_timestep(batch, shift, device, dtype):
    u = torch.randn(batch, device=device).sigmoid()
    sigma = shifted_t_to_sigma(u, shift).to(dtype)
    timestep = (sigma * 1000.0).to(dtype)
    return sigma, timestep


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


def make_bridge(seed: int, device, dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(TRANSFORMER_DIM, CSD_DIM, generator=g) / np.sqrt(TRANSFORMER_DIM)
    return R.to(device, dtype)


def aux_loss_per_token(
    embedded: torch.Tensor,       # (n_tokens, 3840)
    csd_target: torch.Tensor,     # (768,) L2-normalized
    R: torch.Tensor,              # (3840, 768)
) -> torch.Tensor:
    """mean_over_tokens(1 - cos_sim(token_i @ R, csd_target)).

    Per-token CSD aux. Each of the n_tokens output rows is individually
    pressured toward the reference's CSD direction. Combined with the
    diffusion loss varying per-position under offset RoPE, this should
    drive per-token magnitude variation that the mean-pool aux didn't.
    """
    per_token = embedded @ R                                          # (n_tokens, 768)
    pred_n = F.normalize(per_token.float(), dim=-1, p=2)
    targ_n = F.normalize(csd_target.float(), dim=-1, p=2)
    cos_sims = (pred_n * targ_n.unsqueeze(0)).sum(dim=-1)
    return (1.0 - cos_sims).mean()


def structural_correlation(
    proj_outputs: torch.Tensor,
    csd_vectors: torch.Tensor,
) -> float:
    p = F.normalize(proj_outputs.float(), dim=-1, p=2)
    c = F.normalize(csd_vectors.float(), dim=-1, p=2)
    sim_p = (p @ p.T).cpu().numpy()
    sim_c = (c @ c.T).cpu().numpy()
    n = sim_p.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.corrcoef(sim_p[iu], sim_c[iu])[0, 1])


def load_cache():
    print(f"Loading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    entries = cache["entries"]
    n_complete = sum(
        1 for e in entries.values()
        if "latent" in e and "cap_feats" in e and "csd_vector" in e
    )
    print(f"  Loaded: {len(entries)} entries, {n_complete} complete")
    if n_complete < len(entries):
        raise RuntimeError(
            f"Cache has {len(entries) - n_complete} incomplete entries. "
            f"Run src/build_phase4_cache.py first."
        )
    # Convert to list of dicts (with raw_path retained for per-step PIL load)
    return list(entries.values())


# ── training ─────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    print(f"Run: {RUN_NAME}  STEPS={STEPS}  LR={LR}  GRAD_ACCUM={GRAD_ACCUM}  "
          f"LAMBDA_AUX={LAMBDA_AUX}  GRAD_CLIP={GRAD_CLIP}  "
          f"STYLE_ROPE_CONVENTION={STYLE_ROPE_CONVENTION!r}  "
          f"USE_DUAL_TIME={USE_DUAL_TIME}  "
          f"STYLE_DROPOUT_PROB={STYLE_DROPOUT_PROB}  "
          f"USE_GRADIENT_CHECKPOINTING={USE_GRADIENT_CHECKPOINTING}")
    assert STYLE_ROPE_CONVENTION == "offset", (
        f"Phase 4 requires STYLE_ROPE_CONVENTION='offset' but got "
        f"{STYLE_ROPE_CONVENTION!r}. Check src/adapter.py."
    )

    train_data = load_cache()
    print(f"Dataset: {len(train_data)} paintings (all used for both train and val)")

    R_bridge = make_bridge(BRIDGE_SEED, DEVICE, DTYPE)
    print(f"Bridge: R shape={tuple(R_bridge.shape)}, dtype={R_bridge.dtype}, frozen")

    # ── load training models ─────────────────────────────────────
    print("Loading transformer (Z-Image Base, NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    ).to(DEVICE)
    transformer.requires_grad_(False)
    transformer.train()
    if USE_GRADIENT_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()
        print(f"  Gradient checkpointing: ENABLED")
    else:
        print(f"  Gradient checkpointing: DISABLED (faster but more VRAM)")
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
    n_emb = sum(p.numel() for p in style_embedder.parameters())
    print(f"  StyleEmbedder params: {n_emb/1e6:.2f}M (only trainable surface)")

    optim = bnb.optim.AdamW8bit(style_embedder.parameters(), lr=LR)

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    # ── training loop ───────────────────────────────────────────
    rng = random.Random(SEED)
    dropout_rng = random.Random(SEED + 100)  # separate stream for dropout decisions
    print(f"\nTraining {STEPS} steps (grad_accum={GRAD_ACCUM}, eff batch={GRAD_ACCUM})...")
    diff_losses, aux_losses = [], []
    n_dropout_steps = 0
    val_history = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")
    loss_log.write("step\tdiff_loss\taux_loss\tlr\tdropped\n")
    optim.zero_grad(set_to_none=True)

    for step in range(STEPS):
        cur_lr = lr_at(step, LR, WARMUP)
        for g in optim.param_groups:
            g["lr"] = cur_lr

        entry = train_data[rng.randrange(len(train_data))]

        # Per-step augment + SigLIP encode (only step that needs the raw PIL)
        ref_pil = Image.open(entry["raw_path"]).convert("RGB")
        aug_ref = augment_reference(ref_pil, rng)
        siglip_feats, sig_H, sig_W = encode_siglip(
            siglip, processor, aug_ref, DEVICE, DTYPE
        )

        cap_feats = entry["cap_feats"].to(DEVICE, DTYPE)
        cap_len_padded = entry["cap_len_padded"]
        latent = entry["latent"].to(DEVICE, DTYPE)
        csd_target = entry["csd_vector"].to(DEVICE, DTYPE)

        sigma, timestep = sample_training_timestep(1, SHIFT, DEVICE, DTYPE)
        sigma_b = sigma.view(-1, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn_like(latent)
            noisy = sigma_b * noise + (1.0 - sigma_b) * latent
            target = latent - noise

        x_input = noisy.unsqueeze(2)
        x_list = list(x_input.unbind(dim=0))

        # Per-token CSD aux on the StyleEmbedder's own output
        flat_siglip = siglip_feats.reshape(sig_H * sig_W, -1)
        embedded = style_embedder(flat_siglip)
        aux_loss_raw = aux_loss_per_token(embedded, csd_target, R_bridge)

        # Style dropout (IP-Adapter pattern): with probability STYLE_DROPOUT_PROB,
        # train without style this step. drop_style=True makes style_injection a
        # no-op — the trunk processes [x, cap] only. diff_loss flows no gradient
        # to style_embedder (style isn't in the forward path), but aux_loss does
        # (it's computed on the projector's output independently). This teaches
        # the projector to handle the CFG cond-uncond differential cleanly at
        # inference, where the inference path masks style for uncond.
        drop_style_this_step = dropout_rng.random() < STYLE_DROPOUT_PROB
        if drop_style_this_step:
            n_dropout_steps += 1

        # Forward + backward inside the style_injection context. The dual-time
        # AdaLN patches stash data on the transformer; that stash and the
        # layer.forward overrides must survive backward (which under gradient
        # checkpointing re-runs each block's forward during the backward pass).
        # Moving backward outside the with block triggers a NoneType crash in
        # diffusers' transformer_z_image.py:241 because the saved args don't
        # include t_noisy/t_clean.
        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
            timestep=(timestep if USE_DUAL_TIME else None),
            drop_style=drop_style_this_step,
        ):
            pred_list = transformer(x_list, timestep, [cap_feats], return_dict=False)[0]

            pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)
            diff_loss_raw = F.mse_loss(pred.float(), target.float())

            total_loss = (diff_loss_raw + LAMBDA_AUX * aux_loss_raw) / GRAD_ACCUM
            total_loss.backward()

        diff_val = diff_loss_raw.item()
        aux_val = aux_loss_raw.item()

        if (step + 1) % GRAD_ACCUM == 0:
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    style_embedder.parameters(), max_norm=GRAD_CLIP
                )
            optim.step()
            optim.zero_grad(set_to_none=True)

        diff_losses.append(diff_val)
        aux_losses.append(aux_val)
        loss_log.write(
            f"{step}\t{diff_val:.6f}\t{aux_val:.6f}\t{cur_lr:.2e}\t{int(drop_style_this_step)}\n"
        )
        loss_log.flush()

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            recent_d = diff_losses[-LOG_EVERY:] if len(diff_losses) >= LOG_EVERY else diff_losses
            recent_a = aux_losses[-LOG_EVERY:] if len(aux_losses) >= LOG_EVERY else aux_losses
            dropout_rate = n_dropout_steps / (step + 1)
            print(
                f"step {step:4d}  diff={diff_val:.4f} avg_d={np.mean(recent_d):.4f}  "
                f"aux={aux_val:.4f} avg_a={np.mean(recent_a):.4f}  "
                f"lr={cur_lr:.2e}  sigma={sigma.item():.3f}  "
                f"drop={'Y' if drop_style_this_step else '.'} "
                f"drop_rate={dropout_rate:.2%}  "
                f"t={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )

        # ── periodic validation (uses all 22 with deterministic noise) ──
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            style_embedder.eval()
            val_diff, val_aux = [], []
            val_mean_outputs, val_csd_targets = [], []
            val_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
            with torch.no_grad():
                for ventry in train_data:
                    ref_pil_v = Image.open(ventry["raw_path"]).convert("RGB")
                    # Val uses CENTER-crop 0.7 — deterministic, no flip
                    work = ref_pil_v.resize((512, 512), Image.LANCZOS)
                    crop_size = int(512 * 0.7)
                    pad = (512 - crop_size) // 2
                    work = work.crop((pad, pad, pad + crop_size, pad + crop_size))
                    work = work.resize((REF_BASE_SIZE, REF_BASE_SIZE), Image.LANCZOS)
                    sf, vh, vw = encode_siglip(siglip, processor, work, DEVICE, DTYPE)

                    u = torch.randn(1, generator=val_gen, device=DEVICE).sigmoid()
                    sigma_v = shifted_t_to_sigma(u, SHIFT).to(DTYPE)
                    ts_v = (sigma_v * 1000.0).to(DTYPE)
                    sigma_vb = sigma_v.view(-1, 1, 1, 1)
                    v_latent = ventry["latent"].to(DEVICE, DTYPE)
                    v_cap = ventry["cap_feats"].to(DEVICE, DTYPE)
                    v_csd = ventry["csd_vector"].to(DEVICE, DTYPE)
                    n_v = torch.randn(v_latent.shape, generator=val_gen,
                                       device=DEVICE, dtype=DTYPE)
                    noisy_v = sigma_vb * n_v + (1.0 - sigma_vb) * v_latent
                    target_v = v_latent - n_v

                    flat_v = sf.reshape(vh * vw, -1)
                    emb_v = style_embedder(flat_v)
                    val_aux.append(aux_loss_per_token(emb_v, v_csd, R_bridge).item())
                    val_mean_outputs.append(emb_v.mean(dim=0).cpu())
                    val_csd_targets.append(v_csd.cpu())

                    x_v = noisy_v.unsqueeze(2)
                    with style_injection(
                        transformer=transformer,
                        style_embedder=style_embedder,
                        siglip_features=sf,
                        image_size=(image_h_patched, image_w_patched),
                        cap_lens=[ventry["cap_len_padded"]],
                        timestep=(ts_v if USE_DUAL_TIME else None),
                    ):
                        pv = transformer(list(x_v.unbind(dim=0)), ts_v,
                                          [v_cap], return_dict=False)[0]
                    pv_t = torch.stack([p for p in pv], dim=0).squeeze(2)
                    val_diff.append(F.mse_loss(pv_t.float(), target_v.float()).item())

            md = float(np.mean(val_diff))
            ma = float(np.mean(val_aux))
            outputs_t = torch.stack(val_mean_outputs).float()
            csd_t = torch.stack(val_csd_targets).float()
            mc = structural_correlation(outputs_t, csd_t)
            val_history.append((step, md, ma, mc))
            print(f"  -- val @ step {step}: diff={md:.4f}  aux={ma:.4f}  "
                  f"struct_corr={mc:+.4f}  (n={len(val_diff)})")
            with open(VAL_LOG_PATH, "w") as fv:
                fv.write("step\tval_diff\tval_aux\tval_struct_corr\n")
                for s, d, a, c in val_history:
                    fv.write(f"{s}\t{d:.6f}\t{a:.6f}\t{c:+.6f}\n")
            style_embedder.train()

        # ── checkpoint ──────────────────────────────────────────
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
    print(f"Dropout:   {n_dropout_steps}/{STEPS} steps ({n_dropout_steps/STEPS:.2%})")
    if val_history:
        print(f"Val diff:        first={val_history[0][1]:.4f} last={val_history[-1][1]:.4f}")
        print(f"Val aux:         first={val_history[0][2]:.4f} last={val_history[-1][2]:.4f}")
        print(f"Val struct_corr: first={val_history[0][3]:+.4f} last={val_history[-1][3]:+.4f}")


if __name__ == "__main__":
    main()

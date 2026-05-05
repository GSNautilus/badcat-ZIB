"""Phase 5: phase 4 architecture + per-block LoRA on trunk Q, K projections.

Motivation
----------
Phase 4 (2026-05-04) shipped a clean, artifact-free Base-trained adapter on
22 paintings — but block-level masking diagnostics revealed the trained
projector's effective contribution is concentrated in the first 2-3 of 30
main blocks. Masking blocks 0-2 entirely removes visible style; masking
blocks 3-29 has no observable effect. The frozen Z-Image trunk's pretrained
Q projections aren't receptive to OOD sequence-concat tokens beyond the
first few blocks.

Phase 5 follows the OminiControl recipe (arxiv:2411.15098) with three
mechanisms added on top to actively force balanced per-block contribution
(the first phase 5 attempt failed at this — a vanilla LoRA at every block
just amplified the same gradient asymmetry phase 4 had):

  1. **Variance regularization** on per-block effective LoRA magnitudes.
     The loss includes a term that minimizes the variance of
     ||gate_i * up_i @ down_i||_F across the 30 blocks. Outlier blocks
     (much bigger than the mean) get pushed down; small blocks get
     pulled up. This is the load-bearing fix.

  2. **Decoupled learning rates** for the projector vs the LoRA. The
     LoRA, sitting deep in a residual stream, gets larger gradients
     than the projector that lives at the input. Lower LR for LoRA
     (default 1/4 of projector LR) prevents runaway growth at blocks
     that win the gradient lottery.

  3. **Cosine LR decay** to a small floor. The original training kept
     LR at peak indefinitely; with that, fast-growing blocks just kept
     accelerating. Decay throttles back as the LoRA converges.

Architecture summary
--------------------
- Trunk: Z-Image-Base, NF4-quantized, frozen
- StyleEmbedder: RMSNorm + Linear (1152→3840), trainable, ~5M params
- Per-block LoRA: rank-32 on Q and K of each of 30 main blocks, plus per-
  block scalar gate, trainable, ~14.7M params total
- Sequence-concat with offset RoPE (same as phase 4)
- Single-time AdaLN (USE_DUAL_TIME defaults False)
- Style dropout 10% (same as phase 4)

Loss: rectified-flow MSE + per-token CSD aux (LAMBDA_AUX=50, same recipe
and bridge seed as phase 4 for clean comparison) + variance regularization
(LAMBDA_BALANCE=1.0, applies only to the LoRA — the projector doesn't have
per-block structure to balance).

Checkpoint format: a single .pt file containing both the projector and
LoRA stack state, plus a config dict. Backward-incompatible with phase 4
checkpoints (different format); inference loader detects format and falls
back to projector-only behavior for older files. Pre-gate phase 5
checkpoints load via strict=False with gates defaulting to 1.0.

Smoke target: 50 steps, no OOM, loss decreases, LoRA up-matrix norms grow
away from zero, balance loss is non-zero and contributing gradient.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import (
    StyleEmbedder,
    BlockLoRAStack,
    style_injection,
    lora_injection,
    STYLE_ROPE_CONVENTION,
)

from diffusers import BitsAndBytesConfig
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
USE_DUAL_TIME = os.environ.get("USE_DUAL_TIME", "0") == "1"
# Phase 4 finding: dual-time AdaLN is OOD on Base (no learned association
# for t=t_scale on style tokens). Default False here. Override via env if
# we want to test it again.
STYLE_DROPOUT_PROB = float(os.environ.get("STYLE_DROPOUT_PROB", "0.10"))
USE_GRADIENT_CHECKPOINTING = os.environ.get("USE_GRADIENT_CHECKPOINTING", "1") == "1"
USE_NF4 = os.environ.get("USE_NF4", "1") == "1"
# When True (default), load Z-Image-Base with 4-bit NF4 quantization. Required
# for the 12 GB RTX 3060. When False, load the trunk in DTYPE (bf16) directly
# — saves the per-matmul dequantization cost (~10-20% faster per step) but
# uses ~9 GB more VRAM (3 GB quantized vs 12.3 GB bf16). Override to "0" on
# 20-24 GB cards (e.g. A4500). On exactly 20 GB, you may still want
# gradient checkpointing on; on 24 GB+ both can be off.
LORA_RANK = int(os.environ.get("LORA_RANK", "32"))
LORA_LR_RATIO = float(os.environ.get("LORA_LR_RATIO", "0.25"))
# Multiplier applied to LR for the LoRA param group. Projector keeps LR;
# LoRA gets LR * LORA_LR_RATIO. Default 0.25 = LoRA at 1/4 of projector LR.
# Rationale: the LoRA sits deep in the residual stream and amplifies
# gradients more than the input-side projector. Equal LRs on both caused
# block 0's LoRA to accelerate exponentially in the first phase 5 attempt.
LAMBDA_BALANCE = float(os.environ.get("LAMBDA_BALANCE", "1.0"))
# Coefficient on the per-block variance regularization term. The variance
# is in units of (effective LoRA Frobenius norm)^2, which empirically lands
# in single-digit territory once the LoRA has been training a while. Setting
# LAMBDA_BALANCE=1 puts the regularization in roughly the same magnitude
# range as a typical step's diff_loss, which is the right scale to actively
# pull outlier blocks toward the mean without dominating the loss surface.
LR_DECAY_FLOOR = float(os.environ.get("LR_DECAY_FLOOR", "0.1"))
# Floor of cosine LR decay as a fraction of peak LR. Default 0.1 = LR decays
# from peak to peak/10 over (STEPS - WARMUP) steps. The previous schedule
# held LR at peak indefinitely after warmup, which let fast-growing blocks
# keep accelerating throughout training.
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wikimedia_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
RUN_NAME = os.environ.get("RUN_NAME", "phase5_lora")
LOSS_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_loss.txt")
VAL_LOG_PATH = os.path.join(OUT_DIR, f"{RUN_NAME}_val.txt")
CACHE_PATH = os.path.join(OUT_DIR, "phase4_pre_cache.pt")  # reuse phase 4 cache
SEED = 0
TRANSFORMER_DIM = 3840
CSD_DIM = 768


# ── geometric-only augmentation (same as phase 4) ───────────────────
def augment_reference(img: Image.Image, rng: random.Random) -> Image.Image:
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


def lr_at(step: int, base_lr: float, warmup: int, total_steps: int,
          decay_floor: float = LR_DECAY_FLOOR) -> float:
    """Linear warmup then cosine decay from base_lr to base_lr*decay_floor.

    Phase 5's first attempt held LR at peak forever after warmup, which let
    blocks that won early gradient lotteries keep accelerating throughout
    training. Cosine decay throttles back smoothly as the LoRA approaches
    a working state.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, progress)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (decay_floor + (1.0 - decay_floor) * cos_factor)


def make_bridge(seed: int, device, dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(TRANSFORMER_DIM, CSD_DIM, generator=g) / np.sqrt(TRANSFORMER_DIM)
    return R.to(device, dtype)


def aux_loss_per_token(
    embedded: torch.Tensor,
    csd_target: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    per_token = embedded @ R
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
    return list(entries.values())


def lora_norms(block_lora: BlockLoRAStack) -> dict:
    """Quick diagnostics on the LoRA state — used during smoke test to
    verify gradients are reaching the LoRA params (up-matrix norms grow
    away from zero) and to spot if any block is dead.
    """
    q_up_norms = []
    k_up_norms = []
    for layer in block_lora.layers:
        q_up_norms.append(layer.q_up.weight.detach().float().norm().item())
        k_up_norms.append(layer.k_up.weight.detach().float().norm().item())
    return {
        "q_up_mean": float(np.mean(q_up_norms)),
        "q_up_max": float(np.max(q_up_norms)),
        "q_up_min": float(np.min(q_up_norms)),
        "k_up_mean": float(np.mean(k_up_norms)),
        "k_up_max": float(np.max(k_up_norms)),
        "k_up_min": float(np.min(k_up_norms)),
    }


# ── training ─────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  dtype: {DTYPE}")
    print(f"Run: {RUN_NAME}  STEPS={STEPS}  LR={LR}  GRAD_ACCUM={GRAD_ACCUM}  "
          f"LAMBDA_AUX={LAMBDA_AUX}  LAMBDA_BALANCE={LAMBDA_BALANCE}  "
          f"GRAD_CLIP={GRAD_CLIP}  "
          f"STYLE_ROPE_CONVENTION={STYLE_ROPE_CONVENTION!r}  "
          f"USE_DUAL_TIME={USE_DUAL_TIME}  "
          f"STYLE_DROPOUT_PROB={STYLE_DROPOUT_PROB}  "
          f"USE_NF4={USE_NF4}  "
          f"USE_GRADIENT_CHECKPOINTING={USE_GRADIENT_CHECKPOINTING}  "
          f"LORA_RANK={LORA_RANK}  LORA_LR_RATIO={LORA_LR_RATIO}  "
          f"LR_DECAY_FLOOR={LR_DECAY_FLOOR}")
    assert STYLE_ROPE_CONVENTION == "offset", (
        f"Phase 5 inherits offset RoPE from phase 4 — got "
        f"{STYLE_ROPE_CONVENTION!r}. Check src/adapter.py."
    )

    train_data = load_cache()
    print(f"Dataset: {len(train_data)} paintings")

    R_bridge = make_bridge(BRIDGE_SEED, DEVICE, DTYPE)
    print(f"Bridge: R shape={tuple(R_bridge.shape)}, dtype={R_bridge.dtype}, frozen")

    # ── load training models ─────────────────────────────────────
    if USE_NF4:
        print(f"Loading transformer (Z-Image Base, NF4 quantized)...")
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
        )
        transformer = ZImageTransformer2DModel.from_pretrained(
            BASE_MODEL, subfolder="transformer",
            quantization_config=nf4, torch_dtype=DTYPE,
        ).to(DEVICE)
    else:
        print(f"Loading transformer (Z-Image Base, full {DTYPE} — no NF4)...")
        transformer = ZImageTransformer2DModel.from_pretrained(
            BASE_MODEL, subfolder="transformer",
            torch_dtype=DTYPE,
        ).to(DEVICE)
    transformer.requires_grad_(False)
    transformer.train()
    if USE_GRADIENT_CHECKPOINTING:
        transformer.enable_gradient_checkpointing()
        print(f"  Gradient checkpointing: ENABLED")
    else:
        print(f"  Gradient checkpointing: DISABLED")
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
    print(f"  StyleEmbedder params: {n_emb/1e6:.2f}M")

    print(f"Initializing BlockLoRAStack (rank={LORA_RANK})...")
    block_lora = BlockLoRAStack(
        num_blocks=len(transformer.layers),
        dim=transformer.config.dim,
        rank=LORA_RANK,
    ).to(DEVICE, dtype=DTYPE)
    block_lora.train()
    n_lora = sum(p.numel() for p in block_lora.parameters())
    print(f"  BlockLoRAStack params: {n_lora/1e6:.2f}M "
          f"({len(transformer.layers)} blocks × rank {LORA_RANK})")
    print(f"  Total trainable params: {(n_emb + n_lora)/1e6:.2f}M")

    # ── optimizer with separate param groups for projector vs LoRA ───
    # Projector keeps the full LR; LoRA scales by LORA_LR_RATIO. The LoRA
    # otherwise tends to grab gradient at the early/late blocks and runs
    # away faster than the projector can adapt to it. Separate groups also
    # let the cosine decay schedule scale both groups proportionally.
    proj_params = list(style_embedder.parameters())
    lora_params = list(block_lora.parameters())
    optim = bnb.optim.AdamW8bit([
        {"params": proj_params, "lr": LR, "name": "projector"},
        {"params": lora_params, "lr": LR * LORA_LR_RATIO, "name": "lora"},
    ])
    # Stash the per-group base LRs for the cosine schedule update inside the
    # training loop. AdamW8bit doesn't expose the originals after construction.
    base_lrs = [LR, LR * LORA_LR_RATIO]
    print(f"  Optimizer: projector LR={LR:.2e}, LoRA LR={LR * LORA_LR_RATIO:.2e}")
    print(f"  VRAM after optim setup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    # ── training loop ───────────────────────────────────────────
    rng = random.Random(SEED)
    dropout_rng = random.Random(SEED + 100)
    print(f"\nTraining {STEPS} steps (grad_accum={GRAD_ACCUM}, eff batch={GRAD_ACCUM})...")
    diff_losses, aux_losses, balance_losses = [], [], []
    n_dropout_steps = 0
    val_history = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")
    loss_log.write(
        "step\tdiff_loss\taux_loss\tbalance_loss\tlr_proj\tlr_lora\t"
        "dropped\tq_up_mean\tk_up_mean\n"
    )
    optim.zero_grad(set_to_none=True)

    # Initial LoRA norms — should be all zeros (zero-init up matrices).
    init_norms = lora_norms(block_lora)
    print(f"  LoRA init norms: q_up_mean={init_norms['q_up_mean']:.6e} "
          f"k_up_mean={init_norms['k_up_mean']:.6e} "
          f"(both should be 0.0 at init)")

    for step in range(STEPS):
        # Cosine decay scales both param groups proportionally to their base LRs.
        for g_idx, g in enumerate(optim.param_groups):
            g["lr"] = lr_at(step, base_lrs[g_idx], WARMUP, STEPS)
        cur_lr_proj = optim.param_groups[0]["lr"]
        cur_lr_lora = optim.param_groups[1]["lr"]

        entry = train_data[rng.randrange(len(train_data))]

        # The cache stores `raw_path` as the absolute path on the machine
        # where it was built. Use entry["name"] + local DATA_DIR instead so
        # the cache is portable across machines (e.g. 3060 -> A4500).
        ref_pil = Image.open(os.path.join(DATA_DIR, entry["name"])).convert("RGB")
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

        flat_siglip = siglip_feats.reshape(sig_H * sig_W, -1)
        embedded = style_embedder(flat_siglip)
        aux_loss_raw = aux_loss_per_token(embedded, csd_target, R_bridge)

        drop_style_this_step = dropout_rng.random() < STYLE_DROPOUT_PROB
        if drop_style_this_step:
            n_dropout_steps += 1

        # LoRA stays active during BOTH style-present and style-dropout steps.
        # See the docstring on lora_injection for why.
        with lora_injection(transformer, block_lora):
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

                # Per-block variance regularization on effective LoRA
                # magnitudes — pulls outlier blocks toward the mean. At init
                # (zero-init up matrices) this is exactly 0; grows once any
                # blocks start outpacing others.
                balance_loss_raw = block_lora.variance_reg_loss()

                total_loss = (
                    diff_loss_raw
                    + LAMBDA_AUX * aux_loss_raw
                    + LAMBDA_BALANCE * balance_loss_raw
                ) / GRAD_ACCUM
                total_loss.backward()

        diff_val = diff_loss_raw.item()
        aux_val = aux_loss_raw.item()
        balance_val = balance_loss_raw.item()

        if (step + 1) % GRAD_ACCUM == 0:
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    proj_params + lora_params, max_norm=GRAD_CLIP
                )
            optim.step()
            optim.zero_grad(set_to_none=True)

        diff_losses.append(diff_val)
        aux_losses.append(aux_val)
        balance_losses.append(balance_val)

        # Track LoRA norms cheaply for logging
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            norms = lora_norms(block_lora)
            recent_d = diff_losses[-LOG_EVERY:] if len(diff_losses) >= LOG_EVERY else diff_losses
            recent_a = aux_losses[-LOG_EVERY:] if len(aux_losses) >= LOG_EVERY else aux_losses
            recent_b = balance_losses[-LOG_EVERY:] if len(balance_losses) >= LOG_EVERY else balance_losses
            dropout_rate = n_dropout_steps / (step + 1)
            print(
                f"step {step:4d}  diff={diff_val:.4f} avg_d={np.mean(recent_d):.4f}  "
                f"aux={aux_val:.4f} avg_a={np.mean(recent_a):.4f}  "
                f"bal={balance_val:.4e} avg_b={np.mean(recent_b):.4e}  "
                f"q_up_mean={norms['q_up_mean']:.4e}  k_up_mean={norms['k_up_mean']:.4e}  "
                f"lr_p={cur_lr_proj:.2e} lr_l={cur_lr_lora:.2e}  "
                f"drop={'Y' if drop_style_this_step else '.'} "
                f"drop_rate={dropout_rate:.2%}  "
                f"t={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )
            loss_log.write(
                f"{step}\t{diff_val:.6f}\t{aux_val:.6f}\t{balance_val:.6e}\t"
                f"{cur_lr_proj:.2e}\t{cur_lr_lora:.2e}\t"
                f"{int(drop_style_this_step)}\t"
                f"{norms['q_up_mean']:.6e}\t{norms['k_up_mean']:.6e}\n"
            )
        else:
            loss_log.write(
                f"{step}\t{diff_val:.6f}\t{aux_val:.6f}\t{balance_val:.6e}\t"
                f"{cur_lr_proj:.2e}\t{cur_lr_lora:.2e}\t"
                f"{int(drop_style_this_step)}\t\t\n"
            )
        loss_log.flush()

        # ── periodic validation ─────────────────────────────────
        if (step + 1) % VAL_EVERY == 0 or step == STEPS - 1:
            style_embedder.eval()
            block_lora.eval()
            val_diff, val_aux = [], []
            val_mean_outputs, val_csd_targets = [], []
            val_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
            with torch.no_grad():
                for ventry in train_data:
                    ref_pil_v = Image.open(
                        os.path.join(DATA_DIR, ventry["name"])
                    ).convert("RGB")
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
                    with lora_injection(transformer, block_lora):
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
            cur_norms = lora_norms(block_lora)
            print(f"  -- val @ step {step}: diff={md:.4f}  aux={ma:.4f}  "
                  f"struct_corr={mc:+.4f}  q_up_mean={cur_norms['q_up_mean']:.4e}  "
                  f"k_up_mean={cur_norms['k_up_mean']:.4e}  (n={len(val_diff)})")
            with open(VAL_LOG_PATH, "w") as fv:
                fv.write("step\tval_diff\tval_aux\tval_struct_corr\tq_up_mean\tk_up_mean\n")
                for s, d, a, c in val_history:
                    fv.write(f"{s}\t{d:.6f}\t{a:.6f}\t{c:+.6f}\t"
                             f"{cur_norms['q_up_mean']:.6e}\t{cur_norms['k_up_mean']:.6e}\n")
            style_embedder.train()
            block_lora.train()

        # ── checkpoint ──────────────────────────────────────────
        if (step + 1) % CKPT_EVERY == 0 or step == STEPS - 1:
            ckpt_path = os.path.join(OUT_DIR, f"{RUN_NAME}_step{step+1:04d}.pt")
            ckpt = {
                "format": "phase5",
                "config": {
                    "lora_rank": LORA_RANK,
                    "num_blocks": block_lora.num_blocks,
                    "dim": block_lora.dim,
                    "use_dual_time": USE_DUAL_TIME,
                    "style_rope_convention": STYLE_ROPE_CONVENTION,
                    "lora_lr_ratio": LORA_LR_RATIO,
                    "lambda_balance": LAMBDA_BALANCE,
                    "lr_decay_floor": LR_DECAY_FLOOR,
                    "has_gates": True,
                },
                "projector": style_embedder.state_dict(),
                "lora": block_lora.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"  -- saved {ckpt_path}")

    loss_log.close()
    final_path = os.path.join(OUT_DIR, f"{RUN_NAME}_final.pt")
    final_ckpt = {
        "format": "phase5",
        "config": {
            "lora_rank": LORA_RANK,
            "num_blocks": block_lora.num_blocks,
            "dim": block_lora.dim,
            "use_dual_time": USE_DUAL_TIME,
            "style_rope_convention": STYLE_ROPE_CONVENTION,
            "lora_lr_ratio": LORA_LR_RATIO,
            "lambda_balance": LAMBDA_BALANCE,
            "lr_decay_floor": LR_DECAY_FLOOR,
            "has_gates": True,
        },
        "projector": style_embedder.state_dict(),
        "lora": block_lora.state_dict(),
    }
    torch.save(final_ckpt, final_path)
    print(f"\nDone. Final: {final_path}")
    print(f"Diff loss: initial={diff_losses[0]:.4f} final={diff_losses[-1]:.4f}")
    print(f"Aux loss:  initial={aux_losses[0]:.4f} final={aux_losses[-1]:.4f}")
    print(f"Dropout:   {n_dropout_steps}/{STEPS} steps ({n_dropout_steps/STEPS:.2%})")
    final_norms = lora_norms(block_lora)
    print(f"LoRA norms (final): q_up_mean={final_norms['q_up_mean']:.4e} "
          f"q_up_min={final_norms['q_up_min']:.4e} q_up_max={final_norms['q_up_max']:.4e}")
    print(f"                    k_up_mean={final_norms['k_up_mean']:.4e} "
          f"k_up_min={final_norms['k_up_min']:.4e} k_up_max={final_norms['k_up_max']:.4e}")
    if val_history:
        print(f"Val diff:        first={val_history[0][1]:.4f} last={val_history[-1][1]:.4f}")
        print(f"Val aux:         first={val_history[0][2]:.4f} last={val_history[-1][2]:.4f}")
        print(f"Val struct_corr: first={val_history[0][3]:+.4f} last={val_history[-1][3]:+.4f}")


if __name__ == "__main__":
    main()

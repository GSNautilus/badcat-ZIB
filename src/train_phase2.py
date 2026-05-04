"""Phase 2: overfit one (style ref, content image) pair.

Goal: prove the projector can route style information through frozen
self-attention. 500 steps on a single pair. If loss converges and
generations carry style at inference time, the architecture has capacity.

Training setup:
- Frozen Z-Image-Turbo (NF4)
- Frozen SigLIP-2-So400m NaFlex (bf16)
- Frozen Flux VAE (bf16)
- Frozen Qwen3 text encoder (bf16)
- Trainable: StyleEmbedder (~5M, bf16)
- Optimizer: AdamW8bit, lr=1e-3
- Loss: MSE(transformer_output, clean_latent - noise)
  (model is trained to predict the *negated* FM velocity because the
  pipeline does `noise_pred = -noise_pred` before scheduler.step)
- Timestep distribution: logit-normal mapped through shift=3.0
"""
from __future__ import annotations

import math
import os
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


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEIGHT = 512
WIDTH = 512
SHIFT = 3.0  # Turbo's scheduler shift
LR = 1e-3
STEPS = int(os.environ.get("STEPS", "500"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2_overfit.pt")
LOSS_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2_loss.txt")


def shifted_t_to_sigma(t: torch.Tensor, shift: float) -> torch.Tensor:
    """FlowMatchEuler shift mapping: u in (0,1) -> sigma."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_training_timestep(batch: int, shift: float, device, dtype):
    """Logit-normal sampled t, then shifted. Returns (sigma, timestep_for_model)."""
    # logit-normal: u = sigmoid(N(0,1))
    u = torch.randn(batch, device=device).sigmoid()
    sigma = shifted_t_to_sigma(u, shift).to(dtype)
    timestep = (sigma * 1000.0).to(dtype)
    return sigma, timestep


def encode_content_image(vae, image: Image.Image, device, dtype):
    """Encode an image to a Z-Image latent."""
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype)  # (1, 3, H, W)
    with torch.no_grad():
        encoded = vae.encode(t).latent_dist.sample()
    # Z-Image / Flux VAE: (latent - shift) * scale
    encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    return encoded  # (1, 16, H/8, W/8)


def encode_siglip(siglip, processor, image: Image.Image, device, dtype):
    """Encode reference image to SigLIP-2 spatial features (sig_H, sig_W, C)."""
    inputs = processor(images=[image], return_tensors="pt").to(device)
    spatial = inputs.spatial_shapes[0]
    sig_H, sig_W = int(spatial[0]), int(spatial[1])
    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state
    C = hidden.shape[-1]
    feats = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, C).to(dtype)
    return feats, sig_H, sig_W


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    print(f"Device: {DEVICE}, dtype: {DTYPE}")

    # ---- Load frozen NF4 transformer ----
    print("Loading transformer (NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=DTYPE,
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="transformer",
        quantization_config=nf4,
        torch_dtype=DTYPE,
    )
    transformer.requires_grad_(False)
    print(f"  VRAM after transformer: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("Loading rest of pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.vae.eval()
    print(f"  VRAM after pipeline: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Train mode + grad checkpointing on transformer (still frozen weights, just for activation memory)
    transformer.train()
    transformer.enable_gradient_checkpointing()

    # ---- Load SigLIP ----
    print("Loading SigLIP-2...")
    siglip = (
        Siglip2VisionModel.from_pretrained("google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE)
        .to(DEVICE)
        .eval()
    )
    siglip.requires_grad_(False)
    processor = Siglip2ImageProcessorFast.from_pretrained("google/siglip2-so400m-patch16-naflex")
    siglip_dim = siglip.config.hidden_size
    print(f"  VRAM after SigLIP: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ---- Trainable adapter ----
    print("Initializing StyleEmbedder...")
    torch.manual_seed(0)
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        DEVICE, dtype=DTYPE
    )
    style_embedder.train()
    n_params = sum(p.numel() for p in style_embedder.parameters())
    print(f"  Trainable params: {n_params/1e6:.2f}M")

    optim = bnb.optim.AdamW8bit(style_embedder.parameters(), lr=LR)

    # ---- Load and encode images ----
    print("Loading images...")
    content_pil = Image.open("data/cat.jpg").convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
    style_pil = Image.open("data/starry_night.jpg").convert("RGB")
    print(f"  Content: {content_pil.size}; Style ref: {style_pil.size}")

    print("Encoding content latent...")
    clean_latent = encode_content_image(pipe.vae, content_pil, DEVICE, DTYPE)
    print(f"  Latent shape: {tuple(clean_latent.shape)}")

    print("Encoding SigLIP features...")
    siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, style_pil, DEVICE, DTYPE)
    print(f"  SigLIP grid: {sig_H}x{sig_W} = {sig_H*sig_W} tokens")

    print("Encoding caption...")
    caption = "a photograph of an orange tabby cat"
    with torch.no_grad():
        cap_feats_list = pipe._encode_prompt(caption, device=DEVICE)
    cap_len_raw = cap_feats_list[0].shape[0]
    cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
    print(f"  Caption: {cap_len_raw} -> padded {cap_len_padded}")

    # Free text encoder — we don't need it again for training
    del pipe.text_encoder
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    print(f"  VRAM after freeing text encoder: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    image_h_patched = HEIGHT // 16  # 8x VAE * 2x patch
    image_w_patched = WIDTH // 16

    # ---- Training loop ----
    print(f"\nTraining for {STEPS} steps, lr={LR}, shift={SHIFT}...")
    losses = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")
    for step in range(STEPS):
        sigma, timestep = sample_training_timestep(batch=1, shift=SHIFT, device=DEVICE, dtype=DTYPE)
        # sigma shape: (1,); reshape for broadcast over latent
        sigma_b = sigma.view(-1, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(clean_latent)
            noisy = sigma_b * noise + (1.0 - sigma_b) * clean_latent
            target = clean_latent - noise  # = -velocity (model output is negated by pipeline)

        # transformer expects list[Tensor] for x with extra frame dim
        x_input = noisy.unsqueeze(2)  # (1, 16, 1, 64, 64)
        x_list = list(x_input.unbind(dim=0))

        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            pred_list = transformer(x_list, timestep, cap_feats_list, return_dict=False)[0]

        pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)  # (1, 16, 64, 64)
        loss = F.mse_loss(pred.float(), target.float())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        loss_log.write(f"{step}\t{loss.item():.6f}\n")
        loss_log.flush()
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            recent = losses[-LOG_EVERY:] if len(losses) >= LOG_EVERY else losses
            print(
                f"step {step:4d}  loss={loss.item():.4f}  avg_recent={np.mean(recent):.4f}  "
                f"sigma={sigma.item():.3f}  elapsed={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )

    loss_log.close()
    torch.save(style_embedder.state_dict(), SAVE_PATH)
    print(f"\nSaved adapter to {SAVE_PATH}")
    print(f"Final loss: {losses[-1]:.4f}, initial: {losses[0]:.4f}")
    print(f"Loss log: {LOSS_LOG_PATH}")


if __name__ == "__main__":
    main()

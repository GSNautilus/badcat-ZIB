"""Phase 2B: small-scale self-supervised training (~22 images).

Per plan §6.1 self-supervised setup:
- Each step picks a random image from the training set
- Augmented version of the image = style reference
- Clean version of the image = denoising target
- Caption from captions.json

The adapter is forced to learn an abstraction because the target varies
across steps. With only 22 images it will overfit, but the question is
whether it learns something resembling style abstraction (vs. memorization
or constant bias).

Sample-time test (overfit_one_aug.py-style): condition on HELD-OUT
references the model never saw during training. If those references
influence outputs in a style-relevant way, the architecture has the
capacity to abstract style. If outputs are unchanged or look like
training-set memorization, we know to pivot.
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
from PIL import Image, ImageEnhance, ImageFilter

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
SHIFT = 3.0
LR = float(os.environ.get("LR", "5e-4"))  # bit lower than Phase 2 (multi-target = harder)
STEPS = int(os.environ.get("STEPS", "500"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2b_ssl.pt")
LOSS_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2b_loss.txt")
TRAIN_DIR = "data/wikimedia_train"
CAPTION_PATH = "data/wikimedia_train/captions.json"
REF_BASE_SIZE = 384


def shifted_t_to_sigma(t, shift):
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_training_timestep(batch, shift, device, dtype):
    u = torch.randn(batch, device=device).sigmoid()
    sigma = shifted_t_to_sigma(u, shift).to(dtype)
    timestep = (sigma * 1000.0).to(dtype)
    return sigma, timestep


def augment_reference(img: Image.Image, rng: random.Random) -> Image.Image:
    """Plan §6.1 augmentation pipeline. Output: REF_BASE_SIZE square."""
    work = img.resize((512, 512), Image.LANCZOS)
    crop_frac = rng.uniform(0.6, 0.8)
    crop_size = int(512 * crop_frac)
    x0 = rng.randint(0, 512 - crop_size)
    y0 = rng.randint(0, 512 - crop_size)
    work = work.crop((x0, y0, x0 + crop_size, y0 + crop_size))
    work = work.resize((REF_BASE_SIZE, REF_BASE_SIZE), Image.LANCZOS)
    if rng.random() < 0.5:
        work = work.transpose(Image.FLIP_LEFT_RIGHT)
    bf = rng.uniform(0.8, 1.2)
    cf = rng.uniform(0.8, 1.2)
    sf = rng.uniform(0.7, 1.3)
    work = ImageEnhance.Brightness(work).enhance(bf)
    work = ImageEnhance.Contrast(work).enhance(cf)
    work = ImageEnhance.Color(work).enhance(sf)
    if rng.random() < 0.3:
        work = work.resize((256, 256), Image.LANCZOS).resize(
            (REF_BASE_SIZE, REF_BASE_SIZE), Image.LANCZOS
        )
    if rng.random() < 0.15:
        work = work.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return work


def encode_content_image(vae, image, device, dtype):
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        encoded = vae.encode(t).latent_dist.sample()
    encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    return encoded


def encode_siglip(siglip, processor, image, device, dtype):
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

    # Load training set
    with open(CAPTION_PATH) as f:
        captions = json.load(f)
    train_files = sorted(captions.keys())
    print(f"Training set: {len(train_files)} images")

    print("Loading transformer (NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    )
    transformer.requires_grad_(False)

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.vae.eval()
    transformer.train()
    transformer.enable_gradient_checkpointing()

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

    print("Initializing StyleEmbedder...")
    torch.manual_seed(0)
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        DEVICE, dtype=DTYPE
    )
    style_embedder.train()
    n_params = sum(p.numel() for p in style_embedder.parameters())
    print(f"  Trainable params: {n_params/1e6:.2f}M")
    optim = bnb.optim.AdamW8bit(style_embedder.parameters(), lr=LR)

    # Pre-compute content latents and caption embeds (small dataset, cache them all)
    print("Pre-computing content latents and caption embeddings...")
    train_data = []  # list of dicts with keys: pil, latent, cap_feats, cap_len_padded
    for fname in train_files:
        path = os.path.join(TRAIN_DIR, fname)
        try:
            img_pil = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            continue
        latent = encode_content_image(pipe.vae, img_pil, DEVICE, DTYPE)
        with torch.no_grad():
            cap_feats_list = pipe._encode_prompt(captions[fname], device=DEVICE)
        cap_feats = cap_feats_list[0]  # (cap_len_raw, 2560)
        cap_len_raw = cap_feats.shape[0]
        cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
        train_data.append({
            "pil": img_pil,
            "latent": latent,
            "cap_feats": cap_feats,
            "cap_len_padded": cap_len_padded,
            "name": fname,
        })
    print(f"  Cached {len(train_data)} entries")

    # Free text encoder
    del pipe.text_encoder
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    print(f"  VRAM after freeing text encoder: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    rng = random.Random(0)
    print(f"\nTraining {STEPS} steps, lr={LR}, shift={SHIFT}, dataset={len(train_data)}...")
    losses = []
    t0 = time.time()
    loss_log = open(LOSS_LOG_PATH, "w")

    for step in range(STEPS):
        # Pick random training example
        entry = train_data[rng.randrange(len(train_data))]

        # Augment its reference, encode SigLIP fresh
        aug_ref = augment_reference(entry["pil"], rng)
        siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, aug_ref, DEVICE, DTYPE)

        sigma, timestep = sample_training_timestep(1, SHIFT, DEVICE, DTYPE)
        sigma_b = sigma.view(-1, 1, 1, 1)

        with torch.no_grad():
            noise = torch.randn_like(entry["latent"])
            noisy = sigma_b * noise + (1.0 - sigma_b) * entry["latent"]
            target = entry["latent"] - noise

        x_input = noisy.unsqueeze(2)
        x_list = list(x_input.unbind(dim=0))
        cap_feats_list = [entry["cap_feats"]]

        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[entry["cap_len_padded"]],
        ):
            pred_list = transformer(x_list, timestep, cap_feats_list, return_dict=False)[0]

        pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)
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
                f"sigma={sigma.item():.3f}  ref={sig_H}x{sig_W}  "
                f"img={entry['name'][:30]}  elapsed={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f}GB"
            )

    loss_log.close()
    torch.save(style_embedder.state_dict(), SAVE_PATH)
    print(f"\nSaved adapter to {SAVE_PATH}")
    print(f"Final loss: {losses[-1]:.4f}, initial: {losses[0]:.4f}")


if __name__ == "__main__":
    main()

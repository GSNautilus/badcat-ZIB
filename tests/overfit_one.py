"""Phase 2 viability test: sample with the trained adapter on novel prompts.

If the trained adapter produces generations that visibly carry the style of
the reference image, the architecture has capacity to learn style routing
through self-attention. If outputs look like generic Z-Image output (no
style influence), the projector alone is insufficient — pivot to Phase 2B
(add refiner) or Phase 1.5 (side network).
"""
from __future__ import annotations

import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, style_injection

from diffusers import BitsAndBytesConfig, ZImagePipeline
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel


DEVICE = "cuda"
DTYPE = torch.bfloat16
HEIGHT = 512
WIDTH = 512
SEED = 42
STEPS = 9
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2_overfit.pt")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2")

PROMPTS = [
    "a photograph of a dog",
    "a futuristic city skyline at night",
    "a still life with fruit on a wooden table",
]


def encode_siglip(siglip, processor, image, device, dtype):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    spatial = inputs.spatial_shapes[0]
    sig_H, sig_W = int(spatial[0]), int(spatial[1])
    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state
    C = hidden.shape[-1]
    feats = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, C).to(dtype)
    return feats, sig_H, sig_W


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading transformer (NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        subfolder="transformer",
        quantization_config=nf4,
        torch_dtype=DTYPE,
    )
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    transformer.eval()

    print("Loading SigLIP-2...")
    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    siglip_dim = siglip.config.hidden_size

    print(f"Loading trained adapter from {CKPT}")
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        DEVICE, dtype=DTYPE
    )
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    style_embedder.load_state_dict(state)
    style_embedder.eval()

    style_pil = Image.open("data/starry_night.jpg").convert("RGB")
    siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, style_pil, DEVICE, DTYPE)
    print(f"  SigLIP grid: {sig_H}x{sig_W}")

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    for prompt in PROMPTS:
        slug = prompt.replace(" ", "_").replace(",", "")[:40]

        # Need cap_len for style injection
        cap_feats_list = pipe._encode_prompt(prompt, device=DEVICE)
        cap_len_raw = cap_feats_list[0].shape[0]
        cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
        print(f"\nPrompt: {prompt!r} (cap_len_padded={cap_len_padded})")

        # Baseline (no adapter)
        print("  Generating baseline...")
        img = pipe(
            prompt, height=HEIGHT, width=WIDTH,
            num_inference_steps=STEPS, guidance_scale=0.0,
            generator=torch.Generator(DEVICE).manual_seed(SEED),
        ).images[0]
        img.save(os.path.join(OUTDIR, f"{slug}__baseline.png"))

        # With trained adapter
        print("  Generating with adapter...")
        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            img = pipe(
                prompt, height=HEIGHT, width=WIDTH,
                num_inference_steps=STEPS, guidance_scale=0.0,
                generator=torch.Generator(DEVICE).manual_seed(SEED),
            ).images[0]
        img.save(os.path.join(OUTDIR, f"{slug}__with_adapter.png"))

    # Style ref preview
    style_pil.resize((512, 512), Image.LANCZOS).save(os.path.join(OUTDIR, "_style_ref.png"))
    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()

"""Phase 2B diagnostic sampling.

Held-out references the adapter never saw during training:
  - starry: Van Gogh's Starry Night (held out by exclusion from training set)
  - cat:    The orange tabby photo (different distribution: photograph)
  - red:    Solid red square (control: should produce ambiguous/weak style if any)

Test grid: 3 refs x 3 prompts = 9 generations, plus 3 baselines (no adapter).

What we're looking for:
  - Outputs across refs differ in STYLE-relevant ways (palette, texture,
    brushiness vs. photographic) → adapter has style-routing capacity
  - Outputs are similar regardless of ref → adapter learned constant bias
  - Outputs are training-image lookalikes → memorization
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
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "phase2b_ssl.pt")
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase2b")

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


def make_red_ref():
    return Image.new("RGB", (384, 384), "red")


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading models...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    )
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", transformer=transformer, torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    transformer.eval()

    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        ).to(DEVICE).eval()
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

    refs_pil = {
        "starry": Image.open("data/starry_night.jpg").convert("RGB"),
        "cat":    Image.open("data/cat.jpg").convert("RGB"),
        "red":    make_red_ref(),
    }
    refs_feats = {}
    for name, img in refs_pil.items():
        feats, h, w = encode_siglip(siglip, processor, img, DEVICE, DTYPE)
        refs_feats[name] = feats
        print(f"  ref={name}: {h}x{w}")

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    print("\nBaselines (no adapter)...")
    for prompt in PROMPTS:
        slug = prompt.replace(" ", "_").replace(",", "")[:40]
        img = pipe(
            prompt, height=HEIGHT, width=WIDTH,
            num_inference_steps=STEPS, guidance_scale=0.0,
            generator=torch.Generator(DEVICE).manual_seed(SEED),
        ).images[0]
        img.save(os.path.join(OUTDIR, f"baseline__{slug}.png"))
        print(f"  baseline {prompt!r} done")

    print("\nWith adapter (3 refs x 3 prompts)...")
    for ref_name, feats in refs_feats.items():
        for prompt in PROMPTS:
            slug = prompt.replace(" ", "_").replace(",", "")[:40]
            cap_feats_list = pipe._encode_prompt(prompt, device=DEVICE)
            cap_len_raw = cap_feats_list[0].shape[0]
            cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
            with style_injection(
                transformer=transformer,
                style_embedder=style_embedder,
                siglip_features=feats,
                image_size=(image_h_patched, image_w_patched),
                cap_lens=[cap_len_padded],
            ):
                img = pipe(
                    prompt, height=HEIGHT, width=WIDTH,
                    num_inference_steps=STEPS, guidance_scale=0.0,
                    generator=torch.Generator(DEVICE).manual_seed(SEED),
                ).images[0]
            img.save(os.path.join(OUTDIR, f"ref-{ref_name}__{slug}.png"))
            print(f"  ref={ref_name}, {prompt!r} done")

    for name, img in refs_pil.items():
        img.resize((256, 256), Image.LANCZOS).save(os.path.join(OUTDIR, f"_ref_{name}.png"))
    print(f"\nDone. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()

"""Phase 1 deliverable: forward-pass wiring test.

Runs Z-Image-Turbo inference with a randomly-initialized style adapter and
3 visually-distinct reference images. The viability test is whether the
generated outputs vary based on the reference image. If yes, signal flows
from SigLIP -> style_embedder -> unified sequence -> attention -> latent
prediction. If outputs are bit-identical across references, wiring is broken.

Note: We use Turbo (9 steps, no CFG) for fast iteration. Phase 5 will switch
to Base (50 steps, CFG 4.0) for actual training.
"""
from __future__ import annotations

import os
import sys

import torch
from PIL import Image, ImageDraw

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, style_injection

from diffusers import ZImagePipeline
from transformers import Siglip2VisionModel, Siglip2ImageProcessorFast


OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "phase1")


def make_test_references() -> dict[str, Image.Image]:
    """Three visually-distinct synthetic references with very different
    SigLIP embeddings. Phase 1 doesn't need artistic refs; we just need
    the SigLIP embeddings to differ enough that signal flow is detectable.
    """
    refs = {}

    # 1. Bold red/yellow stripes
    img = Image.new("RGB", (384, 384), "red")
    draw = ImageDraw.Draw(img)
    for y in range(0, 384, 48):
        draw.rectangle([0, y, 384, y + 24], fill="yellow")
    refs["stripes"] = img

    # 2. Blue-to-green radial gradient
    img = Image.new("RGB", (384, 384))
    pixels = img.load()
    for y in range(384):
        for x in range(384):
            dx, dy = x - 192, y - 192
            d = min(255, int((dx * dx + dy * dy) ** 0.5))
            pixels[x, y] = (0, d, 255 - d)
    refs["radial"] = img

    # 3. High-frequency noise
    g = torch.Generator().manual_seed(7)
    noise = (torch.rand(384, 384, 3, generator=g) * 255).to(torch.uint8).numpy()
    refs["noise"] = Image.fromarray(noise, mode="RGB")

    return refs


@torch.no_grad()
def encode_siglip(siglip, processor, image: Image.Image, device, dtype) -> torch.Tensor:
    """Returns (sig_H, sig_W, hidden) tensor — the (H, W, C) reshaped features
    matching how the omni pipeline prepares siglip embeds.
    """
    inputs = processor(images=[image], return_tensors="pt").to(device)
    spatial_shape = inputs.spatial_shapes[0]  # (H, W)
    sig_H, sig_W = int(spatial_shape[0]), int(spatial_shape[1])
    hidden = siglip(**inputs).last_hidden_state  # (1, N, C)
    C = hidden.shape[-1]
    hidden = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, C).to(dtype)
    return hidden, sig_H, sig_W


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    device = "cuda"
    dtype = torch.bfloat16

    print("Loading Z-Image-Turbo pipeline (bf16)...")
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=dtype)
    pipe.to(device)
    transformer = pipe.transformer
    transformer.eval()

    print("Loading SigLIP-2-So400m NaFlex...")
    siglip = Siglip2VisionModel.from_pretrained(
        "google/siglip2-so400m-patch16-naflex", torch_dtype=dtype
    ).to(device).eval()
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    siglip_dim = siglip.config.hidden_size  # 1152
    print(f"  SigLIP hidden size: {siglip_dim}")

    print("Initializing StyleEmbedder (random)...")
    torch.manual_seed(0)
    style_embedder = StyleEmbedder(in_dim=siglip_dim, out_dim=transformer.config.dim).to(
        device, dtype=dtype
    ).eval()

    refs = make_test_references()
    for name, img in refs.items():
        img.save(os.path.join(OUTDIR, f"ref_{name}.png"))

    prompt = "a photo of a golden retriever in a field"
    height, width = 512, 512
    seed = 42

    # ---- Baseline: no adapter ----
    print(f"\nBaseline (no adapter): {prompt!r}")
    image = pipe(
        prompt, height=height, width=width,
        num_inference_steps=9, guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(seed),
    ).images[0]
    image.save(os.path.join(OUTDIR, "baseline_no_adapter.png"))
    print(f"  Saved baseline.")

    # ---- Each reference, with adapter ----
    # The image patched grid: at 512x512 with VAE 8x downsample + patch 2:
    #   latent: 64x64, patches: 32x32
    image_h_patched = height // 16  # 8x VAE * 2x patch
    image_w_patched = width // 16

    # Caption length (after SEQ_MULTI_OF padding) — we need to know this for RoPE.
    # Encode the prompt once to figure out cap_len.
    with torch.no_grad():
        prompt_embeds_list = pipe._encode_prompt(prompt, device=device)
    cap_len_raw = prompt_embeds_list[0].shape[0]
    cap_len_padded = cap_len_raw + (-cap_len_raw) % 32
    print(f"  Caption length: raw={cap_len_raw}, padded={cap_len_padded}")

    # Pixel diffs to confirm outputs vary by reference
    images_out = {}

    for name, ref_img in refs.items():
        print(f"\nGenerating with reference={name}")
        siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, ref_img, device, dtype)
        print(f"  SigLIP grid: {sig_H}x{sig_W} = {sig_H*sig_W} tokens")

        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            image = pipe(
                prompt, height=height, width=width,
                num_inference_steps=9, guidance_scale=0.0,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]
        out_path = os.path.join(OUTDIR, f"with_ref_{name}.png")
        image.save(out_path)
        images_out[name] = image
        print(f"  Saved {out_path}")

    # Pixel diff sanity check
    print("\n--- Pixel diff sanity check ---")
    import numpy as np
    arrs = {k: np.array(v).astype(np.int32) for k, v in images_out.items()}
    keys = list(arrs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            d = np.abs(arrs[keys[i]] - arrs[keys[j]]).mean()
            print(f"  mean |pixel diff| {keys[i]} vs {keys[j]}: {d:.2f}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

"""Cheap diagnostic: did the trained Phase 2 projector collapse to a constant?

Compare three different reference images by:
  1. Encoding each through SigLIP (baseline distinctness)
  2. Pushing those features through our trained style_embedder
  3. Measuring whether the projected outputs are still distinct

If projected cosine ~ raw cosine: projector preserves distinctions (working).
If projected cosine >> raw cosine: projector flattens / collapses signal.
"""
from __future__ import annotations

import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder

from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel


DEVICE = "cuda"
DTYPE = torch.bfloat16
CKPT = os.environ.get("CKPT", "checkpoints/phase2_overfit.pt")


def main():
    print(f"Loading SigLIP-2 + trained adapter from {CKPT}...")
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
    style_embedder = StyleEmbedder(in_dim=siglip.config.hidden_size, out_dim=3840).to(
        DEVICE, dtype=DTYPE
    )
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    style_embedder.load_state_dict(state)
    style_embedder.eval()

    refs = {
        "starry": Image.open("data/starry_night.jpg").convert("RGB"),
        "cat":    Image.open("data/cat.jpg").convert("RGB"),
        "red":    Image.new("RGB", (384, 384), "red"),
    }

    outputs, raw_means = {}, {}
    print("\n--- Per-reference projector outputs ---")
    for name, img in refs.items():
        inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
        sH, sW = int(inputs.spatial_shapes[0][0]), int(inputs.spatial_shapes[0][1])
        with torch.no_grad():
            hidden = siglip(**inputs).last_hidden_state
            raw = hidden[0, : sH * sW].float()
            flat = hidden[:, : sH * sW].view(sH, sW, -1).reshape(sH * sW, -1).to(DTYPE)
            projected = style_embedder(flat).float()
        outputs[name] = projected
        raw_means[name] = raw.mean(dim=0)
        print(
            f"[{name:6s}] SigLIP {sH:>2d}x{sW:<2d}  "
            f"proj_mean={projected.mean():+.4f}  proj_std={projected.std():.4f}  "
            f"proj_norm={projected.norm():.2f}"
        )

    print("\n--- Pairwise comparisons (mean-token vector) ---")
    names = list(outputs.keys())
    cos_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ma = outputs[names[i]].mean(0)
            mb = outputs[names[j]].mean(0)
            cos_p = torch.nn.functional.cosine_similarity(
                ma.unsqueeze(0), mb.unsqueeze(0)
            ).item()
            cos_r = torch.nn.functional.cosine_similarity(
                raw_means[names[i]].unsqueeze(0), raw_means[names[j]].unsqueeze(0)
            ).item()
            l2_rel = (ma - mb).norm().item() / max(ma.norm().item(), 1e-8)
            cos_pairs.append(cos_p)
            print(f"\n{names[i]} vs {names[j]}:")
            print(f"  raw SigLIP cosine:  {cos_r:+.4f}")
            print(f"  projected cosine:   {cos_p:+.4f}")
            print(f"  projected rel L2:   {l2_rel:.4f}")

    avg = sum(cos_pairs) / len(cos_pairs)
    print(f"\navg projected cosine: {avg:+.4f}")
    if avg > 0.995:
        print("VERDICT: PROJECTOR COLLAPSED — outputs effectively constant")
    elif avg > 0.95:
        print("VERDICT: PROJECTOR DEGRADED — heavily attenuated SigLIP signal")
    else:
        print("VERDICT: PROJECTOR INTACT — SigLIP signal flows through")


if __name__ == "__main__":
    main()

"""Backfill CSD style vectors into the existing Phase 3 precompute cache.

Loads the cache produced by train_phase3.py, encodes each cached image's
raw_path through CSD (yuxi-liu-wired/CSD), adds a `csd_vector` (768-dim
L2-normalized cpu tensor) to each entry, and writes the cache back.

Idempotent: skips entries that already have a csd_vector.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin
from transformers import CLIPProcessor, CLIPVisionModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "phase3_pre_cache.pt"
)


class CSD_CLIP(nn.Module, PyTorchModelHubMixin):
    """Mirrors the upstream CSD class.

    Backbone is raw CLIP ViT-L (no learned projection head); CSD's own
    1024 -> 768 projection matrix produces the L2-normalized style vector.
    """

    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.last_layer_style = nn.Parameter(torch.empty(1024, 768))
        self.last_layer_content = nn.Parameter(torch.empty(1024, 768))

    def forward(self, pixel_values: torch.Tensor):
        out = self.backbone(pixel_values=pixel_values)
        feature = out.pooler_output  # (B, 1024)
        style_output = F.normalize(feature @ self.last_layer_style, dim=1, p=2)
        content_output = F.normalize(feature @ self.last_layer_content, dim=1, p=2)
        return feature, content_output, style_output


def main():
    if not os.path.exists(CACHE_PATH):
        print(f"ERROR: cache not found at {CACHE_PATH}")
        print("Run train_phase3.py first to produce the base cache.")
        sys.exit(1)

    print(f"Loading cache from {CACHE_PATH}...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    if "entries" in cache:
        entries = cache["entries"]
    else:
        # backwards-compat: old format had separate train/val keys
        entries = {}
        for e in cache.get("train", []):
            entries[e["name"]] = e
        for e in cache.get("val", []):
            entries[e["name"]] = e
    print(f"  Loaded {len(entries)} entries")

    needs_csd = [name for name, e in entries.items() if "csd_vector" not in e]
    print(f"  {len(entries) - len(needs_csd)} already have CSD vectors")
    print(f"  {len(needs_csd)} need encoding")

    if not needs_csd:
        print("All entries have CSD vectors. Nothing to do.")
        return

    print("Loading CSD (yuxi-liu-wired/CSD)...")
    model = CSD_CLIP.from_pretrained("yuxi-liu-wired/CSD").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    print(f"  VRAM after CSD load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    t0 = time.time()
    skipped = 0
    with torch.no_grad():
        for i, name in enumerate(needs_csd):
            entry = entries[name]
            try:
                img = Image.open(entry["raw_path"]).convert("RGB")
            except Exception as e:
                print(f"  SKIP {name}: {e}")
                skipped += 1
                continue
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            _, _, style = model(pixel_values)
            entry["csd_vector"] = style.squeeze(0).cpu().float()  # (768,)
            if (i + 1) % 50 == 0 or (i + 1) == len(needs_csd):
                print(
                    f"  {i+1}/{len(needs_csd)}  elapsed={time.time()-t0:.0f}s  "
                    f"skipped={skipped}"
                )

    print(f"\nSaving updated cache to {CACHE_PATH}...")
    torch.save({"entries": entries}, CACHE_PATH)
    print(f"  Done. Total entries: {len(entries)}, with CSD: "
          f"{sum(1 for e in entries.values() if 'csd_vector' in e)}")


if __name__ == "__main__":
    main()

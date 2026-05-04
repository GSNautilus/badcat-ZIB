"""Compare what CSD vs SigLIP-2 see in our synthetic diagnostic refs.

Prints two pairwise cosine-similarity matrices, one per encoder, side by side.
The cells we care most about:
  solid_red  vs  solid_blue       (does the encoder see them as different?)
  red_square vs  green_diamond    (does color/orientation matter?)
  circle_full vs circle_half      (does scale matter?)
  bw_stripes  vs  rainbow_lines   (does texture/color matter?)

Hypothesis from the ComfyUI test: SigLIP says red ≈ blue (semantic invariance),
which is why solid colors don't transfer. CSD should disagree (it was trained
to discriminate styles including palette).

If CSD disagrees with SigLIP on the cells where we expect it to:
  -> Variant A (CSD aux loss) is targeting a real gap, worth running
If CSD agrees with SigLIP (says red ≈ blue too):
  -> CSD won't fix the issue; we need a different teacher signal
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin
from transformers import (
    CLIPProcessor,
    CLIPVisionModel,
    Siglip2ImageProcessorFast,
    Siglip2VisionModel,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # keep this comparison in fp32 for clean numbers
DEFAULT_REFS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diagnostic_refs")
REFS_DIR = os.environ.get("REFS_DIR", DEFAULT_REFS_DIR)
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))   # 0 = no limit
MATRIX_PRINT_LIMIT = 12   # only print full matrix if <= this many images


# ── CSD model definition (mirrors the upstream class) ─────────────
# The HF checkpoint yuxi-liu-wired/CSD ships its own class via
# PyTorchModelHubMixin. We re-declare it here so we don't depend on
# importing from the published checkpoint's custom code path.
class CSD_CLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        # CSD uses raw CLIP ViT-L (no learned projection head).
        # Backbone pooler_output is 1024-dim; CSD's own matrices project to 768.
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.last_layer_style = nn.Parameter(torch.empty(1024, 768))
        self.last_layer_content = nn.Parameter(torch.empty(1024, 768))

    def forward(self, pixel_values: torch.Tensor):
        out = self.backbone(pixel_values=pixel_values)
        feature = out.pooler_output  # (B, 1024)
        style_output = F.normalize(feature @ self.last_layer_style, dim=1, p=2)
        content_output = F.normalize(feature @ self.last_layer_content, dim=1, p=2)
        return feature, content_output, style_output


def load_refs() -> dict[str, Image.Image]:
    """Load reference images from REFS_DIR. Supports png/jpg/jpeg.

    If MAX_IMAGES > 0, takes a deterministic subset of that size.
    Returns {short_name: image}.
    """
    paths = []
    for ext in ("png", "jpg", "jpeg"):
        paths.extend(glob.glob(os.path.join(REFS_DIR, f"*.{ext}")))
    paths = sorted(paths)
    if MAX_IMAGES > 0 and len(paths) > MAX_IMAGES:
        # deterministic stride sample for representative coverage
        step = len(paths) / MAX_IMAGES
        paths = [paths[int(i * step)] for i in range(MAX_IMAGES)]
    refs = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        # truncate very long names for readability
        if len(name) > 28:
            name = name[:28]
        refs[name] = Image.open(path).convert("RGB")
    return refs


def encode_csd(refs: dict[str, Image.Image]) -> dict[str, torch.Tensor]:
    """Encode each ref with CSD; return {name: 768-dim style vector}."""
    print("Loading CSD (yuxi-liu-wired/CSD)...")
    model = CSD_CLIP.from_pretrained("yuxi-liu-wired/CSD").to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    out = {}
    with torch.no_grad():
        for name, img in refs.items():
            inputs = processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            _, _, style = model(pixel_values)
            out[name] = style.squeeze(0).cpu()
    del model
    torch.cuda.empty_cache()
    return out


def encode_siglip(refs: dict[str, Image.Image]) -> dict[str, torch.Tensor]:
    """Encode each ref with SigLIP-2; return {name: mean-pooled token vector}.

    To get one comparable vector per image (like CSD), we mean-pool over the
    spatial token grid. This is what our projector's auxiliary head would
    operate on after the projection.
    """
    print("Loading SigLIP-2 (google/siglip2-so400m-patch16-naflex)...")
    model = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    out = {}
    with torch.no_grad():
        for name, img in refs.items():
            inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
            spatial = inputs.spatial_shapes[0]
            sig_h, sig_w = int(spatial[0]), int(spatial[1])
            hidden = model(**inputs).last_hidden_state  # (1, S, 1152)
            feats = hidden[:, : sig_h * sig_w, :]       # drop pad tokens
            mean_feat = feats.mean(dim=1).squeeze(0)     # (1152,)
            mean_feat = F.normalize(mean_feat, dim=0, p=2)
            out[name] = mean_feat.cpu()
    del model
    torch.cuda.empty_cache()
    return out


def cosine_matrix(vecs: dict[str, torch.Tensor]) -> tuple[list[str], np.ndarray]:
    names = list(vecs.keys())
    M = torch.stack([vecs[n] for n in names], dim=0).float()  # (N, D)
    M = F.normalize(M, dim=1, p=2)
    sim = (M @ M.t()).numpy()
    return names, sim


def print_matrix(title: str, names: list[str], sim: np.ndarray):
    print(f"\n=== {title} ===")
    short = [n[:13] for n in names]
    print(f"{'':14s} " + " ".join(f"{s:>7s}" for s in short))
    for i, n in enumerate(short):
        cells = " ".join(f"{sim[i, j]:7.3f}" for j in range(len(names)))
        print(f"{n:14s} {cells}")


def print_summary(title: str, names: list[str], sim: np.ndarray):
    """Off-diagonal cosine-sim distribution + most/least similar pairs."""
    n = len(names)
    iu = np.triu_indices(n, k=1)
    off = sim[iu]
    print(f"\n=== {title} (off-diagonal: {len(off)} pairs) ===")
    print(f"  min:    {off.min():.3f}")
    print(f"  median: {np.median(off):.3f}")
    print(f"  mean:   {off.mean():.3f}")
    print(f"  max:    {off.max():.3f}")
    print(f"  std:    {off.std():.3f}")
    print(f"  range:  {off.max() - off.min():.3f}  (wider = more discriminative)")

    # 5 most similar pairs (boring — expect baseline)
    pairs = [(off[k], names[iu[0][k]], names[iu[1][k]]) for k in range(len(off))]
    pairs.sort(key=lambda x: x[0], reverse=True)
    print(f"\n  Most similar (top 5):")
    for s, a, b in pairs[:5]:
        print(f"    {s:.3f}  {a:30s}  vs  {b}")
    print(f"\n  Least similar (bottom 5 — these are the style-distant pairs):")
    for s, a, b in pairs[-5:]:
        print(f"    {s:.3f}  {a:30s}  vs  {b}")


def print_key_pairs(title: str, names: list[str], sim: np.ndarray):
    """Pairs we specifically care about, ranked best-to-worst discrimination."""
    pairs_of_interest = [
        ("solid_red", "solid_blue"),
        ("red_square", "green_diamond"),
        ("circle_full", "circle_half"),
        ("bw_stripes", "rainbow_lines"),
        ("solid_red", "red_square"),
        ("rainbow_lines", "solid_red"),
        ("rainbow_lines", "bw_stripes"),
    ]
    print(f"\n--- {title}: key pair similarities ---")
    print(f"{'pair':40s}  cos_sim   interpretation")
    name_to_idx = {n: i for i, n in enumerate(names)}
    for a, b in pairs_of_interest:
        if a in name_to_idx and b in name_to_idx:
            s = sim[name_to_idx[a], name_to_idx[b]]
            interp = "VERY similar" if s > 0.9 else "similar" if s > 0.7 else "different" if s > 0.3 else "VERY different"
            print(f"  {a:18s} vs {b:18s}  {s:6.3f}   ({interp})")


def main():
    print(f"REFS_DIR: {REFS_DIR}")
    refs = load_refs()
    print(f"Loaded {len(refs)} refs")

    csd_vecs = encode_csd(refs)
    siglip_vecs = encode_siglip(refs)

    print(f"\nCSD vector dim:    {csd_vecs[next(iter(csd_vecs))].shape}")
    print(f"SigLIP vector dim: {siglip_vecs[next(iter(siglip_vecs))].shape}")

    siglip_names, siglip_sim = cosine_matrix(siglip_vecs)
    csd_names, csd_sim = cosine_matrix(csd_vecs)

    if len(refs) <= MATRIX_PRINT_LIMIT:
        print_matrix("SigLIP-2 cosine similarity (mean-pooled tokens)", siglip_names, siglip_sim)
        print_matrix("CSD style-vector cosine similarity", csd_names, csd_sim)
    else:
        print(f"\n(matrix too large to print — {len(refs)} > {MATRIX_PRINT_LIMIT}; showing summary only)")

    print_summary("SigLIP-2", siglip_names, siglip_sim)
    print_summary("CSD", csd_names, csd_sim)

    # Side-by-side comparison of discriminativeness
    n = len(siglip_names)
    iu = np.triu_indices(n, k=1)
    sl_off = siglip_sim[iu]
    cs_off = csd_sim[iu]
    print("\n" + "=" * 60)
    print("DISCRIMINATIVENESS COMPARISON (off-diagonal pair sims)")
    print("=" * 60)
    print(f"{'metric':12s}  {'SigLIP':>10s}  {'CSD':>10s}  {'Δ (CSD-SigLIP)':>16s}")
    for label, sl_v, cs_v in [
        ("min",    sl_off.min(),    cs_off.min()),
        ("median", np.median(sl_off), np.median(cs_off)),
        ("mean",   sl_off.mean(),   cs_off.mean()),
        ("max",    sl_off.max(),    cs_off.max()),
        ("std",    sl_off.std(),    cs_off.std()),
        ("range",  sl_off.max() - sl_off.min(), cs_off.max() - cs_off.min()),
    ]:
        delta = cs_v - sl_v
        print(f"{label:12s}  {sl_v:10.3f}  {cs_v:10.3f}  {delta:+16.3f}")
    print()
    print("Interpretation:")
    print(f"  - Lower mean similarity = encoder sees images as more distinct")
    print(f"  - Wider range / higher std = encoder provides richer gradient signal")
    print(f"  - If CSD has wider range than SigLIP, the auxiliary loss has")
    print(f"    more useful pressure to apply during training.")


if __name__ == "__main__":
    main()

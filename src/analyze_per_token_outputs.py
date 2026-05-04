"""Per-token output analysis: compare what each trained adapter actually
produces at the token level for the same SigLIP inputs.

Hypothesis being tested: phase 2b produces per-token outputs that have
real per-token variance / spatial structure (so Z-Image's attention can
extract signal from individual tokens), while phase 3a/b/c produce per-token
outputs that are essentially noise around a mean (so Z-Image's attention
treats them as noise per-token, equivalent to the zero adapter).

We compute, per (adapter, reference image):
  - per-token L2 norm: mean and std across tokens
  - mean-pool magnitude
  - "concentration ratio": mean-pool magnitude / mean per-token magnitude
       high (-> 1) = tokens point in similar directions (collapsed to mean)
       low (-> 1/sqrt(N)) = tokens point in different directions (spread out)
  - mean pairwise cosine similarity between tokens within the image
       high = tokens are near-duplicates (no per-token info)
       low = tokens carry distinct signal per position

Then aggregates across reference images for each adapter and prints a
side-by-side comparison.
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder

from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DIAG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diagnostic_refs")
WIKI_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wikimedia_train")

# Adapters to compare
ADAPTERS = [
    ("zero",        "zero_adapter.pt"),
    ("phase2b",     "phase2b_ssl.pt"),
    ("phase3a",     "phase3a_step3000.pt"),
    ("phase3b",     "phase3b_step3000.pt"),
    ("phase3c",     "phase3c_smoke200_clip100_lambda50_step0200.pt"),
    ("phase3d_100", "phase3d_halton500_step0100.pt"),
    ("phase3d_500", "phase3d_halton500_step0500.pt"),
    ("phase3e_500",  "phase3e_halton_pertoken3000_step0500.pt"),
    ("phase3e_1000", "phase3e_halton_pertoken3000_step1000.pt"),
    ("phase3e_1500", "phase3e_halton_pertoken3000_step1500.pt"),
    ("phase3e_2000", "phase3e_halton_pertoken3000_step2000.pt"),
    ("phase3e_2500", "phase3e_halton_pertoken3000_step2500.pt"),
    ("phase3e_3000", "phase3e_halton_pertoken3000_step3000.pt"),
    ("phase4_250",   "phase4_offset_500_step0250.pt"),
    ("phase4_500",   "phase4_offset_500_step0500.pt"),
    ("p4drop_500",   "phase4_offset_dropout_16000_step0500.pt"),
    ("p4drop_1000",  "phase4_offset_dropout_16000_step1000.pt"),
    ("p4drop_2000",  "phase4_offset_dropout_16000_step2000.pt"),
    ("p4drop_3000",  "phase4_offset_dropout_16000_step3000.pt"),
    ("p4drop_5000",  "phase4_offset_dropout_16000_step5000.pt"),
    ("p4drop_7000",  "phase4_offset_dropout_16000_step7000.pt"),
    ("p4drop_9000",  "phase4_offset_dropout_16000_step9000.pt"),
    ("p4drop_11500", "phase4_offset_dropout_16000_step11500.pt"),
]


def load_adapter(name: str, fname: str) -> StyleEmbedder:
    path = os.path.join(CKPT_DIR, fname)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    embedder = StyleEmbedder(in_dim=1152, out_dim=3840)
    embedder.load_state_dict(sd, strict=True)
    embedder.to(DEVICE, DTYPE).eval()
    for p in embedder.parameters():
        p.requires_grad_(False)
    return embedder


def encode_siglip(siglip, processor, image: Image.Image):
    inputs = processor(images=[image], return_tensors="pt").to(DEVICE)
    spatial = inputs.spatial_shapes[0]
    sig_H, sig_W = int(spatial[0]), int(spatial[1])
    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state
    feats = hidden[0, : sig_H * sig_W, :].to(DTYPE)   # (sig_H*sig_W, 1152)
    return feats, sig_H, sig_W


def per_image_stats(tokens: torch.Tensor) -> dict:
    """tokens: (N, D). Returns per-image statistics."""
    N, D = tokens.shape
    norms = tokens.norm(dim=-1)                      # (N,) per-token L2 norms
    mean_token_norm = norms.mean().item()
    std_token_norm = norms.std().item()

    mean_pool = tokens.mean(dim=0)                   # (D,)
    mean_pool_norm = mean_pool.norm().item()

    # Concentration ratio: mean_pool_norm / mean_token_norm
    # If all tokens identical: ratio = 1.0 (mean = each token)
    # If tokens random orthogonal: ratio ≈ 1/sqrt(N) -> 0
    concentration = mean_pool_norm / max(mean_token_norm, 1e-9)

    # Pairwise cosine similarity between tokens (sample if N large)
    if N > 256:
        idx = torch.randperm(N)[:256]
        sample = tokens[idx]
    else:
        sample = tokens
    normed = F.normalize(sample, dim=-1, p=2)
    pair_sim = (normed @ normed.T)                   # (n, n)
    n = pair_sim.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sim_off = pair_sim.cpu().numpy()[iu]
    mean_pair_cos = float(np.mean(pair_sim_off))

    return {
        "n_tokens": N,
        "mean_token_norm": mean_token_norm,
        "std_token_norm": std_token_norm,
        "mean_pool_norm": mean_pool_norm,
        "concentration": concentration,
        "mean_pair_cos": mean_pair_cos,
    }


def collect_refs() -> list[tuple[str, Image.Image]]:
    refs = []
    # Synthetic diagnostic refs
    for path in sorted(glob.glob(os.path.join(DIAG_DIR, "*.png"))):
        name = "diag_" + os.path.splitext(os.path.basename(path))[0]
        refs.append((name, Image.open(path).convert("RGB")))
    # A few wikimedia paintings (real-world refs)
    paintings = sorted(glob.glob(os.path.join(WIKI_DIR, "*.jpg")))[:5]
    for path in paintings:
        name = "paint_" + os.path.splitext(os.path.basename(path))[0][:20]
        refs.append((name, Image.open(path).convert("RGB")))
    return refs


def main():
    print(f"Device: {DEVICE}, dtype: {DTYPE}")

    # Load adapters
    print(f"\nLoading {len(ADAPTERS)} adapters...")
    embedders = {}
    for name, fname in ADAPTERS:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"  SKIP {name}: {path} not found")
            continue
        embedders[name] = load_adapter(name, fname)
        print(f"  loaded {name}")

    # Load SigLIP
    print("\nLoading SigLIP-2...")
    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        ).to(DEVICE).eval()
    )
    siglip.requires_grad_(False)
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )

    # Collect references
    refs = collect_refs()
    print(f"\nUsing {len(refs)} reference images:")
    for n, _ in refs:
        print(f"  {n}")

    # Encode each ref through each adapter
    # Storage: results[adapter_name] = list of per-image stat dicts
    results = {name: [] for name in embedders}

    print(f"\nEncoding and analyzing...")
    with torch.no_grad():
        for ref_name, ref_img in refs:
            siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, ref_img)
            for name, embedder in embedders.items():
                tokens = embedder(siglip_feats)              # (sig_H*sig_W, 3840)
                stats = per_image_stats(tokens)
                results[name].append(stats)

    # Aggregate and print
    print("\n" + "=" * 80)
    print("AGGREGATED PER-TOKEN STATISTICS (mean across reference images)")
    print("=" * 80)
    print(f"{'adapter':10s}  {'mean_tok_n':>11s}  {'std_tok_n':>10s}  "
          f"{'mean_pool':>10s}  {'concentr':>9s}  {'mean_pair_cos':>14s}")
    print("-" * 80)
    for name in embedders:
        rs = results[name]
        if not rs:
            continue
        mean_tok_n = np.mean([r["mean_token_norm"] for r in rs])
        std_tok_n = np.mean([r["std_token_norm"] for r in rs])
        mean_pool_n = np.mean([r["mean_pool_norm"] for r in rs])
        conc = np.mean([r["concentration"] for r in rs])
        mean_pair = np.mean([r["mean_pair_cos"] for r in rs])
        print(f"{name:10s}  {mean_tok_n:>11.3f}  {std_tok_n:>10.3f}  "
              f"{mean_pool_n:>10.3f}  {conc:>9.3f}  {mean_pair:>14.3f}")

    print()
    print("How to read:")
    print("  mean_tok_n   - average L2 norm of an individual token output")
    print("  std_tok_n    - variance of token L2 norms within an image")
    print("                  (low = all tokens have similar magnitude)")
    print("  mean_pool    - L2 norm of the mean across tokens")
    print("  concentr     - mean_pool / mean_tok_n. Range:")
    print("                  ~1.0 = all tokens collapsed to same direction")
    print("                  ~1/sqrt(N) = tokens point in random orthogonal directions")
    print("                  ~0.5 = tokens partially aligned, moderate spread")
    print("  mean_pair_cos - average cosine sim between any two tokens in the image")
    print("                  high = tokens are near-duplicates, no per-position info")
    print("                  low = tokens carry distinct content per spatial position")
    print()
    print("Hypothesis: phase 2b should have either higher mean_tok_n (stronger signal")
    print("per token) OR lower concentration / lower mean_pair_cos (tokens carry distinct")
    print("information across positions). Either gives Z-Image attention something to")
    print("extract per token. Phase 3a/b/c should look more like the zero adapter on")
    print("at least one of these metrics.")


if __name__ == "__main__":
    main()

"""Compare bridge variants for the auxiliary CSD loss in isolation.

Each bridge is non-trainable and maps mean(StyleEmbedder output) -> 768-dim
predicted CSD vector. We train the StyleEmbedder ALONE (no Z-Image, no
diffusion loss) against each bridge and compare:

  - Train aux loss curve and final value
  - Held-out val aux loss (memorization vs generalization)
  - Structural correlation: does the trained StyleEmbedder produce outputs
    whose pairwise similarity matrix on val data correlates with CSD's
    pairwise similarity matrix? Higher = projector learned to mirror CSD's
    style discrimination structure.

This test isolates "can this bridge force the projector to extract style
features" from "does the resulting projector also work for diffusion." A
bridge that fails this test will definitely fail in the full training; a
bridge that passes still needs the full pipeline test for final answer.

Bridges compared:
  A - Slice: predicted_csd = mean_output[:768]
  B - Random: predicted_csd = mean_output @ R, R fixed random
  C - CSD-spectrum-weighted random: R columns scaled by CSD's singular values
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder

from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32   # keep this in fp32 for clean numerical comparison
CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "phase3_pre_cache.pt"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase3_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
TRANSFORMER_DIM = 3840
SIGLIP_DIM = 1152
CSD_DIM = 768

STEPS = int(os.environ.get("STEPS", "300"))
LR = float(os.environ.get("LR", "1e-3"))   # higher than full training, no warmup needed
VAL_SIZE = 50
SEED = 0


# ── Bridges ───────────────────────────────────────────────────────
class Bridge:
    """Fixed (non-trainable) mapping from mean StyleEmbedder output (3840)
    to predicted CSD vector (768). Stored as a buffer, not a parameter."""
    name: str
    R: torch.Tensor  # fixed matrix or None for slice

    def apply(self, mean_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def num_constrained_dims(self) -> int:
        """How many output dims are 'in' the constraint (involved in the bridge)."""
        raise NotImplementedError


class SliceBridge(Bridge):
    name = "A_slice"
    R = None

    def apply(self, mean_output: torch.Tensor) -> torch.Tensor:
        return mean_output[..., :CSD_DIM]

    @property
    def num_constrained_dims(self) -> int:
        return CSD_DIM   # only first 768 dims


class RandomBridge(Bridge):
    name = "B_random"

    def __init__(self, seed: int = 42):
        g = torch.Generator(device="cpu").manual_seed(seed)
        # Random Gaussian, scale so that mean_output @ R has reasonable magnitude.
        # Use 1/sqrt(in_dim) scaling (Glorot-ish) so the output norm is in scale.
        self.R = torch.randn(TRANSFORMER_DIM, CSD_DIM, generator=g) / np.sqrt(TRANSFORMER_DIM)
        self.R = self.R.to(DEVICE, DTYPE)

    def apply(self, mean_output: torch.Tensor) -> torch.Tensor:
        return mean_output @ self.R

    @property
    def num_constrained_dims(self) -> int:
        return TRANSFORMER_DIM   # all 3840 dims contribute


class CSDSpectrumBridge(Bridge):
    """Random bridge whose columns are scaled by the singular values of the
    CSD vector dataset. Effect: high-variance principal directions of CSD
    space have larger weights, so the projector is more strongly pressured
    along those directions.
    """
    name = "C_csd_spectrum"

    def __init__(self, csd_vectors: torch.Tensor, seed: int = 42):
        # csd_vectors: (N, 768). When N < 768, only N principal directions
        # are well-defined; the rest live in the null space of the data
        # covariance. Use full_matrices=True so V is always (768, 768).
        U, S, Vt = torch.linalg.svd(csd_vectors.float(), full_matrices=True)
        V = Vt.T  # (768, 768) right singular vectors

        # Pad singular values: only N are meaningful (others are 0).
        # Replace the missing ones with a small floor so unobserved directions
        # don't get fully zeroed out (they'd get zero gradient pressure).
        N = S.shape[0]
        S_full = torch.zeros(CSD_DIM, dtype=S.dtype)
        S_full[:N] = S
        floor = S.max() * 0.1  # unobserved directions get 10% of max weight
        S_full[N:] = floor

        S_norm = S_full / S_full.max()  # in [0, 1]

        g = torch.Generator(device="cpu").manual_seed(seed)
        R_base = torch.randn(TRANSFORMER_DIM, CSD_DIM, generator=g) / np.sqrt(TRANSFORMER_DIM)
        # Rotate to CSD principal basis, scale columns by importance
        self.R = (R_base @ V * S_norm.unsqueeze(0)).to(DEVICE, DTYPE)

    def apply(self, mean_output: torch.Tensor) -> torch.Tensor:
        return mean_output @ self.R

    @property
    def num_constrained_dims(self) -> int:
        return TRANSFORMER_DIM


# ── helpers ───────────────────────────────────────────────────────
def encode_siglip_batch(siglip, processor, pil_images: list[Image.Image]) -> list[torch.Tensor]:
    """Encode each PIL image, return list of (sig_H*sig_W, 1152) tensors on CPU."""
    feats_list = []
    with torch.no_grad():
        for img in pil_images:
            inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
            spatial = inputs.spatial_shapes[0]
            sig_H, sig_W = int(spatial[0]), int(spatial[1])
            hidden = siglip(**inputs).last_hidden_state  # (1, S, 1152)
            feats = hidden[0, : sig_H * sig_W, :].to(DTYPE).cpu()  # (sig_H*sig_W, 1152)
            feats_list.append(feats)
    return feats_list


def project_and_pool(style_embedder: StyleEmbedder, siglip_feats: torch.Tensor) -> torch.Tensor:
    """Forward through StyleEmbedder, return mean-pooled (3840,) output."""
    embedded = style_embedder(siglip_feats)  # (sig_H*sig_W, 3840)
    return embedded.mean(dim=0)              # (3840,)


def aux_loss(predicted_csd: torch.Tensor, target_csd: torch.Tensor) -> torch.Tensor:
    """1 - cosine_sim. predicted_csd may not be L2-normalized; we normalize."""
    pred_n = F.normalize(predicted_csd, dim=-1, p=2)
    targ_n = F.normalize(target_csd, dim=-1, p=2)
    return 1.0 - (pred_n * targ_n).sum()


def structural_correlation(
    proj_outputs: torch.Tensor,   # (N, D)
    csd_vectors: torch.Tensor,    # (N, 768)
) -> float:
    """Pearson correlation of pairwise cosine sim matrices."""
    p = F.normalize(proj_outputs.float(), dim=-1, p=2)
    c = F.normalize(csd_vectors.float(), dim=-1, p=2)
    sim_p = (p @ p.T).cpu().numpy()
    sim_c = (c @ c.T).cpu().numpy()
    n = sim_p.shape[0]
    iu = np.triu_indices(n, k=1)   # upper triangle, exclude diagonal
    return float(np.corrcoef(sim_p[iu], sim_c[iu])[0, 1])


# ── main ──────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print(f"Steps per bridge: {STEPS}, LR: {LR}")
    print()

    # Load cache: need raw_paths and csd_vectors
    print("Loading cache...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    entries = cache["entries"]
    with open(CAPTION_PATH) as f:
        captions = json.load(f)
    all_files = sorted(captions.keys())
    rng_split = random.Random(SEED)
    files_shuffled = list(all_files)
    rng_split.shuffle(files_shuffled)
    val_files = files_shuffled[:VAL_SIZE]
    train_files = files_shuffled[VAL_SIZE:]
    train_data = [entries[f] for f in train_files if f in entries]
    val_data = [entries[f] for f in val_files if f in entries]
    print(f"  train={len(train_data)} val={len(val_data)}")

    # Load SigLIP, encode all images once
    print("Loading SigLIP-2 and encoding all images (one-time)...")
    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        ).to(DEVICE).eval()
    )
    siglip.requires_grad_(False)
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    t0 = time.time()
    train_pil = [Image.open(e["raw_path"]).convert("RGB") for e in train_data]
    val_pil = [Image.open(e["raw_path"]).convert("RGB") for e in val_data]
    train_siglip = encode_siglip_batch(siglip, processor, train_pil)  # list of CPU tensors
    val_siglip = encode_siglip_batch(siglip, processor, val_pil)
    train_csd = torch.stack([e["csd_vector"] for e in train_data]).to(DEVICE, DTYPE)
    val_csd = torch.stack([e["csd_vector"] for e in val_data]).to(DEVICE, DTYPE)
    print(f"  Encoded {len(train_siglip)+len(val_siglip)} images in {time.time()-t0:.0f}s")

    # Free SigLIP — we don't need it for training
    del siglip, processor
    torch.cuda.empty_cache()

    # Build the bridges
    print("\nBuilding bridges...")
    bridges = [
        SliceBridge(),
        RandomBridge(seed=42),
        CSDSpectrumBridge(train_csd.cpu(), seed=42),
    ]
    for b in bridges:
        info = f"R shape {tuple(b.R.shape)}" if b.R is not None else "no matrix"
        print(f"  {b.name}: constrains {b.num_constrained_dims}/3840 dims, {info}")

    results = []
    rng = random.Random(SEED)

    for bridge in bridges:
        print(f"\n{'='*60}\nTraining StyleEmbedder against bridge: {bridge.name}\n{'='*60}")
        torch.manual_seed(SEED)
        style_embedder = StyleEmbedder(in_dim=SIGLIP_DIM, out_dim=TRANSFORMER_DIM).to(DEVICE, DTYPE)
        style_embedder.train()
        optim = torch.optim.AdamW(style_embedder.parameters(), lr=LR)

        train_curve = []
        for step in range(STEPS):
            idx = rng.randrange(len(train_siglip))
            sf = train_siglip[idx].to(DEVICE, DTYPE)
            target = train_csd[idx]
            mean_output = project_and_pool(style_embedder, sf)
            predicted = bridge.apply(mean_output)
            loss = aux_loss(predicted, target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_curve.append(loss.item())

        # Final train and val evaluation
        style_embedder.eval()
        with torch.no_grad():
            # full train aux loss (over all train images)
            train_losses = []
            train_outputs = []
            for sf, target in zip(train_siglip, train_csd):
                sf_g = sf.to(DEVICE, DTYPE)
                mo = project_and_pool(style_embedder, sf_g)
                pred = bridge.apply(mo)
                train_losses.append(aux_loss(pred, target).item())
                train_outputs.append(mo.cpu())
            # full val aux loss
            val_losses = []
            val_outputs = []
            for sf, target in zip(val_siglip, val_csd):
                sf_g = sf.to(DEVICE, DTYPE)
                mo = project_and_pool(style_embedder, sf_g)
                pred = bridge.apply(mo)
                val_losses.append(aux_loss(pred, target).item())
                val_outputs.append(mo.cpu())

            train_outputs_t = torch.stack(train_outputs)
            val_outputs_t = torch.stack(val_outputs)
            train_struct_corr = structural_correlation(train_outputs_t, train_csd.cpu())
            val_struct_corr = structural_correlation(val_outputs_t, val_csd.cpu())

        results.append({
            "name": bridge.name,
            "first_loss": train_curve[0],
            "last_loss": train_curve[-1],
            "rolling_last_25": np.mean(train_curve[-25:]),
            "train_aux_full": np.mean(train_losses),
            "val_aux": np.mean(val_losses),
            "train_struct_corr": train_struct_corr,
            "val_struct_corr": val_struct_corr,
            "gen_gap_aux": np.mean(val_losses) - np.mean(train_losses),
        })
        print(f"  first step loss:    {train_curve[0]:.4f}")
        print(f"  last 25 train avg:  {np.mean(train_curve[-25:]):.4f}")
        print(f"  full train aux:     {np.mean(train_losses):.4f}")
        print(f"  full val aux:       {np.mean(val_losses):.4f}")
        print(f"  generalization gap: {np.mean(val_losses) - np.mean(train_losses):+.4f}")
        print(f"  train struct corr:  {train_struct_corr:+.4f}")
        print(f"  val struct corr:    {val_struct_corr:+.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("BRIDGE COMPARISON SUMMARY")
    print("=" * 60)
    cols = [
        ("name", "name", "{:<18s}"),
        ("first_loss", "init", "{:>7.3f}"),
        ("rolling_last_25", "train_end", "{:>9.3f}"),
        ("val_aux", "val", "{:>7.3f}"),
        ("gen_gap_aux", "gen_gap", "{:>+8.3f}"),
        ("train_struct_corr", "train_corr", "{:>+11.3f}"),
        ("val_struct_corr", "val_corr", "{:>+9.3f}"),
    ]
    header = " ".join(f"{label:>{int(fmt.split(':')[1].lstrip('<>+').rstrip('sdf').split('.')[0]) if any(c.isdigit() for c in fmt) else 18}}"
                       for _, label, fmt in cols)
    print(header)
    for r in results:
        row = " ".join(fmt.format(r[k]) for k, _, fmt in cols)
        print(row)

    print()
    print("Interpretation:")
    print("  - train_end / val: lower = projector satisfies aux loss better")
    print("  - gen_gap: large positive means memorization (train good, val bad)")
    print("  - val_corr: how strongly trained projector outputs mirror CSD's")
    print("    pairwise similarity structure on held-out images.")
    print("    > 0.5 = strong style structure recovered")
    print("    ~ 0.0 = projector outputs unrelated to CSD discrimination")
    print("    < 0   = projector outputs anti-correlated with CSD (bad)")


if __name__ == "__main__":
    main()

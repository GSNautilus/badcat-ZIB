"""Print ground-truth weight statistics for every adapter checkpoint.

Run this BEFORE running ComfyUI with the instrumented node — the loader
prints the same stats at load time, and you compare the printed values
against this table to confirm the loader is honestly reading each file
(Test B from node_troubleshooting.md §6).

Output is a single table: one row per checkpoint, columns for the two
load-bearing trainable weights (proj.weight and proj.bias) plus pad_token
and the always-1.0 norm/gates (sanity-check they didn't drift).
"""
from __future__ import annotations

import os
import glob
import torch


CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")


def stats(t: torch.Tensor) -> dict:
    f = t.float()
    return {
        "shape": tuple(t.shape),
        "norm": f.norm().item(),
        "mean": f.mean().item(),
        "std":  f.std().item(),
        "min":  f.min().item(),
        "max":  f.max().item(),
    }


# Order matters for readability — phase 2b first as known-working reference
PRIORITY = [
    "zero_adapter.pt",
    "random_adapter.pt",
    "phase2b_ssl.pt",
    "phase3a_step3000.pt",
    "phase3b_step3000.pt",
    "phase3c_smoke200_clip100_lambda50_step0200.pt",
]


def main():
    files_present = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CKPT_DIR, "*.pt")))
    # Show priority files first, then anything else (intermediate steps etc.)
    ordered = [f for f in PRIORITY if f in files_present]
    leftover = [f for f in files_present if f not in ordered and not f.startswith("phase3_pre_cache")]
    ordered += sorted(leftover)

    print(f"Inspecting {len(ordered)} checkpoint(s) in {CKPT_DIR}")
    print()

    # Header
    cols = ["checkpoint", "key", "shape", "norm", "mean", "std", "min", "max"]
    widths = [44, 12, 16, 11, 10, 10, 10, 10]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for fname in ordered:
        path = os.path.join(CKPT_DIR, fname)
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(sd, dict):
            print(f"{fname:44s}  (not a state dict, skipping)")
            continue
        # Only print the keys we care about for adapter checkpoints
        keys = [k for k in ("proj.weight", "proj.bias", "pad_token", "norm.weight", "gates") if k in sd]
        if not keys:
            # Likely a training artifact (.pt of cache or final-state dict with optimizer); skip
            continue
        for i, k in enumerate(keys):
            s = stats(sd[k])
            row = [
                fname if i == 0 else "",
                k,
                str(s["shape"]),
                f"{s['norm']:.4f}",
                f"{s['mean']:+.4f}",
                f"{s['std']:.4f}",
                f"{s['min']:+.4f}",
                f"{s['max']:+.4f}",
            ]
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
        print()


if __name__ == "__main__":
    main()

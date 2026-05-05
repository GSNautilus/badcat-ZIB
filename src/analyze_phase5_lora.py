"""Analyze phase 5 LoRA checkpoints to assess training health.

Two modes:

1. SINGLE checkpoint: per-block weight statistics. For each of the 30 blocks,
   compute the LoRA up/down matrix norms and the effective delta matrix
   norm (||q_up @ q_down||_F) which is the actual size of the modification
   the LoRA is making to the trunk's to_q / to_k projections. Tells us:
     - whether all 30 blocks are training (no dead/lagging blocks)
     - whether early blocks have larger LoRA than late blocks (or vice versa)
     - the absolute scale of the modification vs typical Linear weight norms
       (~0.5-1.5 for a Linear with ~3840 in/out dims at fan_in init)

2. TRAJECTORY across steps: pass a glob like "phase5_lora_16000_step*.pt"
   and the script plots the per-block effective delta over training steps.
   Reveals:
     - are all blocks growing at similar rates?
     - which blocks are stuck near zero (sign of dead block)?
     - is growth saturating?

Cheap to run (operates on per-block matrices, no GPU, no trunk needed).
Just point at one checkpoint or a glob of checkpoints.

Usage:

    python -m src.analyze_phase5_lora checkpoints/phase5_smoke_step0050.pt
    python -m src.analyze_phase5_lora "checkpoints/phase5_lora_16000_step*.pt"

The second form auto-sorts by step number so the printed trajectory is in
training order regardless of how the shell expanded the glob.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import torch


def load_lora_state(path: str) -> tuple[dict, dict]:
    """Load a phase 5 checkpoint, return (lora_state_dict, config)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != "phase5":
        raise ValueError(
            f"{path} is not a phase 5 checkpoint (no 'format'='phase5' key). "
            f"This script only analyzes phase 5 LoRA-format checkpoints."
        )
    return state["lora"], state.get("config", {})


def per_block_stats(lora_state: dict, num_blocks: int) -> list[dict]:
    """For each block, compute the LoRA matrix norms and the effective delta
    norm (size of the modification to to_q / to_k).

    For phase 5 checkpoints with per-block gates (post-fix), the effective
    delta is `gate * ||q_up @ q_down||_F` — the actual magnitude added to
    the trunk's to_q. For older phase 5 checkpoints without gates, the gate
    is implicitly 1.0 and the effective delta is just `||q_up @ q_down||_F`.
    """
    rows = []
    for i in range(num_blocks):
        q_down = lora_state[f"layers.{i}.q_down.weight"].float()
        q_up = lora_state[f"layers.{i}.q_up.weight"].float()
        k_down = lora_state[f"layers.{i}.k_down.weight"].float()
        k_up = lora_state[f"layers.{i}.k_up.weight"].float()
        # Per-block gate (added post-fix). Default to 1.0 if missing for
        # backward compat with older phase 5 checkpoints.
        gate_key = f"layers.{i}.gate"
        if gate_key in lora_state:
            gate_val = float(lora_state[gate_key].float().squeeze())
        else:
            gate_val = 1.0

        q_up_n = float(q_up.norm())
        k_up_n = float(k_up.norm())
        q_down_n = float(q_down.norm())
        k_down_n = float(k_down.norm())

        # Effective delta now includes the gate. This is what the
        # variance_reg_loss in BlockLoRAStack actually penalizes.
        q_effective = abs(gate_val) * float((q_up @ q_down).norm())
        k_effective = abs(gate_val) * float((k_up @ k_down).norm())

        rows.append({
            "block": i,
            "gate": gate_val,
            "q_up_norm": q_up_n,
            "k_up_norm": k_up_n,
            "q_down_norm": q_down_n,
            "k_down_norm": k_down_n,
            "q_effective": q_effective,
            "k_effective": k_effective,
        })
    return rows


def print_single_checkpoint(path: str) -> None:
    """Single-checkpoint mode: print per-block stats + summary."""
    print(f"\n=== {os.path.basename(path)} ===\n")
    lora_state, config = load_lora_state(path)
    num_blocks = config.get("num_blocks", 30)
    rank = config.get("lora_rank", "?")
    dim = config.get("dim", "?")
    print(f"num_blocks={num_blocks}  rank={rank}  dim={dim}")

    rows = per_block_stats(lora_state, num_blocks)
    has_gates = any(f"layers.{i}.gate" in lora_state for i in range(num_blocks))
    if has_gates:
        print(f"per-block gates: present (variance_reg fix active)")
    else:
        print(f"per-block gates: absent (pre-fix checkpoint, gates implicit 1.0)")

    print()
    print(f"{'blk':>3} | {'gate':>6} | {'q_up':>9} {'k_up':>9} | "
          f"{'q_down':>9} {'k_down':>9} | "
          f"{'q_eff':>9} {'k_eff':>9}")
    print("-" * 90)
    for r in rows:
        print(f"{r['block']:>3} | {r['gate']:>6.3f} | "
              f"{r['q_up_norm']:>9.4f} {r['k_up_norm']:>9.4f} | "
              f"{r['q_down_norm']:>9.4f} {r['k_down_norm']:>9.4f} | "
              f"{r['q_effective']:>9.4f} {r['k_effective']:>9.4f}")

    q_eff = np.array([r["q_effective"] for r in rows])
    k_eff = np.array([r["k_effective"] for r in rows])
    print()
    print("Summary across blocks:")
    print(f"  q_effective: mean={q_eff.mean():.4f}  "
          f"std={q_eff.std():.4f}  min={q_eff.min():.4f}  "
          f"max={q_eff.max():.4f}  spread={q_eff.max()/(q_eff.min()+1e-9):.2f}x")
    print(f"  k_effective: mean={k_eff.mean():.4f}  "
          f"std={k_eff.std():.4f}  min={k_eff.min():.4f}  "
          f"max={k_eff.max():.4f}  spread={k_eff.max()/(k_eff.min()+1e-9):.2f}x")

    # Read: if max/min spread > ~3x, suggests imbalanced training across
    # blocks (some blocks training much faster than others). If spread is
    # close to 1.0, all blocks trained ~uniformly.
    print()
    if q_eff.min() < 1e-3:
        dead = [i for i, r in enumerate(rows) if r["q_effective"] < 1e-3]
        print(f"  [!] Possibly dead/inactive Q-blocks (q_effective < 1e-3): {dead}")
    if k_eff.min() < 1e-3:
        dead = [i for i, r in enumerate(rows) if r["k_effective"] < 1e-3]
        print(f"  [!] Possibly dead/inactive K-blocks (k_effective < 1e-3): {dead}")

    # Block-position bias indicator: are early blocks (0-2) carrying more
    # LoRA weight than late blocks? Phase 4 finding suggested early blocks
    # were where the projector concentrated; if phase 5's LoRA is also
    # heavily concentrated in early blocks, the architectural change may
    # not have shifted the distribution as hoped.
    if num_blocks >= 30:
        early_q = q_eff[:5].mean()
        late_q = q_eff[-5:].mean()
        early_k = k_eff[:5].mean()
        late_k = k_eff[-5:].mean()
        print()
        print("Early/late block distribution (mean of first 5 vs last 5):")
        print(f"  Q: early={early_q:.4f}  late={late_q:.4f}  ratio={early_q/(late_q+1e-9):.2f}")
        print(f"  K: early={early_k:.4f}  late={late_k:.4f}  ratio={early_k/(late_k+1e-9):.2f}")
        if max(early_q / (late_q + 1e-9), early_k / (late_k + 1e-9)) > 2.0:
            print("  -> Early-block-heavy: LoRA is concentrating in first blocks "
                  "(consistent with phase 4 pathology returning).")
        elif max(late_q / (early_q + 1e-9), late_k / (early_k + 1e-9)) > 2.0:
            print("  -> Late-block-heavy: LoRA is concentrating in last blocks. "
                  "(Surprising; worth investigating — may indicate a different "
                  "training pathology.)")
        else:
            print("  -> Balanced: LoRA distributes across early/late blocks "
                  "roughly evenly. This is the OminiControl-style outcome we "
                  "want to see.")


def step_from_path(path: str) -> int:
    """Extract training step from filenames like 'phase5_lora_16000_step0500.pt'.
    Returns -1 for files without a step suffix (sorted last)."""
    m = re.search(r"step(\d+)\.pt$", path)
    if m:
        return int(m.group(1))
    if path.endswith("_final.pt"):
        return 10**9  # sort final to the end
    return -1


def print_trajectory(paths: list[str]) -> None:
    """Trajectory mode: aggregate per-block stats over multiple checkpoints
    sorted by step number. Prints summary metrics per step + per-block
    trajectory for a few representative blocks."""
    paths = sorted(paths, key=step_from_path)
    print(f"\n=== Trajectory across {len(paths)} checkpoints ===\n")

    step_summary = []
    block_q_eff_history = None  # (num_steps, num_blocks)
    block_k_eff_history = None

    for path in paths:
        step = step_from_path(path)
        try:
            lora_state, config = load_lora_state(path)
        except ValueError as e:
            print(f"  skip {path}: {e}")
            continue
        num_blocks = config.get("num_blocks", 30)
        rows = per_block_stats(lora_state, num_blocks)
        q_eff = np.array([r["q_effective"] for r in rows])
        k_eff = np.array([r["k_effective"] for r in rows])

        if block_q_eff_history is None:
            block_q_eff_history = []
            block_k_eff_history = []
        block_q_eff_history.append(q_eff)
        block_k_eff_history.append(k_eff)

        step_summary.append({
            "step": step,
            "name": os.path.basename(path),
            "q_eff_mean": float(q_eff.mean()),
            "q_eff_min": float(q_eff.min()),
            "q_eff_max": float(q_eff.max()),
            "k_eff_mean": float(k_eff.mean()),
            "k_eff_min": float(k_eff.min()),
            "k_eff_max": float(k_eff.max()),
        })

    # Summary table
    print(f"{'step':>7} | {'q_eff':>9} {'q_min':>8} {'q_max':>8} {'q_spread':>9} | "
          f"{'k_eff':>9} {'k_min':>8} {'k_max':>8} {'k_spread':>9}")
    print("-" * 92)
    for s in step_summary:
        spread_q = s["q_eff_max"] / (s["q_eff_min"] + 1e-9)
        spread_k = s["k_eff_max"] / (s["k_eff_min"] + 1e-9)
        step_label = f"{s['step']}" if s['step'] != 10**9 else "final"
        print(f"{step_label:>7} | "
              f"{s['q_eff_mean']:>9.4f} {s['q_eff_min']:>8.4f} {s['q_eff_max']:>8.4f} "
              f"{spread_q:>8.2f}x | "
              f"{s['k_eff_mean']:>9.4f} {s['k_eff_min']:>8.4f} {s['k_eff_max']:>8.4f} "
              f"{spread_k:>8.2f}x")

    # Per-block growth trajectory at sampled blocks
    if block_q_eff_history is not None and len(block_q_eff_history) >= 2:
        block_q_eff = np.array(block_q_eff_history)  # (steps, blocks)
        n_blocks = block_q_eff.shape[1]
        sample_blocks = [0, 1, 2, 3, n_blocks // 4, n_blocks // 2,
                          3 * n_blocks // 4, n_blocks - 3, n_blocks - 1]
        sample_blocks = sorted(set(b for b in sample_blocks if 0 <= b < n_blocks))
        print()
        print("Per-block q_effective trajectory (sampled blocks):")
        header = "  step   " + " ".join(f"blk{b:>3}" for b in sample_blocks)
        print(header)
        for i, s in enumerate(step_summary):
            step_label = f"{s['step']}" if s['step'] != 10**9 else "final"
            row = f"  {step_label:>5}  " + " ".join(
                f"{block_q_eff[i, b]:>6.3f}" for b in sample_blocks
            )
            print(row)

        # Final-checkpoint balance check
        print()
        print("Block balance at final checkpoint:")
        last = block_q_eff[-1]
        if n_blocks >= 30:
            print(f"  q_eff first 5 mean: {last[:5].mean():.4f}")
            print(f"  q_eff middle 5 mean: {last[n_blocks//2-2:n_blocks//2+3].mean():.4f}")
            print(f"  q_eff last 5 mean:  {last[-5:].mean():.4f}")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1

    pattern = argv[1]
    if any(c in pattern for c in "*?["):
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"No files matched: {pattern}")
            return 1
        if len(paths) == 1:
            print_single_checkpoint(paths[0])
        else:
            print_trajectory(paths)
    else:
        if not os.path.exists(pattern):
            print(f"File not found: {pattern}")
            return 1
        print_single_checkpoint(pattern)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

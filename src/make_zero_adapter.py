"""Create a deliberately all-zero StyleEmbedder checkpoint.

Purpose: a diagnostic test for the ComfyUI node. With this adapter loaded,
the projector output is exactly zero for every spatial position, and the
pad_token is also zero, so the style tokens injected into Z-Image's
unified sequence are all zero vectors at their assigned RoPE positions.

If grid still appears on Base when this is loaded, the grid is being
caused by the act of injecting tokens at certain positions (regardless
of what those tokens contain). If grid disappears, the grid is being
caused by the actual content/magnitude of the projector's output.

State-dict keys + shapes match exactly what BadcatLoadZImageStyleAdapter
expects — drops cleanly into the existing ComfyUI loader.
"""
from __future__ import annotations

import os
import torch


CSD_DIM = 1152            # SigLIP feature dim (norm.weight uses this)
TRANSFORMER_DIM = 3840    # Z-Image hidden dim
NUM_BLOCKS = 30           # gates length

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "zero_adapter.pt"
)


def main():
    state_dict = {
        "norm.weight": torch.ones(CSD_DIM, dtype=torch.bfloat16),       # RMSNorm scale = 1.0 (default)
        "proj.weight": torch.zeros(TRANSFORMER_DIM, CSD_DIM, dtype=torch.bfloat16),
        "proj.bias":   torch.zeros(TRANSFORMER_DIM, dtype=torch.bfloat16),
        "pad_token":   torch.zeros(1, TRANSFORMER_DIM, dtype=torch.bfloat16),
        "gates":       torch.ones(NUM_BLOCKS, dtype=torch.bfloat16),    # gates not used at inference but loader expects them
    }

    print(f"Creating zero adapter:")
    for k, v in state_dict.items():
        sh = str(tuple(v.shape))
        print(f"  {k:14s}  shape={sh:20s}  sum={v.float().sum().item():.4f}")

    torch.save(state_dict, OUT_PATH)
    print(f"\nSaved to: {OUT_PATH}")
    print()
    print("Properties of this adapter:")
    print("  - norm.weight = 1.0  (RMSNorm passes input through scale-wise)")
    print("  - proj.weight = 0,  proj.bias = 0  -> projector output is exactly")
    print("    zero for ALL spatial positions, regardless of SigLIP input")
    print("  - pad_token = 0  -> padding style tokens are also zero")
    print("  - gates = 1.0  (loader requires this key; unused at inference)")
    print()
    print("Net effect at inference: the unified sequence has style tokens")
    print("at the usual RoPE positions, but every style token is the zero vector.")
    print("Z-Image's self-attention attends to/from these zero tokens normally,")
    print("but they contribute zero value and zero key signal.")


if __name__ == "__main__":
    main()

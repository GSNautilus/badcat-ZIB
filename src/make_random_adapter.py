"""Create a freshly-random-initialized StyleEmbedder checkpoint.

Purpose: Test E from node_troubleshooting.md §6 — a direct test of whether
the inference path actually consumes the trained projector weights, or
whether it produces ~the same output regardless of what proj.weight contains.

If random_adapter.pt produces visually-identical output to phase3a_step3000.pt
in ComfyUI, that means trained weights are not surviving the inference path
(or are too close to random for the differences to matter at attention).

If random_adapter.pt produces visually-different output to phase3a_step3000.pt,
then trained weights ARE consumed and the bug is upstream of attention —
i.e., the trained weights themselves are the issue (consistent with the
per-token structure diagnosis in next_steps_briefing.md §4).

State-dict keys + shapes match BadcatLoadZImageStyleAdapter expectations.
The init mirrors what nn.Linear / nn.Parameter use by default in adapter.py.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn

from src.adapter import StyleEmbedder


SEED = 0xBADCA7
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "random_adapter.pt"
)


def main():
    torch.manual_seed(SEED)
    embedder = StyleEmbedder(in_dim=1152, out_dim=3840)

    # Default init from StyleEmbedder.__init__:
    #   norm.weight  = torch.ones(1152)            (RMSNorm default)
    #   proj.weight  = nn.Linear default kaiming_uniform_(a=sqrt(5))
    #   proj.bias    = nn.Linear default uniform_(-1/sqrt(in), 1/sqrt(in))
    #   pad_token    = N(0, 0.02^2)
    #   gates        = torch.ones(30)
    # This matches the at-init state of phase3a/b/c BEFORE training started.

    state_dict = {k: v.detach().clone().to(torch.bfloat16) for k, v in embedder.state_dict().items()}

    print("Creating random adapter (untrained init):")
    for k, v in state_dict.items():
        sh = str(tuple(v.shape))
        f = v.float()
        print(f"  {k:14s}  shape={sh:20s}  norm={f.norm().item():.4f}  "
              f"mean={f.mean().item():+.4f}  std={f.std().item():.4f}  "
              f"min={f.min().item():+.4f}  max={f.max().item():+.4f}")

    torch.save(state_dict, OUT_PATH)
    print(f"\nSaved to: {OUT_PATH}")
    print(f"Seed: {hex(SEED)}")


if __name__ == "__main__":
    main()

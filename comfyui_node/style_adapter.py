"""StyleEmbedder + LoRA modules — same architecture and state_dict shape as
src/adapter.py, but without the diffusers-coupled context managers (the
ComfyUI integration patches the trunk's modules directly; see nodes.py).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn

# Sequence padding multiple from the diffusers Z-Image transformer (line 33).
# ComfyUI's NextDiT uses `pad_tokens_multiple` per-instance, which we read off
# the loaded model rather than hard-coding here. Kept for parity with src/.
SEQ_MULTI_OF = 32


class RMSNorm(nn.Module):
    """Compatible with diffusers.models.normalization.RMSNorm (elementwise_affine=True).
    Same `weight` parameter name and shape, so checkpoints saved against the
    diffusers RMSNorm load cleanly here.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_in = x.dtype
        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dtype_in) * self.weight


class StyleEmbedder(nn.Module):
    def __init__(self, in_dim: int = 1152, out_dim: int = 3840, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(in_dim, eps=eps)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        self.pad_token = nn.Parameter(torch.zeros(1, out_dim))
        nn.init.normal_(self.pad_token, std=0.02)
        self.num_blocks = 30
        self.gates = nn.Parameter(torch.ones(self.num_blocks))

    def forward(self, siglip_features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(siglip_features))


# =============================================================================
# Phase 5: per-block LoRA on Q, K projections of the trunk's main attention.
# =============================================================================
# Mirrors src/adapter.py's QKLoRAPair / BlockLoRAStack so phase 5 checkpoints
# can be loaded into the inference-side classes with the same state_dict
# layout. The forward-time application differs from training:
#
# - Training (diffusers): replaces the attention processor; the LoRA delta is
#   added after to_q(x) / to_k(x), since diffusers uses separate Q/K/V Linears.
# - Inference (ComfyUI's NextDiT): the attention class uses a fused QKV
#   projection (`attn.qkv(x)` then split). The inference patch wraps that
#   fused module so the genuine output is computed first, then LoRA deltas
#   are added to the Q and K slices in-place — see nodes.py:_install_lora_hooks.
#
# Both paths produce identical math: query := to_q(x) + q_up(q_down(x)),
# key := to_k(x) + k_up(k_down(x)).

class QKLoRAPair(nn.Module):
    """Per-block LoRA on Q and K projections.

    Up matrices are zero-initialized so untrained LoRAs are exact no-ops
    (loading a fresh-init pair makes attention identical to the frozen trunk).
    Trained values from a phase 5 checkpoint encode the deltas the model
    learned during training.

    Per-block scalar `gate` is a learnable multiplier (init 1.0). Mirrors the
    training-side definition in src/adapter.py — present so phase 5
    checkpoints with trained gates load correctly. Older phase 5 checkpoints
    (pre-gate version) load with gates defaulting to 1.0 via load_state_dict
    strict=False handling.
    """

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        self.k_down = nn.Linear(dim, rank, bias=False)
        self.k_up = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Parameter(torch.ones(1))
        nn.init.kaiming_uniform_(self.q_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.k_up.weight)

    def q_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * self.q_up(self.q_down(x))

    def k_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * self.k_up(self.k_down(x))


class BlockLoRAStack(nn.Module):
    """ModuleList of QKLoRAPair, one per main transformer block."""

    def __init__(self, num_blocks: int = 30, dim: int = 3840, rank: int = 32):
        super().__init__()
        self.num_blocks = num_blocks
        self.dim = dim
        self.rank = rank
        self.layers = nn.ModuleList(
            [QKLoRAPair(dim=dim, rank=rank) for _ in range(num_blocks)]
        )

    def __getitem__(self, idx: int) -> QKLoRAPair:
        return self.layers[idx]


def detect_checkpoint_format(state) -> str:
    """Return 'phase5' if state has the {format, projector, lora} dict layout,
    else 'phase4' (flat StyleEmbedder state_dict). Used by the loader node to
    decide whether to instantiate a BlockLoRAStack alongside the projector."""
    if isinstance(state, dict) and state.get("format") == "phase5":
        return "phase5"
    return "phase4"

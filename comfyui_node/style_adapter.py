"""StyleEmbedder — same architecture and state_dict shape as src/adapter.py,
but without the diffusers-coupled style_injection context manager (the
ComfyUI integration patches `patchify_and_embed` directly; see nodes.py).
"""
from __future__ import annotations

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

"""Z-Image Style Adapter: style-token injection into the unified sequence.

Phase 1: forward-pass wiring only. No training. The style_embedder is randomly
initialized; the goal is to confirm signal flows from a reference image through
self-attention to the latent prediction.

Approach:
- Stay in basic mode (sequence order [x, cap]).
- Append style tokens at the end: [x, cap, style].
- Style RoPE: temporal coord = cap_len + 2 (one past the image's temporal slot);
  spatial coords rescaled from SigLIP's grid to match the image's patched grid.
- Inject by monkey-patching _build_unified_sequence on a per-call basis via
  a context manager.
"""
from __future__ import annotations

import contextlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import (
    SEQ_MULTI_OF,
    ZSingleStreamAttnProcessor,
)


# RoPE convention for style tokens. See diagnosis in
# C:/Users/Nautilus/.claude/projects/C--Users-Nautilus-ZImage-IPAdapter/memory/project_grid_diagnosis.md
#
# "rescaled" — original convention: style positions rescaled to land exactly on
#   image grid coords. Causes the periodic grid artifact on Base normal trunk
#   because regular alignment compounds across denoising steps. Phase 2b/3a/3b/3c
#   were trained against this. DO NOT USE for Base training — diffusion gradient
#   will be dominated by "suppress your own grid contribution" instead of
#   style discrimination, producing inert per-token outputs.
#
# "halton" — quasi-random positions from a 2D Halton sequence (bases 2 and 3)
#   scaled to the image grid. Phase 3e (Halton + per-token aux, 3000 steps) drove
#   per-token statistics monotonically toward phase 2b's targets but visually
#   produced uniformly-distributed splotches instead of style — splotches are
#   inherent to Halton's irregular geometry. Superseded by "offset" for Base
#   training as of 2026-05-03.
#
# "offset" — Lumina-Accessory recipe: condition tokens placed at the condition's
#   own row/col + the *target* image's grid size, on the same temporal slot as
#   the image. With image_h_patched = 32 (512px target) and a 24x24 SigLIP grid,
#   condition tokens land at (32..55, 32..55) — past the image grid, no spatial
#   alignment with image positions. This is the official position-encoding
#   recipe for non-spatially-aligned conditioning in NextDiT-class models
#   (Alpha-VLLM/Lumina-Accessory; OminiControl on FLUX). Pairs with dual-time
#   AdaLN (style tokens at t=1, image+cap at diffusion t) — see style_injection.
STYLE_ROPE_CONVENTION = "offset"


def _halton(i: int, base: int) -> float:
    """Radical inverse of i in the given base (i-th Halton index).
    Mirror of the inference-side helper in comfyui_node/nodes.py — keep
    these two implementations identical so training-time and inference-time
    style positions match exactly. Index i should start at 1."""
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


class StyleAuxHead(nn.Module):
    """Auxiliary head that predicts a CSD-like style vector from the
    StyleEmbedder's mean-pooled output.

    Used at training time only — its purpose is to push gradient pressure
    through the StyleEmbedder that says "your output should encode style
    information that a pretrained style descriptor (CSD) would recognize."
    Discarded at inference time (not part of the saved StyleEmbedder weights).

    Deliberately small (~1.18M params) so the head can't simply "compensate"
    for a non-style-aware StyleEmbedder. The pressure is forced onto the
    StyleEmbedder itself.
    """

    def __init__(self, in_dim: int = 3840, hidden_dim: int = 256, out_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (in_dim,) or (B, in_dim) — mean-pooled StyleEmbedder output.

        Returns L2-normalized prediction matching CSD's output geometry.
        """
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return F.normalize(out, dim=-1, p=2)


class StyleEmbedder(nn.Module):
    """Mirrors the omni siglip_embedder: RMSNorm + Linear projection."""

    def __init__(self, in_dim: int = 1152, out_dim: int = 3840, eps: float = 1e-5):
        super().__init__()
        self.norm = RMSNorm(in_dim, eps=eps)
        self.proj = nn.Linear(in_dim, out_dim, bias=True)
        # Learned pad token for style sequence padding to SEQ_MULTI_OF.
        self.pad_token = nn.Parameter(torch.zeros(1, out_dim))
        nn.init.normal_(self.pad_token, std=0.02)
        # Per-block zero-init scalar gates (one per main transformer block).
        # Used during training; for Phase 1 we leave gates at default 1.0 to
        # ensure signal can actually reach the model (zero-init would mean no
        # signal flow, defeating the test).
        self.num_blocks = 30
        self.gates = nn.Parameter(torch.ones(self.num_blocks))

    def forward(self, siglip_features: torch.Tensor) -> torch.Tensor:
        """siglip_features: (N, in_dim) → (N, out_dim)"""
        return self.proj(self.norm(siglip_features))


def compute_style_rope_positions(
    siglip_h: int,
    siglip_w: int,
    image_h: int,
    image_w: int,
    cap_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute (t, h, w) coords for style tokens.

    Dispatches based on STYLE_ROPE_CONVENTION (module-level config):
      - "rescaled": original convention (siglip grid rescaled to image grid).
        ONLY use this for Turbo training or for matching legacy phase 2b/3a/b/c
        checkpoints. Causes architectural grid on Base.
      - "halton": quasi-random positions from a 2D Halton sequence. The current
        Base-training default. See module-level docstring for rationale.

    Both conventions place style tokens at temporal coord = cap_len + 2
    (one past the image's temporal slot). Returns float32 tensor of shape
    (siglip_h * siglip_w, 3) — the rope_embedder accepts either int or float.
    """
    t_coord = cap_len + 2  # one past image's temporal id
    n = siglip_h * siglip_w

    if STYLE_ROPE_CONVENTION == "halton":
        # Halton 2D positions in [0, 1)^2 scaled to the image grid range,
        # rounded to integers and clamped to [0, image_dim - 1]. Diffusers'
        # rope_embedder indexes into a precomputed lookup table — positions
        # MUST be integer-valued. Inference-side ComfyUI rope handles floats,
        # but for training/inference consistency we round on both sides.
        h_vals = [min(image_h - 1, max(0, round(_halton(k + 1, 2) * float(image_h))))
                  for k in range(n)]
        w_vals = [min(image_w - 1, max(0, round(_halton(k + 1, 3) * float(image_w))))
                  for k in range(n)]
        h_t = torch.tensor(h_vals, device=device, dtype=torch.int32)
        w_t = torch.tensor(w_vals, device=device, dtype=torch.int32)
        t_t = torch.full_like(h_t, t_coord)
        pos = torch.stack([t_t, h_t, w_t], dim=-1)
        return pos

    if STYLE_ROPE_CONVENTION == "rescaled":
        grid_h = torch.arange(siglip_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(siglip_w, device=device, dtype=torch.float32)
        if siglip_h > 1:
            grid_h = grid_h / (siglip_h - 1) * (image_h - 1)
        if siglip_w > 1:
            grid_w = grid_w / (siglip_w - 1) * (image_w - 1)
        hh, ww = torch.meshgrid(grid_h, grid_w, indexing="ij")
        pos = torch.stack(
            [torch.full_like(hh, float(t_coord)), hh, ww],
            dim=-1,
        ).reshape(-1, 3)
        return pos.to(torch.int32)

    if STYLE_ROPE_CONVENTION == "offset":
        # Lumina-Accessory: condition's own row/col + target image grid size.
        # E.g., 24x24 SigLIP grid + image_h=32 places conditions at h in 32..55.
        # Position-axis budget check: Z-Image axes_lens=[1024, 512, 512] →
        # h/w trained up to 511. For target 1024² (image_h=64) + 27x27 SigLIP,
        # conditions land at 64..90; for 512² (image_h=32) + 24x24, at 32..55.
        # Both are inside the densely-trained range.
        grid_h = torch.arange(siglip_h, device=device, dtype=torch.int32) + image_h
        grid_w = torch.arange(siglip_w, device=device, dtype=torch.int32) + image_w
        hh, ww = torch.meshgrid(grid_h, grid_w, indexing="ij")
        t_t = torch.full_like(hh, t_coord, dtype=torch.int32)
        pos = torch.stack([t_t, hh, ww], dim=-1).reshape(-1, 3)
        return pos

    raise ValueError(f"Unknown STYLE_ROPE_CONVENTION: {STYLE_ROPE_CONVENTION!r}")


def prepare_style_tokens(
    transformer,
    style_embedder: StyleEmbedder,
    siglip_features_3d: torch.Tensor,  # (sig_H, sig_W, in_dim)
    image_h: int,  # image patched height
    image_w: int,  # image patched width
    cap_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run style features through the embedder, pad to SEQ_MULTI_OF, compute
    RoPE freqs.

    Returns: (style_tokens, style_freqs, style_seqlen)
        style_tokens: (style_seqlen, dim)
        style_freqs: (style_seqlen, head_dim/2, 2) complex-as-real RoPE freqs
        style_seqlen: padded length
    """
    device = siglip_features_3d.device
    sig_H, sig_W, _ = siglip_features_3d.shape
    flat = siglip_features_3d.reshape(sig_H * sig_W, -1)
    # Embed
    embedded = style_embedder(flat)
    # Compute RoPE positions (before padding)
    pos = compute_style_rope_positions(sig_H, sig_W, image_h, image_w, cap_len, device)
    # Pad to SEQ_MULTI_OF
    ori_len = embedded.shape[0]
    pad_len = (-ori_len) % SEQ_MULTI_OF
    if pad_len > 0:
        pad_emb = style_embedder.pad_token.expand(pad_len, -1)
        embedded = torch.cat([embedded, pad_emb], dim=0)
        # Padding positions get (0,0,0) per the convention in _pad_with_ids
        pad_pos = torch.zeros((pad_len, 3), dtype=torch.int32, device=device)
        pos = torch.cat([pos, pad_pos], dim=0)
    # Run through rope_embedder to get freqs
    freqs = transformer.rope_embedder(pos)
    return embedded, freqs, embedded.shape[0]


@contextlib.contextmanager
def style_injection(
    transformer,
    style_embedder: StyleEmbedder,
    siglip_features: torch.Tensor,  # (sig_H, sig_W, in_dim) — single reference, batch=1
    image_size: tuple[int, int],  # (image_H_patched, image_W_patched)
    cap_lens: list[int],  # caption length per batch item (after SEQ_MULTI_OF pad)
    timestep: torch.Tensor | None = None,  # if provided, enables dual-time AdaLN
    drop_style: bool = False,
):
    """Context manager that monkey-patches _build_unified_sequence to append
    style tokens, optionally with dual-time AdaLN.

    Limitation: batch=1, single reference image.

    timestep: if provided, enables Lumina-Accessory dual-time AdaLN. Style
    tokens are AdaLN-modulated at t=1 (treated as clean reference); image and
    caption tokens are modulated at the supplied diffusion `t`. Implementation:
    we compute both AdaLN inputs here, stash on the transformer, patch each
    main block's forward and the FinalLayer's forward to substitute the
    dual-time values from the stash, and have patched_build return a noise_mask
    distinguishing noisy (1=image+cap) from clean (0=style) rows. The diffusers
    ZImageTransformerBlock natively supports this dispatch (lines 226-280 of
    transformer_z_image.py) — we just route the parameters through.

    drop_style: if True, no patching at all — original transformer behavior.
    Used for IP-Adapter-style CFG image-dropout during training.
    """
    if drop_style:
        yield
        return

    image_H, image_W = image_size
    assert len(cap_lens) == 1, "batch=1"

    style_tokens, style_freqs, style_seqlen = prepare_style_tokens(
        transformer,
        style_embedder,
        siglip_features,
        image_h=image_H,
        image_w=image_W,
        cap_len=cap_lens[0],
    )

    use_dual_time = timestep is not None
    if use_dual_time:
        ones_t = torch.ones_like(timestep)
        adaln_noisy = transformer.t_embedder(timestep * transformer.t_scale).type_as(style_tokens)
        adaln_clean = transformer.t_embedder(ones_t * transformer.t_scale).type_as(style_tokens)
        transformer._style_dual_adaln_noisy = adaln_noisy
        transformer._style_dual_adaln_clean = adaln_clean

    original_build = transformer._build_unified_sequence

    def patched_build(
        x, x_freqs, x_seqlens, x_noise_mask,
        cap, cap_freqs, cap_seqlens, cap_noise_mask,
        siglip, siglip_freqs, siglip_seqlens, siglip_noise_mask,
        omni_mode, device,
    ):
        assert not omni_mode, "Style injection only patches basic mode"
        bsz = len(x_seqlens)
        assert bsz == 1, "batch=1"
        x_len, cap_len = x_seqlens[0], cap_seqlens[0]
        unified = torch.cat(
            [x[0][:x_len], cap[0][:cap_len], style_tokens]
        ).unsqueeze(0)
        unified_freqs = torch.cat(
            [x_freqs[0][:x_len], cap_freqs[0][:cap_len], style_freqs]
        ).unsqueeze(0)
        # Stash sequence layout for downstream consumers (style_attention_block_mask).
        # Layout is [image (x_len) | cap (cap_len) | style (style_seqlen)]; cleared
        # on style_injection exit. Read lazily by per-block forward wrappers so they
        # can construct masks against this exact layout regardless of sequence size.
        transformer._badcat_seq_lens = (int(x_len), int(cap_len), int(style_seqlen))
        if use_dual_time:
            # 1 = noisy (x + cap), 0 = clean (style). Matches the convention in
            # ZImageTransformerBlock.select_per_token (noisy=1 picks adaln_noisy).
            noise_mask = torch.cat([
                torch.ones(x_len + cap_len, dtype=torch.long, device=device),
                torch.zeros(style_seqlen, dtype=torch.long, device=device),
            ]).unsqueeze(0)
            return unified, unified_freqs, None, noise_mask
        return unified, unified_freqs, None, None

    transformer._build_unified_sequence = patched_build

    layer_patches = []
    final_layer_patches = []

    if use_dual_time:
        # Patch each main block to substitute stashed dual-time inputs when the
        # block is invoked with a noise_mask (which our patched_build provides).
        for layer in transformer.layers:
            original_layer_fwd = layer.forward

            def make_patched_layer(orig_fwd):
                def patched_layer_fwd(
                    x, attn_mask, freqs_cis,
                    adaln_input=None, noise_mask=None,
                    adaln_noisy=None, adaln_clean=None,
                ):
                    if noise_mask is not None and hasattr(transformer, "_style_dual_adaln_noisy"):
                        return orig_fwd(
                            x, attn_mask, freqs_cis,
                            adaln_input=None,
                            noise_mask=noise_mask,
                            adaln_noisy=transformer._style_dual_adaln_noisy,
                            adaln_clean=transformer._style_dual_adaln_clean,
                        )
                    return orig_fwd(
                        x, attn_mask, freqs_cis,
                        adaln_input, noise_mask, adaln_noisy, adaln_clean,
                    )
                return patched_layer_fwd

            layer.forward = make_patched_layer(original_layer_fwd)
            layer_patches.append((layer, original_layer_fwd))

        # FinalLayer is called once per generation. Basic mode passes c=adaln_input;
        # we override to the dual-time path. The unified-sequence noise_mask isn't
        # plumbed through, so reconstruct here from the known style_seqlen suffix.
        for key, final_layer in transformer.all_final_layer.items():
            original_final_fwd = final_layer.forward

            def make_patched_final(orig_fwd):
                def patched_final_fwd(x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
                    if hasattr(transformer, "_style_dual_adaln_noisy"):
                        seq_len = x.shape[1]
                        nm = torch.cat([
                            torch.ones(seq_len - style_seqlen, dtype=torch.long, device=x.device),
                            torch.zeros(style_seqlen, dtype=torch.long, device=x.device),
                        ]).unsqueeze(0)
                        return orig_fwd(
                            x,
                            c=None,
                            noise_mask=nm,
                            c_noisy=transformer._style_dual_adaln_noisy,
                            c_clean=transformer._style_dual_adaln_clean,
                        )
                    return orig_fwd(x, c, noise_mask, c_noisy, c_clean)
                return patched_final_fwd

            final_layer.forward = make_patched_final(original_final_fwd)
            final_layer_patches.append((final_layer, original_final_fwd))

    try:
        yield
    finally:
        transformer._build_unified_sequence = original_build
        for layer, orig in layer_patches:
            layer.forward = orig
        for final_layer, orig in final_layer_patches:
            final_layer.forward = orig
        if use_dual_time:
            if hasattr(transformer, "_style_dual_adaln_noisy"):
                del transformer._style_dual_adaln_noisy
            if hasattr(transformer, "_style_dual_adaln_clean"):
                del transformer._style_dual_adaln_clean
        if hasattr(transformer, "_badcat_seq_lens"):
            del transformer._badcat_seq_lens


# =============================================================================
# Phase 5: per-block LoRA on Q, K projections of the trunk's main layers.
# =============================================================================
# Motivation: phase 4 demonstrated empirically (2026-05-04) that with the
# pure-projector + sequence-concat architecture on the frozen Z-Image trunk,
# the trained adapter's effective contribution is concentrated in the first
# 2-3 of 30 main blocks. Masking blocks 0-2 entirely removes visible style;
# masking blocks 3-29 has no observable effect. The trunk's frozen pretrained
# Q projections are only receptive to OOD sequence-concat tokens at the very
# first blocks. See conversation memory + research notes.
#
# Phase 5 addresses this by adding per-block LoRA on the to_q, to_k linear
# layers. This is the OminiControl recipe (arxiv:2411.15098): instead of
# adding new cross-attention layers (the SDXL/FLUX IP-Adapter pattern),
# modify the trunk's existing attention via low-rank deltas so its Q/K
# computation becomes receptive to OOD tokens at every block.
#
# OminiControl found Q, K LoRA was the load-bearing piece; V LoRA was less
# critical. We follow that here. Rank=32 is the default starting point.
#
# The LoRA modules are trainable parameters that live in the adapter package
# alongside the StyleEmbedder, NOT inside the transformer itself. The trunk
# stays frozen; we attach LoRA at runtime via a context manager that swaps
# each main block's attention processor for a LoRA-aware variant.

class QKLoRAPair(nn.Module):
    """Per-block LoRA on Q and K projections.

    Output of forward isn't used directly — call q_delta(x) / k_delta(x) to
    get the additive deltas applied to the genuine to_q(x) / to_k(x) outputs.
    Up matrices are zero-initialized so the LoRA is a no-op until trained.

    Per-block scalar `gate` is a learnable multiplier on the LoRA contribution.
    Initialized at 1.0 so behavior at init matches a vanilla LoRA. The gate
    exists to make per-block effective magnitude directly addressable by the
    variance regularization loss in BlockLoRAStack — without it, the only
    way to redistribute the per-block LoRA effective magnitude is through
    growing/shrinking the up @ down product itself, which has more degrees
    of freedom than needed and reacts more sluggishly to regularization
    pressure.
    """

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.q_down = nn.Linear(dim, rank, bias=False)
        self.q_up = nn.Linear(rank, dim, bias=False)
        self.k_down = nn.Linear(dim, rank, bias=False)
        self.k_up = nn.Linear(rank, dim, bias=False)
        # Per-block scalar gate, learnable, init 1.0 (no scaling at init).
        self.gate = nn.Parameter(torch.ones(1))
        # Kaiming on the down projections (standard LoRA init).
        nn.init.kaiming_uniform_(self.q_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.k_down.weight, a=math.sqrt(5))
        # Zero-init up: initial output is zero, so adding the LoRA delta is a
        # no-op against the genuine to_q/to_k. Loss starts identical to the
        # un-augmented trunk; gradient pressure determines how it grows.
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.k_up.weight)

    def q_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * self.q_up(self.q_down(x))

    def k_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate * self.k_up(self.k_down(x))

    def effective_q_norm_sq(self) -> torch.Tensor:
        """Squared Frobenius norm of the effective Q delta matrix.

        Equivalent to ||gate * q_up @ q_down||_F^2, but computed without
        materializing the rank-32 product matrix (which would be (dim, dim)).
        Uses the gram-trace identity: for symmetric A_g = A^T A and
        B_g = B B^T, ||A B||_F^2 = trace(A_g B_g) = (A_g * B_g).sum().
        """
        A = self.q_up.weight  # (dim, rank)
        B = self.q_down.weight  # (rank, dim)
        A_g = A.T @ A  # (rank, rank), symmetric
        B_g = B @ B.T  # (rank, rank), symmetric
        return self.gate.pow(2).squeeze() * (A_g * B_g).sum()

    def effective_k_norm_sq(self) -> torch.Tensor:
        A = self.k_up.weight
        B = self.k_down.weight
        A_g = A.T @ A
        B_g = B @ B.T
        return self.gate.pow(2).squeeze() * (A_g * B_g).sum()


class BlockLoRAStack(nn.Module):
    """A ModuleList of QKLoRAPair, one per main transformer block.

    For Z-Image-Base, num_blocks=30, dim=3840. At rank=32 this is ~14.7M
    trainable params total + 30 scalar gates — same order of magnitude as
    the StyleEmbedder.
    """

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

    def variance_reg_loss(self, eps: float = 1e-12) -> torch.Tensor:
        """Variance of effective per-block LoRA magnitudes.

        For each block, computes the Frobenius norm of the effective Q and K
        delta matrices (gate × up @ down). Returns the variance across blocks
        of these magnitudes — minimized when all blocks contribute equally.

        Pressure direction: outlier blocks (much larger than mean) get pushed
        down; small blocks get pushed up. This is the load-bearing fix for
        phase 5's failure mode where vanilla LoRA's gradient asymmetry caused
        early/late blocks to monopolize contribution while middle blocks
        languished. With this regularization, the optimizer has explicit
        gradient pressure to redistribute toward uniform per-block magnitude.

        Computed in float32 for numerical stability — the squared norms can
        span several orders of magnitude across blocks during training, and
        bf16 variance can underflow at the small-blocks end.

        The `eps` inside sqrt is critical for the zero-init regime: at step 0
        all up matrices are zero, so norm_sq = 0; without eps, the derivative
        d sqrt(0) = 1/(2·sqrt(0)) = inf would flow back through the up/down
        matrices and corrupt the optimizer state. With eps=1e-12, the sqrt
        is well-defined at init and the variance starts cleanly at 0
        (all blocks have sqrt(eps) ≈ 1e-6, equal). Once any block grows
        past sqrt(eps), the eps becomes negligible.
        """
        q_norms = torch.stack(
            [(l.effective_q_norm_sq() + eps).sqrt() for l in self.layers]
        ).float()
        k_norms = torch.stack(
            [(l.effective_k_norm_sq() + eps).sqrt() for l in self.layers]
        ).float()
        return q_norms.var() + k_norms.var()


class LoRAAwareAttnProcessor(ZSingleStreamAttnProcessor):
    """Drop-in replacement for ZSingleStreamAttnProcessor that adds LoRA
    deltas to to_q and to_k outputs.

    Reads the LoRA module from `attn._badcat_lora` if present; if absent,
    behaves identically to the genuine processor. We set _badcat_lora on
    each main block's attention from the lora_injection context manager
    and clear it on exit, so this processor reverts to no-op behavior
    when LoRA isn't installed.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Genuine Q, K, V from the frozen trunk's projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # LoRA deltas: only added if a LoRA module is attached for this attn.
        lora = getattr(attn, "_badcat_lora", None)
        if lora is not None:
            query = query + lora.q_delta(hidden_states)
            key = key + lora.k_delta(hidden_states)

        # Match the rest of ZSingleStreamAttnProcessor.__call__ exactly.
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE — copied from the genuine processor.
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(
                    x_in.float().reshape(*x_in.shape[:-1], -1, 2)
                )
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # The genuine processor calls dispatch_attention_fn — we need the same.
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


@contextlib.contextmanager
def lora_injection(transformer, block_lora: BlockLoRAStack):
    """Install per-block LoRA on the transformer's main layers' attention.

    On entry: for each of the 30 main layers, attach the corresponding
    QKLoRAPair to that layer's attention as `_badcat_lora`, and swap the
    attention processor to LoRAAwareAttnProcessor. The genuine processor
    is stashed and restored on exit. The base transformer parameters are
    unchanged; only `attn.processor` and `attn._badcat_lora` are touched.

    Compose with style_injection — typically you want both:

        with lora_injection(transformer, block_lora):
            with style_injection(transformer, ..., drop_style=...):
                pred = transformer(...)

    The LoRA stays active during style_injection's drop_style branch. This
    is intentional: during style-dropout training steps, the LoRA receives
    gradient pressure to NOT modify attention since there's no style for
    its modifications to help with. Combined with the ~90% style-present
    pressure to BE active, the LoRA learns deltas that are useful when
    style is in the sequence and small when it isn't.
    """
    layers = transformer.layers
    if len(layers) != block_lora.num_blocks:
        raise ValueError(
            f"BlockLoRAStack has {block_lora.num_blocks} layers but transformer "
            f"has {len(layers)} main layers — they must match."
        )

    saved_processors = []
    try:
        for i, layer in enumerate(layers):
            attn = layer.attention
            saved_processors.append((attn, attn.processor))
            attn._badcat_lora = block_lora[i]
            attn.processor = LoRAAwareAttnProcessor()
        yield
    finally:
        for attn, original_processor in saved_processors:
            attn.processor = original_processor
            if hasattr(attn, "_badcat_lora"):
                del attn._badcat_lora


@contextlib.contextmanager
def style_attention_block_mask(
    transformer,
    active_mask: list[bool],
):
    """Per-block masking of image→style attention during training.

    For each block where `active_mask[k]` is False, install a forward wrapper
    that injects a 4D attention mask hiding image-query × style-key cells.
    Caption→style and style→style attention are unaffected (only image queries
    are blocked from seeing style at masked blocks).

    This is the training-time analog of the inference-side `start_block`
    diagnostic in comfyui_node/nodes.py:_install_block_wrappers, but with
    arbitrary per-block selection rather than a single threshold.

    Used by phase 5c training: at each step, sample `active_mask` with
    independent Bernoulli(1-p_mask) for each block. The optimizer cannot
    concentrate the LoRA's useful work at any specific blocks because any
    specific blocks might be masked at any step. To reduce loss across all
    mask configurations, every block must learn to contribute when called
    upon.

    Sequence layout (matches `style_injection` patched_build):
        [image (x_len) | cap (cap_len) | style (style_len)]
    Lengths are read at call time from `transformer._badcat_seq_lens`, which
    `style_injection` stashes during patched_build. If absent (e.g. style
    dropout, or this manager entered outside style_injection), wrappers fall
    through to the genuine forward — masking only applies when style is
    actually present.

    Composes with `style_injection` and `lora_injection`:
        with lora_injection(transformer, block_lora):
            with style_injection(...):
                with style_attention_block_mask(transformer, mask):
                    pred = transformer(...)

    Order matters: enter this AFTER style_injection so we wrap whatever
    layer.forward style_injection's dual-time path may have installed. On
    exit we restore the forward found at entry.
    """
    layers = transformer.layers
    if len(active_mask) != len(layers):
        raise ValueError(
            f"active_mask has {len(active_mask)} entries but transformer "
            f"has {len(layers)} main layers — they must match."
        )

    layer_patches = []

    def make_wrapper(orig_fwd):
        def wrapped(x, attn_mask, freqs_cis, *args, **kwargs):
            seq_lens = getattr(transformer, "_badcat_seq_lens", None)
            if seq_lens is None:
                # No style in the sequence (e.g. drop_style step) — pass through.
                return orig_fwd(x, attn_mask, freqs_cis, *args, **kwargs)
            x_len, cap_len, style_len = seq_lens
            total_S = x_len + cap_len + style_len
            if x.shape[1] != total_S:
                # Sequence length mismatch — defensive fallback.
                return orig_fwd(x, attn_mask, freqs_cis, *args, **kwargs)
            image_start = 0
            image_end = x_len
            style_start = x_len + cap_len
            bsz = x.shape[0]
            dev = x.device
            # Build a fresh 4D mask if none was supplied (training default).
            # Otherwise expand/clone the existing one to (B, 1, total_S, total_S).
            if attn_mask is None:
                new_mask = torch.ones(
                    (bsz, 1, total_S, total_S), dtype=torch.bool, device=dev
                )
            elif attn_mask.dim() == 4 and attn_mask.shape[2] == 1:
                new_mask = attn_mask.expand(-1, -1, total_S, -1).clone()
            elif attn_mask.dim() == 2:
                new_mask = attn_mask[:, None, None, :].expand(
                    -1, -1, total_S, -1
                ).clone()
            else:
                new_mask = attn_mask.clone()
            # Hide image→style cells; leave caption→style and style→style alone.
            new_mask[:, :, image_start:image_end, style_start:] = False
            return orig_fwd(x, new_mask, freqs_cis, *args, **kwargs)
        return wrapped

    try:
        for block_idx, layer in enumerate(layers):
            if active_mask[block_idx]:
                continue  # unmasked — image queries can see style normally
            original_fwd = layer.forward
            layer.forward = make_wrapper(original_fwd)
            layer_patches.append((layer, original_fwd))
        yield
    finally:
        for layer, original_fwd in layer_patches:
            layer.forward = original_fwd

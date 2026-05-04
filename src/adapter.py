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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.models.normalization import RMSNorm
from diffusers.models.transformers.transformer_z_image import SEQ_MULTI_OF


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

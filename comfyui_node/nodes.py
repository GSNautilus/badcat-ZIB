"""ComfyUI nodes for the Z-Image Style Adapter (badcat).

Three nodes:
  - badcatLoadSigLIP2          : loads SigLIP-2 NaFlex vision encoder -> SIGLIP2
  - badcatLoadZImageStyleAdapter: loads StyleEmbedder weights        -> ZIMAGE_STYLE_ADAPTER
  - badcatApplyZImageStyleAdapter: MODEL + adapter + siglip + ref + strength -> MODEL

The Apply node hooks ComfyUI's NextDiT (`comfy.ldm.lumina.model.NextDiT`)
by replacing `patchify_and_embed` via ModelPatcher.add_object_patch. The
wrapper calls the original, then appends projected SigLIP tokens to the
unified sequence with rescale-to-image-grid RoPE coords (matching the
diffusers convention the adapter was trained against).
"""
from __future__ import annotations

import os
import time
import torch
from PIL import Image
import numpy as np

from .style_adapter import StyleEmbedder


# ---------------------------------------------------------------------------
# Diagnostic instrumentation
# ---------------------------------------------------------------------------
# When True, every load + apply + first patched_patchify call prints diagnostic
# info. Used to validate Tests B/C/D from node_troubleshooting.md §6:
#   B) loader is honestly reading each .pt file (compare printed weight stats
#      against `python -m src.inspect_adapter_weights`)
#   C) projected style tokens reach attention with non-zero norm
#   D) the patch is intercepting (the [BADCAT-PATCH] line proves it ran)
# Restart ComfyUI after toggling this to pick up the change.
BADCAT_DEBUG = True

# RoPE-position experiments for the Base architectural-grid issue. Mutually
# exclusive (set at most one True). Both default False = original convention
# (rescaled to image grid, the one phase 2b was trained against).
#
# DEGENERATE: all 252 real style tokens at the SAME position (cap+2 + offset, 0, 0).
# Diagnostic only — confirmed the spatial-RoPE-alignment hypothesis but
# introduced a top-left asymmetry artifact (top-left image position is
# RoPE-equivalent to (0,0), so attention to style is asymmetric across the
# image). Keeping the flag for reproducibility / regression checks.
#
# OFFSET: Lumina-Accessory recipe. Each style token's position is
# (cap+2, h_index + image_h_patched, w_index + image_w_patched), where h_index
# and w_index are the token's row/col within the SigLIP grid (NOT rescaled).
# E.g. 24x24 SigLIP at image_h=32 → positions span 32..55 on h and w. Style
# coords sit immediately past the image grid; per-token distinction preserved.
# This is the convention phase 4 trains against (see src/adapter.py:
# STYLE_ROPE_CONVENTION = "offset"). Set this flag True to load phase 4
# checkpoints. Note: previous version of this flag used rescaled-grid + offset,
# which did NOT match training — corrected 2026-05-03.
#
# DEGENERATE / CENTERED / NO_ROPE / HALTON below are diagnostic-only: they
# change inference-side RoPE without matching training, so they're useful for
# probing artifact mechanisms but no checkpoint was trained for them.
BADCAT_DEGENERATE_STYLE_ROPE = False
BADCAT_OFFSET_STYLE_ROPE = False
# CENTERED: like DEGENERATE but with spatial = (image_h_patched/2, image_w_patched/2)
# instead of (0,0). Same mechanism (all style tokens at one point) — center
# image position attends to style with zero spatial-RoPE distance, so style
# enters the residual stream there and spreads via subsequent self-attention.
# But the four corners are roughly equidistant from the center so any
# asymmetric attention artifact should be radially symmetric (center-bright)
# rather than top-left-concentrated. More visually acceptable for natural
# images, and should preserve the same style-propagation behavior the (0,0)
# degenerate showed.
BADCAT_CENTERED_STYLE_ROPE = False
# NO_ROPE: replace style positions' RoPE rotation matrices with identity
# (cos=1, sin=0). Style tokens then participate in attention without any
# position modulation — their Q/K dot products with image queries are
# un-rotated, so attention is uniformly accessible from every image patch
# rather than position-modulated. This is the only convention that's
# structurally bias-free: no privileged image position, no periodic alignment.
#
# The catch: Z-Image's trunk was trained to ALWAYS see RoPE-rotated keys.
# Identity rotation is technically out of distribution — attention may behave
# unpredictably. Per-token spatial-coord choice in _build_style_rope_positions
# is irrelevant when this flag is on (we overwrite the rotation matrices
# directly), so this flag supersedes the three above.
BADCAT_NO_ROPE_STYLE = False
# HALTON: assign each style token a unique spatial RoPE position drawn from
# a 2D Halton sequence (bases 2 and 3) scaled to fill the image's patched
# grid. Halton is a low-discrepancy quasi-random sequence — deterministic
# (same positions every run) and evenly distributed (no clustering, no
# regular grid). The geometric properties:
#   - No regular alignment with the image grid → no periodic compounding
#     that would produce the grid artifact.
#   - Every image position has SOME style positions within ~3 RoPE distance,
#     so coupling is roughly uniform across image positions (no privileged
#     "matches my Q-rotation" position).
#   - Per-token positional distinction preserved.
# This is the most honest "minimize spatial bias without grid" option
# within the sequence-concat + RoPE constraint. Temporal stays at cap+2
# (close, like the original convention) so attention can actually couple.
# Mutually exclusive with the other flags above.
BADCAT_HALTON_STYLE_ROPE = False

# CFG-MASK: when False (default), style tokens are masked OUT of the uncond
# branch's attention so the (cond - uncond) differential carries the full
# style contribution and CFG amplifies it (canonical IP-Adapter pattern).
# When True, style is visible in BOTH cond and uncond branches → CFG no longer
# amplifies style (or any structured contribution from the privileged image
# position created by the offset RoPE convention). Diagnostic: if the
# bottom-right artifact disappears with this flag True, the artifact is
# specifically in the cond-uncond differential, not the absolute attention
# contribution. Trade-off when this is True: CFG can't be used to control
# style strength via cfg_scale.
BADCAT_DISABLE_CFG_MASK = False


def _log(msg: str) -> None:
    if BADCAT_DEBUG:
        print(f"[BADCAT] {msg}", flush=True)


def _tensor_stats(t: torch.Tensor) -> str:
    f = t.detach().float()
    return (
        f"shape={tuple(t.shape)} norm={f.norm().item():.4f} "
        f"mean={f.mean().item():+.4f} std={f.std().item():.4f} "
        f"min={f.min().item():+.4f} max={f.max().item():+.4f}"
    )


# ---------------------------------------------------------------------------
# Module-level cache for SigLIP (1.6 GB; don't reload per workflow run)
# ---------------------------------------------------------------------------
_SIGLIP_CACHE: dict = {}


def _comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE is (B, H, W, C) float32 in [0,1]. Take batch[0] -> PIL RGB."""
    if image.ndim == 4:
        image = image[0]
    arr = (image.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Node 1: SigLIP-2 loader
# ---------------------------------------------------------------------------
class BadcatLoadSigLIP2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "google/siglip2-so400m-patch16-naflex"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("SIGLIP2",)
    RETURN_NAMES = ("siglip2",)
    FUNCTION = "load"
    CATEGORY = "badcat/ZImage"

    def load(self, model_id: str, dtype: str):
        from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel

        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        cache_key = (model_id, dtype)
        if cache_key not in _SIGLIP_CACHE:
            model = Siglip2VisionModel.from_pretrained(model_id, torch_dtype=torch_dtype).eval()
            processor = Siglip2ImageProcessorFast.from_pretrained(model_id)
            _SIGLIP_CACHE[cache_key] = {
                "model": model,
                "processor": processor,
                "dtype": torch_dtype,
                "hidden_size": model.config.hidden_size,
            }
        return (_SIGLIP_CACHE[cache_key],)


# ---------------------------------------------------------------------------
# Node 2: Style adapter loader
# ---------------------------------------------------------------------------
class BadcatLoadZImageStyleAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_path": ("STRING", {
                    "default": "phase2b_ssl.pt",
                    "tooltip": "Absolute path or path relative to ComfyUI/models/style_adapters/",
                }),
                "in_dim": ("INT", {"default": 1152, "min": 1, "max": 4096}),
                "out_dim": ("INT", {"default": 3840, "min": 1, "max": 8192}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("ZIMAGE_STYLE_ADAPTER",)
    RETURN_NAMES = ("style_adapter",)
    FUNCTION = "load"
    CATEGORY = "badcat/ZImage"

    def load(self, adapter_path: str, in_dim: int, out_dim: int, dtype: str):
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        path = adapter_path
        if not os.path.isabs(path) and not os.path.exists(path):
            try:
                import folder_paths
                models_dir = os.path.join(folder_paths.models_dir, "style_adapters")
                candidate = os.path.join(models_dir, path)
                if os.path.exists(candidate):
                    path = candidate
            except Exception:
                pass

        if not os.path.exists(path):
            raise FileNotFoundError(f"Style adapter weights not found: {adapter_path}")

        embedder = StyleEmbedder(in_dim=in_dim, out_dim=out_dim)
        state = torch.load(path, map_location="cpu", weights_only=True)
        embedder.load_state_dict(state)
        embedder = embedder.to(dtype=torch_dtype).eval()

        if BADCAT_DEBUG:
            try:
                st = os.stat(path)
                mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                _log(f"LOAD adapter input='{adapter_path}'")
                _log(f"  resolved path: {path}")
                _log(f"  file: size={st.st_size} bytes, mtime={mtime}")
                _log(f"  dtype: {torch_dtype}")
                # Print stats from the *loaded embedder* (post-cast to torch_dtype),
                # not from the raw state dict — this is what attention will see.
                sd = embedder.state_dict()
                for k in ("proj.weight", "proj.bias", "pad_token"):
                    if k in sd:
                        _log(f"  {k:14s} {_tensor_stats(sd[k])}")
            except Exception as e:
                _log(f"  (debug print failed: {e})")

        return ({"embedder": embedder, "dtype": torch_dtype, "in_dim": in_dim, "out_dim": out_dim},)


# ---------------------------------------------------------------------------
# Node 3: Apply adapter to a Z-Image MODEL
# ---------------------------------------------------------------------------
def _is_nextdit(obj) -> bool:
    return (
        hasattr(obj, "patchify_and_embed")
        and hasattr(obj, "rope_embedder")
        and hasattr(obj, "patch_size")
        and hasattr(obj, "layers")
    )


def _find_inner_transformer(model_patcher):
    """Walk known ComfyUI conventions to find the NextDiT (Z-Image) transformer."""
    inspected = []
    m = getattr(model_patcher, "model", model_patcher)
    candidates = [m]
    if hasattr(m, "diffusion_model"):
        candidates.append(m.diffusion_model)
    for c in list(candidates):
        if hasattr(c, "model") and c.model is not c:
            candidates.append(c.model)
    for c in candidates:
        inspected.append(type(c).__name__)
        if _is_nextdit(c):
            return c
    raise RuntimeError(
        "Could not locate Z-Image NextDiT transformer. "
        "Inspected: " + ", ".join(inspected)
    )


def _halton(i: int, base: int) -> float:
    """Radical inverse of i in the given base — i.e., the i-th Halton index.
    Used to generate low-discrepancy quasi-random positions deterministically.
    Index i should start at 1 (i=0 always returns 0)."""
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def _build_style_rope_positions(
    sig_H: int,
    sig_W: int,
    image_h_patched: int,
    image_w_patched: int,
    cap_padded_len: int,
    n_padded: int,
    bsz: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute (B, n_padded, 3) RoPE positions for style tokens.

    Diffusers omni siglip coord convention (matches what the adapter was
    trained against in src/adapter.py / Phase 2B):
      - temporal: cap_padded_len + 2 (one past x's `cap_padded_len + 1`)
      - spatial: rescale sig grid to image's patched grid
      - padding tokens: (0, 0, 0)

    When BADCAT_DEGENERATE_STYLE_ROPE is True, all real style tokens are
    placed at a single far-away point with zero spatial coords — the
    diagnostic test for whether spatial-RoPE alignment causes the Base
    normal-trunk grid artifact. Padding tokens stay at (0,0,0).
    """
    pos = torch.zeros((bsz, n_padded, 3), dtype=torch.float32, device=device)
    n_real = sig_H * sig_W
    if BADCAT_DEGENERATE_STYLE_ROPE:
        # All real style tokens share the same RoPE position: temporal far
        # from the image's slot, spatial = (0, 0). No spatial structure;
        # no alignment with the image's spatial grid.
        far_t = float(cap_padded_len + 2 + max(image_h_patched, image_w_patched) + 32)
        pos[:, :n_real, 0] = far_t
        # spatial dims left at 0 from the zeros() initialization
        if BADCAT_DEBUG:
            _log(f"DEGENERATE-ROPE: all {n_real} real style tokens at "
                 f"(t={far_t:.1f}, h=0, w=0); padding {n_padded - n_real} at (0,0,0)")
        return pos
    if BADCAT_CENTERED_STYLE_ROPE:
        # All real style tokens share the same RoPE position: temporal far
        # from image's slot, spatial = (image_h_patched/2, image_w_patched/2).
        # Center image position has zero spatial-RoPE distance (style enters
        # residual stream there); corners equidistant (artifact symmetric).
        far_t = float(cap_padded_len + 2 + max(image_h_patched, image_w_patched) + 32)
        center_h = float(image_h_patched) / 2.0
        center_w = float(image_w_patched) / 2.0
        pos[:, :n_real, 0] = far_t
        pos[:, :n_real, 1] = center_h
        pos[:, :n_real, 2] = center_w
        if BADCAT_DEBUG:
            _log(f"CENTERED-ROPE: all {n_real} real style tokens at "
                 f"(t={far_t:.1f}, h={center_h:.1f}, w={center_w:.1f}); "
                 f"padding {n_padded - n_real} at (0,0,0)")
        return pos
    if BADCAT_HALTON_STYLE_ROPE:
        # Each real style token gets a unique deterministic position from a
        # 2D Halton sequence (bases 2 and 3) scaled to the image's patched
        # grid, rounded to integers (for training/inference consistency —
        # diffusers' rope_embedder requires int positions; we round here too
        # so both sides see identical positions). Temporal stays at cap+2.
        pos[:, :n_real, 0] = float(cap_padded_len + 2)
        h_vals = [min(image_h_patched - 1, max(0, round(_halton(k + 1, 2) * float(image_h_patched))))
                  for k in range(n_real)]
        w_vals = [min(image_w_patched - 1, max(0, round(_halton(k + 1, 3) * float(image_w_patched))))
                  for k in range(n_real)]
        h_coords = torch.tensor(h_vals, dtype=torch.float32, device=device)
        w_coords = torch.tensor(w_vals, dtype=torch.float32, device=device)
        pos[:, :n_real, 1] = h_coords.unsqueeze(0)
        pos[:, :n_real, 2] = w_coords.unsqueeze(0)
        if BADCAT_DEBUG:
            _log(f"HALTON-ROPE: {n_real} real style tokens at integer-rounded "
                 f"quasi-random positions; t={cap_padded_len + 2}, "
                 f"h range [{int(h_coords.min().item())}, {int(h_coords.max().item())}], "
                 f"w range [{int(w_coords.min().item())}, {int(w_coords.max().item())}], "
                 f"unique (h,w) pairs={len(set(zip(h_vals, w_vals)))}/{n_real}; "
                 f"padding {n_padded - n_real} at (0,0,0)")
        return pos
    pos[:, :n_real, 0] = float(cap_padded_len + 2)
    if BADCAT_OFFSET_STYLE_ROPE:
        # Lumina-Accessory recipe: condition's own row/col (NOT rescaled) +
        # image_h/w_patched offset. Mirrors src/adapter.py "offset" branch.
        # E.g. 24x24 SigLIP at image_h=32 → positions span 32..55 on each axis.
        h_idx = torch.arange(sig_H, dtype=torch.float32, device=device) + float(image_h_patched)
        w_idx = torch.arange(sig_W, dtype=torch.float32, device=device) + float(image_w_patched)
        hh = h_idx.view(-1, 1).repeat(1, sig_W).flatten()
        ww = w_idx.view(1, -1).repeat(sig_H, 1).flatten()
        if BADCAT_DEBUG:
            _log(f"OFFSET-ROPE (Lumina-Accessory): siglip-grid + image-size; "
                 f"h range [{int(hh.min().item())}, {int(hh.max().item())}], "
                 f"w range [{int(ww.min().item())}, {int(ww.max().item())}]; "
                 f"unique (h,w) pairs={n_real}/{n_real}")
    else:
        # Default: rescaled grid (siglip → image grid coords). What phase 2b/3a/b/c
        # were trained against. Causes the periodic grid artifact on Base.
        if sig_H > 1:
            grid_h = torch.arange(sig_H, dtype=torch.float32, device=device) / (sig_H - 1) * (image_h_patched - 1)
        else:
            grid_h = torch.zeros(sig_H, dtype=torch.float32, device=device)
        if sig_W > 1:
            grid_w = torch.arange(sig_W, dtype=torch.float32, device=device) / (sig_W - 1) * (image_w_patched - 1)
        else:
            grid_w = torch.zeros(sig_W, dtype=torch.float32, device=device)
        hh = grid_h.view(-1, 1).repeat(1, sig_W).flatten()
        ww = grid_w.view(1, -1).repeat(sig_H, 1).flatten()
    pos[:, :n_real, 1] = hh.unsqueeze(0)
    pos[:, :n_real, 2] = ww.unsqueeze(0)
    return pos


class BadcatApplyZImageStyleAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "style_adapter": ("ZIMAGE_STYLE_ADAPTER",),
                "siglip2": ("SIGLIP2",),
                "reference_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "badcat/ZImage"

    def _encode_reference(self, siglip_pkg, pil_image, device):
        siglip = siglip_pkg["model"]
        processor = siglip_pkg["processor"]
        dtype = siglip_pkg["dtype"]

        if next(siglip.parameters()).device != torch.device(device):
            siglip.to(device)

        inputs = processor(images=[pil_image], return_tensors="pt").to(device)
        spatial = inputs.spatial_shapes[0]
        sig_H, sig_W = int(spatial[0]), int(spatial[1])
        with torch.no_grad():
            hidden = siglip(**inputs).last_hidden_state
        feats = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, hidden.shape[-1]).to(dtype)
        return feats, sig_H, sig_W

    def apply(self, model, style_adapter, siglip2, reference_image, strength):
        m = model.clone()
        transformer = _find_inner_transformer(m)

        device = getattr(model, "load_device", None) or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        target_dtype = style_adapter["dtype"]

        embedder = style_adapter["embedder"].to(device=device, dtype=target_dtype)

        pil = _comfy_image_to_pil(reference_image)
        siglip_feats, sig_H, sig_W = self._encode_reference(siglip2, pil, device)

        # Stash the genuine original on the transformer the first time we see
        # it. ComfyUI's patch lifecycle can leave a previous patched version as
        # the live `patchify_and_embed` (no guaranteed unpatch between runs).
        # If we naively captured `transformer.patchify_and_embed` here on a
        # later invocation, we'd wrap our own previous wrapper -> recursive
        # double-injection -> noise. The stash is set once and outlives any
        # number of patch/unpatch cycles.
        current = transformer.patchify_and_embed
        already_patched = getattr(current, "_is_badcat_patch", False)
        if not already_patched:
            transformer._badcat_genuine_patchify_and_embed = current

        if BADCAT_DEBUG:
            sd = embedder.state_dict()
            _log(f"APPLY transformer={type(transformer).__name__} device={device} "
                 f"dtype={target_dtype} strength={float(strength):.3f}")
            _log(f"  reference: PIL size={pil.size}, siglip grid={sig_H}x{sig_W}, "
                 f"siglip_feats {_tensor_stats(siglip_feats)}")
            _log(f"  embedder.proj.weight {_tensor_stats(sd['proj.weight'])}")
            _log(f"  embedder.proj.bias   {_tensor_stats(sd['proj.bias'])}")
            _log(f"  embedder.pad_token   {_tensor_stats(sd['pad_token'])}")
            _log(f"  patch state on entry: live patchify_and_embed is_badcat={already_patched}")

        rope_embedder = transformer.rope_embedder
        patch_size = transformer.patch_size
        pad_multiple = transformer.pad_tokens_multiple
        alpha = float(strength)

        # Track which cond_or_uncond branches we've printed for, so we capture
        # both the cond and uncond paths at cfg>1 (ComfyUI runs them as
        # separate forward passes, not batched into one call).
        debug_state = {"printed_branches": set()}

        def patched_patchify(*args, **kwargs):
            # Signature: patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens,
            #   ref_latents=[], ref_contexts=[], siglip_feats=[], transformer_options={})
            # We replaced an instance attribute, so `self` is NOT auto-passed.
            # Always call the stashed genuine — never the live attribute, which
            # may be another (older) wrap.
            genuine = transformer._badcat_genuine_patchify_and_embed
            padded_full_embed, mask, img_size, cap_size, freqs_cis, timestep_zero_index = genuine(
                *args, **kwargs
            )
            # transformer_options is the last positional arg (idx 8) or a kwarg.
            xformer_opts = kwargs.get("transformer_options", None)
            if xformer_opts is None and len(args) >= 9:
                xformer_opts = args[8]
            cond_or_uncond = (xformer_opts or {}).get("cond_or_uncond", None)

            # If omni mode is active (ref_latents present), don't inject — let the
            # native omni path handle conditioning.
            ref_latents = kwargs.get("ref_latents", None)
            if ref_latents is None and len(args) >= 6:
                ref_latents = args[5]
            if ref_latents and len(ref_latents) > 0:
                return padded_full_embed, mask, img_size, cap_size, freqs_cis, timestep_zero_index

            bsz = padded_full_embed.shape[0]
            dev = padded_full_embed.device
            dt = padded_full_embed.dtype

            # Project SigLIP -> dim, scale by strength
            feats = siglip_feats.to(device=dev, dtype=dt)
            sig_C = feats.shape[-1]
            flat = feats.reshape(sig_H * sig_W, sig_C)
            style = embedder.to(device=dev, dtype=dt)(flat) * alpha   # (N, dim)
            style = style.unsqueeze(0).expand(bsz, -1, -1).contiguous()  # (B, N, dim)

            # Pad to pad_tokens_multiple using our trained pad_token
            n_real = sig_H * sig_W
            if pad_multiple is not None:
                pad_extra = (-style.shape[1]) % pad_multiple
            else:
                pad_extra = 0
            if pad_extra > 0:
                pad_tok = embedder.pad_token.to(device=dev, dtype=dt).unsqueeze(0).expand(
                    bsz, pad_extra, -1
                )
                style = torch.cat([style, pad_tok], dim=1)
            n_padded = style.shape[1]

            # RoPE for style tokens
            cap_padded_len = int(cap_size[0])
            H, W = img_size[0]
            image_h_patched = H // patch_size
            image_w_patched = W // patch_size
            style_pos = _build_style_rope_positions(
                sig_H, sig_W, image_h_patched, image_w_patched,
                cap_padded_len, n_padded, bsz, dev,
            )
            style_freqs = rope_embedder(style_pos).movedim(1, 2).to(freqs_cis.dtype)

            if BADCAT_NO_ROPE_STYLE:
                # Replace the per-position rotation matrices with identity
                # rotations [[1,0],[0,1]] so style tokens' Q/K dot products
                # are NOT position-modulated. Format from
                # comfy.ldm.flux.math.rope: each (..., 2, 2) is the 2D rotation
                # matrix [[cos, -sin], [sin, cos]]; identity is [[1,0],[0,1]].
                style_freqs = torch.zeros_like(style_freqs)
                style_freqs[..., 0, 0] = 1.0
                style_freqs[..., 1, 1] = 1.0
                if BADCAT_DEBUG and debug_state.get("no_rope_logged") is None:
                    debug_state["no_rope_logged"] = True
                    _log(f"NO-ROPE: replaced style_freqs with identity rotations "
                         f"(shape={tuple(style_freqs.shape)} dtype={style_freqs.dtype})")

            new_padded = torch.cat([padded_full_embed, style], dim=1)
            new_freqs = torch.cat([freqs_cis, style_freqs], dim=1)

            # Print once per distinct cond_or_uncond pattern we see (so cfg>1
            # gets two prints — one for the cond pass, one for the uncond
            # pass — instead of just the first call).
            branch_key = tuple(cond_or_uncond) if cond_or_uncond is not None else ("none",)
            if BADCAT_DEBUG and branch_key not in debug_state["printed_branches"]:
                debug_state["printed_branches"].add(branch_key)
                _log(f"PATCH call branch={branch_key}")
                _log(f"  genuine returned: padded_full_embed shape={tuple(padded_full_embed.shape)}, "
                     f"mask={'None' if mask is None else f'tensor shape={tuple(mask.shape)}'}, "
                     f"img_size={img_size}, cap_size={cap_size}")
                # Per-row style norms (one per batch item)
                style_row_norms = style.float().norm(dim=-1).norm(dim=-1).tolist()
                _log(f"  style tokens: shape={tuple(style.shape)} n_real={n_real} "
                     f"n_padded={n_padded} per_row_total_norm={[f'{x:.3f}' for x in style_row_norms]}")
                # Per-token magnitude stats (the load-bearing diagnosis from
                # next_steps_briefing.md §4)
                tok_norms = style[0].float().norm(dim=-1)  # (n_padded,)
                _log(f"  per-token L2 across positions: "
                     f"mean={tok_norms.mean().item():.4f} std={tok_norms.std().item():.4f} "
                     f"min={tok_norms.min().item():.4f} max={tok_norms.max().item():.4f}")
                _log(f"  cond_or_uncond={cond_or_uncond}, alpha={alpha:.3f}, "
                     f"image_grid={image_h_patched}x{image_w_patched}, "
                     f"cap_padded_len={cap_padded_len}")
                _log(f"  new sequence: total_len={new_padded.shape[1]} "
                     f"(was {padded_full_embed.shape[1]}, +{n_padded} style)")

            # CFG-aware: hide style tokens from uncond rows so the CFG
            # subtraction (cond - uncond) carries the full style direction
            # instead of cancelling. We mask style keys for uncond rows; queries
            # at any position in those rows then can't attend to style.
            # ComfyUI convention (comfy_types/__init__.py:29): 0=cond, 1=uncond.
            # Mask shape: 4D (B, 1, 1, S_k) avoids attention_pytorch's 2D/3D
            # unsqueeze logic which would misinterpret a (B, S) mask.
            #
            # When BADCAT_DISABLE_CFG_MASK is True, this masking is skipped:
            # style is available to BOTH cond and uncond branches, CFG no
            # longer amplifies style (or the privileged-position artifact).
            new_mask = mask
            mask_path = "passthrough (mask unchanged)"
            if BADCAT_DISABLE_CFG_MASK:
                mask_path = "cfg-mask DISABLED (style visible to both cond and uncond)"
            elif (
                cond_or_uncond is not None
                and len(cond_or_uncond) == bsz
                and any(c == 1 for c in cond_or_uncond)
            ):
                style_start = padded_full_embed.shape[1]
                total_S = new_padded.shape[1]
                new_mask = torch.ones((bsz, 1, 1, total_S), dtype=torch.bool, device=dev)
                for row, c in enumerate(cond_or_uncond):
                    if c == 1:
                        new_mask[row, 0, 0, style_start:] = False
                mask_path = f"cfg-aware 4D, uncond rows={[i for i,c in enumerate(cond_or_uncond) if c==1]}"

            # Log mask state alongside each unique-branch print above
            if BADCAT_DEBUG and branch_key in debug_state["printed_branches"] and \
               (branch_key, "mask") not in debug_state["printed_branches"]:
                debug_state["printed_branches"].add((branch_key, "mask"))
                _log(f"  mask path: {mask_path}, "
                     f"new_mask={'None' if new_mask is None else f'tensor shape={tuple(new_mask.shape)} dtype={new_mask.dtype}'}")

            return new_padded, new_mask, img_size, cap_size, new_freqs, timestep_zero_index

        # Mark our patched function so a future apply() call can detect that
        # the live `patchify_and_embed` is already a badcat wrap and avoid
        # capturing it as "genuine".
        patched_patchify._is_badcat_patch = True

        m.add_object_patch("diffusion_model.patchify_and_embed", patched_patchify)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "BadcatLoadSigLIP2": BadcatLoadSigLIP2,
    "BadcatLoadZImageStyleAdapter": BadcatLoadZImageStyleAdapter,
    "BadcatApplyZImageStyleAdapter": BadcatApplyZImageStyleAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BadcatLoadSigLIP2": "badcat Load SigLIP-2",
    "BadcatLoadZImageStyleAdapter": "badcat Load Z-Image Style Adapter",
    "BadcatApplyZImageStyleAdapter": "badcat Apply Z-Image Style Adapter",
}

"""Measure gradient L2 norms from diffusion loss vs auxiliary loss
flowing into the StyleEmbedder. Compute the λ that balances them.

Method: at each of N samples, run the full forward pass, then call
backward separately for each loss component (using retain_graph) and
record the L2 norm of the StyleEmbedder gradient. Take the ratio of
mean diffusion-grad-norm to mean aux-grad-norm. That ratio is the λ
which makes the two losses contribute equal gradient magnitude to
the projector.

Output: a single number, plus per-sample variance for context.

This avoids picking λ by guesswork and gives us a principled scale.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapter import StyleEmbedder, style_injection

from diffusers import BitsAndBytesConfig
from diffusers.models.transformers import ZImageTransformer2DModel
from transformers import Siglip2ImageProcessorFast, Siglip2VisionModel


DEVICE = "cuda"
DTYPE = torch.bfloat16
BASE_MODEL = "Tongyi-MAI/Z-Image"
HEIGHT = 512
WIDTH = 512
SHIFT = 3.0
N_SAMPLES = int(os.environ.get("N_SAMPLES", "30"))
BRIDGE_SEED = 42
SEED = 0
TRANSFORMER_DIM = 3840
CSD_DIM = 768
CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "phase3_pre_cache.pt"
)


def shifted_t_to_sigma(t, shift):
    return shift * t / (1.0 + (shift - 1.0) * t)


def sample_training_timestep(batch, shift, device, dtype, gen):
    u = torch.randn(batch, generator=gen, device=device).sigmoid()
    sigma = shifted_t_to_sigma(u, shift).to(dtype)
    timestep = (sigma * 1000.0).to(dtype)
    return sigma, timestep


def encode_siglip(siglip, processor, image, device, dtype):
    inputs = processor(images=[image], return_tensors="pt").to(device)
    spatial = inputs.spatial_shapes[0]
    sig_H, sig_W = int(spatial[0]), int(spatial[1])
    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state
    C = hidden.shape[-1]
    feats = hidden[:, : sig_H * sig_W].view(sig_H, sig_W, C).to(dtype)
    return feats, sig_H, sig_W


def make_bridge(seed: int, device, dtype) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(TRANSFORMER_DIM, CSD_DIM, generator=g) / np.sqrt(TRANSFORMER_DIM)
    return R.to(device, dtype)


def proj_grad_norm(style_embedder: StyleEmbedder) -> float:
    """L2 norm of all StyleEmbedder parameter gradients combined."""
    total = 0.0
    for p in style_embedder.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return float(np.sqrt(total))


def zero_grads(style_embedder: StyleEmbedder):
    for p in style_embedder.parameters():
        p.grad = None


def main():
    print(f"Device: {DEVICE}, dtype: {DTYPE}, N_SAMPLES: {N_SAMPLES}")

    # Load cache (need raw_paths, latents, cap_feats, csd_vectors)
    print("Loading cache...")
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    entries = list(cache["entries"].values())
    print(f"  {len(entries)} entries available")

    # Load training models (same setup as train_phase3c.py)
    print("Loading transformer (NF4)...")
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    ).to(DEVICE)
    transformer.requires_grad_(False)
    transformer.train()
    transformer.enable_gradient_checkpointing()

    print("Loading SigLIP-2...")
    siglip = (
        Siglip2VisionModel.from_pretrained(
            "google/siglip2-so400m-patch16-naflex", torch_dtype=DTYPE
        ).to(DEVICE).eval()
    )
    siglip.requires_grad_(False)
    processor = Siglip2ImageProcessorFast.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )

    print("Initializing StyleEmbedder...")
    torch.manual_seed(SEED)
    style_embedder = StyleEmbedder(in_dim=siglip.config.hidden_size,
                                    out_dim=transformer.config.dim).to(DEVICE, DTYPE)
    style_embedder.train()

    R_bridge = make_bridge(BRIDGE_SEED, DEVICE, DTYPE)
    print(f"Bridge R shape: {tuple(R_bridge.shape)}")

    image_h_patched = HEIGHT // 16
    image_w_patched = WIDTH // 16

    rng = random.Random(SEED)
    sample_gen = torch.Generator(device=DEVICE).manual_seed(SEED + 100)

    diff_grads, aux_grads = [], []
    diff_losses, aux_losses = [], []

    print(f"\nMeasuring on {N_SAMPLES} random samples...")
    for i in range(N_SAMPLES):
        entry = entries[rng.randrange(len(entries))]
        ref_pil = Image.open(entry["raw_path"]).convert("RGB")
        siglip_feats, sig_H, sig_W = encode_siglip(siglip, processor, ref_pil, DEVICE, DTYPE)

        cap_feats = entry["cap_feats"].to(DEVICE, DTYPE)
        cap_len_padded = entry["cap_len_padded"]
        latent = entry["latent"].to(DEVICE, DTYPE)
        csd_target = entry["csd_vector"].to(DEVICE, DTYPE)

        sigma, timestep = sample_training_timestep(1, SHIFT, DEVICE, DTYPE, sample_gen)
        sigma_b = sigma.view(-1, 1, 1, 1)
        with torch.no_grad():
            noise = torch.randn(latent.shape, generator=sample_gen, device=DEVICE, dtype=DTYPE)
            noisy = sigma_b * noise + (1.0 - sigma_b) * latent
            target = latent - noise

        x_input = noisy.unsqueeze(2)
        x_list = list(x_input.unbind(dim=0))

        # === aux loss path (need separate forward to get clean grad)
        zero_grads(style_embedder)
        flat_siglip = siglip_feats.reshape(sig_H * sig_W, -1)
        embedded = style_embedder(flat_siglip)
        mean_pooled = embedded.mean(dim=0)
        predicted = mean_pooled @ R_bridge
        pred_n = F.normalize(predicted.float(), dim=-1, p=2)
        targ_n = F.normalize(csd_target.float(), dim=-1, p=2)
        aux_loss = 1.0 - (pred_n * targ_n).sum()
        aux_loss.backward()
        aux_grad_norm = proj_grad_norm(style_embedder)
        aux_grads.append(aux_grad_norm)
        aux_losses.append(aux_loss.item())

        # === diff loss path (separate forward, run transformer too)
        zero_grads(style_embedder)
        with style_injection(
            transformer=transformer,
            style_embedder=style_embedder,
            siglip_features=siglip_feats,
            image_size=(image_h_patched, image_w_patched),
            cap_lens=[cap_len_padded],
        ):
            pred_list = transformer(x_list, timestep, [cap_feats], return_dict=False)[0]
        pred = torch.stack([p for p in pred_list], dim=0).squeeze(2)
        diff_loss = F.mse_loss(pred.float(), target.float())
        diff_loss.backward()
        diff_grad_norm = proj_grad_norm(style_embedder)
        diff_grads.append(diff_grad_norm)
        diff_losses.append(diff_loss.item())

        if (i + 1) % 5 == 0:
            print(
                f"  sample {i+1}/{N_SAMPLES}: "
                f"diff_loss={diff_loss.item():>8.2f} aux_loss={aux_loss.item():>5.3f}  "
                f"diff_grad={diff_grad_norm:>10.5f}  aux_grad={aux_grad_norm:>10.5f}  "
                f"sigma={sigma.item():.3f}"
            )

    diff_grads = np.array(diff_grads)
    aux_grads = np.array(aux_grads)
    diff_losses = np.array(diff_losses)
    aux_losses = np.array(aux_losses)
    ratios = diff_grads / np.maximum(aux_grads, 1e-12)

    print("\n" + "=" * 70)
    print("GRADIENT NORM ANALYSIS")
    print("=" * 70)
    print(f"{'metric':30s}  {'mean':>10s}  {'median':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print(f"{'diff loss':30s}  {diff_losses.mean():>10.3f}  {np.median(diff_losses):>10.3f}  "
          f"{diff_losses.std():>10.3f}  {diff_losses.min():>10.3f}  {diff_losses.max():>10.3f}")
    print(f"{'aux loss':30s}  {aux_losses.mean():>10.3f}  {np.median(aux_losses):>10.3f}  "
          f"{aux_losses.std():>10.3f}  {aux_losses.min():>10.3f}  {aux_losses.max():>10.3f}")
    print(f"{'||grad_proj L_diff||':30s}  {diff_grads.mean():>10.5f}  {np.median(diff_grads):>10.5f}  "
          f"{diff_grads.std():>10.5f}  {diff_grads.min():>10.5f}  {diff_grads.max():>10.5f}")
    print(f"{'||grad_proj L_aux||':30s}  {aux_grads.mean():>10.5f}  {np.median(aux_grads):>10.5f}  "
          f"{aux_grads.std():>10.5f}  {aux_grads.min():>10.5f}  {aux_grads.max():>10.5f}")
    print(f"{'ratio (diff/aux)':30s}  {ratios.mean():>10.2f}  {np.median(ratios):>10.2f}  "
          f"{ratios.std():>10.2f}  {ratios.min():>10.2f}  {ratios.max():>10.2f}")

    # The principal answer (no clipping)
    lambda_balanced = float(np.median(ratios))
    print()
    print("=" * 70)
    print("UNCLIPPED RECOMMENDATION")
    print("=" * 70)
    print(f"  λ_balanced (median ratio):     {lambda_balanced:.1f}")
    print(f"    — Phase 3b/3c smoke at λ=10 had zero struct_corr movement.")
    print(f"    — λ=50 smoke had +0.05 movement and then plateau.")
    print(f"    — The plateau suggests heavy-tailed diff gradient is the issue.")
    print()

    # === Clipping analysis ===
    print("=" * 70)
    print("EFFECT OF GRADIENT CLIPPING ON DIFF GRADIENT")
    print("=" * 70)
    print(f"{'clip':>8s}  {'mean_clip':>10s}  {'median_clip':>11s}  "
          f"{'p90_clip':>10s}  {'%clipped':>10s}  {'lambda_med':>11s}  {'lambda_mean':>11s}")
    for clip in [None, 1000, 500, 200, 100, 50, 30, 10, 5, 2]:
        if clip is None:
            clipped = diff_grads.copy()
            clip_label = "none"
            pct_clipped = 0.0
        else:
            clipped = np.minimum(diff_grads, clip)
            clip_label = f"{clip}"
            pct_clipped = 100.0 * (diff_grads > clip).mean()
        m = np.mean(clipped)
        med = np.median(clipped)
        p90 = np.percentile(clipped, 90)
        # Recompute aux_grad-balanced lambda
        clipped_ratios = clipped / np.maximum(aux_grads, 1e-12)
        lambda_med = np.median(clipped_ratios)
        lambda_mean = np.mean(clipped_ratios)
        print(f"{clip_label:>8s}  {m:>10.2f}  {med:>11.2f}  {p90:>10.2f}  "
              f"{pct_clipped:>9.1f}%  {lambda_med:>11.1f}  {lambda_mean:>11.1f}")
    print()
    print("Reading: rows below 'none' show what the diff grad distribution")
    print("would look like with that clip value, and what λ would balance the")
    print("clipped distribution against aux. We want a clip value that:")
    print("  - clips ~10-30% of samples (the heavy tail)")
    print("  - brings mean and median much closer together")
    print("  - keeps median diff grad similar to pre-clip (preserves typical signal)")
    print()
    print("After picking a clip value, the recalibrated λ ≈ lambda_med for that row.")
    print("That's the principled (clip, λ) combination to try in the smoke test.")


if __name__ == "__main__":
    main()

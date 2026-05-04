# badcat-ZIB

A trained-from-scratch, image-conditioned **style adapter for [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image)** (Tongyi-MAI). Drop in any reference image at workflow time and the generated output adopts its style — no per-style fine-tuning. Functionally analogous to IP-Adapter on SDXL or InstantX/XLabs IP-Adapter on FLUX, but for Z-Image.

This repo contains the training pipeline, the ComfyUI custom node, and the diagnostics. Trained checkpoints, datasets, and project notes live outside the repo.

## Status (2026-05-04)

A Z-Image-Base trained adapter exists that:

- **Produces no visible artifacts** at normal CFG (3–4) on Base — meets the "no artifacts" quality bar
- **Applies style uniformly** across the full image — no spatial bias
- Surpasses the prior best (phase 2b) on every per-token discrimination metric (std/mean **19.3%** at step 11500 vs phase 2b's gold-standard 14.9%; trajectory had not plateaued)

What it does **not** yet do:

- **Within-painting-style discrimination** is sub-threshold — references currently transfer as "generic painterly" for paintings or as abstract perceptual features (color saturation, frequency content) for non-paintings. This is a data ceiling (22-painting training set clusters tightly in CSD space), not an architectural one.
- **Z-Image-Turbo**: the adapter is trained against Base's CFG×30-step amplification dynamics and has no observable effect on Turbo (cfg=1, 8 steps). Phase 2b had the inverse property — both adapters are regime-specific to their training trunk.

See [§7 of the project state doc](https://github.com/GSNautilus/badcat-ZIB) (kept locally) for full open questions; brief summary in [Open questions](#open-questions) below.

## The recipe

| Component | Setting |
|---|---|
| Base model | Z-Image-Base, NF4-quantized, frozen |
| Reference encoder | SigLIP-2 SO400M (`google/siglip2-so400m-patch16-naflex`), frozen, base size 384 |
| Trainable surface | `StyleEmbedder` = `RMSNorm` + `Linear(1152→3840)`, ~4.4M params |
| Style RoPE convention | **"offset"** — condition's own row/col + image grid size (e.g., 24×24 SigLIP at image_h=32 → positions span (32..55, 32..55)), all at temporal `cap_len+2` |
| AdaLN modulation | **Single-time** — style tokens get the same diffusion-t modulation as image+caption tokens (dual-time tested, no benefit on Base) |
| Style dropout | **10%** — 10% of training steps drop style tokens entirely. Cleans the CFG cond-uncond differential. |
| Per-token CSD aux loss | λ=50, fixed random bridge `R: (3840, 768)`, seed=42. Mean-over-tokens `1 − cos_sim(token_i @ R, csd_target)`. |
| Diffusion loss | Pure rectified-flow MSE on velocity prediction, shift=3.0 |
| Optimizer | AdamW8bit (bnb), LR=1e-4, warmup=500, grad_clip=100, grad_accum=4 |
| Resolution | 512×512 |
| Augmentation | **Geometric only** (random crop 0.6–0.8 + h-flip). No photometric. |
| Hardware | RTX 3060 12 GB, ~2.7s/step with gradient checkpointing |

The three load-bearing pieces:

1. **Offset RoPE** removes the periodic-grid pressure that aligned style positions create on Base. It introduces a *different* geometric bias (bottom-right has the strongest attention coupling), which the trained projector compensates for.
2. **Style dropout** trains the projector to keep its cond-uncond differential clean, so CFG amplifies real style instead of amplifying a privileged-position bias.
3. **Per-token CSD aux** drives real per-token magnitude variation. Pure rectified flow on a coherent dataset doesn't produce per-token discrimination on its own (phase 3a/b/c demonstrated this); the aux is what makes individual tokens carry style instead of collapsing to uniform-magnitude noise.

Full mechanistic discussion is in the project's local research / state docs, not committed.

## Repo layout

```
src/                  # training, dataset/cache prep, diagnostics
comfyui_node/         # ComfyUI custom node (loader + ModelPatcher + RoPE flags)
tests/                # forward-pass smoke tests, projector diagnostics
configs/              # (reserved)
```

### `src/`
- `adapter.py` — `StyleEmbedder`, `style_injection` context manager, `compute_style_rope_positions`, dual-time AdaLN scaffolding (currently unused)
- `train_phase4.py` — **the working training recipe** (offset + dropout + per-token CSD)
- `train_phase2.py` … `train_phase3c.py` — historical recipes (see [Ruled-out approaches](#ruled-out-approaches))
- `build_phase4_cache.py`, `add_csd_to_cache.py`, `download_*.py` — dataset and pre-computed-feature cache builders
- `analyze_per_token_outputs.py` — per-token statistics; quantitatively distinguishes "useful style tokens" from "uniform noise"
- `inspect_adapter_weights.py` — bit-for-bit checkpoint audit
- `test_encoder_discrimination.py` — pairwise CSD/SigLIP cosine analysis on a reference set
- `test_bridge_variants.py` — aux-loss design tests in isolation from diffusion training
- `measure_gradient_balance.py` — diffusion vs aux gradient scale measurement
- `make_random_adapter.py`, `make_zero_adapter.py` — diagnostic baselines (not adapters)
- `make_diagnostic_refs.py` — synthetic controlled references for per-position behavior tests

### `comfyui_node/`
Three nodes:
- `badcatLoadSigLIP2` — loads SigLIP-2 NaFlex
- `badcatLoadZImageStyleAdapter` — loads `StyleEmbedder` weights
- `badcatApplyZImageStyleAdapter` — patches ComfyUI's `NextDiT.patchify_and_embed` via `ModelPatcher.add_object_patch` to append projected style tokens to the unified sequence, with CFG-aware mask handling

`BADCAT_*` env flags select alternate RoPE conventions (`rescaled` / `halton` / `offset` / `degenerate` / `centered` / `no_rope`) — used during the architectural investigation. The shipped recipe runs with `BADCAT_OFFSET_STYLE_ROPE=True`.

### `tests/`
Forward-pass wiring, single-image overfit sanity checks, projector diagnostics, a bare diffusers T2I baseline.

## Ruled-out approaches

For provenance — the things that didn't work and why, so this repo isn't read as "a clean linear path":

| Approach | Why it failed |
|---|---|
| Rescaled RoPE (style positions on image grid) | Periodic grid artifact on Base — alignment compounds across 30 denoising steps × CFG amplification |
| Halton RoPE (quasi-random positions) | Splotches inherent to the irregular geometry, regardless of training quality |
| Mean-pool aux loss (phase 3a/b/c) | Doesn't drive per-token discrimination; projector collapses to uniform-magnitude noise tokens |
| Dual-time AdaLN on Base | OOD: t=t_scale modulation has a learned association on Z-Image-Omni but not Base |
| Photometric augmentation | Modifies style attributes — incompatible with the "preserve style" principle |
| Inference-time RoPE override of phase 2b | Each tested convention produced *some* localized artifact, but this didn't generalize: from-scratch training under offset is artifact-free. Inference-override results don't predict from-scratch training results. |

## Open questions

- **Style discrimination quality.** Generic painterly is the current ceiling on 22 paintings. Diverse paintings (200–1000 across distinct categories) is the cheapest next test; triplet supervision (CSGO-style) is the published path to specific-style transfer.
- **Base/Turbo compatibility.** Currently regime-specific to training trunk. Mixed-trunk batches or post-hoc Turbo fine-tune are candidate paths if Turbo is wanted.
- **Multi-reference composition.** Required by the original goal; architecturally probably straightforward but unimplemented.
- **Resolution scaling.** Trained at 512×512. At 1024 target the offset positions land at 64..127, mildly extrapolated for Z-Image's training curriculum.
- **Strength range / very high CFG.** Tested at default strength and CFG 3–4; high-strength / high-CFG behavior is uncharacterized.
- **Encoder choice / bridge.** SigLIP-2 + random CSD bridge is inherited. DINOv2 keys (Splicing-ViT pattern) or an alternate style descriptor is untested.
- **Training ceiling.** Trajectory had not plateaued at step 11500. Whether longer training continues to improve, saturates, or overfits is open.

## Requirements

- A working ComfyUI install with Z-Image / Z-Image-Turbo
- Python with `torch`, `diffusers`, `transformers`, `Pillow`, `numpy`, `bitsandbytes` (for NF4 + AdamW8bit)
- For training: a CUDA GPU. The recipe is tuned for 12 GB.

## ComfyUI install

```
cp -r comfyui_node <ComfyUI>/custom_nodes/zimage_style_adapter
```

Place adapter weights at `<ComfyUI>/models/style_adapters/<name>.pt`, or pass an absolute path to the loader node. Adapter `.pt` files are not bundled.

## License

[MIT](LICENSE)

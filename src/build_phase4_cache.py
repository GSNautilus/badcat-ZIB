"""Pre-compute training cache for Phase 4 (22 wikimedia paintings).

Mirrors the structure of phase3_pre_cache.pt but on the 22-painting corpus
that originally produced phase 2b. For each image computes:
  - VAE latent of the clean 512x512 image
  - Qwen3 caption hidden states (penultimate layer, 2560-dim)
  - CSD style vector (768-dim, L2-normalized)

Per-step augmentation (geometric only) is done at training time on the raw
PIL image, so the SigLIP encoding is computed fresh each step — only the
latent / cap_feats / csd_vector are cacheable.

Idempotent: skips images that already have all three. Stores raw_path so the
training loop can re-load PIL for augmentation.
"""
from __future__ import annotations

import json
import os
import sys
import time
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import PyTorchModelHubMixin
from transformers import CLIPProcessor, CLIPVisionModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
HEIGHT = 512
WIDTH = 512
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wikimedia_train")
CAPTION_PATH = os.path.join(DATA_DIR, "captions.json")
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "phase4_pre_cache.pt"
)


class CSD_CLIP(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.last_layer_style = nn.Parameter(torch.empty(1024, 768))
        self.last_layer_content = nn.Parameter(torch.empty(1024, 768))

    def forward(self, pixel_values: torch.Tensor):
        out = self.backbone(pixel_values=pixel_values)
        feature = out.pooler_output
        style_output = F.normalize(feature @ self.last_layer_style, dim=1, p=2)
        content_output = F.normalize(feature @ self.last_layer_content, dim=1, p=2)
        return feature, content_output, style_output


def resolve_image_path(data_dir: str, fname: str) -> str:
    """Resolve a captions.json key to an actual on-disk path.

    captions.json was authored on a system using NFD normalization (e.g.
    "Eug" + e + U+0300 + "ne") but the files on this Windows disk are NFC
    (single "è"). Direct os.path.join + open fails. This helper normalizes
    to NFC, then NFD, then falls back to listing the directory and matching
    on normalized form.
    """
    candidates = [fname, unicodedata.normalize("NFC", fname), unicodedata.normalize("NFD", fname)]
    for cand in candidates:
        p = os.path.join(data_dir, cand)
        if os.path.exists(p):
            return p
    # Last resort: scan the directory and match on NFC-normalized form
    target_nfc = unicodedata.normalize("NFC", fname)
    for disk_name in os.listdir(data_dir):
        if unicodedata.normalize("NFC", disk_name) == target_nfc:
            return os.path.join(data_dir, disk_name)
    raise FileNotFoundError(f"Could not resolve {fname!r} in {data_dir}")


def encode_content_image(vae, image, device, dtype):
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        encoded = vae.encode(t).latent_dist.sample()
    encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    return encoded


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Device: {DEVICE}  dtype: {DTYPE}")

    with open(CAPTION_PATH) as f:
        captions = json.load(f)
    train_files = sorted(captions.keys())
    print(f"Training set: {len(train_files)} images")

    # Try to resume an existing partial cache
    if os.path.exists(OUT_PATH):
        print(f"Loading existing cache from {OUT_PATH}...")
        cache = torch.load(OUT_PATH, map_location="cpu", weights_only=False)
        entries = cache.get("entries", {}) if isinstance(cache, dict) else {}
        print(f"  {len(entries)} existing entries")
    else:
        entries = {}

    needs_encoding = [
        f for f in train_files
        if f not in entries
        or "latent" not in entries.get(f, {})
        or "cap_feats" not in entries.get(f, {})
        or "csd_vector" not in entries.get(f, {})
    ]
    print(f"  {len(train_files) - len(needs_encoding)} fully cached")
    print(f"  {len(needs_encoding)} need encoding")

    if not needs_encoding:
        print("All entries cached. Nothing to do.")
        return

    # ── Z-Image pipeline (VAE + text encoder) ─────────────────────
    print("\nLoading Z-Image pipeline (VAE + text encoder)...")
    from diffusers import BitsAndBytesConfig, ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image", subfolder="transformer",
        quantization_config=nf4, torch_dtype=DTYPE,
    )
    transformer.requires_grad_(False)
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image", transformer=transformer, torch_dtype=DTYPE
    )
    pipe.to(DEVICE)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.vae.eval()

    print(f"  VRAM after pipeline load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ── encode VAE latents and caption features ──────────────────
    print("\nEncoding VAE latents and caption features...")
    t0 = time.time()
    for i, fname in enumerate(needs_encoding):
        try:
            path = resolve_image_path(DATA_DIR, fname)
            img_pil = Image.open(path).convert("RGB").resize(
                (WIDTH, HEIGHT), Image.LANCZOS
            )
        except Exception as e:
            # Print a length so any console-encoding issues with the filename
            # don't mask the failure
            print(f"  SKIP (len={len(fname)}): {type(e).__name__}: {e}")
            continue
        latent = encode_content_image(pipe.vae, img_pil, DEVICE, DTYPE).cpu()
        with torch.no_grad():
            cap_feats_list = pipe._encode_prompt(captions[fname], device=DEVICE)
        cap_feats = cap_feats_list[0]
        cap_len_raw = cap_feats.shape[0]
        cap_len_padded = cap_len_raw + (-cap_len_raw) % 32

        entries[fname] = {
            "name": fname,
            "raw_path": path,
            "latent": latent,
            "cap_feats": cap_feats.cpu(),
            "cap_len_padded": cap_len_padded,
            "caption": captions[fname],
        }
        if (i + 1) % 5 == 0 or (i + 1) == len(needs_encoding):
            print(
                f"  {i+1}/{len(needs_encoding)}  elapsed={time.time()-t0:.0f}s  "
                f"vram={torch.cuda.memory_allocated()/1024**3:.1f} GB"
            )

    # Save partial cache (latent + cap_feats) before swapping models
    torch.save({"entries": entries}, OUT_PATH)
    print(f"  Saved partial cache (without CSD) to {OUT_PATH}")

    # ── Free Z-Image, load CSD ───────────────────────────────────
    print("\nFreeing Z-Image pipeline...")
    del pipe.transformer
    del pipe.text_encoder
    del pipe
    del transformer
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print("\nLoading CSD (yuxi-liu-wired/CSD)...")
    csd = CSD_CLIP.from_pretrained("yuxi-liu-wired/CSD").to(DEVICE).eval()
    for p in csd.parameters():
        p.requires_grad_(False)
    csd_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Only files that successfully completed the first pass (have an entry
    # in entries) AND don't yet have a csd_vector
    needs_csd = [
        f for f in train_files
        if f in entries and "csd_vector" not in entries[f]
    ]
    print(f"  {len(needs_csd)} entries need CSD encoding")

    t0 = time.time()
    with torch.no_grad():
        for i, fname in enumerate(needs_csd):
            entry = entries[fname]
            try:
                img = Image.open(entry["raw_path"]).convert("RGB")
            except Exception as e:
                print(f"  SKIP (len={len(fname)}): {type(e).__name__}: {e}")
                continue
            inputs = csd_processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(DEVICE)
            _, _, style = csd(pixel_values)
            entry["csd_vector"] = style.squeeze(0).cpu().float()
            if (i + 1) % 10 == 0 or (i + 1) == len(needs_csd):
                print(
                    f"  {i+1}/{len(needs_csd)}  elapsed={time.time()-t0:.0f}s"
                )

    print(f"\nSaving final cache to {OUT_PATH}...")
    torch.save({"entries": entries}, OUT_PATH)
    print(f"  Total entries: {len(entries)}")
    print(f"  With latent:    {sum(1 for e in entries.values() if 'latent' in e)}")
    print(f"  With cap_feats: {sum(1 for e in entries.values() if 'cap_feats' in e)}")
    print(f"  With CSD:       {sum(1 for e in entries.values() if 'csd_vector' in e)}")


if __name__ == "__main__":
    main()

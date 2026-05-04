"""Download ~500 diverse images from Spawning/PD12M for Phase 3 Run A.

PD12M is metadata-only (URLs to a Spawning-controlled S3 bucket).
We stream the dataset, shuffle, filter to reasonable sizes, and download
the first N that succeed. Captions are saved alongside.

Output layout:
    data/phase3_train/{id}.jpg
    data/phase3_train/captions.json   {filename: caption}
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request

from datasets import load_dataset
from PIL import Image


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "phase3_train")
CAPTIONS_PATH = os.path.join(OUT_DIR, "captions.json")
TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "500"))
MAX_ATTEMPTS = TARGET_COUNT * 3
MIN_SIDE = 384       # below this, the SigLIP grid gets too small
MAX_SIDE = 2048      # above this, save bandwidth — resize on the fly
SHUFFLE_BUFFER = 10000
SHUFFLE_SEED = 0
TIMEOUT = 20
UA = "ZImageStyleAdapter/0.3 (azrack@gmail.com) urllib"


def fetch(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except Exception:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    captions: dict[str, str] = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH) as f:
            captions = json.load(f)
        print(f"Resuming: {len(captions)} already on disk")

    if len(captions) >= TARGET_COUNT:
        print(f"Already have {len(captions)} >= target {TARGET_COUNT}. Done.")
        return

    print("Streaming PD12M metadata...")
    ds = load_dataset("Spawning/PD12M", split="train", streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)

    attempts = 0
    ok = len(captions)
    skip_size = 0
    skip_dl = 0
    skip_decode = 0
    t0 = time.time()

    for sample in ds:
        if ok >= TARGET_COUNT or attempts >= MAX_ATTEMPTS:
            break
        attempts += 1

        sid = sample["id"]
        fname = f"{sid}.jpg"
        if fname in captions:
            continue

        w, h = int(sample["width"]), int(sample["height"])
        if min(w, h) < MIN_SIDE:
            skip_size += 1
            continue

        data = fetch(sample["url"])
        if data is None:
            skip_dl += 1
            continue

        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            skip_decode += 1
            continue

        if max(img.size) > MAX_SIDE:
            scale = MAX_SIDE / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        out_path = os.path.join(OUT_DIR, fname)
        try:
            img.save(out_path, "JPEG", quality=92)
        except Exception:
            skip_decode += 1
            continue

        captions[fname] = sample["caption"]
        ok += 1

        if ok % 25 == 0:
            with open(CAPTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)
            elapsed = time.time() - t0
            rate = ok / max(elapsed, 1.0)
            print(
                f"[{ok:4d}/{TARGET_COUNT}] attempts={attempts} "
                f"skip(size/dl/decode)={skip_size}/{skip_dl}/{skip_decode} "
                f"rate={rate:.1f}/s elapsed={elapsed:.0f}s last={sid[:12]}"
            )

    with open(CAPTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(
        f"\nDone. ok={ok} attempts={attempts} "
        f"skip(size/dl/decode)={skip_size}/{skip_dl}/{skip_decode} "
        f"elapsed={time.time()-t0:.0f}s"
    )
    print(f"Captions: {CAPTIONS_PATH}")


if __name__ == "__main__":
    main()

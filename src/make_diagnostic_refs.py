"""Generate synthetic 1024x1024 reference images for diagnosing what
information actually transfers from reference to output.

Each image isolates a different attribute so we can see what the projector
picks up:
  solid_red          - pure color, no shape (does color come through?)
  solid_blue         - pure color (different hue)
  red_square         - centered red square, 50% scale (color + shape + position)
  green_diamond      - centered green square rotated 45 (orientation isolation)
  circle_full        - white circle tangent to edges (large round shape)
  circle_half        - white circle 50% scale (small round shape)
  bw_stripes         - diagonal B/W stripes (high-freq texture)
  rainbow_lines      - horizontal rainbow stripes (color spectrum + line pattern)
"""
from __future__ import annotations

import os
import math
import colorsys

import numpy as np
from PIL import Image, ImageDraw


SIZE = 1024
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "diagnostic_refs")


def _new(color=(0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), color)


def solid_red() -> Image.Image:
    return _new((255, 0, 0))


def solid_blue() -> Image.Image:
    return _new((0, 0, 255))


def red_square() -> Image.Image:
    """Black background, centered red square at 50% of canvas."""
    img = _new((0, 0, 0))
    s = SIZE // 2          # 512x512 square
    off = (SIZE - s) // 2  # 256
    draw = ImageDraw.Draw(img)
    draw.rectangle([off, off, off + s, off + s], fill=(255, 0, 0))
    return img


def green_diamond() -> Image.Image:
    """Black background, centered green square at 50% scale rotated 45 deg.

    Drawn on a transparent canvas, rotated, then composited onto black so the
    rotation is exact (no PIL aliasing on rectangle redraws).
    """
    s = SIZE // 2
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    off = (SIZE - s) // 2
    draw = ImageDraw.Draw(layer)
    draw.rectangle([off, off, off + s, off + s], fill=(0, 255, 0, 255))
    layer = layer.rotate(45, resample=Image.BICUBIC, expand=False)
    base = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    base.paste(layer, (0, 0), layer)
    return base


def circle_full() -> Image.Image:
    """White circle tangent to image edges (diameter == SIZE) on black."""
    img = _new((0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, SIZE - 1, SIZE - 1], fill=(255, 255, 255))
    return img


def circle_half() -> Image.Image:
    """White circle, 50% scale (diameter == SIZE/2), centered, on black."""
    img = _new((0, 0, 0))
    d = SIZE // 2
    off = (SIZE - d) // 2
    draw = ImageDraw.Draw(img)
    draw.ellipse([off, off, off + d, off + d], fill=(255, 255, 255))
    return img


def bw_stripes() -> Image.Image:
    """Black/white diagonal alternating stripes.

    Stripe width = 32px measured perpendicular to the diagonal.
    """
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    yy, xx = np.indices((SIZE, SIZE))
    # Project onto the [1,1] diagonal and threshold by stripe width
    diag = (xx + yy) // 32
    mask = (diag % 2 == 0)
    arr[mask] = 255
    return Image.fromarray(arr)


def rainbow_lines() -> Image.Image:
    """Horizontal rainbow stripes spanning the full hue spectrum."""
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for y in range(SIZE):
        h = y / SIZE  # 0..1 hue
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        arr[y, :] = (int(r * 255), int(g * 255), int(b * 255))
    return Image.fromarray(arr)


GENERATORS = {
    "solid_red.png": solid_red,
    "solid_blue.png": solid_blue,
    "red_square.png": red_square,
    "green_diamond.png": green_diamond,
    "circle_full.png": circle_full,
    "circle_half.png": circle_half,
    "bw_stripes.png": bw_stripes,
    "rainbow_lines.png": rainbow_lines,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname, gen in GENERATORS.items():
        path = os.path.join(OUT_DIR, fname)
        gen().save(path, "PNG", optimize=True)
        print(f"  wrote {path}")
    print(f"\nDone. {len(GENERATORS)} images in {OUT_DIR}")


if __name__ == "__main__":
    main()

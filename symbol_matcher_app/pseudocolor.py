"""Multi-scale pseudo-coloring (per-channel Gaussian pyramid).

SAM-family models are trained on natural photos and struggle with razor-thin,
1-px monochrome CAD lines. This turns a monochrome line drawing into a colourful
"chromatic aberration" image by blurring each RGB channel with a different
Gaussian kernel:

  Red   (high freq) : tight kernel  (~3x3)  -> sharp structural boundaries
  Green (mid  freq) : medium kernel (~7x7)  -> local structural context
  Blue  (low  freq) : wide kernel   (~13x13)-> broad gradient "glow"

Merging the three channels produces a coloured gradient halo around every line,
which SAM's patch-attention perceives like natural depth/lighting transitions.

Two modes:
- ``invert=False`` (default): operate on the raw grayscale (dark lines on white)
  -> chromatic halos on a white background (keeps the drawing readable).
- ``invert=True``: operate on the ink (lines bright) -> coloured glow on black.
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def pseudo_color(
    image: np.ndarray,
    kernels: tuple[int, int, int] = (3, 6, 8),
    invert: bool = False,
) -> np.ndarray:
    """Return a BGR uint8 image with per-channel multi-scale Gaussian blur.

    ``kernels`` are the (red, green, blue) Gaussian kernel sizes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    src = (255 - gray) if invert else gray
    kr, kg, kb = (_odd(k) for k in kernels)
    r = cv2.GaussianBlur(src, (kr, kr), 0)
    g = cv2.GaussianBlur(src, (kg, kg), 0)
    b = cv2.GaussianBlur(src, (kb, kb), 0)
    return cv2.merge([b, g, r])  # OpenCV is BGR: blue=widest, red=tightest


def sharpen(
    image: np.ndarray,
    amount: float = 1.5,
    radius: int = 3,
) -> np.ndarray:
    """Return an unsharp-masked (crisper) copy of ``image``.

    ``amount`` scales the high-frequency detail added back: ~1.0 is moderate,
    the default 1.5 is deliberately a little extreme so razor-thin CAD strokes
    pop for SAM. ``radius`` is the Gaussian blur kernel used to build the mask.
    """
    blur = cv2.GaussianBlur(image, (_odd(radius), _odd(radius)), 0)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


if __name__ == "__main__":
    import argparse

    import worksheet_loader as wl
    import sam_boxes as sb

    ap = argparse.ArgumentParser(description="Preview multi-scale pseudo-coloring")
    ap.add_argument("--rid", default="92cce256-6e75-4574-ab59-71ee3e9d9e32")
    ap.add_argument("--wid", default="e88dc05a-11fb-4df1-ac36-bfd91f83f5ab")
    ap.add_argument("--kernels", default="3,7,13")
    ap.add_argument("--crop", type=int, default=350, help="half-size of the dense preview crop")
    ap.add_argument("--out", default="pseudocolor_preview")
    args = ap.parse_args()

    kernels = tuple(int(x) for x in args.kernels.split(","))
    img = wl.load_worksheet_image(args.rid, args.wid)
    H, W = img.shape[:2]

    # centre the crop on the densest cluster of reference points if available
    pts = sb.load_reference_points(args.rid, args.wid, W, H)
    if pts:
        c = np.array([[p.x, p.y] for p in pts], float)
        from scipy.spatial import cKDTree

        tree = cKDTree(c)
        counts = [len(tree.query_ball_point(p, 250)) for p in c]
        cxp, cyp = c[int(np.argmax(counts))].astype(int)
    else:
        cxp, cyp = W // 2, H // 2

    R = args.crop
    x0, y0 = max(0, cxp - R), max(0, cyp - R)
    x1, y1 = min(W, cxp + R), min(H, cyp + R)

    pc_white = pseudo_color(img, kernels, invert=False)
    pc_glow = pseudo_color(img, kernels, invert=True)

    def save(name, full):
        cv2.imwrite(f"{args.out}_{name}_full.png", cv2.resize(full, (full.shape[1] // 4, full.shape[0] // 4)))
        cv2.imwrite(f"{args.out}_{name}_crop.png", full[y0:y1, x0:x1])

    save("orig", img)
    save("white", pc_white)
    save("glow", pc_glow)
    print(f"kernels={kernels} crop=({x0},{y0})-({x1},{y1}) — wrote {args.out}_*_(full|crop).png")

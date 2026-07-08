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
    kernels: tuple[int, int, int] = (6, 8, 3),
    invert: bool = False,
    keep_black: bool = False,
) -> np.ndarray:
    """Return a BGR uint8 image with per-channel multi-scale Gaussian blur.

    ``kernels`` are the (red, green, blue) Gaussian kernel sizes.

    ``keep_black`` composites the sharp original ink back into all three channels
    so the **line core stays solid black** while the multi-scale color survives
    only as a halo around it. It scales the glow by ``alpha = gray/255`` (1 at the
    background/halo, 0 at solid ink), so ``gray == 0`` -> ``(0, 0, 0)`` and the
    off-line halo keeps its full colour, with anti-aliased edges blending smoothly.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    src = (255 - gray) if invert else gray
    kr, kg, kb = (_odd(k) for k in kernels)
    r = cv2.GaussianBlur(src, (kr, kr), 0)
    g = cv2.GaussianBlur(src, (kg, kg), 0)
    b = cv2.GaussianBlur(src, (kb, kb), 0)
    merged = cv2.merge([b, g, r])  # OpenCV is BGR: blue=widest, red=tightest
    if keep_black:
        alpha = (gray.astype(np.float32) / 255.0)[:, :, None]
        merged = (merged.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return merged


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


def gaussian(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Return a plain Gaussian-blurred copy of ``image`` (kernel ``ksize``)."""
    k = _odd(max(1, ksize))
    return cv2.GaussianBlur(image, (k, k), 0)


def laplacian(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Return a Laplacian-of-Gaussian edge image as BGR uint8.

    ``ksize`` is the Gaussian pre-blur kernel (a larger value smooths more before
    the Laplacian). The response is rectified and normalized to 0-255 so thin CAD
    strokes become bright edges on a dark field.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    k = _odd(max(1, ksize))
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    lap = np.absolute(cv2.Laplacian(blur, cv2.CV_64F, ksize=3))
    lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)


def median(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur (good at knocking out salt-and-pepper speckle)."""
    return cv2.medianBlur(image, _odd(max(1, ksize)))


def bilateral(image: np.ndarray, ksize: int = 9) -> np.ndarray:
    """Edge-preserving bilateral smoothing (``ksize`` is the pixel diameter)."""
    return cv2.bilateralFilter(image, max(1, ksize), 75, 75)


def canny(image: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    """Canny edge map as BGR uint8."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.cvtColor(cv2.Canny(gray, lo, hi), cv2.COLOR_GRAY2BGR)


def clahe(image: np.ndarray, tile: int = 8) -> np.ndarray:
    """Local contrast enhancement (CLAHE); ``tile`` is the grid size."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    t = max(1, tile)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(t, t)).apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def adaptive_threshold(image: np.ndarray, block: int = 11) -> np.ndarray:
    """Adaptive (Gaussian) binarization; ``block`` is the neighbourhood size."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    b = _odd(max(3, block))
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, 2
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def blur_threshold(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian blur then global (Otsu) binarization as BGR uint8.

    The textbook denoise-then-threshold recipe: a light Gaussian blur (kernel
    ``ksize``) smooths anti-aliasing / speckle so Otsu finds a clean split, then the
    whole region is thresholded to solid black ink on white (``THRESH_BINARY``, same
    polarity as ``adaptive_threshold``). Otsu is global, so it is run per tile/crop
    where the histogram is reliably bimodal (dark ink on a white field).

    Caveat: a large ``ksize`` blurs razor-thin (1-px) CAD strokes toward the
    background and Otsu can then drop them entirely -- keep the kernel small (3-5).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    k = _odd(max(1, ksize))
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def invert(image: np.ndarray) -> np.ndarray:
    """Invert intensities (dark lines become bright)."""
    return cv2.bitwise_not(image)


# Filters whose single ``ksize`` field is meaningful (the rest ignore it).
KSIZE_FILTERS = {
    "gaussian", "laplace", "sharpen", "median", "bilateral", "clahe", "threshold",
    "blur_threshold",
}


def preprocess(
    image: np.ndarray,
    filt: str = "none",
    ksize: int = 5,
    kernels: tuple[int, int, int] = (6, 8, 3),
) -> np.ndarray:
    """Apply the selected preprocessing ``filt`` to ``image`` for SAM input.

    - ``"none"``      : return the image unchanged
    - ``"gaussian"``  : Gaussian blur (kernel ``ksize``)
    - ``"laplace"``   : Laplacian-of-Gaussian edges (pre-blur kernel ``ksize``)
    - ``"channels"``  : multi-scale per-channel pseudo-coloring (``kernels`` = R,G,B)
    - ``"sharpen"``   : unsharp mask (radius ``ksize``)
    - ``"median"``    : median blur (kernel ``ksize``)
    - ``"bilateral"`` : edge-preserving smoothing (diameter ``ksize``)
    - ``"canny"``     : Canny edges
    - ``"clahe"``     : local contrast enhancement (tile ``ksize``)
    - ``"threshold"`` : adaptive binarization (block ``ksize``)
    - ``"blur_threshold"`` : Gaussian blur (kernel ``ksize``) then Otsu binarization
    - ``"invert"``    : intensity inversion
    """
    if filt == "gaussian":
        return gaussian(image, ksize)
    if filt == "laplace":
        return laplacian(image, ksize)
    if filt == "channels":
        return pseudo_color(image, kernels, invert=True, keep_black=True)
    if filt == "sharpen":
        return sharpen(image, radius=ksize)
    if filt == "median":
        return median(image, ksize)
    if filt == "bilateral":
        return bilateral(image, ksize)
    if filt == "canny":
        return canny(image)
    if filt == "clahe":
        return clahe(image, ksize)
    if filt == "threshold":
        return adaptive_threshold(image, ksize)
    if filt == "blur_threshold":
        return blur_threshold(image, ksize)
    if filt == "invert":
        return invert(image)
    return image


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

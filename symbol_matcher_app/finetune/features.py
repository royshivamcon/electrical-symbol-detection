"""Build the mask-confidence head's inputs.

One candidate = a point-centered grayscale crop + one FastSAM mask (in crop coords) +
the prompt-point location. Two inputs feed the head:

* an image tensor with three channels so the head can judge *is this the correct symbol
  mask for the prompted point*:
    0. grayscale crop (line art context), normalized to [0, 1]
    1. the candidate binary mask
    2. a Gaussian heatmap at the prompt point
* a small scalar-feature vector (``build_feats``) that the head fuses with the pooled
  image embedding: FastSAM objectness + cheap mask geometry (size / shape) that carry
  signal the raw pixels don't expose directly.

All resized to ``size x size``. Shared by ``dataset.py`` (cached crops) and ``infer.py``
(live crops) so training and inference see identical inputs.
"""

from __future__ import annotations

import cv2
import numpy as np

# Length of the scalar feature vector from ``build_feats`` (keep in sync with the model).
N_FEATS = 4


def build_input(gray_u8: np.ndarray, mask_bool: np.ndarray, cx: float, cy: float,
                size: int) -> np.ndarray:
    """Return a ``[3, size, size]`` float32 array for one (crop, mask, point)."""
    ch, cw = gray_u8.shape[:2]
    if (cw, ch) != (size, size):
        g = cv2.resize(gray_u8, (size, size), interpolation=cv2.INTER_AREA)
    else:
        g = gray_u8
    g = g.astype(np.float32) / 255.0

    m = mask_bool.astype(np.uint8)
    if m.shape[:2] != (size, size):
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    m = m.astype(np.float32)

    sx = size / cw if cw else 1.0
    sy = size / ch if ch else 1.0
    pcx, pcy = cx * sx, cy * sy
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    sigma = max(1.0, size / 16.0)
    hm = np.exp(-((xx - pcx) ** 2 + (yy - pcy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)

    return np.stack([g, m, hm], axis=0)


def build_feats(mask_bool: np.ndarray, conf: float) -> np.ndarray:
    """Return the ``[N_FEATS]`` float32 scalar features for one (mask, FastSAM conf).

    Resolution-independent so training (cached masks) and inference (live masks) agree:

      0. ``conf``   — FastSAM objectness for this mask (the current baseline score)
      1. ``norm_area`` — bbox area / crop area (how big the mask is, matches the label gate)
      2. ``aspect`` — min(w,h)/max(w,h) in [0,1] (squareness; symbols aren't slivers)
      3. ``extent`` — mask pixels / bbox area in [0,1] (fill/solidity of the blob)
    """
    ch, cw = mask_bool.shape[:2]
    crop_area = max(1, ch * cw)
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return np.array([float(conf), 0.0, 0.0, 0.0], dtype=np.float32)
    w = int(xs.max() - xs.min()) + 1
    h = int(ys.max() - ys.min()) + 1
    bbox_area = max(1, w * h)
    norm_area = bbox_area / crop_area
    aspect = min(w, h) / max(w, h)
    extent = float(xs.size) / bbox_area
    return np.array([float(conf), norm_area, aspect, extent], dtype=np.float32)

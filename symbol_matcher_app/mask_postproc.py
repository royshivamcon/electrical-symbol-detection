"""symbol_det-style mask post-processing for the point-prompted SAM finders.

The standalone ``symbol_det`` localizer keeps the *glyph strokes* inside a SAM
region rather than the filled blob: it intersects each mask with the binarized
ink (``local = ink & mask``) and drops proposals that are too small, too sparse,
or line-like (see ``_extract_masks`` / ``localize.py``). This module reuses those
exact primitives so the app's HQ-SAM / FastSAM boxes can be refined the same way.

Import is centralised here so the finder modules don't each need the sys.path
juggling to reach the standalone package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# --- locate + import the standalone symbol_det package ---------------------
APP_DIR = Path(__file__).resolve().parent
_CANDIDATES = [
    APP_DIR.parents[1] / "agent-nightwing" / "trades" / "electrical" / "metrics"
    / "symbol_det_standalone",
    Path(
        "/Users/shivam.roy/Desktop/Shivam_Attentive/agent-nightwing/trades/"
        "electrical/metrics/symbol_det_standalone"
    ),
]
for _p in _CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from symbol_det.models import PipelineConfig  # noqa: E402
from symbol_det.pipeline.localize import (  # noqa: E402
    _is_line_like,
    binarize,
    line_mask,
)

DEFAULT_CFG = PipelineConfig()


def _gray(img: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale view of a BGR or already-gray image."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def ink_of(img: np.ndarray) -> np.ndarray:
    """Boolean ink map (ink = dark strokes on white paper) via invert + Otsu."""
    return binarize(_gray(img)) > 0


def suppressed_of(img: np.ndarray, cfg: PipelineConfig = DEFAULT_CFG) -> np.ndarray:
    """Binary ink with long straight lines (leaders/gridlines) subtracted.

    Mirrors ``localize.localize``'s line-suppression stage (Hough line mask +
    subtract, optional morphological reconnect). Returns a uint8 (0/255) image.
    """
    bw = binarize(_gray(img))
    lm = line_mask(
        bw,
        min_len=int(cfg.area_max * cfg.line_min_len_factor),
        thickness=cfg.line_erase_thickness,
    )
    sup = cv2.subtract(bw, lm)
    if cfg.reconnect_kernel > 0:
        k = np.ones((cfg.reconnect_kernel, cfg.reconnect_kernel), np.uint8)
        sup = cv2.morphologyEx(sup, cv2.MORPH_CLOSE, k)
    return sup


def refine(
    mask_bool: np.ndarray,
    ink_bool: np.ndarray,
    cfg: PipelineConfig = DEFAULT_CFG,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """symbol_det ``_extract_masks`` content filter for one mask.

    Intersect the mask with the ink inside the mask's bbox and keep it only if
    the ink strokes are big enough (``min_pixels``), dense enough (``min_fill``)
    and not line-like. Returns ``(local_ink_bool, (x0, y0, w, h))`` in the same
    (crop/tile) coordinates as the inputs, or ``None`` to reject the mask.
    """
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    w = int(xs.max() - x0 + 1)
    h = int(ys.max() - y0 + 1)
    local = ink_bool[y0:y0 + h, x0:x0 + w] & mask_bool[y0:y0 + h, x0:x0 + w]
    area = float(local.sum())
    if area < cfg.min_pixels:
        return None
    if area / float(max(1, w * h)) < cfg.min_fill:
        return None
    if _is_line_like(local, cfg):
        return None
    return local, (x0, y0, w, h)

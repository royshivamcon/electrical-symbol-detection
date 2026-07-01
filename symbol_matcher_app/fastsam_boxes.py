"""FastSAM box-finding from worksheet reference points.

MobileSAM's single-point masks were too loose on these schematics. FastSAM
(YOLOv8-seg based) gives much tighter instance masks. Because the sheets are
huge (~7000x5000) and the symbols tiny, we run FastSAM on a **small crop around
each reference point** and keep the smallest mask that covers the point — this
yields a tight bounding box per symbol. If FastSAM finds nothing for a point we
fall back to a dark connected-component box inside the crop so the symbol is
still boxed.

Reference points use the same "electrical" filtering as the EDA notebooks
(``feature.geometry_type == 1``); the loader lives in ``sam_boxes``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import worksheet_loader as wl
from sam_boxes import RefPoint, load_reference_points  # reuse point loader

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINT = PROJECT_ROOT / "models" / "FastSAM-s.pt"

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    """Lazily load a single shared FastSAM model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from ultralytics import FastSAM

            path = str(CHECKPOINT) if CHECKPOINT.exists() else "FastSAM-s.pt"
            _MODEL = FastSAM(path)
    return _MODEL


@dataclass
class SamBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    name: str
    source: str  # "fastsam" | "cc"

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 4), "name": self.name, "source": self.source,
        }


def _cc_fallback(crop: np.ndarray, cx: int, cy: int, min_px: int) -> tuple | None:
    """Tight box of the dark connected component under (cx, cy) inside the crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # dilate a touch so thin strokes of one symbol connect
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lbl = labels[cy, cx]
        if lbl > 0:
            x, y, w, h, _ = stats[lbl]
            if w >= min_px and h >= min_px:
                return int(x), int(y), int(w), int(h)
    return None


def boxes_from_points(
    image_bgr: np.ndarray,
    points: list[RefPoint],
    crop: int = 90,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.9,
    max_box_frac: float = 0.45,  # of the crop area (drops background masks)
    min_box_px: int = 6,
    min_symbol_px: int = 28,     # boxes smaller than this are grown around the point
    max_symbol_px: int = 90,     # boxes larger than this fall back to the core mask
    size_ratio: float = 1.6,     # union only masks within size_ratio x the median
    pad: int = 4,                # extra pixels added on every side
    fallback: bool = True,
    pseudocolor: bool = False,   # glow pseudo-color the model input (not the fallback)
    sharpen: bool = False,       # unsharp-mask the model input (not the fallback)
) -> list[SamBox]:
    """Return one box per reference point using crop-wise FastSAM.

    For each point we collect every (non-background) mask covering it, then union
    only those near the **median size** — this merges fragmented sub-strokes into
    the whole symbol while dropping an oversized enveloping mask. The result is
    floored to ``min_symbol_px`` and, if it still exceeds ``max_symbol_px``, falls
    back to the single smallest covering mask so boxes don't blow up.
    """
    if not points:
        return []
    model = get_model()
    H, W = image_bgr.shape[:2]

    # Preprocessing (sharpen and/or glow pseudo-color) feeds SAM an enhanced
    # image; the CC fallback and all coordinate/size math stay on the original
    # BGR frame.
    model_img = image_bgr
    if sharpen or pseudocolor:
        import pseudocolor as pc

        if sharpen:
            model_img = pc.sharpen(model_img)
        if pseudocolor:
            model_img = pc.pseudo_color(model_img, invert=True)

    def finalize(gx0, gy0, gx1, gy1, score, name, source):
        gx0, gy0 = gx0 - pad, gy0 - pad
        gx1, gy1 = gx1 + pad, gy1 + pad
        w_, h_ = gx1 - gx0, gy1 - gy0
        # grow tiny boxes around their center to a sensible minimum
        if w_ < min_symbol_px:
            cxg = (gx0 + gx1) / 2
            gx0, gx1 = cxg - min_symbol_px / 2, cxg + min_symbol_px / 2
        if h_ < min_symbol_px:
            cyg = (gy0 + gy1) / 2
            gy0, gy1 = cyg - min_symbol_px / 2, cyg + min_symbol_px / 2
        gx0, gy0 = max(0, int(round(gx0))), max(0, int(round(gy0)))
        gx1, gy1 = min(W, int(round(gx1))), min(H, int(round(gy1)))
        return SamBox(gx0, gy0, gx1 - gx0, gy1 - gy0, score, name, source)

    out: list[SamBox] = []
    for p in points:
        x0, y0 = max(0, p.x - crop), max(0, p.y - crop)
        x1, y1 = min(W, p.x + crop), min(H, p.y + crop)
        sub = image_bgr[y0:y1, x0:x1]
        sub_in = model_img[y0:y1, x0:x1]
        ch, cw = sub.shape[:2]
        if ch < min_box_px or cw < min_box_px:
            continue
        cx, cy = p.x - x0, p.y - y0
        crop_area = ch * cw

        res = model(sub_in, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]

        # Collect every (non-background) mask covering the point.
        covering = []  # (area, bb, conf)
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy()
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for mk, bb, cf in zip(masks, xyxy, confs):
                mh, mw = mk.shape
                yy = int(cy * mh / ch)
                xx = int(cx * mw / cw)
                if not (0 <= yy < mh and 0 <= xx < mw and mk[yy, xx] > 0.5):
                    continue
                area = float((bb[2] - bb[0]) * (bb[3] - bb[1]))
                if area > max_box_frac * crop_area:
                    continue  # skip background-sized masks
                covering.append((area, bb, float(cf)))

        if covering:
            covering.sort(key=lambda t: t[0])
            areas = [a for a, _, _ in covering]
            med = areas[len(areas) // 2]
            kept = [(a, bb, cf) for a, bb, cf in covering if a <= size_ratio * med]
            ux0 = min(bb[0] for _, bb, _ in kept)
            uy0 = min(bb[1] for _, bb, _ in kept)
            ux1 = max(bb[2] for _, bb, _ in kept)
            uy1 = max(bb[3] for _, bb, _ in kept)
            score = max(cf for _, _, cf in kept)
            # Ceiling: if the union is still too big, use the smallest core mask.
            if (ux1 - ux0) > max_symbol_px or (uy1 - uy0) > max_symbol_px:
                a, bb, score = covering[0]
                ux0, uy0, ux1, uy1 = bb[0], bb[1], bb[2], bb[3]
            out.append(finalize(x0 + ux0, y0 + uy0, x0 + ux1, y0 + uy1, score, p.name, "fastsam"))
            continue

        if fallback:
            fb = _cc_fallback(sub, cx, cy, min_box_px)
            if fb is not None:
                fx, fy, fw, fh = fb
                if fw <= max_box_frac * cw or fh <= max_box_frac * ch:
                    out.append(finalize(x0 + fx, y0 + fy, x0 + fx + fw, y0 + fy + fh, 0.0, p.name, "cc"))
    return out

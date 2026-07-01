"""SAM 2.1 box-finding from worksheet reference points.

SAM 2.1 (Hiera backbone) is the SAM-family model that actually runs on Apple
MPS, so it's the Mac-friendly stand-in for SAM 3 (which aborts on MPS — see the
README "SAM 3 status" note). We load it through Ultralytics, which auto-downloads
the checkpoint from the HuggingFace/Ultralytics assets.

Same crop-wise recipe as the HQ-SAM path: an **adaptive crop** shrunk toward the
nearest neighbouring reference point, with the remaining in-crop neighbours fed
as **negative** point prompts to separate symbols in dense grids.

Reference points use the same "electrical" filtering (``geometry_type == 1``);
the loader lives in ``sam_boxes``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sam_boxes import RefPoint  # reuse point loader types

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINT = PROJECT_ROOT / "models" / "sam2.1_s.pt"

_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    """Lazily load a single shared SAM 2.1 model (Ultralytics)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from ultralytics import SAM

            path = str(CHECKPOINT) if CHECKPOINT.exists() else "sam2.1_s.pt"
            _MODEL = SAM(path)
    return _MODEL


@dataclass
class SamBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    name: str
    source: str  # "sam2" | "cc"

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 4), "name": self.name, "source": self.source,
        }


def _cc_fallback(crop: np.ndarray, cx: int, cy: int, min_px: int) -> tuple | None:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lbl = labels[cy, cx]
        if lbl > 0:
            x, y, w, h, _ = stats[lbl]
            if w >= min_px and h >= min_px:
                return int(x), int(y), int(w), int(h)
    return None


def _mask_bbox(mask, cy, cx, min_box_px, max_box_frac, crop_area, max_symbol_px):
    if cy >= mask.shape[0] or cx >= mask.shape[1] or not mask[cy, cx]:
        return None
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw_, bh_ = bx1 - bx0, by1 - by0
    if bw_ < min_box_px or bh_ < min_box_px:
        return None
    area = bw_ * bh_
    if area > max_box_frac * crop_area:
        return None
    within = bw_ <= max_symbol_px and bh_ <= max_symbol_px
    return (bx0, by0, bx1, by1), area, within


def boxes_from_points(
    image_bgr: np.ndarray,
    points: list[RefPoint],
    crop: int = 90,
    min_crop: int = 34,
    crop_nn_frac: float = 0.75,
    max_box_frac: float = 0.45,
    min_box_px: int = 6,
    min_symbol_px: int = 28,
    max_symbol_px: int = 90,
    pad: int = 4,
    use_negatives: bool = True,
    max_negatives: int = 8,
    fallback: bool = True,
    pseudocolor: bool = False,   # glow pseudo-color the model input (not the fallback)
    sharpen: bool = False,       # unsharp-mask the model input (not the fallback)
) -> list[SamBox]:
    """Return one box per reference point using crop-wise SAM 2.1."""
    if not points:
        return []
    model = get_model()
    H, W = image_bgr.shape[:2]
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)

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

    if len(points) > 1:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_dist = np.sqrt(d2.min(1))
    else:
        nn_dist = np.array([float(crop) * 2])

    def finalize(gx0, gy0, gx1, gy1, score, name, source):
        gx0, gy0, gx1, gy1 = gx0 - pad, gy0 - pad, gx1 + pad, gy1 + pad
        if (gx1 - gx0) < min_symbol_px:
            c = (gx0 + gx1) / 2
            gx0, gx1 = c - min_symbol_px / 2, c + min_symbol_px / 2
        if (gy1 - gy0) < min_symbol_px:
            c = (gy0 + gy1) / 2
            gy0, gy1 = c - min_symbol_px / 2, c + min_symbol_px / 2
        gx0, gy0 = max(0, int(round(gx0))), max(0, int(round(gy0)))
        gx1, gy1 = min(W, int(round(gx1))), min(H, int(round(gy1)))
        return SamBox(gx0, gy0, gx1 - gx0, gy1 - gy0, score, name, source)

    out: list[SamBox] = []
    for i, p in enumerate(points):
        c = int(round(min(crop, max(min_crop, crop_nn_frac * nn_dist[i]))))
        x0, y0 = max(0, p.x - c), max(0, p.y - c)
        x1, y1 = min(W, p.x + c), min(H, p.y + c)
        sub = image_bgr[y0:y1, x0:x1]
        sub_in = model_img[y0:y1, x0:x1]
        ch, cw = sub.shape[:2]
        if ch < min_box_px or cw < min_box_px:
            continue
        cx, cy = p.x - x0, p.y - y0
        crop_area = ch * cw

        pts_xy = [[cx, cy]]
        labels = [1]
        if use_negatives:
            dx, dy = coords[:, 0] - p.x, coords[:, 1] - p.y
            inside = (np.abs(dx) < c) & (np.abs(dy) < c)
            inside[i] = False
            idx = np.where(inside)[0]
            if idx.size:
                idx = idx[np.argsort(dx[idx] ** 2 + dy[idx] ** 2)][:max_negatives]
                for j in idx:
                    pts_xy.append([int(coords[j, 0] - x0), int(coords[j, 1] - y0)])
                    labels.append(0)

        res = model(sub_in, points=pts_xy, labels=labels, verbose=False)[0]

        best = None  # (bbox, score)
        best_key = None
        if res.masks is not None:
            masks = res.masks.data.cpu().numpy() > 0.5
            confs = (
                res.boxes.conf.cpu().numpy()
                if getattr(res, "boxes", None) is not None and res.boxes is not None
                else np.ones(len(masks))
            )
            for mk, cf in zip(masks, confs):
                r = _mask_bbox(mk, cy, cx, min_box_px, max_box_frac, crop_area, max_symbol_px)
                if r is None:
                    continue
                bbox, area, within = r
                key = (within, float(cf), -area)
                if best_key is None or key > best_key:
                    best_key, best = key, (bbox, float(cf))

        if best is not None:
            (bx0, by0, bx1, by1), sc = best
            out.append(finalize(x0 + bx0, y0 + by0, x0 + bx1, y0 + by1, sc, p.name, "sam2"))
            continue

        if fallback:
            fb = _cc_fallback(sub, cx, cy, min_box_px)
            if fb is not None:
                fx, fy, fw, fh = fb
                if fw <= max_box_frac * cw or fh <= max_box_frac * ch:
                    out.append(finalize(x0 + fx, y0 + fy, x0 + fx + fw, y0 + fy + fh, 0.0, p.name, "cc"))
    return out

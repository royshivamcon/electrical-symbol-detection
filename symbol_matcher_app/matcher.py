"""Multi-scale, tiled template matching with non-maximum suppression.

Given a worksheet image and a user-selected template patch, find every region
in the image that looks similar to the template using normalized
cross-correlation (``cv2.TM_CCOEFF_NORMED``) evaluated across several scales.

The image is processed in overlapping tiles (patches). Tiling keeps each
``matchTemplate`` call small/cacheable and lets tiles run in parallel; the
overlap between neighbouring tiles is at least the template size (plus a small
margin) so a symbol straddling a tile boundary is still fully contained in some
tile. Detections from all tiles are merged and deduplicated with a global NMS.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Match:
    x: int
    y: int
    w: int
    h: int
    score: float

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 4)}


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _nms(boxes: list[Match], iou_thresh: float = 0.3) -> list[Match]:
    """Greedy non-maximum suppression, keeping the highest-scoring boxes."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    keep: list[Match] = []
    for cand in boxes:
        cx1, cy1, cx2, cy2 = cand.x, cand.y, cand.x + cand.w, cand.y + cand.h
        cand_area = cand.w * cand.h
        suppressed = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k.x, k.y, k.x + k.w, k.y + k.h
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            union = cand_area + k.w * k.h - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed = True
                break
        if not suppressed:
            keep.append(cand)
    return keep


def _match_tile(
    tile: np.ndarray,
    templates: list[tuple[np.ndarray, int, int]],
    threshold: float,
    ox: int,
    oy: int,
) -> list[Match]:
    """Match all pre-scaled templates against a single tile.

    ``templates`` is a list of ``(template_img, tile_w, tile_h)``. Returned
    boxes are offset by ``(ox, oy)`` into working-image coordinates.
    """
    found: list[Match] = []
    th_tile, tw_tile = tile.shape[:2]
    for tpl, tw, th in templates:
        if th >= th_tile or tw >= tw_tile:
            continue
        res = cv2.matchTemplate(tile, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for x, y in zip(xs.tolist(), ys.tolist()):
            found.append(Match(ox + int(x), oy + int(y), tw, th, float(res[y, x])))
    return found


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
    scales: list[float] | None = None,
    max_matches: int = 4000,
    iou_thresh: float = 0.3,
    max_image_dim: int = 2500,
    tile_size: int = 1024,
    overlap_margin: int = 16,
    workers: int = 4,
) -> list[Match]:
    """Find all regions in ``image`` similar to ``template`` via tiled matching.

    Parameters
    ----------
    image, template : BGR or grayscale uint8 arrays (original resolution).
    threshold : minimum normalized correlation score (0-1) to keep a hit.
    scales : template scale factors to try (defaults span 0.8x - 1.25x).
    max_matches : hard cap on raw detections before NMS (guards against blow-ups).
    iou_thresh : IoU above which overlapping detections are merged.
    max_image_dim : longest side (px) of the working image; larger inputs are
        downscaled before matching and results are mapped back.
    tile_size : nominal tile (patch) side length in working-image pixels.
    overlap_margin : extra overlap (px) added on top of the template size, so a
        little slack remains around symbols that straddle a tile boundary.
    workers : number of tiles matched in parallel.
    """
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.25]

    img_gray = _to_gray(image)
    tpl_gray = _to_gray(template)
    if tpl_gray.shape[0] < 4 or tpl_gray.shape[1] < 4:
        return []

    # Downscale image + template together to bound the work.
    longest = max(img_gray.shape[:2])
    f = min(1.0, max_image_dim / longest) if longest > 0 else 1.0
    if f < 1.0:
        img_gray = cv2.resize(
            img_gray, (int(img_gray.shape[1] * f), int(img_gray.shape[0] * f)),
            interpolation=cv2.INTER_AREA,
        )
        tpl_gray = cv2.resize(
            tpl_gray, (max(4, int(tpl_gray.shape[1] * f)), max(4, int(tpl_gray.shape[0] * f))),
            interpolation=cv2.INTER_AREA,
        )

    H, W = img_gray.shape[:2]
    th0, tw0 = tpl_gray.shape[:2]
    inv = 1.0 / f

    # Pre-build scaled templates once (reused across every tile).
    templates: list[tuple[np.ndarray, int, int]] = []
    max_tw = max_th = 0
    for s in scales:
        tw, th = max(4, int(round(tw0 * s))), max(4, int(round(th0 * s)))
        if th >= H or tw >= W:
            continue
        tpl = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
        templates.append((tpl, tw, th))
        max_tw, max_th = max(max_tw, tw), max(max_th, th)
    if not templates:
        return []

    # Overlap must cover the largest template so a symbol is whole in some tile.
    ov_x = max_tw + overlap_margin
    ov_y = max_th + overlap_margin
    eff_tile = max(tile_size, max_tw + 1, max_th + 1)
    step_x = max(1, eff_tile - ov_x)
    step_y = max(1, eff_tile - ov_y)

    def tile_starts(total: int, tile: int, step: int) -> list[int]:
        if total <= tile:
            return [0]
        starts = list(range(0, total - tile + 1, step))
        if starts[-1] != total - tile:  # flush a final tile to the far edge
            starts.append(total - tile)
        return starts

    xs = tile_starts(W, eff_tile, step_x)
    ys = tile_starts(H, eff_tile, step_y)

    jobs = []
    for oy in ys:
        for ox in xs:
            tile = img_gray[oy : min(oy + eff_tile, H), ox : min(ox + eff_tile, W)]
            jobs.append((tile, ox, oy))

    raw: list[Match] = []
    if workers and workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(
                lambda j: _match_tile(j[0], templates, threshold, j[1], j[2]), jobs
            )
            for r in results:
                raw.extend(r)
    else:
        for tile, ox, oy in jobs:
            raw.extend(_match_tile(tile, templates, threshold, ox, oy))

    if len(raw) > max_matches:
        raw = sorted(raw, key=lambda b: b.score, reverse=True)[:max_matches]

    # NMS in working coordinates, then map survivors back to original pixels.
    kept = _nms(raw, iou_thresh=iou_thresh)
    if f < 1.0:
        for m in kept:
            m.x = int(round(m.x * inv))
            m.y = int(round(m.y * inv))
            m.w = int(round(m.w * inv))
            m.h = int(round(m.h * inv))
    return kept

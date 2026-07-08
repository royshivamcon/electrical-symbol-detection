"""Training-free neural template matching on the frozen HQ-SAM backbone.

Two extra "template match" methods that reuse the SAM-HQ ViT encoder already
loaded by :mod:`seg_models` (``get_predictor``). Neither needs any new weights
or fine-tuning; both operate on the encoder's **dense feature map**
(``get_image_embedding`` -> ``[256, 64, 64]`` per encoded crop).

Methods
-------
* ``"tmr"``  — *Template Matching & Regression* (training-free flavour):
  build a small **feature kernel** from the exemplar's foreground cells and slide
  it (as a normalized cross-correlation, at several scales) across each target
  tile's feature map. Peaks in the correlation map are candidate locations; every
  peak is then **refined by the SAM decoder** into a tight box (the "regression"
  step is done by the mask decoder rather than a learned head).

* ``"persam"`` — *PerSAM* one-shot (training-free): pool a **target embedding**
  from the exemplar's foreground features, score every target cell by cosine
  similarity to it (the PerSAM "location confidence" map), and prompt SAM at each
  peak with a **positive point + a negative point** at the least-similar cell so
  the mask latches onto the matched instance.

Why different sizes are still found
-----------------------------------
Localization is driven by *appearance similarity in feature space*, which is
largely scale-agnostic (a peak marks a symbol centre regardless of its size). The
actual box extent comes from the SAM decoder mask at that point, not from the
correlation window — so instances larger or smaller than the exemplar are still
detected and get correctly sized boxes.

The whole sheet is processed in overlapping tiles (each ~1024 px so the encoder's
64x64 grid stays ~16 px/cell — fine enough for small CAD symbols); boxes from all
tiles are merged with a global NMS.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import seg_models as sm
import tiling as tl
from matcher import Match, _nms

# SAM ViT patch stride: img_size (1024) / feature grid (64).
_FEAT_STRIDE = 16


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _fg_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground (ink) mask: dark strokes on a light background."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw > 0


def _valid_grid(predictor) -> tuple[int, int]:
    """Feature-grid size (rows, cols) covering the *valid* (non-padded) region."""
    ih, iw = predictor.input_size  # resized (h, w) before padding to 1024
    return int(math.ceil(ih / _FEAT_STRIDE)), int(math.ceil(iw / _FEAT_STRIDE))


def _l2norm(t, dim: int = 0, eps: float = 1e-6):
    return t / (t.norm(dim=dim, keepdim=True) + eps)


def _scale(predictor) -> tuple[float, float]:
    """(sx, sy): resized-pixels per original-pixel for the current encoding."""
    ih, iw = predictor.input_size
    oh, ow = predictor.original_size
    return (iw / ow if ow else 1.0), (ih / oh if oh else 1.0)


def _cell_to_pixel(predictor, r: int, c: int) -> tuple[float, float]:
    """Feature cell (r, c) centre -> pixel in the (unpadded) tile image."""
    sx, sy = _scale(predictor)
    rx = c * _FEAT_STRIDE + _FEAT_STRIDE / 2.0
    ry = r * _FEAT_STRIDE + _FEAT_STRIDE / 2.0
    return rx / sx, ry / sy


def _grid_bbox(predictor, box_local, gh: int, gw: int):
    """Tile-local pixel box (x0, y0, x1, y1) -> clamped feature-cell (r0, r1, c0, c1)."""
    sx, sy = _scale(predictor)
    x0, y0, x1, y1 = box_local
    c0 = max(0, int(math.floor(x0 * sx / _FEAT_STRIDE)))
    c1 = min(gw, max(c0 + 1, int(math.ceil(x1 * sx / _FEAT_STRIDE))))
    r0 = max(0, int(math.floor(y0 * sy / _FEAT_STRIDE)))
    r1 = min(gh, max(r0 + 1, int(math.ceil(y1 * sy / _FEAT_STRIDE))))
    return r0, r1, c0, c1


def _exemplar(predictor, featn, sub_bgr: np.ndarray, box_local):
    """Sample the exemplar descriptor + kernel from a feature map encoded at the
    **same scale as the target tiles** (avoids the scale mismatch you get from
    encoding the tiny crop on its own, which SAM would upscale to 1024).

    Returns ``(target_emb, kernel)``: the L2-normalized mean of the exemplar's
    foreground cells (PerSAM location prior) and the per-cell unit feature patch
    covering the exemplar box (TMR correlation kernel).
    """
    import torch

    gh, gw = featn.shape[1], featn.shape[2]
    r0, r1, c0, c1 = _grid_bbox(predictor, box_local, gh, gw)
    patch = featn[:, r0:r1, c0:c1].contiguous()  # [C, kh, kw], unit per cell
    kh, kw = patch.shape[1], patch.shape[2]

    x0, y0, x1, y1 = (int(round(v)) for v in box_local)
    crop = sub_bgr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size:
        fg = _fg_mask(_to_gray(crop)).astype(np.uint8)
        fg_g = cv2.resize(fg, (kw, kh), interpolation=cv2.INTER_NEAREST) > 0
    else:
        fg_g = np.ones((kh, kw), dtype=bool)
    if fg_g.sum() < 1:
        fg_g = np.ones((kh, kw), dtype=bool)

    fg_t = torch.as_tensor(fg_g, device=patch.device)
    target_emb = _l2norm(patch.permute(1, 2, 0)[fg_t].mean(0), dim=0)  # [C]
    return target_emb, patch


def _tile_for_box(tiles, cx: float, cy: float) -> int:
    """Index of the tile whose interior most fully contains point (cx, cy)."""
    best_t, best_margin = 0, -1e18
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        margin = min(cx - x0, x1 - cx, cy - y0, y1 - cy)
        if margin > best_margin:
            best_margin, best_t = margin, ti
    return best_t


def _heatmap(featn, method: str, target_emb, kernel, scales):
    """Per-cell match score map for one tile's normalized feature map."""
    import torch
    import torch.nn.functional as F

    if method == "persam":
        # Cosine similarity to the pooled exemplar embedding (unit vectors).
        sim = torch.einsum("chw,c->hw", featn, target_emb)
        return sim.detach().cpu().numpy()

    # TMR: multi-scale normalized cross-correlation with the exemplar kernel.
    C, gh, gw = featn.shape
    x = featn.unsqueeze(0)  # [1, C, gh, gw]
    kh0, kw0 = int(kernel.shape[1]), int(kernel.shape[2])
    best = None
    for s in scales or [1.0]:
        kh = max(1, int(round(kh0 * s)))
        kw = max(1, int(round(kw0 * s)))
        if kh > gh or kw > gw:
            continue
        k = kernel.unsqueeze(0)
        if (kh, kw) != (kh0, kw0):
            k = F.interpolate(k, size=(kh, kw), mode="bilinear", align_corners=False)
            k = _l2norm(k[0], dim=0).unsqueeze(0)  # renormalize per cell after resize
        corr = F.conv2d(x, k, padding=(kh // 2, kw // 2))[0, 0] / float(kh * kw)
        corr = corr[:gh, :gw]
        best = corr if best is None else torch.maximum(best, corr)
    if best is None:  # kernel bigger than every tile grid -> fall back to cosine
        best = torch.einsum("chw,c->hw", featn, _l2norm(kernel.mean((1, 2)), 0))
    return best.detach().cpu().numpy()


def _peaks(heat: np.ndarray, thr: float, min_dist: int, topk: int) -> list[tuple[int, int, float]]:
    """Threshold + greedy grid NMS -> list of (row, col, score) peak cells."""
    gh, gw = heat.shape
    ys, xs = np.where(heat >= thr)
    if ys.size == 0:
        return []
    order = np.argsort(heat[ys, xs])[::-1][:topk]
    ys, xs = ys[order], xs[order]
    taken = np.zeros((gh, gw), dtype=bool)
    md = max(1, int(min_dist))
    out: list[tuple[int, int, float]] = []
    for r, c in zip(ys.tolist(), xs.tolist()):
        if taken[max(0, r - md): r + md + 1, max(0, c - md): c + md + 1].any():
            continue
        taken[r, c] = True
        out.append((r, c, float(heat[r, c])))
    return out


def _decode_box(predictor, px, py, neg_xy, min_px, max_px, shape):
    """Prompt the (already-encoded) SAM decoder at (px, py); return a tight bbox."""
    pts = [[float(px), float(py)]]
    labels = [1]
    if neg_xy is not None:
        pts.append([float(neg_xy[0]), float(neg_xy[1])])
        labels.append(0)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(pts, dtype=np.float32),
        point_labels=np.array(labels),
        multimask_output=True,
        hq_token_only=True,
    )
    hs, ws = shape[:2]
    icx, icy = int(round(px)), int(round(py))
    if not (0 <= icx < ws and 0 <= icy < hs):
        return None
    best, best_key = None, None
    for mk, sc in zip(masks, scores):
        if not mk[icy, icx]:
            continue
        ys, xs = np.where(mk)
        if xs.size == 0:
            continue
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = bx1 - bx0, by1 - by0
        if bw < 1 or bh < 1 or bw > max_px or bh > max_px:
            continue
        key = (float(sc), -(bw * bh))
        if best_key is None or key > best_key:
            best_key, best = key, (bx0, by0, bx1, by1)
    if best is None:
        return None
    bx0, by0, bx1, by1 = best
    if (bx1 - bx0) < min_px:
        cx = (bx0 + bx1) / 2.0
        bx0, bx1 = int(round(cx - min_px / 2)), int(round(cx + min_px / 2))
    if (by1 - by0) < min_px:
        cy = (by0 + by1) / 2.0
        by0, by1 = int(round(cy - min_px / 2)), int(round(cy + min_px / 2))
    return bx0, by0, bx1, by1


def match_template(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    method: str = "tmr",
    threshold: float = 0.7,
    scales: list[float] | None = None,
    tile: int = 1024,
    overlap: int = 128,
    min_symbol_px: int = 6,
    max_symbol_px: int = 220,
    pad: int = 2,
    peak_min_dist: int = 2,
    topk_per_tile: int = 250,
    max_matches: int = 4000,
    nms_iou: float = 0.35,
) -> list[Match]:
    """Find every region similar to the exemplar ``box`` via SAM-feature matching.

    Parameters
    ----------
    box : ``(x, y, w, h)`` exemplar rectangle in original image pixels.
    method : ``"tmr"`` (feature cross-correlation) or ``"persam"`` (cosine prior).
    threshold : minimum match score (cosine / correlation, 0-1) to keep a peak.
    scales : TMR kernel scale factors (defaults span 0.75x - 1.5x).
    tile / overlap : tiling of the target sheet (px) for the dense encoder.
    min_symbol_px / max_symbol_px : reject decoder masks outside this size range.
    pad : pixels added around each refined box.
    """
    x, y, w, h = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    if w < 3 or h < 3:
        return []
    if scales is None:
        scales = [0.75, 1.0, 1.5]

    method = method.lower()
    predictor = sm.get_predictor()

    import progress as pg

    H, W = image_bgr.shape[:2]
    tiles = tl.tile_grid(W, H, tile, overlap)

    # Sample the exemplar from its containing tile so its features share the
    # target tiles' pixel scale (encoding the bare crop would rescale it to 1024).
    ex_ti = _tile_for_box(tiles, x + w / 2.0, y + h / 2.0)
    etx0, ety0, etx1, ety1 = tiles[ex_ti]
    esub = image_bgr[ety0:ety1, etx0:etx1]
    predictor.set_image(cv2.cvtColor(esub, cv2.COLOR_BGR2RGB))
    egh, egw = _valid_grid(predictor)
    efeatn = _l2norm(predictor.get_image_embedding()[0][:, :egh, :egw], dim=0)
    box_local = (x - etx0, y - ety0, x + w - etx0, y + h - ety0)
    target_emb, kernel = _exemplar(predictor, efeatn, esub, box_local)

    prog = pg.Progress(len(tiles), f"{method.upper()} match {len(tiles)}tiles", every=1)

    raw: list[Match] = []
    for (tx0, ty0, tx1, ty1) in tiles:
        prog.update()
        sub = image_bgr[ty0:ty1, tx0:tx1]
        if sub.shape[0] < 8 or sub.shape[1] < 8:
            continue
        predictor.set_image(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
        gh, gw = _valid_grid(predictor)
        featn = _l2norm(predictor.get_image_embedding()[0][:, :gh, :gw], dim=0)
        heat = _heatmap(featn, method, target_emb, kernel, scales)

        neg_xy = None
        if method == "persam":
            nr, nc = np.unravel_index(int(np.argmin(heat)), heat.shape)
            neg_xy = _cell_to_pixel(predictor, int(nr), int(nc))

        for r, c, score in _peaks(heat, threshold, peak_min_dist, topk_per_tile):
            px, py = _cell_to_pixel(predictor, r, c)
            bx = _decode_box(predictor, px, py, neg_xy, min_symbol_px, max_symbol_px, sub.shape)
            if bx is None:
                continue
            bx0, by0, bx1, by1 = bx
            raw.append(
                Match(
                    tx0 + bx0 - pad, ty0 + by0 - pad,
                    (bx1 - bx0) + 2 * pad, (by1 - by0) + 2 * pad,
                    float(score),
                )
            )
    prog.done()

    if len(raw) > max_matches:
        raw = sorted(raw, key=lambda m: m.score, reverse=True)[:max_matches]
    kept = _nms(raw, iou_thresh=nms_iou)

    for m in kept:  # clip to image bounds
        m.x = max(0, m.x)
        m.y = max(0, m.y)
        m.w = min(W - m.x, m.w)
        m.h = min(H - m.y, m.h)
    return [m for m in kept if m.w > 0 and m.h > 0]

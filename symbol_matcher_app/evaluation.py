"""Evaluate predicted symbol boxes against ground-truth polygon patches.

Two matching criteria are reported side by side:

- **Center hit** (detection): a GT polygon counts as found if some predicted box
  contains its center ``(x + w/2, y + h/2)``; a predicted box counts as a true
  positive if it contains at least one GT center. This is robust to the scale
  mismatch between point-derived boxes and polygon patches.
- **BBox alignment** (IoU): predictions are greedily matched to GT boxes by IoU
  at a threshold ``iou_thr``; we report precision / recall / F1 plus the mean IoU
  of the matched pairs (how tightly the boxes line up).

``evaluate`` also tags every prediction (``tp``/``fp``) and every GT
(``matched``/``missed``) so the UI can color the overlay.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]  # (x, y, w, h)


def iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _center(b: Box) -> tuple[float, float]:
    x, y, w, h = b
    return x + w / 2.0, y + h / 2.0


def _contains(box: Box, px: float, py: float) -> bool:
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def _prf(tp: int, n_pred: int, n_gt: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate(pred_boxes: list[Box], gt_boxes: list[Box], iou_thr: float = 0.5) -> dict:
    """Score ``pred_boxes`` against ``gt_boxes`` with both criteria.

    Returns a dict with ``n_pred``, ``n_gt``, a ``center`` block, an ``iou`` block
    (including ``mean_iou``), and ``pred_status`` / ``gt_status`` lists for the UI.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    gt_centers = [_center(g) for g in gt_boxes]

    # --- Center-hit detection -------------------------------------------------
    gt_hit = [False] * n_gt
    pred_hit = [False] * n_pred
    for pi, pb in enumerate(pred_boxes):
        for gi, (gx, gy) in enumerate(gt_centers):
            if _contains(pb, gx, gy):
                pred_hit[pi] = True
                gt_hit[gi] = True
    center_tp = sum(gt_hit)  # GT found (recall numerator)
    center = _prf(sum(pred_hit), n_pred, n_gt)
    center["gt_found"] = center_tp
    center["pred_hits"] = sum(pred_hit)

    # --- BBox alignment via greedy IoU matching ------------------------------
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if v >= iou_thr:
                pairs.append((v, pi, gi))
    pairs.sort(reverse=True)
    used_pred, used_gt = set(), set()
    matched_iou = []
    pred_tp = [False] * n_pred
    for v, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched_iou.append(v)
        pred_tp[pi] = True
    n_match = len(matched_iou)
    iou_block = _prf(n_match, n_pred, n_gt)
    iou_block["mean_iou"] = round(sum(matched_iou) / n_match, 4) if n_match else 0.0
    iou_block["matches"] = n_match

    return {
        "mode": "bboxes",
        "n_pred": n_pred,
        "n_gt": n_gt,
        "iou_thr": iou_thr,
        "center": center,
        "iou": iou_block,
        # Status (and overlay coloring) is IoU-based: a prediction is a TP only if
        # it aligns with a GT box at ``iou_thr``. Center-hit stays informational.
        "pred_status": ["tp" if pred_tp[i] else "fp" for i in range(n_pred)],
        "gt_status": ["matched" if i in used_gt else "missed" for i in range(n_gt)],
    }


def evaluate_points(pred_boxes: list[Box], gt_points: list[tuple[float, float]]) -> dict:
    """Score ``pred_boxes`` against ground-truth **points** by containment.

    Used when the GT is point features (no box extent), so IoU is undefined. A GT
    point is *found* if some predicted box contains it; a predicted box is a TP if
    it contains at least one GT point. Returns the same shape as ``evaluate`` with a
    zeroed ``iou`` block so callers can treat both modes uniformly.
    """
    n_pred, n_gt = len(pred_boxes), len(gt_points)
    gt_hit = [False] * n_gt
    pred_hit = [False] * n_pred
    for pi, pb in enumerate(pred_boxes):
        for gi, (gx, gy) in enumerate(gt_points):
            if _contains(pb, gx, gy):
                pred_hit[pi] = True
                gt_hit[gi] = True
    gt_found = sum(gt_hit)
    precision = sum(pred_hit) / n_pred if n_pred else 0.0
    recall = gt_found / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    center = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_found": gt_found,
        "pred_hits": sum(pred_hit),
    }
    return {
        "mode": "points",
        "n_pred": n_pred,
        "n_gt": n_gt,
        "iou_thr": 0.0,
        "center": center,
        "iou": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "matches": 0},
        "pred_status": ["tp" if pred_hit[i] else "fp" for i in range(n_pred)],
        "gt_status": ["matched" if gt_hit[i] else "missed" for i in range(n_gt)],
    }

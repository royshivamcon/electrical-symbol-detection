"""Segmentation-model adapters for the unified box pipeline.

Each adapter encodes a small crop around one reference point and returns that
point's symbol mask/box. Two backends share one interface so the caller only ever
calls ``encode_predict`` and switching model is a one-line change:

- ``FastAdapter``: FastSAM (YOLOv8-seg "segment everything" on the crop, then union
  the masks that cover the point).
- ``HQAdapter``: Light HQ-SAM (``vit_tiny`` + high-quality head), point-prompted with
  optional neighbour **negatives** and a **refine** pass for dense regions.

The per-point crop / grow-on-clip / finalize / fallback scaffold lives in
``boxes_from_points``; this module is only "encode a crop -> mask for the point".
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FASTSAM_CKPT = PROJECT_ROOT / "models" / "FastSAM-s.pt"
HQSAM_CKPT = PROJECT_ROOT / "models" / "sam_hq_vit_tiny.pth"

# HQ-SAM masks hug the ink tightly and can clip the symbol, so it runs with a
# larger box margin than FastSAM: callers add this to the shared ``pad``.
HQ_EXTRA_PAD = 3


@dataclass
class Cfg:
    """Mask-selection knobs passed per ``encode_predict`` call.

    Each adapter uses the subset it needs and ignores the rest.
    """

    min_box_px: int = 16
    max_symbol_px: int = 90
    max_box_frac: float = 0.4
    size_ratio: float = 1.6   # FastSAM: union masks within this x the median area
    imgsz: int = 1024         # FastSAM input size
    conf: float = 0.25        # FastSAM confidence
    iou: float = 0.9          # FastSAM NMS IoU
    refine: bool = True       # HQ-SAM refinement pass


@dataclass
class Pred:
    """Adapter output for one point: bbox in crop coords + optional full-crop mask."""

    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    clipped: bool                     # bbox touches the crop edge (-> grow)
    mask: np.ndarray | None = None    # full crop-sized boolean mask, or None


def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --- FastSAM helpers -------------------------------------------------------
def _covering_masks(res, cx, cy, ch, cw, crop_area, max_box_frac):
    """Every non-background mask covering (cx, cy) as (area, bbox, conf, mask)."""
    covering = []
    if res.masks is None:
        return covering
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
            continue
        covering.append((area, bb, float(cf), mk))
    return covering


def _union_full(used, ch, cw):
    """OR the selected masks (resized to the crop) onto a full (ch, cw) canvas."""
    m = np.zeros((ch, cw), dtype=bool)
    for _, _, _, mk in used:
        b = mk > 0.5
        if b.shape != (ch, cw):
            b = cv2.resize(b.astype(np.uint8), (cw, ch),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        m |= b
    return m


def _mask_extent(used, ch, cw):
    """Tight (ux0, uy0, ux1, uy1, union) of the OR'd masks in ``used``, from the
    mask pixels themselves (matches the segment-everything localizer), not the YOLO
    detection boxes which run looser than the mask. Returns None if the union is
    empty."""
    union = _union_full(used, ch, cw)
    ys, xs = np.where(union)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), union


def _select_from_covering(covering, size_ratio, max_symbol_px, ch, cw, want_mask):
    """Union the near-median-size covering masks; fall back to the smallest core
    mask if that union is still oversized. Returns (ux0, uy0, ux1, uy1, score, mask)
    where mask is a full (ch, cw) bool (or None when ``want_mask`` is False).

    The box extent is derived from the actual mask pixels, so it hugs the glyph the
    same way the segment-everything localizer does (not the looser YOLO boxes)."""
    covering.sort(key=lambda t: t[0])
    areas = [a for a, _, _, _ in covering]
    med = areas[len(areas) // 2]
    kept = [(a, bb, cf, mk) for a, bb, cf, mk in covering if a <= size_ratio * med]
    score = max(cf for _, _, cf, _ in kept)
    ext = _mask_extent(kept, ch, cw)
    if ext is None:  # empty union (shouldn't happen); fall back to detection boxes
        ux0 = min(bb[0] for _, bb, _, _ in kept)
        uy0 = min(bb[1] for _, bb, _, _ in kept)
        ux1 = max(bb[2] for _, bb, _, _ in kept)
        uy1 = max(bb[3] for _, bb, _, _ in kept)
        union = _union_full(kept, ch, cw)
    else:
        ux0, uy0, ux1, uy1, union = ext
    if (ux1 - ux0) > max_symbol_px or (uy1 - uy0) > max_symbol_px:
        _, bb, score, _ = covering[0]
        ext = _mask_extent([covering[0]], ch, cw)
        if ext is not None:
            ux0, uy0, ux1, uy1, union = ext
        else:
            ux0, uy0, ux1, uy1 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            union = _union_full([covering[0]], ch, cw)
    mask = union if want_mask else None
    return ux0, uy0, ux1, uy1, score, mask


# --- HQ-SAM helpers --------------------------------------------------------
def _mask_bbox(mask, cy, cx, min_box_px, max_box_frac, crop_area, max_symbol_px):
    """Return (bbox, area, within_max) for a mask if it covers the point and
    isn't background-sized, else None."""
    if not mask[cy, cx]:
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


# --- adapters --------------------------------------------------------------
class FastAdapter:
    name = "fastsam"
    needs_negatives = False

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def get_model(self):
        """Lazily load a single shared FastSAM model."""
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from ultralytics import FastSAM

                path = str(FASTSAM_CKPT) if FASTSAM_CKPT.exists() else "FastSAM-s.pt"
                self._model = FastSAM(path)
        return self._model

    def encode_predict(self, model_crop, cx, cy, negatives, *, want_mask, cfg: Cfg):
        model = self.get_model()
        ch, cw = model_crop.shape[:2]
        res = model(model_crop, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                    retina_masks=True, verbose=False)[0]
        covering = _covering_masks(res, cx, cy, ch, cw, ch * cw, cfg.max_box_frac)
        if not covering:
            return None
        ux0, uy0, ux1, uy1, score, mask = _select_from_covering(
            covering, cfg.size_ratio, cfg.max_symbol_px, ch, cw, want_mask)
        clipped = ux0 <= 1 or uy0 <= 1 or ux1 >= cw - 2 or uy1 >= ch - 2
        return Pred(int(round(ux0)), int(round(uy0)), int(round(ux1)), int(round(uy1)),
                    float(score), clipped, mask)


class HQAdapter:
    name = "hqsam"
    needs_negatives = True

    def __init__(self):
        self._predictor = None
        self._lock = threading.Lock()

    def get_predictor(self):
        """Lazily build a single shared Light HQ-SAM predictor."""
        if self._predictor is not None:
            return self._predictor
        with self._lock:
            if self._predictor is None:
                import torch
                from segment_anything_hq import SamPredictor, sam_model_registry

                if not HQSAM_CKPT.exists():
                    raise FileNotFoundError(f"HQ-SAM checkpoint not found: {HQSAM_CKPT}")
                sam = sam_model_registry["vit_tiny"](checkpoint=None)
                sam.load_state_dict(torch.load(str(HQSAM_CKPT), map_location="cpu"))
                sam.to(_device())
                sam.eval()
                self._predictor = SamPredictor(sam)
        return self._predictor

    def encode_predict(self, model_crop, cx, cy, negatives, *, want_mask, cfg: Cfg):
        predictor = self.get_predictor()
        ch, cw = model_crop.shape[:2]
        area_cap = cfg.max_box_frac * ch * cw
        predictor.set_image(cv2.cvtColor(model_crop, cv2.COLOR_BGR2RGB))
        masks, scores, logits = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=True,
            hq_token_only=True,
        )
        best, best_key, best_mask = None, None, None
        for k, (mk, sc) in enumerate(zip(masks, scores)):
            r = _mask_bbox(mk, cy, cx, cfg.min_box_px, 1.0, area_cap, cfg.max_symbol_px)
            if r is None:
                continue
            bbox, area, within = r
            key = (within, float(sc), -area)
            if best_key is None or key > best_key:
                best_key, best = key, (bbox, float(sc), k)
                best_mask = mk if want_mask else None
        if best is not None and cfg.refine:
            pcpts = [[cx, cy]] + list(negatives)
            pl = [1] + [0] * len(negatives)
            r_masks, r_scores, _ = predictor.predict(
                point_coords=np.array(pcpts),
                point_labels=np.array(pl),
                mask_input=logits[best[2]][None, :, :],
                multimask_output=False,
                hq_token_only=True,
            )
            r = _mask_bbox(r_masks[0], cy, cx, cfg.min_box_px, 1.0, area_cap, cfg.max_symbol_px)
            if r is not None:
                best = (r[0], float(r_scores[0]), best[2])
                best_mask = r_masks[0] if want_mask else None
        if best is None:
            return None
        (bx0, by0, bx1, by1), sc, _ = best
        clipped = bx0 <= 1 or by0 <= 1 or bx1 >= cw - 2 or by1 >= ch - 2
        return Pred(bx0, by0, bx1, by1, float(sc), clipped,
                    np.asarray(best_mask, bool) if best_mask is not None else None)


_ADAPTERS: dict[str, object] = {}
_ADAPTERS_LOCK = threading.Lock()


def _make_adapter(name: str):
    if name == "fastsam":
        return FastAdapter()
    if name == "hqsam":
        return HQAdapter()
    raise ValueError(f"unknown seg model: {name!r}")


def get_model(name: str):
    """Return a cached adapter singleton for ``name`` ("fastsam" | "hqsam")."""
    a = _ADAPTERS.get(name)
    if a is not None:
        return a
    with _ADAPTERS_LOCK:
        if name not in _ADAPTERS:
            _ADAPTERS[name] = _make_adapter(name)
    return _ADAPTERS[name]


# Per-thread adapters for the parallel box finder. The FastSAM/HQ-SAM adapters
# hold mutable inference state (HQ-SAM's predictor stores the encoded image
# between ``set_image``/``predict``), so a shared singleton cannot be called
# from several worker threads at once. Each worker thread lazily builds and
# reuses its own adapter instead; the single-threaded path keeps ``get_model``.
_THREAD_LOCAL = threading.local()


def get_thread_model(name: str):
    """Return an adapter for ``name`` that is unique to the calling thread.

    Unlike ``get_model`` (one shared singleton), this hands each thread its own
    model instance so concurrent ``encode_predict`` calls don't clobber each
    other's state. Instances are cached per thread, so the model weights are
    loaded once per worker and reused across calls.
    """
    cache = getattr(_THREAD_LOCAL, "adapters", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL.adapters = cache
    a = cache.get(name)
    if a is None:
        a = _make_adapter(name)
        cache[name] = a
    return a


def get_predictor():
    """Shared HQ-SAM predictor (reused by feat_match's training-free matchers)."""
    return get_model("hqsam").get_predictor()

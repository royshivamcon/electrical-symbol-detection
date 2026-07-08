"""Unified per-point box finder for FastSAM / HQ-SAM (and their per-point ``mix``).

Detection is **always per-point**: a small crop is taken around each reference
point, optionally grown when the mask clips the crop edge, and the selected
``seg_models`` adapter returns the symbol mask/box. This single routine drives
every model -- only the ``encode_predict`` call differs.

**Tiling is a memory strategy, not a detection mode.** When ``tile > 0`` the sheet
is covered with overlapping tiles and the identical per-point routine runs inside
each tile (tiles are sliced from the passed image, or fetched via ``tile_provider``
for on-demand high-res rendering). It never changes how a symbol is boxed: each
point is assigned to exactly one tile and produces exactly one box (no NMS).

Per-model behaviour (negatives / refine / grow / nn-adaptive crop / extra pad) is
selected here from ``model`` so callers pass one flat set of knobs. FastSAM uses a
fixed ``crop`` half-window per point; HQ-SAM shrinks that window toward each point's
nearest neighbour (``crop_nn_frac``) so a dense cluster's crop never spans two
symbols -- the single-point encoder is far more sensitive to a neighbour in-frame
than FastSAM's segment-everything pass.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

import pseudocolor as pc
import seg_models as sm
import tiling as tl
from sam_boxes import RefPoint  # noqa: F401 (re-exported for callers)


def _resolve_workers(workers: int) -> int:
    """Number of worker threads to use. ``0`` = auto (env override or a small
    cap so we don't oversubscribe cores against PyTorch's own intra-op threads);
    ``1`` keeps the sequential path; any other value is used as-is (>= 1)."""
    if workers and workers > 0:
        return int(workers)
    env = os.environ.get("SYMBOL_MATCHER_WORKERS")
    if env:
        try:
            v = int(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return max(1, min(4, os.cpu_count() or 1))


# One reusable pool per worker-count, shared across requests, so each worker
# thread builds its per-thread model once (via ``sm.get_thread_model``) and
# reuses it instead of reloading weights on every call.
_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _get_executor(n_workers: int) -> ThreadPoolExecutor:
    ex = _EXECUTORS.get(n_workers)
    if ex is not None:
        return ex
    with _EXECUTORS_LOCK:
        if n_workers not in _EXECUTORS:
            _EXECUTORS[n_workers] = ThreadPoolExecutor(
                max_workers=n_workers, thread_name_prefix="bfp"
            )
    return _EXECUTORS[n_workers]


@dataclass
class SamBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    name: str
    source: str  # "fastsam" | "hqsam" | "cc"
    # Raw segmentation, only when ``collect_masks``/``postproc``. ``mask`` is a bool
    # array cropped to its bbox; ``mx``/``my`` are its global top-left corner.
    mask: np.ndarray | None = None
    mx: int = 0
    my: int = 0

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 4), "name": self.name, "source": self.source,
        }


def _cc_fallback(crop: np.ndarray, cx: int, cy: int, min_px: int):
    """Tight box of the dark connected component under (cx, cy) inside the crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lbl = labels[cy, cx]
        if lbl > 0:
            x, y, w, h, _ = stats[lbl]
            if w >= min_px and h >= min_px:
                return int(x), int(y), int(w), int(h)
    return None


# --- per-model behaviour ---------------------------------------------------
@dataclass
class _Flavor:
    use_negatives: bool
    refine: bool
    grow: bool
    extra_pad: int
    nn_adapt: bool = False


def _flavor(model: str, grow_on_clip: bool) -> _Flavor:
    if model == "fastsam":
        # Segment-everything on a fixed crop; no point negatives / grow / nn-adapt.
        return _Flavor(False, False, False, 0, False)
    if model == "hqsam":
        # Single-point encoder: neighbour negatives push the mask off crossing wires,
        # a refine pass tightens it to the glyph, and the nearest-neighbour adaptive
        # crop keeps the window from spanning two symbols in a dense cluster -- the
        # combination that made the old HQ-SAM boxes good. ``grow`` (off by default)
        # optionally starts tighter and enlarges only when the mask clips the edge.
        return _Flavor(True, True, grow_on_clip, 0, True)
    raise ValueError(f"unknown model: {model!r}")


def boxes_from_points(
    image_bgr: np.ndarray,
    points: list,
    model: str = "fastsam",
    crop: int = 50,
    max_box_frac: float = 1,
    min_box_px: int = 8,
    min_symbol_px: int = 8,
    max_symbol_px: int = 50,
    size_ratio: float = 1.4,
    crop_nn_frac: float = 0.75,
    pad: int = 0,
    max_negatives: int = 16,
    fallback: bool = True,
    filt: str = "none",
    ksize: int = 5,
    kernels: tuple[int, int, int] = (6, 8, 3),
    grow_on_clip: bool = False,
    start_crop_frac: float = 0.3,
    grow_factor: float = 1.2,
    max_grows: int = 2,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.9,
    tile: int = 0,
    tile_overlap: int = 96,
    collect_masks: bool = False,
    postproc: bool = False,
    tile_provider=None,
    image_shape: tuple[int, int] | None = None,
    workers: int = 0,
) -> list[SamBox]:
    """Return one box per reference point using ``model``.

    ``model`` = ``"mix"`` runs FastSAM and HQ-SAM per point and keeps the tighter
    (smaller-area) box for each point. ``tile > 0`` chunks the sheet for bounded
    memory only -- the per-point routine is identical inside each tile.

    When ``tile_provider`` is given (memory-bounded rendering) ``image_bgr`` may be
    ``None``; pass ``image_shape=(H, W)`` of the full rendered sheet so boxes clamp
    and tiles are laid out correctly without holding the whole image in RAM.
    """
    if not points:
        return []
    kw = dict(
        crop=crop,
        max_box_frac=max_box_frac, min_box_px=min_box_px, min_symbol_px=min_symbol_px,
        max_symbol_px=max_symbol_px, size_ratio=size_ratio, crop_nn_frac=crop_nn_frac,
        pad=pad,
        max_negatives=max_negatives, fallback=fallback, filt=filt, ksize=ksize,
        kernels=kernels, grow_on_clip=grow_on_clip, start_crop_frac=start_crop_frac,
        grow_factor=grow_factor, max_grows=max_grows, imgsz=imgsz, conf=conf, iou=iou,
        tile=tile, tile_overlap=tile_overlap, collect_masks=collect_masks,
        tile_provider=tile_provider, image_shape=image_shape, workers=workers,
    )
    if model == "mix":
        fp = _detect_dict(image_bgr, points, "fastsam", postproc=postproc, **kw)
        hp = _detect_dict(image_bgr, points, "hqsam", postproc=False, **kw)
        return _merge_mix(fp, hp)
    return list(_detect_dict(image_bgr, points, model, postproc=postproc, **kw).values())


def _merge_mix(fp: dict, hp: dict) -> list[SamBox]:
    """Per-point union of the two model dicts: keep the smaller-area box per point."""
    out: list[SamBox] = []
    for pi in set(fp) | set(hp):
        a, b = fp.get(pi), hp.get(pi)
        if a is not None and b is not None:
            out.append(a if a.w * a.h <= b.w * b.h else b)
        else:
            out.append(a if a is not None else b)
    return out


def _detect_dict(
    image_bgr, points, model, *, crop, max_box_frac,
    min_box_px, min_symbol_px, max_symbol_px, size_ratio, crop_nn_frac, pad,
    max_negatives, fallback, filt, ksize, kernels, grow_on_clip, start_crop_frac,
    grow_factor, max_grows, imgsz, conf, iou, tile, tile_overlap, collect_masks,
    postproc, tile_provider, image_shape, workers,
) -> dict:
    """Run one model over ``points`` and return {point_index: SamBox}."""
    import progress as pg

    n_workers = _resolve_workers(workers)
    fl = _flavor(model, grow_on_clip)
    # ``needs_negatives`` is a class-level constant, so we can read it off the
    # shared singleton without loading per-thread weights; ``_process`` fetches
    # the actual per-thread adapter for the (stateful) ``encode_predict`` call.
    needs_negatives = sm.get_model(model).needs_negatives
    adapter_name = model
    pad_eff = pad + fl.extra_pad
    # HQ-SAM keeps its raw grow-on-clip masks: the symbol_det ink-intersect/refine
    # rejects most of them at high zoom (recall 32->8 at 4x), so postproc is off.
    if model == "hqsam":
        postproc = False
    want_mask = collect_masks or postproc
    if postproc:
        import mask_postproc as mpp
    H, W = image_shape if image_shape is not None else image_bgr.shape[:2]
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    # HQ-SAM (``fl.nn_adapt``) sizes each crop by the distance to the nearest other
    # point so a crowded cluster gets a tight window (used in ``_process``); other
    # models keep the fixed ``crop`` and skip this O(n^2) pass. Coords are in the same
    # (possibly zoomed) space as ``crop``, so the adaptive size scales with zoom too.
    nn_dist = None
    if fl.nn_adapt and len(points) > 1:
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn_dist = np.sqrt(d2.min(1))
    cfg = sm.Cfg(min_box_px=min_box_px, max_symbol_px=max_symbol_px,
                 max_box_frac=max_box_frac, size_ratio=size_ratio, imgsz=imgsz,
                 conf=conf, iou=iou, refine=fl.refine)

    def _neg(i, ox, oy, x0, y0, half):
        if not (fl.use_negatives and needs_negatives):
            return []
        dx, dy = coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1]
        inside = (np.abs(dx) < half) & (np.abs(dy) < half)
        inside[i] = False
        idx = np.where(inside)[0]
        if not idx.size:
            return []
        idx = idx[np.argsort(dx[idx] ** 2 + dy[idx] ** 2)][:max_negatives]
        return [[int(coords[j, 0] - ox - x0), int(coords[j, 1] - oy - y0)] for j in idx]

    def _finalize(gx0, gy0, gx1, gy1, score, source):
        gx0, gy0, gx1, gy1 = gx0 - pad_eff, gy0 - pad_eff, gx1 + pad_eff, gy1 + pad_eff
        if (gx1 - gx0) < min_symbol_px:
            c = (gx0 + gx1) / 2
            gx0, gx1 = c - min_symbol_px / 2, c + min_symbol_px / 2
        if (gy1 - gy0) < min_symbol_px:
            c = (gy0 + gy1) / 2
            gy0, gy1 = c - min_symbol_px / 2, c + min_symbol_px / 2
        gx0, gy0 = max(0, int(round(gx0))), max(0, int(round(gy0)))
        gx1, gy1 = min(W, int(round(gx1))), min(H, int(round(gy1)))
        return gx0, gy0, gx1, gy1

    def _process(i, region_model, region_src, ox, oy):
        adapter = sm.get_thread_model(model)
        th, tw = region_src.shape[:2]
        p = points[i]
        lpx, lpy = p.x - ox, p.y - oy
        c_target = crop
        if fl.nn_adapt and nn_dist is not None:
            # Shrink the window toward the nearest neighbour so the crop can't span two
            # symbols in a dense cluster (old HQ-SAM's key mechanism). Floor at one
            # symbol so it never under-covers; cap at the configured ``crop``.
            floor = max(min_box_px, min_symbol_px)
            c_target = int(round(min(crop, max(floor, crop_nn_frac * nn_dist[i]))))
        c_cur = c_target
        if fl.grow:
            c_cur = max(min_box_px, min(c_target, int(round(start_crop_frac * c_target))))

        chosen = None
        chosen_mask = None
        for _ in range(max_grows + 1):
            x0, y0 = max(0, lpx - c_cur), max(0, lpy - c_cur)
            x1, y1 = min(tw, lpx + c_cur), min(th, lpy + c_cur)
            sub_in = region_model[y0:y1, x0:x1]
            ch, cw = sub_in.shape[:2]
            if ch < min_box_px or cw < min_box_px:
                break
            cx, cy = lpx - x0, lpy - y0
            neg = _neg(i, ox, oy, x0, y0, c_cur)
            ink_sub = mpp.ink_of(region_src[y0:y1, x0:x1]) if postproc else None
            pred = adapter.encode_predict(sub_in, cx, cy, neg, want_mask=want_mask, cfg=cfg)
            if pred is not None:
                bx0, by0, bx1, by1 = pred.x0, pred.y0, pred.x1, pred.y1
                local = None
                # postproc is refine-only: tighten the box/mask to the symbol_det ink
                # when refine succeeds, but keep the raw detection when it rejects.
                # (Prompts sit on real symbols, so dropping on reject is a false
                # negative -- e.g. it halved FastSAM recall at 4x.)
                if postproc and pred.mask is not None:
                    ref = mpp.refine(np.asarray(pred.mask, bool), ink_sub)
                    if ref is not None:
                        loc, (rx0, ry0, rw, rh) = ref
                        bx0, by0, bx1, by1 = rx0, ry0, rx0 + rw - 1, ry0 + rh - 1
                        local = (loc, rx0, ry0)
                chosen = (ox + x0 + bx0, oy + y0 + by0, ox + x0 + bx1, oy + y0 + by1,
                          pred.score)
                if want_mask and pred.mask is not None:
                    if local is not None:
                        loc, rx0, ry0 = local
                        chosen_mask = (loc.copy(), ox + x0 + rx0, oy + y0 + ry0)
                    else:
                        chosen_mask = (pred.mask[by0:by1 + 1, bx0:bx1 + 1].copy(),
                                       ox + x0 + bx0, oy + y0 + by0)
                if not pred.clipped or c_cur >= c_target:
                    break
            if c_cur >= c_target:
                break
            c_cur = min(c_target, max(c_cur + 1, int(round(c_cur * grow_factor))))

        if chosen is not None:
            gx0, gy0, gx1, gy1, sc = chosen
            fx0, fy0, fx1, fy1 = _finalize(gx0, gy0, gx1, gy1, sc, adapter_name)
            box = SamBox(fx0, fy0, fx1 - fx0, fy1 - fy0, sc, p.name, adapter_name)
            if chosen_mask is not None:
                box.mask, box.mx, box.my = chosen_mask
            return box

        if fallback:
            x0, y0 = max(0, lpx - c_target), max(0, lpy - c_target)
            x1, y1 = min(tw, lpx + c_target), min(th, lpy + c_target)
            sub = region_src[y0:y1, x0:x1]
            fh_c, fw_c = sub.shape[:2]
            fb = _cc_fallback(sub, lpx - x0, lpy - y0, min_box_px)
            if fb is not None:
                fx, fy, fw, fh = fb
                if fw <= max_box_frac * fw_c or fh <= max_box_frac * fh_c:
                    fx0, fy0, fx1, fy1 = _finalize(
                        ox + x0 + fx, oy + y0 + fy, ox + x0 + fx + fw, oy + y0 + fy + fh,
                        0.0, "cc")
                    return SamBox(fx0, fy0, fx1 - fx0, fy1 - fy0, 0.0, p.name, "cc")
        return None

    def _iter_processed(pis, region_model, region_src, ox, oy):
        """Yield ``(point_index, box)`` for each index in ``pis``. Runs on the
        shared thread pool when ``n_workers > 1`` (each worker uses its own
        per-thread model), otherwise sequentially."""
        if n_workers <= 1 or len(pis) <= 1:
            for i in pis:
                yield i, _process(i, region_model, region_src, ox, oy)
            return
        ex = _get_executor(n_workers)
        futs = {ex.submit(_process, i, region_model, region_src, ox, oy): i for i in pis}
        for fut in as_completed(futs):
            yield futs[fut], fut.result()

    result: dict[int, SamBox] = {}
    if tile and tile > 0:
        ov = max(tile_overlap, 2 * crop)
        tiles = tl.tile_grid(W, H, tile, ov)
        assign = tl.assign_points_to_tiles(points, tiles)
        n_tiles = sum(1 for ti in range(len(tiles)) if assign.get(ti))
        prog = pg.Progress(n_tiles, f"{model} tiled@{tile} {len(points)}pts/{n_tiles}tiles"
                           f" x{n_workers}w", every=1)
        # Tiles run sequentially (the ``tile_provider`` PDF renderer is not
        # thread-safe); the points inside each rendered tile run in parallel.
        for ti, (tx0, ty0, tx1, ty1) in enumerate(tiles):
            pis = assign.get(ti, [])
            if not pis:
                continue
            prog.update(note=f"{len(pis)}pts")
            region_src = tile_provider(tx0, ty0, tx1, ty1) if tile_provider is not None \
                else image_bgr[ty0:ty1, tx0:tx1]
            region_model = pc.preprocess(region_src, filt=filt, ksize=ksize, kernels=kernels)
            for i, box in _iter_processed(pis, region_model, region_src, tx0, ty0):
                if box is not None:
                    result[i] = box
        prog.done()
    else:
        region_model = pc.preprocess(image_bgr, filt=filt, ksize=ksize, kernels=kernels)
        prog = pg.Progress(len(points), f"{model} {len(points)}pts x{n_workers}w")
        for i, box in _iter_processed(range(len(points)), region_model, image_bgr, 0, 0):
            prog.update()
            if box is not None:
                result[i] = box
        prog.done()
    return result

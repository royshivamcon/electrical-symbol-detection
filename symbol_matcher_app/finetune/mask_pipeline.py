"""Whole-sheet segment-everything mask creation pipeline.

Shared by ``prep_dataset.py``, ``visualise.py``, and the live app's
``/segment_masks`` API.  Tiles the sheet, runs frozen FastSAM / FastSAM-X on each
tile (optionally after a preprocessing filter and/or symbol_det processed view),
filters candidates by size, and optionally re-scores with the trained
``MaskConfidenceNet`` head.

Run standalone for debugging::

    ../.envs/vsam/bin/python finetune/mask_pipeline.py --rid <id> --wid <id>
"""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent
for _p in (str(APP_DIR), str(FT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mask_colors as mc  # noqa: E402
import pseudocolor as pc  # noqa: E402
import progress as pg  # noqa: E402
import seg_models as sm  # noqa: E402
import tiling as tl  # noqa: E402
import worksheet_loader as wl  # noqa: E402
from config import PrepCfg  # noqa: E402


RenderFn = Callable[[int, int, int, int], np.ndarray]


@dataclass
class ScanCfg:
    """Knobs for a whole-sheet segment-everything scan."""

    zoom: float = 4.0
    remove_text: bool = True
    proc_view: str = "original"  # original | binary | suppressed
    tile: int = 1024
    overlap: int = 128
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.9
    min_px: int = 12
    max_frac: float = 0.1
    batch: int = 8
    chunk: int = 64
    # preprocessing filter fed to FastSAM (same names as pseudocolor.preprocess)
    filt: str = "none"
    ksize: int = 5
    kernels: tuple[int, int, int] = (6, 8, 3)
    # optional head re-scoring
    use_head: bool = False
    head_ckpt: str | Path | None = None
    min_score: float = 0.0
    zcrop: int | None = None  # centroid window for head; default from PrepCfg.rendered_gates


def _tile_windows(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """``(ty0, tx0, ty1, tx1)`` windows covering a ``h x w`` sheet."""
    return [
        (ty0, tx0, ty1, tx1)
        for tx0, ty0, tx1, ty1 in tl.tile_grid(w, h, tile, overlap)
    ]


def _apply_proc_view(gray: np.ndarray, proc_view: str) -> np.ndarray:
    """symbol_det processed view as BGR (black ink on white, matching the UI)."""
    import mask_postproc as mpp

    if proc_view == "binary":
        m = (mpp.ink_of(gray).astype(np.uint8)) * 255
    elif proc_view == "suppressed":
        m = mpp.suppressed_of(gray)
    else:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = 255 - m
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _prepare_tile(crop_bgr: np.ndarray, cfg: ScanCfg) -> np.ndarray:
    """Apply processing view + preprocessing filter before FastSAM."""
    if cfg.proc_view != "original":
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        crop_bgr = _apply_proc_view(gray, cfg.proc_view)
    if cfg.filt and cfg.filt != "none":
        crop_bgr = pc.preprocess(crop_bgr, filt=cfg.filt, ksize=cfg.ksize, kernels=cfg.kernels)
    return crop_bgr


@contextmanager
def tile_renderer(rid: str, wid: str, cfg: ScanCfg):
    """Yield ``(W, H, render_fn, base_W, base_H)``.

    ``render_fn(x0, y0, x1, y1)`` returns a BGR tile at rendered resolution.
    ``base_W``/``base_H`` are the worksheet's base raster dimensions (for mapping
    masks back to the UI coordinate system).
    """
    base = wl.load_worksheet_image(rid, wid)
    base_h, base_w = base.shape[:2]
    Z = float(cfg.zoom) if cfg.zoom and cfg.zoom > 1.0 else 1.0

    def wrap(render_fn: RenderFn) -> RenderFn:
        def fn(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
            return _prepare_tile(render_fn(x0, y0, x1, y1), cfg)
        return fn

    if Z > 1.0 or cfg.remove_text:
        with wl.pdf_tile_renderer(rid, wid, Z, remove_text=cfg.remove_text) as (tw, th, raw_fn):
            yield tw, th, wrap(raw_fn), base_w, base_h
    else:
        def full_fn(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
            return _prepare_tile(base[y0:y1, x0:x1].copy(), cfg)
        yield base_w, base_h, full_fn, base_w, base_h


def mask_candidates(
    res,
    ty0: int,
    tx0: int,
    ch: int,
    cw: int,
    min_px: int,
    max_frac: float,
) -> Iterator[tuple[np.ndarray, tuple[int, int, int, int], float]]:
    """Yield ``(mask_bool_tile, global_bbox, fastsam_conf)`` for each valid mask."""
    if res.masks is None or res.boxes is None:
        return
    masks = res.masks.data.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    tile_area = ch * cw
    for mk, cf in zip(masks, confs):
        b = mk > 0.5
        if b.shape != (ch, cw):
            b = cv2.resize(b.astype(np.uint8), (cw, ch),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.where(b)
        if xs.size == 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        w, h = x1 - x0, y1 - y0
        if min(w, h) < min_px or (w * h) > max_frac * tile_area:
            continue
        gbox = (x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0)
        yield b, gbox, float(cf)


@torch.no_grad()
def _head_scores(scorer, gray: np.ndarray, masks: list[np.ndarray],
                 confs: list[float], zcrop: int) -> np.ndarray:
    """Calibrated head confidence per mask (centroid-centred window)."""
    if not masks:
        return np.zeros((0,), np.float32)
    from features import build_feats, build_input

    ch, cw = gray.shape[:2]
    inputs, feats = [], []
    for m, cf in zip(masks, confs):
        ys, xs = np.where(m)
        mcx, mcy = float(xs.mean()), float(ys.mean())
        x0 = max(0, int(round(mcx - zcrop)))
        y0 = max(0, int(round(mcy - zcrop)))
        x1 = min(cw, int(round(mcx + zcrop)))
        y1 = min(ch, int(round(mcy + zcrop)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            x0, y0, x1, y1 = 0, 0, cw, ch
        sub_gray = gray[y0:y1, x0:x1]
        sub_mask = m[y0:y1, x0:x1]
        inputs.append(build_input(sub_gray, sub_mask, mcx - x0, mcy - y0, scorer.input_size))
        feats.append(build_feats(sub_mask, cf))
    t = torch.from_numpy(np.stack(inputs, 0)).to(scorer.device)
    ft = torch.from_numpy(np.stack(feats, 0)).to(scorer.device)
    logits = scorer.model(t, ft) / scorer.temperature
    return torch.sigmoid(logits).cpu().numpy().astype(np.float32)


@dataclass
class MaskHit:
    """One segment-everything mask in global rendered coordinates."""

    mask: np.ndarray  # bool, tile-local
    ty0: int
    tx0: int
    score: float
    conf: float  # raw FastSAM objectness


def scan_sheet(
    rid: str,
    wid: str,
    model_name: str,
    cfg: ScanCfg,
    *,
    limit_tiles: int = 0,
) -> tuple[list[MaskHit], int, int, int, int]:
    """Run segment-everything across the sheet.

    Returns ``(hits, render_W, render_H, base_W, base_H)``.
    """
    model = sm.get_model(model_name).get_model()
    prep = PrepCfg(zoom=cfg.zoom)
    zcrop = cfg.zcrop if cfg.zcrop is not None else prep.rendered_gates()[0]

    scorer = None
    if cfg.use_head:
        from infer import MaskConfidenceScorer
        from config import CKPT_PATH
        ckpt = Path(cfg.head_ckpt) if cfg.head_ckpt else CKPT_PATH
        if not ckpt.exists():
            raise FileNotFoundError(f"head checkpoint not found: {ckpt}")
        scorer = MaskConfidenceScorer.load(ckpt)

    device = sm._device()
    hits: list[MaskHit] = []

    with tile_renderer(rid, wid, cfg) as (rw, rh, render_fn, base_w, base_h):
        tile_list = _tile_windows(rh, rw, cfg.tile, cfg.overlap)
        if limit_tiles and limit_tiles < len(tile_list):
            import random
            tile_list = random.Random(0).sample(tile_list, limit_tiles)

        prog = pg.Progress(
            len(tile_list), f"Segment masks {model_name} {len(tile_list)}tiles", every=1,
        )
        n_chunks = max(1, (len(tile_list) + cfg.chunk - 1) // cfg.chunk)
        for ci, c0 in enumerate(range(0, len(tile_list), cfg.chunk)):
            chunk_tiles = tile_list[c0:c0 + cfg.chunk]
            chunk_crops = [render_fn(tx0, ty0, tx1, ty1) for (ty0, tx0, ty1, tx1) in chunk_tiles]

            for b0 in range(0, len(chunk_crops), cfg.batch):
                batch_tiles = chunk_tiles[b0:b0 + cfg.batch]
                batch_crops = chunk_crops[b0:b0 + cfg.batch]
                with torch.no_grad():
                    batch_res = model(
                        [sm._fastsam_rgb(c) for c in batch_crops],
                        imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                        retina_masks=True, verbose=False,
                    )

                for res, crop, (ty0, tx0, ty1, tx1) in zip(batch_res, batch_crops, batch_tiles):
                    ch, cw = crop.shape[:2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    cands = list(mask_candidates(res, ty0, tx0, ch, cw, cfg.min_px, cfg.max_frac))
                    if not cands:
                        continue
                    masks = [b for b, _, _ in cands]
                    confs = [cf for _, _, cf in cands]
                    if scorer is not None:
                        scores = _head_scores(scorer, gray, masks, confs, zcrop)
                    else:
                        scores = np.asarray(confs, dtype=np.float32)
                    for (b, _gbox, cf), sc in zip(cands, scores):
                        hits.append(MaskHit(mask=b, ty0=ty0, tx0=tx0, score=float(sc), conf=cf))

                del batch_res
            del chunk_crops
            prog.update(
                inc=len(chunk_tiles),
                note=f"chunk {ci + 1}/{n_chunks} · {len(hits)} masks",
            )
            if device == "mps" and (ci + 1) % 4 == 0:
                torch.mps.empty_cache()
        prog.done()

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return hits, rw, rh, base_w, base_h


def save_hits_cache(
    hits: list[MaskHit],
    path: Path,
    render_w: int,
    render_h: int,
    base_w: int,
    base_h: int,
) -> None:
    """Persist scan results so overlays can be re-filtered without re-running FastSAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        scores=np.array([h.score for h in hits], dtype=np.float32),
        confs=np.array([h.conf for h in hits], dtype=np.float32),
        ty0=np.array([h.ty0 for h in hits], dtype=np.int32),
        tx0=np.array([h.tx0 for h in hits], dtype=np.int32),
        masks=np.array([h.mask.astype(np.uint8) for h in hits], dtype=object),
        render_w=np.int32(render_w),
        render_h=np.int32(render_h),
        base_w=np.int32(base_w),
        base_h=np.int32(base_h),
    )


def load_hits_cache(path: Path) -> tuple[list[MaskHit], int, int, int, int]:
    """Load hits written by ``save_hits_cache``."""
    d = np.load(path, allow_pickle=True)
    hits = [
        MaskHit(
            mask=np.asarray(m, dtype=bool),
            ty0=int(ty0),
            tx0=int(tx0),
            score=float(sc),
            conf=float(cf),
        )
        for m, ty0, tx0, sc, cf in zip(d["masks"], d["ty0"], d["tx0"], d["scores"], d["confs"])
    ]
    return (
        hits,
        int(d["render_w"]),
        int(d["render_h"]),
        int(d["base_w"]),
        int(d["base_h"]),
    )


def render_rgba_overlay(
    hits: list[MaskHit],
    base_w: int,
    base_h: int,
    render_w: int,
    render_h: int,
    *,
    min_score: float = 0.0,
) -> np.ndarray:
    """Composite segment-everything masks onto a transparent BGRA canvas at base resolution."""
    canvas = np.zeros((render_h, render_w, 4), dtype=np.uint8)

    # Stable color index per hit; draw low-confidence visible masks first.
    visible = [(i, h) for i, h in enumerate(hits) if h.score >= min_score]
    ordered = sorted(visible, key=lambda x: x[1].score)
    for color_i, hit in ordered:
        m = hit.mask
        ty0, tx0 = hit.ty0, hit.tx0
        mh, mw = m.shape
        by1 = min(render_h, ty0 + mh)
        bx1 = min(render_w, tx0 + mw)
        if bx1 <= tx0 or by1 <= ty0:
            continue
        sub = m[:by1 - ty0, :bx1 - tx0]
        b, g, r, a = mc.mask_color_bgra(color_i)
        region = canvas[ty0:by1, tx0:bx1]
        region[sub, 0] = b
        region[sub, 1] = g
        region[sub, 2] = r
        region[sub, 3] = a

    if render_w != base_w or render_h != base_h:
        return cv2.resize(canvas, (base_w, base_h), interpolation=cv2.INTER_AREA)
    return canvas


def run_scan_overlay(
    rid: str,
    wid: str,
    model_name: str,
    cfg: ScanCfg,
    *,
    limit_tiles: int = 0,
    min_score: float | None = None,
) -> np.ndarray:
    """Convenience: scan + return base-resolution BGRA overlay."""
    hits, rw, rh, base_w, base_h = scan_sheet(rid, wid, model_name, cfg, limit_tiles=limit_tiles)
    thr = cfg.min_score if min_score is None else min_score
    return render_rgba_overlay(hits, base_w, base_h, rw, rh, min_score=thr)


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rid", required=True)
    ap.add_argument("--wid", required=True)
    ap.add_argument("--model", default="fastsam", choices=("fastsam", "fastsamx"))
    ap.add_argument("--filt", default="none")
    ap.add_argument("--ksize", type=int, default=5)
    ap.add_argument("--proc-view", default="original", choices=("original", "binary", "suppressed"))
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--use-head", action="store_true")
    ap.add_argument("--limit-tiles", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = ScanCfg(
        proc_view=args.proc_view,
        filt=args.filt,
        ksize=args.ksize,
        min_score=args.min_score,
        use_head=args.use_head,
    )
    overlay = run_scan_overlay(
        args.rid, args.wid, args.model, cfg,
        limit_tiles=args.limit_tiles, min_score=args.min_score,
    )
    out = Path(args.out) if args.out else (FT_DIR / "dataset" / "viz_sheets" / f"{args.rid[:8]}_{args.wid[:8]}" / "segment_masks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    n = int((overlay[:, :, 3] > 0).any(axis=1).sum())
    print(f"[mask_pipeline] wrote {out} ({overlay.shape[1]}x{overlay.shape[0]}, ~{n} mask rows)")


if __name__ == "__main__":
    _main()

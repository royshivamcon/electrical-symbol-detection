"""Training prep: build the mask-confidence dataset as a whole-sheet symbol detector.

Instead of labeling by "covers the centered reference point + size gate" (which forced
every crop to be point-centered and made training diverge from ``visualise.py``), we now
**target the app's own bounding boxes**:

1. Ground truth: run the app finder (``boxes_from_points``, ``gt_model``) at the sheet's
   electrical reference points -> per-symbol GT boxes (kept in rendered space).
2. Candidates: run frozen FastSAM **segment-everything** across the whole sheet in tiles
   (same scan as ``visualise.py``) -> every mask FastSAM proposes.
3. Label each candidate by IoU with the nearest GT box: ``>= pos_iou`` positive,
   ``< neg_iou`` negative, drop the ambiguous band between.
4. Save a centroid-centered crop per kept mask (same window ``visualise._head_scores``
   scores on) so training and inference see the same distribution.

The full ``base*zoom`` sheet is never materialized: tiles are rendered on demand via
``worksheet_loader.pdf_tile_renderer``. FastSAM runs under ``torch.no_grad()``.

Run from the app dir (so bare imports resolve):
    ../.envs/vsam/bin/python finetune/prep_dataset.py --limit-sheets 5
    ../.envs/vsam/bin/python finetune/prep_dataset.py --limit-rids 10 --sheets-per-rid 10
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import sys
import time
from pathlib import Path

FT_DIR = Path(__file__).resolve().parent
APP_DIR = FT_DIR.parent
for _p in (str(APP_DIR), str(FT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boxes_from_points as bfp  # noqa: E402
import sam_boxes as sb  # noqa: E402
import seg_models as sm  # noqa: E402
import worksheet_loader as wl  # noqa: E402
from config import CROPS_DIR, DATASET_DIR, MANIFEST, PrepCfg  # noqa: E402

FS_BATCH = 8       # FastSAM tiles per forward
CHUNK = 64         # tiles rendered per chunk (memory bound)


def _peak_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _tiles(h: int, w: int, tile: int, overlap: int):
    """Yield (y0, x0, y1, x1) tile windows covering the image with overlap
    (edge tiles clipped, not padded) -- same convention as ``visualise._tiles``."""
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y0 in ys:
        for x0 in xs:
            yield y0, x0, min(y0 + tile, h), min(x0 + tile, w)


def _iou(a: tuple, b: tuple) -> float:
    """IoU of two (x0, y0, x1, y1) boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gt_boxes(rid: str, wid: str, zpts: list, render_fn, TW: int, TH: int, cfg: PrepCfg):
    """App per-point boxes in RENDERED space (x0, y0, x1, y1).

    Mirrors ``run_sam``'s zoom branch: points are already scaled to rendered space, so
    the returned boxes stay in rendered space (no map-back to base)."""
    zcrop, zmax, zmin = cfg.rendered_gates()
    sam = bfp.boxes_from_points(
        None, zpts, model=cfg.gt_model, crop=zcrop, min_symbol_px=zmin,
        max_symbol_px=zmax, pad=cfg.pad, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
        tile=cfg.tile, tile_provider=render_fn, image_shape=(TH, TW), postproc=True,
    )
    return [(b.x, b.y, b.x + b.w, b.y + b.h) for b in sam if b.w > 0 and b.h > 0]


def _tile_candidates(res, ty0: int, tx0: int, ch: int, cw: int, cfg: PrepCfg):
    """Yield (mask_bool_tile, global_bbox, conf) for every FastSAM mask on a tile that
    passes the candidate size gate. ``global_bbox`` is (x0, y0, x1, y1) in full-sheet
    rendered coords; ``mask_bool_tile`` stays in tile-local coords."""
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
        if min(w, h) < cfg.cand_min_px or (w * h) > cfg.cand_max_frac * tile_area:
            continue
        gbox = (x0 + tx0, y0 + ty0, x1 + tx0, y1 + ty0)
        yield b, gbox, float(cf)


def _select(cands: list, neg_per_pos: int, seed: int) -> list:
    """Keep all positives + ``neg_per_pos * n_pos`` negatives (half near-miss by IoU,
    half random). ``cands`` items are dicts with 'label' and 'iou'."""
    pos = [c for c in cands if c["label"] == 1]
    neg = [c for c in cands if c["label"] == 0]
    if not pos:
        return []
    budget = neg_per_pos * len(pos)
    if len(neg) <= budget:
        return pos + neg
    rng = random.Random(seed)
    neg_sorted = sorted(neg, key=lambda c: c["iou"], reverse=True)  # near-miss first
    n_hard = budget // 2
    hard = neg_sorted[:n_hard]
    rest = neg_sorted[n_hard:]
    rng.shuffle(rest)
    return pos + hard + rest[:budget - n_hard]


def _assign_splits(rows: list, val_frac: float, seed: int) -> None:
    """Sheet-level train/val split; falls back to a per-npz split for a single sheet."""
    rng = random.Random(seed)
    sheets = sorted({(r["rid"], r["wid"]) for r in rows})
    if len(sheets) > 1:
        rng.shuffle(sheets)
        n_val = max(1, int(round(len(sheets) * val_frac)))
        val = set(sheets[:n_val])
        for r in rows:
            r["split"] = "val" if (r["rid"], r["wid"]) in val else "train"
        return
    npzs = sorted({r["npz"] for r in rows})
    rng.shuffle(npzs)
    n_val = max(1, int(round(len(npzs) * val_frac)))
    val = set(npzs[:n_val])
    for r in rows:
        r["split"] = "val" if r["npz"] in val else "train"


def _process_sheet(rid: str, wid: str, base_w: int, base_h: int, cfg: PrepCfg,
                   model, crops_dir: Path, rows: list, limit_points: int) -> dict:
    """Render one sheet's GT boxes + segment-everything candidates, label by IoU, cache
    the kept masks' centroid crops, and append manifest rows. Returns per-sheet stats."""
    Z = float(cfg.zoom) if cfg.zoom and cfg.zoom > 1.0 else 1.0
    zcrop, _zmax, _zmin = cfg.rendered_gates()

    pts = sb.load_reference_points(rid, wid, base_w, base_h)
    if cfg.electrical_only:
        pts = [p for p in pts if sb._is_electrical(p.name)]
    if not pts:
        return {"pts": 0, "gt": 0, "pos": 0, "neg": 0}
    if limit_points:
        pts = pts[:limit_points]
    zpts = [sb.RefPoint(int(round(p.x * Z)), int(round(p.y * Z)), p.name) for p in pts]

    device = sm._device()
    sheet_cands: list[dict] = []

    with wl.pdf_tile_renderer(rid, wid, Z, remove_text=cfg.remove_text) as (TW, TH, render_fn):
        gt = _gt_boxes(rid, wid, zpts, render_fn, TW, TH, cfg)
        if not gt:
            return {"pts": len(pts), "gt": 0, "pos": 0, "neg": 0}

        tiles = list(_tiles(TH, TW, cfg.tile, cfg.overlap))
        # pending stores (mask_tile, conf, window) alongside the label so we only cache
        # the crops we actually keep after per-sheet negative sampling.
        pending: list[dict] = []
        for c0 in range(0, len(tiles), CHUNK):
            chunk_tiles = tiles[c0:c0 + CHUNK]
            chunk_crops = [render_fn(tx0, ty0, tx1, ty1) for (ty0, tx0, ty1, tx1) in chunk_tiles]
            for b0 in range(0, len(chunk_crops), FS_BATCH):
                bt = chunk_tiles[b0:b0 + FS_BATCH]
                bc = chunk_crops[b0:b0 + FS_BATCH]
                with torch.no_grad():
                    bres = model([sm._fastsam_rgb(c) for c in bc], imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                                 retina_masks=True, verbose=False)
                for res, crop, (ty0, tx0, ty1, tx1) in zip(bres, bc, bt):
                    ch, cw = crop.shape[:2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    for b, gbox, cf in _tile_candidates(res, ty0, tx0, ch, cw, cfg):
                        miou = max((_iou(gbox, g) for g in gt), default=0.0)
                        if miou >= cfg.pos_iou:
                            label = 1
                        elif miou < cfg.neg_iou:
                            label = 0
                        else:
                            continue  # ambiguous band -> drop
                        ys, xs = np.where(b)
                        mcx, mcy = float(xs.mean()), float(ys.mean())
                        wx0 = max(0, int(round(mcx - zcrop))); wy0 = max(0, int(round(mcy - zcrop)))
                        wx1 = min(cw, int(round(mcx + zcrop))); wy1 = min(ch, int(round(mcy + zcrop)))
                        if wx1 - wx0 < 2 or wy1 - wy0 < 2:
                            continue
                        sub_gray = gray[wy0:wy1, wx0:wx1].copy()
                        sub_mask = b[wy0:wy1, wx0:wx1].copy()
                        gw, gh = gbox[2] - gbox[0], gbox[3] - gbox[1]
                        pending.append({
                            "label": label, "iou": round(miou, 4), "conf": round(cf, 5),
                            "gray": sub_gray, "mask": sub_mask,
                            "cx": mcx - wx0, "cy": mcy - wy0,
                            "w": gw, "h": gh, "area": gw * gh,
                        })
                del bres
            del chunk_crops
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()

        sheet_cands = _select(pending, cfg.neg_per_pos, cfg.seed)

    # cache crops + emit rows for the selected candidates
    n_pos = n_neg = 0
    for k, c in enumerate(sheet_cands):
        npz_name = f"{rid[:8]}_{wid[:8]}_{k:06d}.npz"
        ch, cw = c["gray"].shape[:2]
        np.savez_compressed(
            crops_dir / npz_name,
            gray=c["gray"],
            masks=np.packbits(c["mask"][None, ...]),
            mask_shape=np.array((1, ch, cw), dtype=np.int32),
            cxcy=np.array([int(round(c["cx"])), int(round(c["cy"]))], dtype=np.int32),
        )
        rows.append({
            "rid": rid, "wid": wid, "npz": npz_name, "k": 0,
            "label": c["label"], "iou": c["iou"], "conf": c["conf"],
            "w": c["w"], "h": c["h"], "area": c["area"],
            "cx": int(round(c["cx"])), "cy": int(round(c["cy"])), "ch": ch, "cw": cw,
        })
        n_pos += c["label"]; n_neg += (1 - c["label"])
    return {"pts": len(pts), "gt": len(gt), "pos": n_pos, "neg": n_neg}


def _resolve_rids(explicit: list[str] | None, limit_rids: int, seed: int) -> list[str]:
    """Pick which request ids to process.

    Explicit ``--rid`` list wins; otherwise randomly sample ``limit_rids`` from
    cached requests (or return all when ``limit_rids`` is 0).
    """
    if explicit:
        return list(explicit)
    all_rids = wl.list_requests()
    if not limit_rids or limit_rids >= len(all_rids):
        return all_rids
    rng = random.Random(seed)
    return rng.sample(all_rids, limit_rids)


def _sample_sheets(wsheets: list[dict], sheets_per_rid: int, seed: int, rid: str) -> list[dict]:
    """Optionally shuffle and cap eligible worksheets for one request."""
    if not sheets_per_rid or sheets_per_rid >= len(wsheets):
        return wsheets
    # Per-rid seed so different requests don't share the same sheet shuffle order.
    rng = random.Random(seed ^ hash(rid))
    picked = list(wsheets)
    rng.shuffle(picked)
    return picked[:sheets_per_rid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-sheets", type=int, default=0,
                    help="global sheet cap across all rids (0 = no global cap)")
    ap.add_argument("--limit-rids", type=int, default=0,
                    help="randomly sample N request ids (0 = all; ignored if --rid is set)")
    ap.add_argument("--sheets-per-rid", type=int, default=0,
                    help="randomly sample up to N sheets per rid (0 = all eligible)")
    ap.add_argument("--limit-points", type=int, default=0, help="0 = all GT points per sheet")
    ap.add_argument("--zoom", type=float, default=None, help="override PrepCfg.zoom")
    ap.add_argument("--rid", type=str, nargs="*", default=None,
                    help="restrict to one or more request ids (skips --limit-rids)")
    ap.add_argument("--out", type=str, default=None, help="override dataset dir")
    args = ap.parse_args()

    cfg = PrepCfg()
    if args.zoom is not None:
        cfg.zoom = args.zoom
    out_dir = Path(args.out) if args.out else DATASET_DIR
    crops_dir = (out_dir / "crops") if args.out else CROPS_DIR
    manifest_path = (out_dir / "manifest.jsonl") if args.out else MANIFEST
    crops_dir.mkdir(parents=True, exist_ok=True)

    Z = float(cfg.zoom) if cfg.zoom and cfg.zoom > 1.0 else 1.0
    zcrop, zmax, _zmin = cfg.rendered_gates()
    print(f"[prep] whole-sheet detector | zoom={Z} zcrop={zcrop} tile={cfg.tile}/{cfg.overlap} "
          f"gt_model={cfg.gt_model} pos_iou={cfg.pos_iou} neg_iou={cfg.neg_iou} device={sm._device()}")

    model = sm.get_model("fastsam").get_model()  # frozen ultralytics FastSAM (shared)

    rids = _resolve_rids(args.rid, args.limit_rids, cfg.seed)
    print(f"[prep] rids={len(rids)} limit_rids={args.limit_rids} "
          f"sheets_per_rid={args.sheets_per_rid} limit_sheets={args.limit_sheets} seed={cfg.seed}")

    rows: list[dict] = []
    n_sheets = 0
    t0 = time.time()

    for rid in rids:
        if args.limit_sheets and n_sheets >= args.limit_sheets:
            break
        try:
            wsheets = [w for w in wl.list_worksheets(rid)
                       if w["has_geometry"] and w["width"] and w["height"]]
        except Exception as exc:
            print(f"[prep] skip rid {rid[:8]}: {exc}")
            continue
        wsheets = _sample_sheets(wsheets, args.sheets_per_rid, cfg.seed, rid)
        for w in wsheets:
            if args.limit_sheets and n_sheets >= args.limit_sheets:
                break
            wid, base_w, base_h = w["wid"], w["width"], w["height"]
            try:
                st = _process_sheet(rid, wid, base_w, base_h, cfg, model, crops_dir,
                                    rows, args.limit_points)
            except Exception as exc:
                print(f"[prep] sheet {rid[:8]}/{wid[:8]} error: {exc}")
                continue
            if st["pts"] == 0:
                continue
            n_sheets += 1
            gc.collect()
            print(f"[prep] sheet {n_sheets} {rid[:8]}/{wid[:8]} pts={st['pts']} gt={st['gt']} "
                  f"pos={st['pos']} neg={st['neg']} total={len(rows)} "
                  f"peakRSS={_peak_mb():.0f}MB t={time.time()-t0:.0f}s")

    if not rows:
        print("[prep] no candidates produced — check that the chosen sheets have electrical points")
        return
    _assign_splits(rows, cfg.val_frac, cfg.seed)
    with open(manifest_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_pos = sum(r["label"] for r in rows)
    n_tr = sum(1 for r in rows if r["split"] == "train")
    print(f"[prep] DONE sheets={n_sheets} candidates={len(rows)} "
          f"(pos={n_pos} neg={len(rows)-n_pos}) train={n_tr} val={len(rows)-n_tr}")
    print(f"[prep] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

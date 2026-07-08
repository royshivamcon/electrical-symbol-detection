"""Segment-everything sheet visualisation, re-scored by the fine-tuned head.

This is the ``symbol_matcher.ipynb`` FastSAM "segment-everything" cell (chop the sheet
into overlapping tiles, run frozen FastSAM on each, overlay the masks, stitch one
full-sheet composite with a self-documenting params banner) -- except the masks are NOT
coloured at random. Each candidate mask is re-scored by the trained
``MaskConfidenceNet`` head (``checkpoints/mask_conf.pt``) and coloured green->red by
that learned confidence, so the saved PNG shows *what the head thinks is a real symbol
mask* instead of raw FastSAM output.

FastSAM stays frozen; we only use the head to score. To match the head's training
distribution (point-centred crops of one candidate mask + a prompt-point heatmap), each
segment-everything mask is scored on a ``crop``-sized window re-centred on that mask's
own centroid, with the centroid as the prompt point.

The full sheet is never materialised as one array: tiles are rendered on demand via
``worksheet_loader.pdf_tile_renderer`` and the composite is pasted onto a downscaled
2x canvas.

Run from ``Electrical/symbol_matcher_app/`` (vsam env):

    PY=../.envs/vsam/bin/python

    # full sheet (defaults to the first sheet in the dataset manifest)
    $PY finetune/visualise.py

    # a specific sheet, keeping only masks the head is >=0.5 confident about
    $PY finetune/visualise.py --rid 0ead8522 --wid 516d8877 --min-score 0.5

    # quick dry-run on a random sample of tiles
    $PY finetune/visualise.py --limit-tiles 12
"""

from __future__ import annotations

import argparse
import gc
import json
import random
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

import seg_models as sm  # noqa: E402
import worksheet_loader as wl  # noqa: E402
from config import CKPT_PATH, DATASET_DIR, MANIFEST, PrepCfg  # noqa: E402
from features import build_feats, build_input  # noqa: E402


def _app_gt_boxes(rid, wid, render_fn, W, H, Z, cfg):
    """App per-point GT boxes in rendered space (x0, y0, x1, y1) -- same call as
    ``prep_dataset._gt_boxes``, used only for the optional overlay."""
    import boxes_from_points as bfp
    import sam_boxes as sb
    base_w = int(round(W / Z)); base_h = int(round(H / Z))
    pts = sb.load_reference_points(rid, wid, base_w, base_h)
    if cfg.electrical_only:
        pts = [p for p in pts if sb._is_electrical(p.name)]
    if not pts:
        return []
    zpts = [sb.RefPoint(int(round(p.x * Z)), int(round(p.y * Z)), p.name) for p in pts]
    zcrop, zmax, zmin = cfg.rendered_gates()
    sam = bfp.boxes_from_points(
        None, zpts, model=cfg.gt_model, crop=zcrop, min_symbol_px=zmin, max_symbol_px=zmax,
        pad=cfg.pad, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
        tile=cfg.tile, tile_provider=render_fn, image_shape=(H, W), postproc=True,
    )
    return [(b.x, b.y, b.x + b.w, b.y + b.h) for b in sam if b.w > 0 and b.h > 0]


# --- tiling / drawing helpers (mirrors symbol_matcher.ipynb) ---------------
def _tiles(h: int, w: int, tile: int, overlap: int):
    """Yield (y0, x0, y1, x1) tile windows covering the image with overlap.

    Same convention as the notebook's ``_tiles`` (edge tiles are clipped, not padded).
    """
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y0 in ys:
        for x0 in xs:
            yield y0, x0, min(y0 + tile, h), min(x0 + tile, w)


def _score_to_bgr(score: float) -> tuple[int, int, int]:
    """Green (high confidence) -> red (low confidence) in BGR."""
    return (0, int(round(255 * score)), int(round(255 * (1.0 - score))))


def paste_tile(canvas: np.ndarray, tile_box, vis: np.ndarray, scale: float) -> np.ndarray:
    """Downscale a full-res tile visualisation and paste it at its true location on a
    lower-resolution ``canvas``. ``scale`` is full-res px per canvas px. Mutates and
    returns ``canvas`` -- the whole-sheet composite is built up tile by tile without
    ever holding a full-resolution sheet array."""
    ty0, tx0, ty1, tx1 = tile_box
    px0, py0 = int(tx0 / scale), int(ty0 / scale)
    px1, py1 = int(tx1 / scale), int(ty1 / scale)
    pw, ph = px1 - px0, py1 - py0
    if pw > 0 and ph > 0:
        canvas[py0:py1, px0:px1] = cv2.resize(vis, (pw, ph), interpolation=cv2.INTER_AREA)
    return canvas


def draw_params_banner(img, title, params, origin=(24, 24), alpha=0.55):
    """Burn a titled parameter list into the top-left corner of ``img`` (in place) so a
    saved composite is self-documenting. Font size scales with image width."""
    lines = [title] + [f"{k}: {v}" for k, v in params.items()]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, img.shape[1] / 2200)
    thick = max(1, int(round(scale * 1.4)))
    line_h = int(cv2.getTextSize("Ag", font, scale, thick)[0][1] * 2.0)
    pad = int(line_h * 0.6)
    text_w = max(cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines)
    x0, y0 = origin
    box = (x0, y0, x0 + text_w + 2 * pad, y0 + line_h * len(lines) + 2 * pad)
    overlay = img.copy()
    cv2.rectangle(overlay, box[:2], box[2:], (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    for i, s in enumerate(lines):
        y = y0 + pad + line_h * (i + 1) - int(line_h * 0.35)
        color = (0, 255, 255) if i == 0 else (255, 255, 255)  # yellow title, white rows
        cv2.putText(img, s, (x0 + pad, y), font, scale, color, thick, cv2.LINE_AA)
    return img


# --- head scoring ----------------------------------------------------------
def _mask_candidates(res, ch: int, cw: int, min_px: int, max_frac: float):
    """Yield (mask_bool, fastsam_conf) for every FastSAM mask that passes the size
    gates (drop sub-``min_px`` slivers and background-sized blobs > ``max_frac`` of the
    tile). Masks are resized to the crop resolution if FastSAM returned them smaller."""
    if res.masks is None or res.boxes is None:
        return
    masks = res.masks.data.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    crop_area = ch * cw
    for mk, cf in zip(masks, confs):
        b = mk > 0.5
        if b.shape != (ch, cw):
            b = cv2.resize(b.astype(np.uint8), (cw, ch),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.where(b)
        if xs.size == 0:
            continue
        w = int(xs.max() - xs.min())
        h = int(ys.max() - ys.min())
        if min(w, h) < min_px:
            continue
        if (w * h) > max_frac * crop_area:
            continue
        yield b, float(cf)


@torch.no_grad()
def _head_scores(scorer, gray: np.ndarray, masks: list[np.ndarray],
                 confs: list[float], zcrop: int) -> np.ndarray:
    """Calibrated head confidence in [0,1] for each mask. Each mask is scored on a
    ``zcrop``-radius window re-centred on its own centroid (with the centroid as the
    prompt point), so the input matches what the head saw in training (point-centred
    crop + one mask). ``confs`` are the FastSAM objectness scores fused by the head."""
    if not masks:
        return np.zeros((0,), np.float32)
    ch, cw = gray.shape[:2]
    inputs, feats = [], []
    for m, cf in zip(masks, confs):
        ys, xs = np.where(m)
        mcx, mcy = float(xs.mean()), float(ys.mean())
        x0 = max(0, int(round(mcx - zcrop))); y0 = max(0, int(round(mcy - zcrop)))
        x1 = min(cw, int(round(mcx + zcrop))); y1 = min(ch, int(round(mcy + zcrop)))
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


def _overlay(crop: np.ndarray, masks: list[np.ndarray], scores: np.ndarray,
             min_score: float, annotate: bool) -> tuple[np.ndarray, int]:
    """Blend each mask onto the crop coloured by its head score (green->red), outline
    it, and (optionally) annotate the score. Returns (vis, n_drawn)."""
    vis = crop.copy()
    n = 0
    order = np.argsort(scores)  # draw low-confidence first so confident masks land on top
    for i in order:
        s = float(scores[i])
        if s < min_score:
            continue
        m = masks[i]
        color = _score_to_bgr(s)
        vis[m] = (0.5 * vis[m].astype(np.float32) + 0.5 * np.asarray(color, np.float32)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1)
        if annotate:
            ys, xs = np.where(m)
            cv2.putText(vis, f"{s:.2f}", (int(xs.min()), max(10, int(ys.min()) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        n += 1
    return vis, n


def _default_sheet(manifest: Path) -> tuple[str, str]:
    """First (rid, wid) in the dataset manifest, so the script works out of the box."""
    if not manifest.exists():
        raise SystemExit(f"[viz] no --rid/--wid given and no manifest at {manifest}. "
                         f"Pass --rid/--wid, or run prep_dataset.py first.")
    with manifest.open() as fh:
        r = json.loads(fh.readline())
    return r["rid"], r["wid"]


def main() -> None:
    cfg = PrepCfg()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rid", type=str, default=None, help="request id (full id; default: first in manifest)")
    ap.add_argument("--wid", type=str, default=None, help="sheet id (full id; default: first in manifest)")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_PATH), help="trained head checkpoint")
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--zoom", type=float, default=cfg.zoom, help="detection render zoom")
    ap.add_argument("--stitch-zoom", type=float, default=2.0, help="full-sheet composite zoom")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=cfg.imgsz, help="FastSAM input size")
    ap.add_argument("--conf", type=float, default=0.85, help="FastSAM confidence")
    ap.add_argument("--iou", type=float, default=cfg.iou, help="FastSAM NMS IoU")
    ap.add_argument("--min-px", type=int, default=40, help="drop masks smaller than this (px)")
    ap.add_argument("--max-frac", type=float, default=cfg.cand_max_frac, help="drop masks bigger than this frac of the tile")
    ap.add_argument("--min-score", type=float, default=0.0, help="only draw masks with head score >= this")
    ap.add_argument("--gt-boxes", action="store_true",
                    help="overlay the app GT boxes (thin blue) to eyeball head vs. ground truth")
    ap.add_argument("--batch", type=int, default=8, help="FastSAM tile batch size")
    ap.add_argument("--chunk", type=int, default=64, help="tiles rendered per chunk (memory bound)")
    ap.add_argument("--limit-tiles", type=int, default=0, help="0 = whole sheet; else a random sample of N tiles")
    ap.add_argument("--max-tiles", type=int, default=4000, help="hard stop so a bad config errors fast")
    ap.add_argument("--no-annotate", action="store_true", help="don't burn the per-mask score text")
    ap.add_argument("--no-tiles", action="store_true", help="skip per-tile PNGs; only write the stitched sheet")
    ap.add_argument("--out", type=str, default=None, help="output dir (default: dataset/viz_sheets/<rid8>_<wid8>)")
    ap.add_argument("--remove-text", dest="remove_text", action="store_true", default=cfg.remove_text)
    ap.add_argument("--keep-text", dest="remove_text", action="store_false")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    rid = args.rid
    wid = args.wid
    if not (rid and wid):
        d_rid, d_wid = _default_sheet(manifest)
        rid = rid or d_rid
        wid = wid or d_wid

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"[viz] head checkpoint not found: {ckpt}. Train it first "
                         f"(finetune/train.py) -- this script visualises the *fine-tuned* scores.")
    from infer import MaskConfidenceScorer
    scorer = MaskConfidenceScorer.load(ckpt)

    zcrop, _zmax, _zmin = PrepCfg(zoom=args.zoom).rendered_gates()
    Z = float(args.zoom)
    model = sm.get_model("fastsam").get_model()  # frozen ultralytics FastSAM (same as prep)

    out_dir = Path(args.out) if args.out else (DATASET_DIR / "viz_sheets" / f"{rid[:8]}_{wid[:8]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = sm._device()
    print(f"[viz] sheet rid={rid[:8]} wid={wid[:8]} @ {Z}x  head={ckpt.name} (in={scorer.input_size}) "
          f"device={device} zcrop={zcrop}")

    with wl.pdf_tile_renderer(rid, wid, Z, remove_text=args.remove_text) as (W, H, render_fn):
        tiles = list(_tiles(H, W, args.tile, args.overlap))
        if args.limit_tiles and args.limit_tiles < len(tiles):
            tiles = random.Random(cfg.seed).sample(tiles, args.limit_tiles)
        if len(tiles) > args.max_tiles:
            raise SystemExit(f"[viz] {len(tiles)} tiles exceeds --max-tiles={args.max_tiles}; "
                             f"raise it, use --limit-tiles, or a larger --tile.")
        print(f"[viz] sheet {W}x{H} ({W*H/1e6:.0f} MP) -> {len(tiles)} tiles "
              f"(tile={args.tile}, overlap={args.overlap})")

        stitched = wl.render_pdf_image(rid, wid, zoom=args.stitch_zoom, remove_text=args.remove_text)
        stitch_scale = Z / float(args.stitch_zoom)  # detection px -> stitch-canvas px

        gt_boxes = []
        if args.gt_boxes:
            gt_boxes = _app_gt_boxes(rid, wid, render_fn, W, H, Z, cfg)
            print(f"[viz] overlaying {len(gt_boxes)} app GT boxes (model={cfg.gt_model})")

        t0 = time.time()
        idx = n_masks_total = 0
        for c0 in range(0, len(tiles), args.chunk):
            chunk_tiles = tiles[c0:c0 + args.chunk]
            chunk_crops = [render_fn(tx0, ty0, tx1, ty1) for (ty0, tx0, ty1, tx1) in chunk_tiles]

            for b0 in range(0, len(chunk_crops), args.batch):
                batch_tiles = chunk_tiles[b0:b0 + args.batch]
                batch_crops = chunk_crops[b0:b0 + args.batch]
                with torch.no_grad():
                    batch_res = model([sm._fastsam_rgb(c) for c in batch_crops], imgsz=args.imgsz, conf=args.conf,
                                      iou=args.iou, retina_masks=True, verbose=False)

                for res, crop, tile_box in zip(batch_res, batch_crops, batch_tiles):
                    ch, cw = crop.shape[:2]
                    cands = list(_mask_candidates(res, ch, cw, args.min_px, args.max_frac))
                    masks = [b for b, _ in cands]
                    confs = [cf for _, cf in cands]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    scores = _head_scores(scorer, gray, masks, confs, zcrop)
                    vis, n_drawn = _overlay(crop, masks, scores, args.min_score,
                                            annotate=not args.no_annotate)
                    n_masks_total += n_drawn

                    ty0, tx0, ty1, tx1 = tile_box
                    for gx0, gy0, gx1, gy1 in gt_boxes:
                        if gx1 <= tx0 or gx0 >= tx1 or gy1 <= ty0 or gy0 >= ty1:
                            continue
                        cv2.rectangle(vis, (int(gx0 - tx0), int(gy0 - ty0)),
                                      (int(gx1 - tx0), int(gy1 - ty0)), (255, 0, 0), 1)
                    if not args.no_tiles:
                        cv2.imwrite(str(out_dir / f"tile_{idx:04d}_y{ty0}_x{tx0}_m{n_drawn}.png"), vis)
                    paste_tile(stitched, tile_box, vis, stitch_scale)
                    idx += 1

                del batch_res
            del chunk_crops
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            print(f"[viz]  {min(c0 + args.chunk, len(tiles))}/{len(tiles)} tiles | "
                  f"{n_masks_total} masks drawn | t={time.time()-t0:.0f}s")

    draw_params_banner(stitched, "FastSAM segment-everything + fine-tuned head", {
        "rid/wid": f"{rid[:8]}/{wid[:8]}",
        "head_ckpt": ckpt.name,
        "detect_zoom": f"{Z}x  (stitch {args.stitch_zoom}x)",
        "tile/overlap": f"{args.tile}/{args.overlap}",
        "conf/iou": f"{args.conf}/{args.iou}",
        "min_px/max_frac": f"{args.min_px}/{args.max_frac}",
        "min_score": args.min_score,
        "colour": "green=high head conf -> red=low",
        "gt_boxes": (f"{len(gt_boxes)} blue (app {cfg.gt_model})" if args.gt_boxes else "off"),
        "tiles/masks": f"{len(tiles)} tiles, {n_masks_total} masks",
    })
    stitched_path = out_dir / "stitched_full_sheet.png"
    cv2.imwrite(str(stitched_path), stitched)
    print(f"[viz] DONE {idx} tiles, {n_masks_total} masks drawn")
    print(f"[viz] stitched full-sheet composite -> {stitched_path}")
    if not args.no_tiles:
        print(f"[viz] per-tile overlays -> {out_dir}")


if __name__ == "__main__":
    main()

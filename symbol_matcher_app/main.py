"""FastAPI app: select a request (rid) + worksheet (wid), load its image,
drag-select a symbol, and highlight every patch similar to it.

Run from this directory:
    ../.envs/vsam/bin/uvicorn main:app --reload --port 8000
Then open http://127.0.0.1:8000/
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import matcher
import worksheet_loader as wl

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"    

app = FastAPI(title="Symbol Matcher", version="1.0")


class MatchRequest(BaseModel):
    x: int = Field(..., ge=0, description="Template left, in original image pixels")
    y: int = Field(..., ge=0, description="Template top, in original image pixels")
    w: int = Field(..., gt=0, description="Template width in pixels")
    h: int = Field(..., gt=0, description="Template height in pixels")
    threshold: float = Field(0.7, ge=0.1, le=1.0)
    scales: list[float] | None = None
    method: str = Field(
        "classical",
        description="'classical' (NCC), 'tmr', 'persam', or 'sam3' (SAM 3 concept seg)",
    )


@app.get("/api/requests")
def api_requests() -> dict:
    return {"requests": wl.list_requests()}


@app.get("/api/requests/{rid}/worksheets")
def api_worksheets(rid: str, patches_only: bool = False) -> dict:
    """List worksheets for a request. With ``patches_only=1`` only sheets that
    have at least one ground-truth polygon "true patch" (post electrical/lighting
    filter) are returned."""
    try:
        worksheets = wl.list_worksheets(rid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if patches_only:
        import sam_boxes as sb

        worksheets = [
            w
            for w in worksheets
            if w.get("has_geometry") and sb.worksheet_has_patches(rid, w["wid"])
        ]
    return {"rid": rid, "worksheets": worksheets}


@app.get("/api/worksheet/{rid}/{wid}/image")
def api_image(rid: str, wid: str) -> Response:
    try:
        png = wl.get_image_png_bytes(rid, wid)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # network/download failures
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")
    return Response(content=png, media_type="image/png")


@app.get("/api/worksheet/{rid}/{wid}/meta")
def api_meta(rid: str, wid: str) -> dict:
    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")
    h, w = img.shape[:2]
    return {"rid": rid, "wid": wid, "width": int(w), "height": int(h)}


@app.get("/api/worksheet/{rid}/{wid}/ref_points")
def api_ref_points(rid: str, wid: str) -> dict:
    """Ground-truth reference points (electrical Point features) for the sheet,
    mapped to image pixels. This is the same set the SAM models are prompted
    from, exposed on its own so the UI can show it as a layer."""
    import sam_boxes as sb

    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")
    H, W = img.shape[:2]
    points = sb.load_reference_points(rid, wid, W, H)
    return {
        "count": len(points),
        "points": [{"x": p.x, "y": p.y, "name": p.name} for p in points],
    }


@app.get("/api/worksheet/{rid}/{wid}/ref_polygons")
def api_ref_polygons(rid: str, wid: str, electrical: bool = True) -> dict:
    """Ground-truth "true patches": Polygon features (geometry_type == 3) reduced
    to bounding boxes, mapped to image pixels. Exposed as its own layer. With
    ``electrical=1`` (default) only electrical polygons are kept; ``electrical=0``
    returns every polygon. Only worksheets with polygon geometries return anything."""
    import sam_boxes as sb

    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")
    H, W = img.shape[:2]
    polys = sb.load_reference_polygons(rid, wid, W, H, electrical=electrical)
    return {
        "count": len(polys),
        "polygons": [
            {"x": p.x, "y": p.y, "w": p.w, "h": p.h, "name": p.name} for p in polys
        ],
    }


def run_sam(
    img,
    points,
    model: str = "fastsam",
    crop: int = 100,
    min_symbol_px: int = 16,
    max_symbol_px: int = 50,
    pad: int = 0,
    filt: str = "none",
    ksize: int = 5,
    kernels: tuple[int, int, int] = (6, 8, 3),
    tile: int = 0,
    nms_iou: float = 0.5,
    grow_on_clip: bool = False,
    start_crop_frac: float = 0.3,
    imgsz: int = 1024,
    rid: str | None = None,
    wid: str | None = None,
    zoom: float = 1.0,
    remove_text: bool = False,
    postproc: bool = False,
    workers: int = 0,
) -> list[dict]:
    """Run the selected model over ``points`` (unified per-point finder) and return
    box dicts.

    ``model`` is ``"fastsam"``, ``"hqsam"`` or ``"mix"`` (runs both per point and
    keeps the tighter box; HQ-SAM uses a larger ``pad``). Detection is always
    per-point; ``tile`` > 0 only chunks the sheet for bounded memory -- the per-point
    routine is identical inside each tile (no shared per-tile encode, no NMS).
    ``grow_on_clip`` (HQ-SAM) starts each crop at ``start_crop_frac`` of the adaptive
    size for tighter boxes and only enlarges when the mask clips the crop edge.
    Shared by the ``sam_points``, ``evaluate`` and ``analysis`` endpoints.

    ``zoom`` > 1 (with ``rid``/``wid``) is quill_forge's PDF render zoom: the sheet is
    re-rendered from its source PDF at ``zoom`` x the base raster (``fitz``); the model
    runs on that sharper image and boxes are mapped back to base pixels. Point coords
    and the size gates ``crop`` / ``min_symbol_px`` / ``max_symbol_px`` are scaled by
    ``zoom`` so they mean the same physical size as on the base raster -- otherwise the
    crop clips the enlarged symbol and ``max_symbol_px`` trips the oversize fallback on
    every one, collapsing each box to its smallest sub-mask. ``tile`` / ``imgsz`` and the
    box-fit margin ``pad`` apply directly in rendered space; a rendered ``pad`` of N
    becomes N/zoom base px after the box is divided back down.
    When ``tile`` > 0 under zoom, tiles are rendered on demand
    (``pdf_tile_renderer``) so peak RAM is one tile, not the whole ``base*zoom`` sheet.

    ``remove_text=True`` renders from the PDF (via ``fitz``) with all text words
    redacted out — line art / images kept — so labels and dimensions don't distract
    the detector. It forces the PDF-render path even at ``zoom == 1``.
    """
    import boxes_from_points as bfp  # unified per-point finder (fastsam/hqsam/mix)

    if not points:
        return []
    if rid and wid and ((zoom and zoom > 1.0) or remove_text):
        from sam_boxes import RefPoint

        Z = float(zoom) if zoom and zoom > 1.0 else 1.0
        zpts = [RefPoint(int(round(p.x * Z)), int(round(p.y * Z)), p.name) for p in points]
        # Symbols are Z x larger in rendered space, so the size *gates* must scale
        # with them. With base-raster values at zoom, the crop window clipped every
        # symbol and max_symbol_px tripped the oversize fallback on all of them --
        # collapsing each box to its smallest sub-mask. Scale crop / max_symbol_px /
        # min_symbol_px by Z so they mean the same physical size as on the base raster.
        # pad stays a rendered-space margin (box is measured at zoom then divided by Z,
        # so a rendered pad of N becomes N/Z base px); scaling it over-padded every box.
        zcrop = max(1, int(round(crop * Z)))
        zmax = max(1, int(round(max_symbol_px * Z)))
        zmin = max(1, int(round(min_symbol_px * Z)))
        common = dict(
            model=model, crop=zcrop, min_symbol_px=zmin, max_symbol_px=zmax,
            pad=pad, filt=filt, ksize=ksize, kernels=kernels, tile=tile,
            grow_on_clip=grow_on_clip, start_crop_frac=start_crop_frac, imgsz=imgsz,
            postproc=postproc, workers=workers,
        )
        if tile and tile > 0:
            # Memory-bounded: render one tile at a time instead of the whole
            # base*zoom sheet; the per-point routine is identical inside each tile.
            with wl.pdf_tile_renderer(rid, wid, Z, remove_text=remove_text) as (tw, th, render_fn):
                boxes = [
                    b.as_dict()
                    for b in bfp.boxes_from_points(
                        None, zpts, tile_provider=render_fn, image_shape=(th, tw), **common)
                ]
        else:
            zimg = wl.render_pdf_image(rid, wid, Z, remove_text=remove_text)
            boxes = [b.as_dict() for b in bfp.boxes_from_points(zimg, zpts, **common)]
        inv = 1.0 / Z
        for b in boxes:  # map boxes back to the base raster
            b["x"], b["y"] = int(round(b["x"] * inv)), int(round(b["y"] * inv))
            b["w"], b["h"] = int(round(b["w"] * inv)), int(round(b["h"] * inv))
        return boxes
    return [
        b.as_dict()
        for b in bfp.boxes_from_points(
            img, points, model=model, crop=crop, min_symbol_px=min_symbol_px,
            max_symbol_px=max_symbol_px, pad=pad, filt=filt, ksize=ksize,
            kernels=kernels, tile=tile, grow_on_clip=grow_on_clip,
            start_crop_frac=start_crop_frac, imgsz=imgsz, postproc=postproc,
            workers=workers,
        )
    ]


@app.get("/api/worksheet/{rid}/{wid}/sam_points")
def api_sam_points(
    rid: str,
    wid: str,
    limit: int = 0,
    model: str = "fastsam",
    crop: int = 90,
    min_symbol_px: int = 16,
    max_symbol_px: int = 50,
    pad: int = 3,
    filt: str = "none",
    ksize: int = 5,
    kr: int = 6,
    kg: int = 8,
    kb: int = 3,
    tile: int = 1024,
    nms_iou: float = 0.5,
    imgsz: int = 1024,
    zoom: float = 4.0,
    remove_text: bool = True,
    postproc: bool = True,
    workers: int = 0,
) -> dict:
    """Segment symbols around the worksheet's reference points (electrical Point
    features) and return their bounding boxes.

    ``model`` selects "fastsam" (default, tighter boxes), "hqsam" or "mix".
    ``crop`` is the half-window around each point; ``min_symbol_px`` /
    ``max_symbol_px`` floor and cap box size, and ``pad`` adds a margin.
    ``filt`` picks a preprocessing filter fed to the SAM model: ``"gaussian"`` /
    ``"laplace"`` / ``"sharpen"`` use ``ksize``, and ``"channels"`` (multi-scale
    pseudo-coloring) uses the per-channel kernels ``kr`` / ``kg`` / ``kb``. The
    filter affects only what the model sees.
    ``tile`` > 0 chunks the sheet into ``tile`` px overlapping tiles for bounded
    memory only; the identical per-point routine runs inside each tile.
    ``limit`` > 0 caps the number of points processed (handy for a quick look).
    """
    import sam_boxes as sb  # point loader (lightweight import)

    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")

    H, W = img.shape[:2]
    points = sb.load_reference_points(rid, wid, W, H)
    if not points:
        return {"count": 0, "boxes": [], "points": [], "total_points": 0, "model": model}
    used = points[:limit] if limit and limit > 0 else points

    try:
        boxes = run_sam(
            img, used, model=model, crop=crop, min_symbol_px=min_symbol_px,
            max_symbol_px=max_symbol_px, pad=pad, filt=filt, ksize=ksize,
            kernels=(kr, kg, kb), tile=tile, nms_iou=nms_iou, imgsz=imgsz,
            rid=rid, wid=wid, zoom=zoom, remove_text=remove_text, postproc=postproc,
            workers=workers,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAM run failed: {exc}")

    return {
        "model": model,
        "count": len(boxes),
        "total_points": len(points),
        "used_points": len(used),
        "boxes": boxes,
        "points": [{"x": p.x, "y": p.y, "name": p.name} for p in used],
    }


def _base_gray(rid: str, wid: str, zoom: float, remove_text: bool):
    """Grayscale source at the **base raster** resolution (H, W match /image).

    When ``zoom``/``remove_text`` request the sharper PDF render, it is rendered
    then resized back to the base dimensions so processed-view overlays stay
    pixel-aligned with the displayed worksheet image.
    """
    import cv2

    base = wl.load_worksheet_image(rid, wid)
    H, W = base.shape[:2]
    if (zoom and zoom > 1.0) or remove_text:
        Z = float(zoom) if zoom and zoom > 1.0 else 1.0
        src = wl.render_pdf_image(rid, wid, Z, remove_text=remove_text)
        if src.shape[:2] != (H, W):
            src = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA)
    else:
        src = base
    return cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), H, W


@app.get("/api/worksheet/{rid}/{wid}/processed")
def api_processed(
    rid: str,
    wid: str,
    view: str = "binary",
    zoom: float = 1.0,
    remove_text: bool = False,
) -> Response:
    """Full-sheet processed background at base-raster resolution.

    ``view`` = ``"binary"`` (symbol_det ink binarization) or ``"suppressed"``
    (ink with long straight lines removed). The result is inverted to black ink on
    white paper so it reads like the original drawing, and cached on disk."""
    import cv2 as _cv2
    import mask_postproc as mpp

    if view not in ("binary", "suppressed"):
        raise HTTPException(status_code=400, detail="view must be binary|suppressed")
    key = f"proc__{rid}__{wid}__{view}__z{zoom}__t{int(bool(remove_text))}.png"
    cache_path = wl.CACHE_DIR / key
    if cache_path.exists():
        return Response(content=cache_path.read_bytes(), media_type="image/png")

    try:
        gray, _H, _W = _base_gray(rid, wid, zoom, remove_text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")

    if view == "binary":
        m = (mpp.ink_of(gray).astype("uint8")) * 255
    else:
        m = mpp.suppressed_of(gray)
    out = 255 - m  # black ink on white paper
    ok, buf = _cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    cache_path.write_bytes(buf.tobytes())
    return Response(content=buf.tobytes(), media_type="image/png")


def _boxes_with_masks(
    img, points, model, crop, min_symbol_px, max_symbol_px, pad, filt, ksize,
    kernels, tile, nms_iou, postproc, imgsz=1024, workers=0,
) -> list:
    """Run the unified finder with ``collect_masks=True`` and return SamBox objects
    (with ``.mask``/``.mx``/``.my``). Only hqsam/fastsam/mix carry masks."""
    import boxes_from_points as bfp

    return bfp.boxes_from_points(
        img, points, model=model, crop=crop, min_symbol_px=min_symbol_px,
        max_symbol_px=max_symbol_px, pad=pad, filt=filt, ksize=ksize,
        kernels=kernels, tile=tile, imgsz=imgsz, collect_masks=True, postproc=postproc,
        workers=workers,
    )


@app.get("/api/worksheet/{rid}/{wid}/sam_masks")
def api_sam_masks(
    rid: str,
    wid: str,
    limit: int = 0,
    model: str = "fastsam",
    crop: int = 150,
    min_symbol_px: int = 16,
    max_symbol_px: int = 50,
    pad: int = 0,
    filt: str = "none",
    ksize: int = 5,
    kr: int = 6,
    kg: int = 8,
    kb: int = 3,
    tile: int = 0,
    nms_iou: float = 0.5,
    zoom: float = 4.0,
    remove_text: bool = True,
    postproc: bool = True,
    workers: int = 0,
) -> Response:
    """Transparent full-sheet RGBA overlay of the (ink-intersected) SAM masks.

    Runs the point-prompted finder with ``collect_masks=True`` (and ``postproc`` on
    by default so the strokes are the symbol_det-filtered ink), then composites each
    per-point mask onto a transparent canvas at base-raster resolution. Only
    hqsam/fastsam/mix carry masks; other models return an empty overlay. Cached
    per query on disk."""
    import hashlib

    import cv2 as _cv2
    import numpy as _np
    import sam_boxes as sb

    key_raw = (
        f"{rid}|{wid}|{limit}|{model}|{crop}|{min_symbol_px}|{max_symbol_px}|{pad}|"
        f"{filt}|{ksize}|{kr}|{kg}|{kb}|{tile}|{nms_iou}|{zoom}|{int(bool(remove_text))}|"
        f"{int(bool(postproc))}"
    )
    key = "masks__" + hashlib.sha1(key_raw.encode()).hexdigest()[:20] + ".png"
    cache_path = wl.CACHE_DIR / key
    if cache_path.exists():
        return Response(content=cache_path.read_bytes(), media_type="image/png")

    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")
    H, W = img.shape[:2]
    points = sb.load_reference_points(rid, wid, W, H)
    used = points[:limit] if limit and limit > 0 else points

    Z = float(zoom) if zoom and zoom > 1.0 else 1.0
    if used and ((Z > 1.0) or remove_text):
        from sam_boxes import RefPoint

        det_img = wl.render_pdf_image(rid, wid, Z, remove_text=remove_text)
        det_pts = [RefPoint(int(round(p.x * Z)), int(round(p.y * Z)), p.name) for p in used]
        sc, sms, sp = int(round(crop * Z)), int(round(min_symbol_px * Z)), int(round(max_symbol_px * Z))
        spad, stile = int(round(pad * Z)), (int(round(tile * Z)) if tile else tile)
    else:
        det_img, det_pts = img, used
        sc, sms, sp, spad, stile, Z = crop, min_symbol_px, max_symbol_px, pad, tile, 1.0

    try:
        boxes = _boxes_with_masks(
            det_img, det_pts, model, sc, sms, sp, spad, filt, ksize,
            (kr, kg, kb), stile, nms_iou, postproc, workers=workers,
        ) if det_pts else []
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SAM mask run failed: {exc}")

    canvas = _np.zeros((H, W, 4), dtype=_np.uint8)  # BGRA, transparent
    B, G, R, A = 255, 0, 255, 170  # magenta, semi-transparent
    inv = 1.0 / Z
    for b in boxes:
        m = getattr(b, "mask", None)
        if m is None:
            continue
        mx, my = int(round(b.mx * inv)), int(round(b.my * inv))
        if Z != 1.0:
            nh, nw = max(1, int(round(m.shape[0] * inv))), max(1, int(round(m.shape[1] * inv)))
            m = _cv2.resize(m.astype("uint8"), (nw, nh), interpolation=_cv2.INTER_NEAREST).astype(bool)
        mh, mw = m.shape
        x0, y0 = max(0, mx), max(0, my)
        x1, y1 = min(W, mx + mw), min(H, my + mh)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = m[y0 - my:y1 - my, x0 - mx:x1 - mx]
        region = canvas[y0:y1, x0:x1]
        region[sub, 0] = B
        region[sub, 1] = G
        region[sub, 2] = R
        region[sub, 3] = A

    ok, buf = _cv2.imencode(".png", canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    cache_path.write_bytes(buf.tobytes())
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/worksheet/{rid}/{wid}/evaluate")
def api_evaluate(
    rid: str,
    wid: str,
    model: str = "fastsam",
    iou_thr: float = 0.5,
    gt: str = "bboxes",
    limit: int = 0,
    crop: int = 90,
    min_symbol_px: int = 20,
    max_symbol_px: int = 90,
    pad: int = 0,
    filt: str = "none",
    ksize: int = 5,
    kr: int = 6,
    kg: int = 8,
    kb: int = 3,
    tile: int = 1024,
    nms_iou: float = 0.5,
    imgsz: int = 1024,
    zoom: float = 4.0,
    remove_text: bool = True,
    postproc: bool = True,
    workers: int = 0,
) -> dict:
    """Score a model's boxes on this sheet against the ground truth.

    ``gt`` picks the ground-truth source / metric:

    - ``"bboxes"`` (default): prompt the model at the center of each wiring-filtered
      GT box and match predictions to those boxes by **IoU** at ``iou_thr`` (how well
      it recovers the true box). Center-hit is reported as an informational number.
    - ``"points"``: prompt the model at the electrical GT reference points and score
      by **center-hit** — a prediction is a TP if it contains a GT point (no IoU).

    The detection config (``crop`` / ``min_symbol_px`` / ``max_symbol_px`` / ``pad`` /
    ``tile`` / ``imgsz`` / ``zoom`` / ``remove_text`` / ``postproc``) mirrors
    ``api_sam_points`` exactly, so the scored boxes are the same ones the interactive
    endpoint produces — the metrics reflect the app's real detector, not a tighter
    eval-only variant. Only the *prompt source* differs by ``gt`` (GT box centers vs
    reference points), since that is what the ground truth is measured against.

    Returns precision/recall/F1, plus the predicted + GT items tagged
    tp/fp/matched/missed for the overlay.
    """
    import sam_boxes as sb
    import evaluation as ev
    from sam_boxes import RefPoint

    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")

    H, W = img.shape[:2]
    if gt == "points":
        # Prompt at (and score against) the electrical GT reference points.
        used_prompts = sb.load_reference_points(rid, wid, W, H)
    else:
        polys = sb.load_reference_polygons(rid, wid, W, H, electrical=True)
        used_prompts = [
            RefPoint(int(g.x + g.w / 2), int(g.y + g.h / 2), g.name) for g in polys
        ]
    used = used_prompts[:limit] if limit and limit > 0 else used_prompts

    try:
        boxes = run_sam(
            img, used, model=model, crop=crop, min_symbol_px=min_symbol_px,
            max_symbol_px=max_symbol_px, pad=pad, filt=filt, ksize=ksize,
            kernels=(kr, kg, kb), tile=tile, nms_iou=nms_iou, imgsz=imgsz,
            rid=rid, wid=wid, zoom=zoom, remove_text=remove_text, postproc=postproc,
            workers=workers,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation run failed: {exc}")

    pred_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in boxes]
    if gt == "points":
        gt_points = [(p.x, p.y) for p in used]
        metrics = ev.evaluate_points(pred_boxes, gt_points)
        gt_out = [
            {"x": p.x, "y": p.y, "name": p.name, "status": metrics["gt_status"][i]}
            for i, p in enumerate(used)
        ]
    else:
        gt_boxes = [(g.x, g.y, g.w, g.h) for g in polys]
        metrics = ev.evaluate(pred_boxes, gt_boxes, iou_thr=iou_thr)
        gt_out = [
            {"x": g.x, "y": g.y, "w": g.w, "h": g.h, "name": g.name,
             "status": metrics["gt_status"][i]}
            for i, g in enumerate(polys)
        ]

    return {
        "rid": rid, "wid": wid, "model": model, "gt_mode": gt,
        "metrics": metrics,
        "boxes": [
            {**b, "status": metrics["pred_status"][i]} for i, b in enumerate(boxes)
        ],
        "gt": gt_out,
    }


@app.get("/api/analysis/{rid}")
def api_analysis(
    rid: str,
    models: str = "fastsam,hqsam",
    iou_thr: float = 0.5,
    gt: str = "bboxes",
    limit: int = 0,
    max_sheets: int = 6,
    filt: str = "none",
    ksize: int = 5,
    kr: int = 6,
    kg: int = 8,
    kb: int = 3,
    tile: int = 1024,
    zoom: float = 4.0,
    remove_text: bool = True,
    workers: int = 0,
) -> dict:
    """Aggregate evaluation across the request's sheets that have GT patches.

    ``models``, ``filt`` and ``gt`` are each **comma-separated** so a single run can
    sweep multiple models × preprocessing filters × ground-truth modes. For every
    combination it evaluates up to ``max_sheets`` worksheets and returns a result
    block with micro-averaged precision/recall/F1 and a per-sheet breakdown.

    - ``gt`` values: ``"bboxes"`` (IoU vs polygon patches) / ``"points"`` (center-hit
      vs reference points).
    - ``filt`` values are any preprocessing filter; ``ksize`` / ``kr`` / ``kg`` /
      ``kb`` supply the kernel sizes shared by all selected filters.
    """
    import sam_boxes as sb
    import evaluation as ev
    from sam_boxes import RefPoint

    try:
        worksheets = wl.list_worksheets(rid)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    sheets = [
        w for w in worksheets
        if w.get("has_geometry") and sb.worksheet_has_patches(rid, w["wid"])
    ]
    if max_sheets and max_sheets > 0:
        sheets = sheets[:max_sheets]

    model_list = [m.strip() for m in models.split(",") if m.strip()] or ["fastsam"]
    filt_list = [f.strip() for f in filt.split(",") if f.strip()] or ["none"]
    gt_list = [g.strip() for g in gt.split(",") if g.strip()] or ["bboxes"]

    # Cache decoded images + GT per worksheet so the model/filter/gt sweep doesn't
    # redo that work for every combination.
    img_cache: dict[str, object] = {}
    poly_cache: dict[str, list] = {}
    point_cache: dict[str, list] = {}

    def get_img(wid):
        if wid not in img_cache:
            try:
                img_cache[wid] = wl.load_worksheet_image(rid, wid)
            except Exception:
                img_cache[wid] = None
        return img_cache[wid]

    import progress as pg

    combos = [(g, m, f) for g in gt_list for m in model_list for f in filt_list]
    prog = pg.Progress(
        max(1, len(combos) * len(sheets)),
        f"Analysis {rid[:8]} {len(model_list)}m×{len(filt_list)}f×{len(gt_list)}gt "
        f"x {len(sheets)}sheets",
        every=1,
    )

    def prf(tp, np_, ng):
        p = tp / np_ if np_ else 0.0
        r = tp / ng if ng else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}

    results: list[dict] = []
    for g_mode, model, f_name in combos:
        agg = {"tp_c": 0, "hit_c": 0, "tp_i": 0, "iou_sum": 0.0, "n_pred": 0, "n_gt": 0}
        per_sheet = []
        for w in sheets:
            wid = w["wid"]
            prog.update(note=f"{g_mode}/{model}/{f_name} {wid[:8]}")
            img = get_img(wid)
            if img is None:
                continue
            H, W = img.shape[:2]
            if g_mode == "points":
                if wid not in point_cache:
                    point_cache[wid] = sb.load_reference_points(rid, wid, W, H)
                used = point_cache[wid]
            else:
                if wid not in poly_cache:
                    poly_cache[wid] = sb.load_reference_polygons(rid, wid, W, H, electrical=True)
                polys = poly_cache[wid]
                used = [
                    RefPoint(int(p.x + p.w / 2), int(p.y + p.h / 2), p.name) for p in polys
                ]
            if limit and limit > 0:
                used = used[:limit]
            try:
                boxes = run_sam(
                    img, used, model=model, filt=f_name, ksize=ksize,
                    kernels=(kr, kg, kb), tile=tile,
                    rid=rid, wid=wid, zoom=zoom, remove_text=remove_text,
                    workers=workers,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Analysis run failed: {exc}")
            pred_boxes = [(b["x"], b["y"], b["w"], b["h"]) for b in boxes]
            if g_mode == "points":
                m = ev.evaluate_points(pred_boxes, [(p.x, p.y) for p in used])
            else:
                gt_boxes = [(p.x, p.y, p.w, p.h) for p in polys]
                m = ev.evaluate(pred_boxes, gt_boxes, iou_thr=iou_thr)
            agg["n_pred"] += m["n_pred"]
            agg["n_gt"] += m["n_gt"]
            agg["hit_c"] += m["center"]["pred_hits"]
            agg["tp_c"] += m["center"]["gt_found"]
            agg["tp_i"] += m["iou"]["matches"]
            agg["iou_sum"] += m["iou"]["mean_iou"] * m["iou"]["matches"]
            per_sheet.append({
                "wid": wid, "n_pred": m["n_pred"], "n_gt": m["n_gt"],
                "center_f1": m["center"]["f1"], "iou_f1": m["iou"]["f1"],
                "mean_iou": m["iou"]["mean_iou"],
            })

        center = prf(agg["hit_c"], agg["n_pred"], agg["n_gt"])
        center["recall"] = round(agg["tp_c"] / agg["n_gt"], 4) if agg["n_gt"] else 0.0
        iou_b = prf(agg["tp_i"], agg["n_pred"], agg["n_gt"])
        iou_b["mean_iou"] = round(agg["iou_sum"] / agg["tp_i"], 4) if agg["tp_i"] else 0.0
        label = f"{model} · {f_name} · {g_mode}"
        results.append({
            "label": label, "model": model, "filt": f_name, "gt_mode": g_mode,
            "n_pred": agg["n_pred"], "n_gt": agg["n_gt"],
            "center": center, "iou": iou_b, "sheets": per_sheet,
        })

    prog.done()
    return {
        "rid": rid, "iou_thr": iou_thr,
        "n_sheets": len(sheets), "results": results,
    }


@app.post("/api/worksheet/{rid}/{wid}/match")
def api_match(rid: str, wid: str, req: MatchRequest) -> dict:
    try:
        img = wl.load_worksheet_image(rid, wid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load image: {exc}")

    H, W = img.shape[:2]
    x2, y2 = min(req.x + req.w, W), min(req.y + req.h, H)
    if req.x >= W or req.y >= H or x2 <= req.x or y2 <= req.y:
        raise HTTPException(status_code=400, detail="Selection is outside the image bounds")

    method = (req.method or "classical").lower()
    tpl_w, tpl_h = x2 - req.x, y2 - req.y
    if method == "sam3":
        import sam3_match
        try:
            matches = sam3_match.match_template(
                img, (req.x, req.y, tpl_w, tpl_h), threshold=req.threshold,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            # Missing sam3.pt weights or the CLIP dependency: actionable 503.
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SAM 3 match failed: {exc}")
    elif method in ("tmr", "persam"):
        import feat_match
        matches = feat_match.match_template(
            img, (req.x, req.y, tpl_w, tpl_h),
            method=method, threshold=req.threshold, scales=req.scales,
        )
    else:
        template = img[req.y : y2, req.x : x2]
        matches = matcher.match_template(
            img, template, threshold=req.threshold, scales=req.scales
        )
    return {
        "count": len(matches),
        "method": method,
        "template": {"x": req.x, "y": req.y, "w": tpl_w, "h": tpl_h},
        "matches": [m.as_dict() for m in matches],
    }


@app.get("/")
def index() -> FileResponse:
    # Don't cache the HTML so the versioned ?v= asset links are always re-read.
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

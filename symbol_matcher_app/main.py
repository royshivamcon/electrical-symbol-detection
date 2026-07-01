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


@app.get("/api/requests")
def api_requests() -> dict:
    return {"requests": wl.list_requests()}


@app.get("/api/requests/{rid}/worksheets")
def api_worksheets(rid: str) -> dict:
    try:
        return {"rid": rid, "worksheets": wl.list_worksheets(rid)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


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


@app.get("/api/worksheet/{rid}/{wid}/sam_points")
def api_sam_points(
    rid: str,
    wid: str,
    limit: int = 0,
    model: str = "fastsam",
    crop: int = 90,
    min_symbol_px: int = 28,
    max_symbol_px: int = 90,
    pad: int = 4,
    pseudocolor: bool = False,
    sharpen: bool = False,
) -> dict:
    """Segment symbols around the worksheet's reference points (electrical Point
    features) and return their bounding boxes.

    ``model`` selects "fastsam" (default, tighter boxes) or "mobilesam".
    ``crop`` is the half-window around each point; ``min_symbol_px`` /
    ``max_symbol_px`` floor and cap box size, and ``pad`` adds a margin
    (FastSAM only).
    ``pseudocolor`` glow pseudo-colors the crop fed to the SAM model (FastSAM,
    HQ-SAM, SAM 2.1) so thin monochrome lines read as chromatic gradients.
    ``sharpen`` unsharp-masks the crop fed to the SAM model (a little aggressive)
    to make razor-thin strokes pop; it can be combined with ``pseudocolor``.
    Both affect only what the model sees — box coordinates and the
    connected-component fallback stay on the original image.
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
        if model == "mobilesam":
            boxes = [b.as_dict() for b in sb.boxes_from_points(img, used)]
        elif model == "hqsam":
            import hqsam_boxes as hsb  # lazy: loads segment_anything_hq on first use

            boxes = [
                b.as_dict()
                for b in hsb.boxes_from_points(
                    img, used, crop=crop, min_symbol_px=min_symbol_px,
                    max_symbol_px=max_symbol_px, pad=pad, pseudocolor=pseudocolor,
                    sharpen=sharpen,
                )
            ]
        elif model == "sam2":
            import sam2_boxes as s2b  # lazy: loads SAM 2.1 (ultralytics) on first use

            boxes = [
                b.as_dict()
                for b in s2b.boxes_from_points(
                    img, used, crop=crop, min_symbol_px=min_symbol_px,
                    max_symbol_px=max_symbol_px, pad=pad, pseudocolor=pseudocolor,
                    sharpen=sharpen,
                )
            ]
        else:
            import fastsam_boxes as fsb  # lazy: loads ultralytics on first use

            boxes = [
                b.as_dict()
                for b in fsb.boxes_from_points(
                    img, used, crop=crop, min_symbol_px=min_symbol_px,
                    max_symbol_px=max_symbol_px, pad=pad, pseudocolor=pseudocolor,
                    sharpen=sharpen,
                )
            ]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "model": model,
        "count": len(boxes),
        "total_points": len(points),
        "used_points": len(used),
        "boxes": boxes,
        "points": [{"x": p.x, "y": p.y, "name": p.name} for p in used],
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

    template = img[req.y : y2, req.x : x2]
    matches = matcher.match_template(
        img, template, threshold=req.threshold, scales=req.scales
    )
    return {
        "count": len(matches),
        "template": {"x": req.x, "y": req.y, "w": x2 - req.x, "h": y2 - req.y},
        "matches": [m.as_dict() for m in matches],
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

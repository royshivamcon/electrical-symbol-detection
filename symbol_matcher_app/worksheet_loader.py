"""Worksheet discovery and image loading.

Reads cached request metadata from ``data/requests/<rid>/worksheets_metadata.json``
and downloads worksheet rasters from their remote ``image_url`` (GCS). Downloaded
images are cached on disk so the same worksheet is only fetched once.
"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# Project paths -------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
REQUESTS_DIR = PROJECT_ROOT / "data" / "requests"
CACHE_DIR = APP_DIR / ".image_cache"
CACHE_DIR.mkdir(exist_ok=True)
PDF_CACHE_DIR = CACHE_DIR / "pdfs"

DOWNLOAD_TIMEOUT = 120


def list_requests() -> list[str]:
    """Return request ids (rid) that have cached worksheet metadata."""
    if not REQUESTS_DIR.exists():
        return []
    rids = [
        p.name
        for p in REQUESTS_DIR.iterdir()
        if p.is_dir() and (p / "worksheets_metadata.json").exists()
    ]
    return sorted(rids)


@lru_cache(maxsize=64)
def _load_metadata(rid: str) -> list[dict]:
    meta_path = REQUESTS_DIR / rid / "worksheets_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No worksheet metadata for request {rid!r}")
    with open(meta_path) as fh:
        return json.load(fh)


def list_worksheets(rid: str) -> list[dict]:
    """Return a compact list of worksheets for a request.

    Only worksheets that expose an ``image_url`` are returned, since the app
    loads rasters from that URL. Each entry is flagged with ``has_geometry`` so
    the UI can prefer sheets that actually have annotations, and carries
    ``page_no`` for reference.
    """
    geom_dir = REQUESTS_DIR / rid / "worksheet_geometries"
    out = []
    for w in _load_metadata(rid):
        image = w.get("image") or {}
        url = image.get("image_url")
        if not url:
            continue
        wid = w.get("id")
        has_geom = (geom_dir / f"{wid}_geometries.json").exists()
        out.append(
            {
                "wid": wid,
                "name": w.get("name") or "",
                "title": w.get("title") or "",
                "page_no": w.get("page_no"),
                "width": image.get("width"),
                "height": image.get("height"),
                "has_geometry": has_geom,
            }
        )
    # Sheets with geometries first, then by page number.
    out.sort(key=lambda d: (not d["has_geometry"], d["page_no"] if d["page_no"] is not None else 1e9))
    return out


def _worksheet_entry(rid: str, wid: str) -> dict:
    for w in _load_metadata(rid):
        if w.get("id") == wid:
            return w
    raise KeyError(f"Worksheet {wid!r} not found in request {rid!r}")


def get_image_url(rid: str, wid: str) -> str:
    entry = _worksheet_entry(rid, wid)
    url = (entry.get("image") or {}).get("image_url")
    if not url:
        raise ValueError(f"Worksheet {wid!r} has no image_url")
    return url


def load_worksheet_image(rid: str, wid: str) -> np.ndarray:
    """Return the worksheet raster as a BGR uint8 numpy array.

    The decoded PNG is cached under ``.image_cache`` keyed by rid/wid.
    """
    cache_path = CACHE_DIR / f"{rid}__{wid}.png"
    if cache_path.exists():
        img = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    url = get_image_url(rid, wid)
    resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(cache_path), img)
    return img


def get_image_png_bytes(rid: str, wid: str) -> bytes:
    """Return the worksheet image encoded as PNG bytes (uses the disk cache)."""
    load_worksheet_image(rid, wid)  # ensures cache populated
    cache_path = CACHE_DIR / f"{rid}__{wid}.png"
    return cache_path.read_bytes()


def _download_pdf(url: str) -> Path:
    """Download (and disk-cache) the source blueprint PDF, keyed by URL hash.

    Worksheets of the same request usually share one combined drawing-set PDF, so
    caching by URL means the (large) file is fetched once and reused across sheets.
    """
    PDF_CACHE_DIR.mkdir(exist_ok=True)
    path = PDF_CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.pdf"
    if not path.exists():
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


# MuPDF refuses to allocate a single pixmap beyond an internal size cap ("Overly
# large image"). At high zoom a whole sheet exceeds it, so we rasterize in
# horizontal bands (each well under the cap) and stitch them. Keep each band's
# pixel budget conservative (~96 MP -> ~288 MB at 3 channels).
_MAX_BAND_PX = 96_000_000


def _render_banded(fitz, page, rect, sx: float, sy: float, target_w: int, target_h: int) -> np.ndarray:
    """Rasterize ``page`` to a BGR array of ~(target_h, target_w) in row bands.

    Each band is rendered with a ``clip`` rect (in page/point coordinates) so no
    single pixmap trips MuPDF's maximum-image limit; bands are normalized to
    ``target_w`` and vertically stacked.
    """
    mtx = fitz.Matrix(sx, sy)
    band_rows = max(1, int(_MAX_BAND_PX // max(1, target_w)))
    if band_rows >= target_h:  # small enough to render in one shot
        pix = page.get_pixmap(matrix=mtx, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)

    bands: list[np.ndarray] = []
    y = 0
    while y < target_h:
        y1 = min(target_h, y + band_rows)
        # Target rows [y, y1) -> page-space y via the render scale.
        clip = fitz.Rect(rect.x0, rect.y0 + y / sy, rect.x1, rect.y0 + y1 / sy)
        pix = page.get_pixmap(matrix=mtx, clip=clip, alpha=False)
        b = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
        if b.shape[1] != target_w:  # keep widths identical so vstack lines up
            b = cv2.resize(b, (target_w, b.shape[0]), interpolation=cv2.INTER_AREA)
        bands.append(b)
        y = y1
    return np.vstack(bands)


def render_pdf_image(rid: str, wid: str, zoom: float, remove_text: bool = False) -> np.ndarray:
    """Render the worksheet's source PDF page at ``zoom`` and return BGR uint8.

    This is quill_forge's zoom method: the page is rasterized with
    ``fitz.Matrix(zoom, zoom)`` (``zoom = 1`` == the base raster resolution), giving
    crisper thin lines than the pre-rendered ``image_url``. The output is sized to
    exactly ``base_width*zoom x base_height*zoom`` so ground-truth coordinates (which
    map to the base raster) scale to it by the same factor.

    ``remove_text=True`` strips every text word from the page (redaction) while
    keeping vector line art and raster images intact, so labels/dimensions don't
    distract the detector. Cached per rid/wid/zoom (+ a ``nt`` tag when text-free).
    """
    entry = _worksheet_entry(rid, wid)
    image = entry.get("image") or {}
    base_w, base_h = int(image.get("width") or 0), int(image.get("height") or 0)
    url = entry.get("blueprint_file_url")
    if not url:
        raise ValueError(f"Worksheet {wid!r} has no blueprint_file_url to render")
    if not (base_w and base_h):
        raise ValueError(f"Worksheet {wid!r} has no base image dimensions")

    tag = f"{zoom:g}".replace(".", "p")
    if remove_text:
        tag += "_nt"
    cache_path = CACHE_DIR / f"{rid}__{wid}__z{tag}.png"
    if cache_path.exists():
        img = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    import fitz  # PyMuPDF; imported lazily so base image loading has no hard dep

    page_no = int(entry.get("page_no") or 1)
    rotation = int(entry.get("rotation_angle") or 0)
    pdf_path = _download_pdf(url)
    with fitz.open(str(pdf_path)) as doc:
        page = doc.load_page(max(0, page_no - 1))  # fitz 0-indexed, Feathers 1-indexed
        if remove_text:
            # Redact each text word but leave graphics/images (see pdf_renderring.ipynb).
            for w in page.get_text("words"):
                page.add_redact_annot(fitz.Rect(w[:4]))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
        if rotation:
            page.set_rotation(rotation)
        rect = page.rect
        sx = (zoom * base_w / rect.width) if rect.width else zoom
        sy = (zoom * base_h / rect.height) if rect.height else zoom
        target_w = int(round(base_w * zoom))
        target_h = int(round(base_h * zoom))
        arr = _render_banded(fitz, page, rect, sx, sy, target_w, target_h)

    target = (target_w, target_h)
    if (arr.shape[1], arr.shape[0]) != target:
        arr = cv2.resize(arr, target, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(cache_path), arr)
    return arr


@contextmanager
def pdf_tile_renderer(rid: str, wid: str, zoom: float, remove_text: bool = False):
    """Open the source PDF page once and yield ``(target_w, target_h, render_fn)``.

    ``target_w``/``target_h`` are the full rendered-sheet dimensions (``base*zoom``);
    ``render_fn(x0, y0, x1, y1)`` rasterizes just that rendered-pixel rectangle via a
    MuPDF ``clip`` and returns a BGR uint8 array of exactly ``(y1-y0, x1-x0)``.

    This keeps peak memory to one tile instead of the whole ``base*zoom`` sheet, so
    high-zoom tiled detection doesn't hold the full raster in RAM. The page is
    redacted (``remove_text``) once here, then every tile renders from that page.
    """
    entry = _worksheet_entry(rid, wid)
    image = entry.get("image") or {}
    base_w, base_h = int(image.get("width") or 0), int(image.get("height") or 0)
    url = entry.get("blueprint_file_url")
    if not url:
        raise ValueError(f"Worksheet {wid!r} has no blueprint_file_url to render")
    if not (base_w and base_h):
        raise ValueError(f"Worksheet {wid!r} has no base image dimensions")

    import fitz  # PyMuPDF; lazy import (base image loading has no hard dep)

    page_no = int(entry.get("page_no") or 1)
    rotation = int(entry.get("rotation_angle") or 0)
    pdf_path = _download_pdf(url)
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(max(0, page_no - 1))
        if remove_text:
            for w in page.get_text("words"):
                page.add_redact_annot(fitz.Rect(w[:4]))
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_NONE,
                graphics=fitz.PDF_REDACT_LINE_ART_NONE,
            )
        if rotation:
            page.set_rotation(rotation)
        rect = page.rect
        sx = (zoom * base_w / rect.width) if rect.width else zoom
        sy = (zoom * base_h / rect.height) if rect.height else zoom
        target_w = int(round(base_w * zoom))
        target_h = int(round(base_h * zoom))
        mtx = fitz.Matrix(sx, sy)

        def render_fn(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(target_w, int(x1)), min(target_h, int(y1))
            clip = fitz.Rect(rect.x0 + x0 / sx, rect.y0 + y0 / sy,
                             rect.x0 + x1 / sx, rect.y0 + y1 / sy)
            pix = page.get_pixmap(matrix=mtx, clip=clip, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
            want = (x1 - x0, y1 - y0)
            if (arr.shape[1], arr.shape[0]) != want:
                arr = cv2.resize(arr, want, interpolation=cv2.INTER_AREA)
            return arr

        yield target_w, target_h, render_fn
    finally:
        doc.close()

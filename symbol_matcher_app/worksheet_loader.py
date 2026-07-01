"""Worksheet discovery and image loading.

Reads cached request metadata from ``data/requests/<rid>/worksheets_metadata.json``
and downloads worksheet rasters from their remote ``image_url`` (GCS). Downloaded
images are cached on disk so the same worksheet is only fetched once.
"""

from __future__ import annotations

import io
import json
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

DOWNLOAD_TIMEOUT = 60


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

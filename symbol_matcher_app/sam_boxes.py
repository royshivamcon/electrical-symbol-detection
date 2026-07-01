"""MobileSAM box finding from worksheet reference points.

The worksheet ``*_geometries.json`` files contain annotated **point** features
(the symbol reference points). We apply the same "electrical" filtering used in
the EDA notebooks — keep ``feature.geometry_type == 1`` (Point) outputs — map
each point from Feathers coordinates to image pixels, then prompt MobileSAM with
those points to segment each symbol and return its bounding box.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import worksheet_loader as wl

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CHECKPOINT = PROJECT_ROOT / "models" / "mobile_sam.pt"

_PREDICTOR = None
_PREDICTOR_LOCK = threading.Lock()


# --- model -----------------------------------------------------------------
def _device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_predictor():
    """Lazily build a single shared MobileSAM predictor."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    with _PREDICTOR_LOCK:
        if _PREDICTOR is None:
            from mobile_sam import SamPredictor, sam_model_registry

            if not CHECKPOINT.exists():
                raise FileNotFoundError(f"MobileSAM checkpoint not found: {CHECKPOINT}")
            sam = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
            sam.to(_device())
            sam.eval()
            _PREDICTOR = SamPredictor(sam)
    return _PREDICTOR


# --- reference points (electrical filtering) -------------------------------
@dataclass
class RefPoint:
    x: int  # pixel coords on the worksheet raster
    y: int
    name: str


def _geometry_path(rid: str, wid: str) -> Path:
    return wl.REQUESTS_DIR / rid / "worksheet_geometries" / f"{wid}_geometries.json"


@lru_cache(maxsize=32)
def _worksheet_fe_dims(rid: str, wid: str) -> tuple[int, int]:
    for w in json.load(open(wl.REQUESTS_DIR / rid / "worksheets_metadata.json")):
        if w.get("id") == wid:
            img = w.get("image") or {}
            return int(img.get("width") or 0), int(img.get("height") or 0)
    raise KeyError(f"Worksheet {wid!r} not found in request {rid!r}")


def load_reference_points(rid: str, wid: str, img_w: int, img_h: int) -> list[RefPoint]:
    """Load Point features (geometry_type == 1) mapped to ``img_w x img_h`` pixels.

    Feathers point coordinates are ``[x, y]`` with **y negative** (y-up), defined
    in the worksheet's FE canvas (``image.width x image.height``). We scale to the
    raster we actually loaded and flip Y.
    """
    gp = _geometry_path(rid, wid)
    if not gp.exists():
        return []
    fe_w, fe_h = _worksheet_fe_dims(rid, wid)
    sx = img_w / fe_w if fe_w else 1.0
    sy = img_h / fe_h if fe_h else 1.0

    pts: list[RefPoint] = []
    data = json.load(open(gp))
    for out in data.get("outputs", []):
        feat = out.get("feature", {}) or {}
        if feat.get("geometry_type") != 1:  # 1 == Point
            continue
        name = feat.get("name", "") or ""
        for f in out.get("output_geojson", {}).get("features", []):
            c = (f.get("geometry", {}) or {}).get("coordinates", [])
            if len(c) == 2 and isinstance(c[0], (int, float)):
                px = int(round(c[0] * sx))
                py = int(round(abs(c[1]) * sy))
                if 0 <= px < img_w and 0 <= py < img_h:
                    pts.append(RefPoint(px, py, name))
    return pts


# --- reference polygons (ground-truth "true patches") ----------------------
# Electrical keyword filtering, ported verbatim from the EDA notebook
# (electrical_data_testing.ipynb). Polygons are filtered on their resolved
# ``category + source`` tag string (see ``_poly_tag_str``); e.g. "LIGHTINGLEGEND".
_KEYWORDS = {
    "outlets": ["RECEPTACLE", "RECEPTABLE", "OUTLET", "DUPLEX", "QUADPLEX",
                "T-SLOT", "GROUND FAULT", "SWITCHED", "SINGLE USE",
                "SPECIAL USE", "_WP"],
    "switches": ["SWITCH", "GANG", "SINGLE POLE", "THREE WAY", "FOUR WAY",
                 "3 WAY", "DIMMER", "OCCUPANCY SENSOR", "KEYED",
                 "LOW VOLTAGE", "LV", "LINE VOLTAGE", "T-SWITCH",
                 "PUSH BUTTON", "PUSHBUTTON", "DISCONNECT",
                 "TRANSFER SWITCH", "TAMPER", "FLOW SWITCH", "SWITCH BANK"],
    "lighting": ["LIGHT", "LIGHTING", "FIXTURE", "LUMINAIRE", "DOWNLIGHT",
                 "RECESSED", "SURFACE", "SUSPENDED", "WALL MOUNTED",
                 "CEILING MOUNTED", "PENDANT", "TRACK", "STRIP", "LINEAR",
                 "LED", "FLUORESCENT", "EMERGENCY LIGHT", "EXIT SIGN",
                 "BATTERY PACK", "REMOTE HEAD"],
    "panels": ["PANEL", "SUITE PANEL", "LIGHTING CONTROL PANEL",
               "CIRCUIT BREAKER", "MS", "MSA", "MSB", "MPP", "ATS",
               "DISTRIBUTION", "TRANSFORMER", "KVA", "KAIC", "SPD",
               "208Y/120V", "600Y/347V", "120/208V", "3PH", "1P", "3P"],
    "wiring": ["WIRING_DEVICESLEGEND", "WIRING_DEVICESSCHEDULE",
               "WIRING_ROUTINGLEGEND"],
    "conduit": ["CONDUIT", "CONDUIT CONTINUATION", "CONDUIT CONCEALED",
                "CONDUIT - UP", "CONDUIT - DOWN", "CONDUIT STUB", "UNDERGROUND"],
    "motors": ["MOTOR", "MECHANICAL MOTOR"],
    "generators": ["GENERATOR", "GENSET", "BACKUP GENERATOR", "KW"],
    "meters": ["METER", "UTILITY METER", "CURRENT TRANSFORMER",
               "POTENTIAL TRANSFORMER"],
    "ev_charging": ["PEV", "PEVA", "PEVB"],
    "fire_alarm": ["FIRE ALARM", "PULL STATION", "PULLSTATION",
                   "SMOKE DETECTOR", "CARBON MONOXIDE", "STROBE", "HORN",
                   "SPEAKER", "MINI HORN", "BELL", "CM", "MM", "IM", "AD",
                   "FP", "END OF LINE RESISTOR", "15CD", "60CD"],
    "data_lowv": ["DATA OUTLET", "TELECOM", "NETWORK SWITCH", "CARD READER",
                  "INTERCOM", "SURVEILLANCE", "DOOR SECURITY MONITOR",
                  "ELECTRIC STRIKE", "MAGNETIC LOCK", "ELECTRIC HINGE",
                  "MOTION SENSOR", "OCCUPANCY SENSOR", "THERMOSTAT",
                  "PHOTOELECTRIC", "DAYLIGHT", "POWER PACK", "0-10V", "NLIGHT"],
}
_ELECTRICAL_TERMS = {term.lower() for terms in _KEYWORDS.values() for term in terms}


def _is_electrical(name: str) -> bool:
    n = (name or "").lower()
    return any(term in n for term in _ELECTRICAL_TERMS)


@dataclass
class RefPolygon:
    x: int  # bounding box on the worksheet raster
    y: int
    w: int
    h: int
    name: str


@lru_cache(maxsize=8)
def _tag_maps(rid: str) -> tuple[dict, dict]:
    """Offline replacement for quill_forge ``resolve_tags_info``: return
    ``(tag_id -> tag_name, tag_type_id -> tag_type_name)`` from ``tag_library.json``.
    """
    tl_path = wl.REQUESTS_DIR / rid / "tag_library.json"
    if not tl_path.exists():
        return {}, {}
    lib = json.load(open(tl_path))
    tagid_to_name = {
        tag["id"]: tag.get("name")
        for tt in lib.get("tag_types", [])
        for tag in tt.get("tags", [])
    }
    typeid_to_name = {tt["id"]: tt.get("name") for tt in lib.get("tag_types", [])}
    return tagid_to_name, typeid_to_name


def _poly_tag_str(rid: str, tags_info: dict) -> str:
    """Resolve a polygon's ``tags_info`` to the EDA ``category + source`` string."""
    tagid_to_name, typeid_to_name = _tag_maps(rid)
    resolved: dict[str, str] = {}
    for type_id, v in (tags_info or {}).items():
        tname = typeid_to_name.get(type_id)
        if tname:
            resolved[tname] = tagid_to_name.get((v or {}).get("tagId")) or ""
    return (resolved.get("category") or "") + (resolved.get("source") or "")


def load_reference_polygons(
    rid: str, wid: str, img_w: int, img_h: int, electrical: bool = True
) -> list[RefPolygon]:
    """Load Polygon features (geometry_type == 3) as ground-truth bounding-box
    "true patches", mapped to ``img_w x img_h`` pixels.

    When ``electrical`` is True (default) this mirrors the EDA notebook: keep only
    polygons whose resolved ``category + source`` tag string passes the electrical
    keyword filter (drops ``ROI`` and non-electrical categories). When False, every
    polygon is kept. Each polygon's outer ring is reduced to a bbox.
    """
    gp = _geometry_path(rid, wid)
    if not gp.exists():
        return []
    fe_w, fe_h = _worksheet_fe_dims(rid, wid)
    sx = img_w / fe_w if fe_w else 1.0
    sy = img_h / fe_h if fe_h else 1.0

    polys: list[RefPolygon] = []
    data = json.load(open(gp))
    for out in data.get("outputs", []):
        feat = out.get("feature", {}) or {}
        if feat.get("geometry_type") != 3:  # 3 == Polygon
            continue
        name = feat.get("name", "") or ""
        for f in out.get("output_geojson", {}).get("features", []):
            if electrical:
                tags_info = (f.get("properties", {}) or {}).get("tags_info", {})
                if not _is_electrical(_poly_tag_str(rid, tags_info)):
                    continue
            ring = (f.get("geometry", {}) or {}).get("coordinates", [])
            if not ring or not isinstance(ring[0], (list, tuple)):
                continue
            xs = [c[0] * sx for c in ring[0] if len(c) >= 2]
            ys = [abs(c[1]) * sy for c in ring[0] if len(c) >= 2]
            if not xs or not ys:
                continue
            x0 = max(0, int(round(min(xs))))
            y0 = max(0, int(round(min(ys))))
            x1 = min(img_w, int(round(max(xs))))
            y1 = min(img_h, int(round(max(ys))))
            if x1 > x0 and y1 > y0:
                polys.append(RefPolygon(x0, y0, x1 - x0, y1 - y0, name))
    return polys


# --- segmentation ----------------------------------------------------------
@dataclass
class SamBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    name: str

    def as_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "score": round(self.score, 4), "name": self.name,
        }


def boxes_from_points(
    image_bgr: np.ndarray,
    points: list[RefPoint],
    max_box_frac: float = 0.04,
    min_box_px: int = 6,
) -> list[SamBox]:
    """Prompt MobileSAM with each reference point and return symbol boxes.

    ``max_box_frac`` drops masks whose box covers more than this fraction of the
    image area (a guard against the mask leaking into the background).
    """
    if not points:
        return []
    predictor = get_predictor()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    img_area = float(H * W)
    predictor.set_image(image_rgb)

    out: list[SamBox] = []
    labels = np.array([1])
    for p in points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[p.x, p.y]]),
            point_labels=labels,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        mask = masks[best]
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if bw < min_box_px or bh < min_box_px:
            continue
        if (bw * bh) / img_area > max_box_frac:
            continue
        out.append(SamBox(x1, y1, bw, bh, float(scores[best]), p.name))
    return out

"""Export symbol crops from quill_local bbox_* layers at zoom=4x.

Renders a local PDF window around each bbox (never loads a full 4x sheet).
Writes under ``data/requests/<rid>/symbol_crops_quill/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from symbol_embed.config import DEFAULT_ZOOM, MATCHER_APP, Paths, VAL_RIDS
from symbol_embed.dataset import load_rids_file
from symbol_embed.export_crops import crop_from_box, slugify

if str(MATCHER_APP) not in sys.path:
    sys.path.insert(0, str(MATCHER_APP))

import worksheet_loader as wl  # noqa: E402

BBOX_PREFIX = "bbox_"
PAD_PX = 4  # pad at zoom resolution


def _label_from_layer(name: str) -> str:
    n = (name or "").strip()
    if n.startswith(BBOX_PREFIX):
        n = n[len(BBOX_PREFIX) :]
    return n.strip() or "_unnamed"


def _fe_to_base_xy(
    x: float, y: float, *, fe_w: int, fe_h: int, base_w: int, base_h: int
) -> tuple[float, float]:
    sx = base_w / fe_w if fe_w else 1.0
    sy = base_h / fe_h if fe_h else 1.0
    return float(x) * sx, abs(float(y)) * sy


def iter_quill_bboxes(
    data: dict,
    *,
    fe_w: int,
    fe_h: int,
    base_w: int,
    base_h: int,
    zoom: float,
) -> list[dict]:
    """Parse bbox_* polygon layers → axis-aligned boxes in zoomed pixels."""
    out: list[dict] = []
    for output in data.get("outputs") or []:
        feature = output.get("feature") or {}
        layer_name = str(feature.get("name") or "")
        if not layer_name.startswith(BBOX_PREFIX):
            continue
        label = _label_from_layer(layer_name)
        features = (output.get("output_geojson") or {}).get("features") or []
        for fi, row in enumerate(features):
            geom = row.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            ring = (geom.get("coordinates") or [[]])[0]
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                continue
            xs, ys = [], []
            for pt in ring:
                if len(pt) < 2:
                    continue
                bx, by = _fe_to_base_xy(
                    pt[0], pt[1], fe_w=fe_w, fe_h=fe_h, base_w=base_w, base_h=base_h
                )
                xs.append(bx * zoom)
                ys.append(by * zoom)
            if not xs:
                continue
            x0, y0 = min(xs), min(ys)
            x1, y1 = max(xs), max(ys)
            w, h = x1 - x0, y1 - y0
            if w < 2 or h < 2:
                continue
            out.append(
                {
                    "name": label,
                    "layer": layer_name,
                    "x": int(round(x0)),
                    "y": int(round(y0)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                    "feat_idx": fi,
                }
            )
    return out


def _crop_via_tile(
    session: wl.PdfTileSession,
    box: dict,
    *,
    pad: int = PAD_PX,
) -> np.ndarray | None:
    """Render a padded window around ``box`` and extract the AABB crop."""
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(session.target_w, x + w + pad)
    y1 = min(session.target_h, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    tile = session.render(x0, y0, x1, y1)
    # Box relative to tile
    local = {
        "x": x - x0,
        "y": y - y0,
        "w": w,
        "h": h,
    }
    return crop_from_box(tile, local)


def export_rid_quill(
    rid: str,
    *,
    zoom: float = DEFAULT_ZOOM,
    force: bool = False,
    max_sheets: int = 0,
    max_boxes: int = 0,
    paths: Paths | None = None,
) -> Path:
    paths = paths or Paths()
    out_root = paths.crops_quill_dir(rid)
    quill_dir = paths.quill_local / rid
    if not quill_dir.is_dir():
        raise FileNotFoundError(f"missing quill_local pack: {quill_dir}")

    if out_root.exists() and any(out_root.glob("*/*.png")) and not force:
        print(f"[export_quill] exists {out_root} (pass --force to redo)")
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if force and manifest_path.exists():
        for p in out_root.rglob("*.png"):
            p.unlink()
        manifest_path.unlink(missing_ok=True)

    geom_files = sorted(quill_dir.glob("*_geometries.json"))
    if max_sheets > 0:
        geom_files = geom_files[:max_sheets]
    print(
        f"[export_quill] rid={rid[:8]} sheets={len(geom_files)} zoom={zoom:g}",
        flush=True,
    )

    counts: Counter[str] = Counter()
    name_to_slug: dict[str, str] = {}
    n_written = 0
    n_skip = 0

    with open(manifest_path, "w") as mf:
        for si, gpath in enumerate(geom_files):
            wid = gpath.name.removesuffix("_geometries.json")
            print(f"[export_quill] [{si + 1}/{len(geom_files)}] {wid[:8]} …", flush=True)
            try:
                data = json.loads(gpath.read_text(encoding="utf-8"))
                entry = wl._worksheet_entry(rid, wid) or {}
                image_meta = entry.get("image") or {}
                fe_w = int(image_meta.get("width") or 0)
                fe_h = int(image_meta.get("height") or 0)
                if not (fe_w and fe_h):
                    # Fall back to cached base raster dims
                    base = wl.load_worksheet_image(rid, wid)
                    fe_h, fe_w = base.shape[:2]
                base_w, base_h = fe_w, fe_h
            except Exception as exc:  # noqa: BLE001
                print(f"  skip meta: {exc}")
                n_skip += 1
                continue

            boxes = iter_quill_bboxes(
                data,
                fe_w=fe_w,
                fe_h=fe_h,
                base_w=base_w,
                base_h=base_h,
                zoom=zoom,
            )
            if not boxes:
                print("  no bbox layers")
                continue
            if max_boxes > 0:
                boxes = boxes[:max_boxes]

            try:
                session = wl.open_pdf_tile_session(rid, wid, zoom=zoom, remove_text=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip pdf: {exc}")
                n_skip += 1
                continue

            try:
                for i, box in enumerate(boxes):
                    name = box["name"]
                    try:
                        crop = _crop_via_tile(session, box)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  crop fail {i}: {exc}")
                        continue
                    if crop is None or crop.size == 0:
                        continue
                    slug = name_to_slug.setdefault(name, slugify(name))
                    class_dir = out_root / slug
                    class_dir.mkdir(exist_ok=True)
                    rel = f"{slug}/{wid[:8]}_{i:06d}.png"
                    cv2.imwrite(str(out_root / rel), crop)
                    row = {
                        "path": rel,
                        "name": name,
                        "slug": slug,
                        "wid": wid,
                        "rid": rid,
                        "box": {
                            "x": box["x"],
                            "y": box["y"],
                            "w": box["w"],
                            "h": box["h"],
                        },
                        "layer": box["layer"],
                        "zoom": zoom,
                        "source": "quill_local",
                    }
                    mf.write(json.dumps(row) + "\n")
                    counts[name] += 1
                    n_written += 1
            finally:
                session.close()
            print(f"  boxes={len(boxes)} written_total={n_written}")

    classes = {
        "by_name": {
            name: {"slug": name_to_slug[name], "count": counts[name]}
            for name in sorted(counts)
        },
        "n_images": n_written,
        "n_classes": len(counts),
        "rid": rid,
        "zoom": zoom,
        "source": "quill_local",
    }
    (out_root / "classes.json").write_text(json.dumps(classes, indent=2))
    print(
        f"[export_quill] done: {n_written} crops, {len(counts)} classes, "
        f"skip_sheets={n_skip} → {out_root}"
    )
    return out_root


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rid", action="append", default=None)
    ap.add_argument(
        "--rids-file",
        default=None,
        help="RID list (default: symbol_embed/train_rids.txt)",
    )
    ap.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-sheets", type=int, default=0)
    ap.add_argument("--max-boxes", type=int, default=0, help="Cap boxes per sheet (0=all)")
    args = ap.parse_args(argv)

    paths = Paths()
    if args.rid:
        rid_list = list(args.rid)
    else:
        rids_path = Path(args.rids_file) if args.rids_file else paths.train_rids_file()
        rid_list = load_rids_file(rids_path)

    # Include val RIDs so the full 10-RID pool can be built.
    for vr in VAL_RIDS:
        if vr not in rid_list:
            rid_list.append(vr)

    print(f"[export_quill] {len(rid_list)} rid(s) zoom={args.zoom:g}", flush=True)
    for rid in rid_list:
        export_rid_quill(
            rid,
            zoom=args.zoom,
            force=args.force,
            max_sheets=args.max_sheets,
            max_boxes=args.max_boxes,
            paths=paths,
        )


if __name__ == "__main__":
    main()

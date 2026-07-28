"""Export convex-hull crops from fastsamx_sam2 for one RID."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from symbol_embed.config import MATCHER_APP, Paths

if str(MATCHER_APP) not in sys.path:
    sys.path.insert(0, str(MATCHER_APP))

import sam_boxes as sb  # noqa: E402
import worksheet_loader as wl  # noqa: E402
from detection import DetectionParams, get_strategy  # noqa: E402


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(name: str, *, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("_", (name or "").strip()).strip("_")
    if not s:
        return "_unnamed"
    return s[:max_len]


def _order_hull_corners(hull: list[list[int]]) -> np.ndarray:
    """Order 4 hull corners as tl, tr, br, bl for perspective warp."""
    pts = np.asarray(hull, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"expected 4 hull corners, got {pts.shape}")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def crop_from_box(image_bgr: np.ndarray, box: dict, *, out_size: int | None = None) -> np.ndarray | None:
    """Upright crop from hull (preferred) or AABB. Returns BGR uint8 or None."""
    H, W = image_bgr.shape[:2]
    hull = box.get("hull")
    if hull and len(hull) >= 4:
        try:
            src = _order_hull_corners(hull[:4])
        except ValueError:
            src = None
        if src is not None:
            w = float(np.linalg.norm(src[1] - src[0]))
            h = float(np.linalg.norm(src[3] - src[0]))
            if w >= 2 and h >= 2:
                dw = max(2, int(round(w)))
                dh = max(2, int(round(h)))
                dst = np.array(
                    [[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(src, dst)
                crop = cv2.warpPerspective(image_bgr, M, (dw, dh))
                if out_size:
                    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
                return crop

    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image_bgr[y0:y1, x0:x1].copy()
    if out_size:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def _detect_boxes(rid: str, wid: str, img, points, *, workers: int = 0) -> list[dict]:
    params = DetectionParams(
        model="fastsamx_sam2",
        rid=rid,
        wid=wid,
        hull=True,
        postproc=True,
        workers=workers,
    )
    boxes = get_strategy("fastsamx_sam2").detect(img, points, params, want_masks=False)
    return [b.as_dict() for b in boxes]


def export_rid(
    rid: str,
    *,
    wids: list[str] | None = None,
    limit_points: int = 0,
    workers: int = 0,
    force: bool = False,
    patches_only: bool = True,
    max_sheets: int = 0,
    paths: Paths | None = None,
) -> Path:
    paths = paths or Paths()
    out_root = paths.crops_dir(rid)
    if out_root.exists() and any(out_root.glob("*/*.png")) and not force:
        print(f"[export] exists {out_root} (pass --force to redo)")
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    if force and manifest_path.exists():
        for p in out_root.rglob("*.png"):
            p.unlink()
        manifest_path.unlink(missing_ok=True)

    sheets = wl.list_worksheets(rid)
    if wids:
        want = set(wids)
        sheets = [w for w in sheets if w["wid"] in want]
    sheets = [w for w in sheets if w.get("has_geometry")]
    if patches_only:
        sheets = [w for w in sheets if sb.worksheet_has_patches(rid, w["wid"])]
    if max_sheets > 0:
        sheets = sheets[:max_sheets]
    print(f"[export] rid={rid[:8]} sheets={len(sheets)} patches_only={patches_only}")

    counts: Counter[str] = Counter()
    name_to_slug: dict[str, str] = {}
    n_written = 0

    with open(manifest_path, "w") as mf:
        for si, sheet in enumerate(sheets):
            wid = sheet["wid"]
            print(f"[export] [{si + 1}/{len(sheets)}] {wid[:8]} …", flush=True)
            try:
                img = wl.load_worksheet_image(rid, wid)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip load: {exc}")
                continue
            H, W = img.shape[:2]
            points = sb.load_reference_points(rid, wid, W, H)
            if limit_points > 0:
                points = points[:limit_points]
            if not points:
                print("  no reference points")
                continue
            try:
                boxes = _detect_boxes(rid, wid, img, points, workers=workers)
            except Exception as exc:  # noqa: BLE001
                print(f"  detect failed: {exc}")
                continue

            for i, box in enumerate(boxes):
                name = (box.get("name") or "").strip()
                if not name:
                    continue
                crop = crop_from_box(img, box)
                if crop is None or crop.size == 0:
                    continue
                slug = name_to_slug.setdefault(name, slugify(name))
                class_dir = out_root / slug
                class_dir.mkdir(exist_ok=True)
                rel = f"{slug}/{wid[:8]}_{i:06d}.png"
                abs_path = out_root / rel
                cv2.imwrite(str(abs_path), crop)
                row = {
                    "path": rel,
                    "name": name,
                    "slug": slug,
                    "wid": wid,
                    "rid": rid,
                    "box": {k: box[k] for k in ("x", "y", "w", "h") if k in box},
                    "hull": box.get("hull"),
                    "score": box.get("score"),
                    "source": box.get("source"),
                }
                mf.write(json.dumps(row) + "\n")
                counts[name] += 1
                n_written += 1
            print(f"  boxes={len(boxes)} written_total={n_written}")

    classes = {
        "by_name": {
            name: {"slug": name_to_slug[name], "count": counts[name]}
            for name in sorted(counts)
        },
        "n_images": n_written,
        "n_classes": len(counts),
        "rid": rid,
    }
    (out_root / "classes.json").write_text(json.dumps(classes, indent=2))
    print(f"[export] done: {n_written} crops, {len(counts)} classes → {out_root}")
    return out_root


def main(argv: list[str] | None = None) -> None:
    from symbol_embed.config import DEFAULT_N_RIDS, Paths
    from symbol_embed.dataset import list_available_rids, select_train_rids

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rid", action="append", default=None, help="Repeatable RID (or use --n-rids)")
    ap.add_argument(
        "--n-rids",
        type=int,
        default=None,
        help=f"Export this many RIDs (default {DEFAULT_N_RIDS} when --rid omitted)",
    )
    ap.add_argument("--wid", action="append", default=None, help="Repeatable WID filter")
    ap.add_argument("--limit-points", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Detection postproc workers (default 16 for 16-vCPU host)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--patches-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only worksheets with wiring-device GT patches (default: true)",
    )
    ap.add_argument("--max-sheets", type=int, default=0, help="Cap sheets per RID (0=all)")
    ap.add_argument(
        "--rids-file",
        default=None,
        help="Text file with one RID per line (overrides --n-rids sampling)",
    )
    args = ap.parse_args(argv)

    paths = Paths()
    if args.rids_file:
        from symbol_embed.dataset import load_rids_file

        rid_list = load_rids_file(args.rids_file)
    elif args.rid:
        rid_list = list(args.rid)
    else:
        n = args.n_rids if args.n_rids is not None else DEFAULT_N_RIDS
        rid_list = select_train_rids(
            rids=None,
            n_rids=n,
            seed=args.seed,
            paths=paths,
            require_crops=False,
        )
        if not rid_list:
            rid_list = list_available_rids(paths)[:n]

    print(f"[export] {len(rid_list)} rid(s)", flush=True)
    for rid in rid_list:
        export_rid(
            rid,
            wids=args.wid,
            limit_points=args.limit_points,
            workers=args.workers,
            force=args.force,
            patches_only=args.patches_only,
            max_sheets=args.max_sheets,
        )


if __name__ == "__main__":
    main()

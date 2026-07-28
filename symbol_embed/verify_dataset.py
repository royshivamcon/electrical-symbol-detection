"""Verify quill 10-RID pool before training; write report + sample grids."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from symbol_embed.augment import GreySpeckNoise, StrikethroughNoise
from symbol_embed.config import DEFAULT_ZOOM, Paths
from symbol_embed.dataset import (
    build_pooled_quill_dataset,
    load_manifest,
    resolve_train_val_rids,
)


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _denorm_to_pil(t) -> Image.Image:
    """CHW ImageNet-normalized or [0,1] tensor → RGB PIL."""
    import torch

    from symbol_embed.dataset import IMAGENET_MEAN, IMAGENET_STD

    x = t.detach().cpu().clone()
    if x.min() < -0.1:  # likely normalized
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = x * std + mean
    x = x.clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _make_grid(
    pairs: list[tuple[Image.Image, Image.Image, str]],
    out_path: Path,
    *,
    cell: int = 128,
) -> None:
    """pairs: (clean, noised, label)."""
    if not pairs:
        return
    n = len(pairs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    # each sample: clean | noised side by side
    gw, gh = cell * 2 + 8, cell + 28
    canvas = Image.new("RGB", (cols * gw, rows * gh), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    for i, (clean, noised, label) in enumerate(pairs):
        r, c = divmod(i, cols)
        x0, y0 = c * gw + 4, r * gh + 4
        canvas.paste(clean.resize((cell, cell)), (x0, y0))
        canvas.paste(noised.resize((cell, cell)), (x0 + cell + 4, y0))
        text = (label or "")[:40]
        draw.text((x0, y0 + cell + 2), text, fill=(20, 20, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_verify(
    *,
    rebuild_pool: bool = False,
    min_per_class: int = 5,
    n_samples: int = 12,
    seed: int = 0,
    paths: Paths | None = None,
) -> dict:
    paths = paths or Paths()
    train_rids, val_rids = resolve_train_val_rids(paths=paths)
    all_rids = train_rids + val_rids
    checks: list[dict] = []

    # --- RID coverage (quill packs + crops) ---
    missing_quill = [r for r in all_rids if not (paths.quill_local / r).is_dir()]
    checks.append(
        _check(
            "quill_packs",
            not missing_quill,
            f"missing quill packs: {missing_quill}" if missing_quill else f"{len(all_rids)}/10 packs present",
        )
    )
    sheet_counts = {}
    bbox_counts = {}
    for rid in all_rids:
        qdir = paths.quill_local / rid
        n_sheets = sum(1 for _ in qdir.glob("*_geometries.json")) if qdir.is_dir() else 0
        sheet_counts[rid] = n_sheets
        n_bbox = 0
        if qdir.is_dir():
            for gp in qdir.glob("*_geometries.json"):
                try:
                    data = json.loads(gp.read_text())
                except Exception:  # noqa: BLE001
                    continue
                for out in data.get("outputs") or []:
                    feat = out.get("feature") or {}
                    if str(feat.get("name") or "").startswith("bbox_"):
                        n_bbox += len((out.get("output_geojson") or {}).get("features") or [])
        bbox_counts[rid] = n_bbox
    checks.append(
        _check(
            "quill_bbox_layers",
            all(bbox_counts[r] > 0 for r in all_rids),
            f"bbox features per rid: { {r[:8]: bbox_counts[r] for r in all_rids} }",
        )
    )

    missing_crops = [
        r for r in all_rids if not (paths.crops_quill_dir(r) / "manifest.jsonl").exists()
    ]
    checks.append(
        _check(
            "quill_crops_exported",
            not missing_crops,
            (
                f"missing crops for: {[m[:8] for m in missing_crops]} — run export"
                if missing_crops
                else "all RIDs have symbol_crops_quill/manifest.jsonl"
            ),
        )
    )

    pool_dir = paths.pooled_quill_dir()
    if missing_crops:
        report = {
            "ok": False,
            "checks": checks,
            "train_rids": train_rids,
            "val_rids": val_rids,
            "pool": str(pool_dir),
        }
        _write_reports(pool_dir, report)
        return report

    if rebuild_pool or not (pool_dir / "splits.json").exists():
        build_pooled_quill_dataset(
            train_rids=train_rids,
            val_rids=val_rids,
            out_dir=pool_dir,
            min_per_class=min_per_class,
            seed=seed,
            paths=paths,
        )

    splits = json.loads((pool_dir / "splits.json").read_text())
    rids_meta = json.loads((pool_dir / "rids.json").read_text())
    rows = load_manifest(pool_dir)

    # --- Split integrity ---
    tr = set(splits.get("train_rids") or rids_meta.get("train_rids") or [])
    vr = set(splits.get("val_rids") or rids_meta.get("val_rids") or [])
    overlap = tr & vr
    checks.append(
        _check(
            "rid_overlap",
            not overlap,
            f"overlap={sorted(overlap)}" if overlap else "train∩val = ∅",
        )
    )
    checks.append(
        _check(
            "val_rids_locked",
            vr == set(val_rids),
            f"val={sorted(x[:8] for x in vr)} expected={sorted(x[:8] for x in val_rids)}",
        )
    )
    checks.append(
        _check(
            "split_mode",
            splits.get("split_mode") == "rid_holdout",
            f"split_mode={splits.get('split_mode')!r}",
        )
    )

    # --- Crop counts ---
    per_rid = Counter((r.get("rid") or "") for r in rows)
    checks.append(
        _check(
            "crop_counts",
            splits["n_train"] > 0 and splits["n_val"] > 0,
            f"train={splits['n_train']} val={splits['n_val']} per_rid={ {k[:8]: v for k, v in per_rid.items()} }",
        )
    )

    # --- Class stats ---
    train_paths = set(splits["train"])
    train_names = [
        r["name"]
        for r in rows
        if str(Path(r["path"])) in train_paths or r["path"] in train_paths
    ]
    # paths in splits are absolute; match both
    path_to_name = {str(Path(r["path"])): r["name"] for r in rows}
    train_names = [path_to_name[p] for p in splits["train"] if p in path_to_name]
    cc = Counter(train_names)
    if cc:
        vals = list(cc.values())
        class_detail = (
            f"n_classes={len(cc)} min={min(vals)} median={int(np.median(vals))} "
            f"max={max(vals)} dropped_val_classes={len(splits.get('dropped_val_classes') or [])}"
        )
    else:
        class_detail = "no train classes"
    checks.append(_check("class_stats", len(cc) >= 2, class_detail))

    # --- Label sanity ---
    sample_labels = sorted(cc.keys())[:8]
    checks.append(
        _check(
            "label_sanity",
            all(isinstance(n, str) and n and not n.startswith("bbox_") for n in sample_labels),
            f"sample classes: {sample_labels}",
        )
    )

    # --- Leakage ---
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    path_overlap = train_set & val_set
    train_wids = {
        r.get("wid")
        for r in rows
        if str(Path(r["path"])) in train_set or r["path"] in train_set
    }
    # rebuild via rid membership
    train_wids = {r.get("wid") for r in rows if (r.get("rid") or "") in tr}
    val_wids = {r.get("wid") for r in rows if (r.get("rid") or "") in vr}
    wid_overlap = train_wids & val_wids
    checks.append(
        _check(
            "path_leakage",
            not path_overlap,
            f"shared paths={len(path_overlap)}",
        )
    )
    checks.append(
        _check(
            "wid_leakage",
            not wid_overlap,
            f"shared wids={len(wid_overlap)}" if wid_overlap else "no shared WIDs",
        )
    )

    # --- Zoom sanity ---
    zooms = {float(r.get("zoom") or 0) for r in rows}
    bad_zoom = [z for z in zooms if abs(z - DEFAULT_ZOOM) > 1e-6]
    sample_sides = []
    for r in rows[:50]:
        box = r.get("box") or {}
        if "w" in box and "h" in box:
            sample_sides.append((box["w"], box["h"]))
    checks.append(
        _check(
            "zoom_sanity",
            not bad_zoom and (DEFAULT_ZOOM in zooms or all(abs(z - DEFAULT_ZOOM) < 1e-6 for z in zooms)),
            f"zooms={sorted(zooms)} sample_box_wh={sample_sides[:5]}",
        )
    )

    # --- Sample grids (clean vs noised) ---
    rng = random.Random(seed)
    samples_dir = pool_dir / "verify_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    speck = GreySpeckNoise(p=1.0)
    strike = StrikethroughNoise(p=1.0)

    def _sample_split(split_name: str, n: int) -> list[tuple[Image.Image, Image.Image, str]]:
        paths_list = list(splits[split_name])
        rng.shuffle(paths_list)
        out = []
        for p in paths_list:
            if len(out) >= n:
                break
            pp = Path(p)
            if not pp.is_file():
                continue
            try:
                clean = Image.open(pp).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            import torch
            from torchvision import transforms as T

            t = T.Compose([T.Resize((224, 224)), T.ToTensor()])(clean)
            noised = strike(speck(t.clone()))
            name = path_to_name.get(str(pp), path_to_name.get(p, "?"))
            out.append((_denorm_to_pil(t), _denorm_to_pil(noised), f"{split_name}:{name}"))
        return out

    train_pairs = _sample_split("train", n_samples)
    val_pairs = _sample_split("val", max(4, n_samples // 2))
    _make_grid(train_pairs, samples_dir / "train_clean_vs_noise.png")
    _make_grid(val_pairs, samples_dir / "val_clean_vs_noise.png")
    # val grid still shows what noise *would* look like; training won't apply it to val
    checks.append(
        _check(
            "sample_grids",
            (samples_dir / "train_clean_vs_noise.png").is_file(),
            f"wrote {samples_dir}",
        )
    )

    ok = all(c["ok"] for c in checks)
    report = {
        "ok": ok,
        "checks": checks,
        "train_rids": train_rids,
        "val_rids": val_rids,
        "n_train": splits["n_train"],
        "n_val": splits["n_val"],
        "n_classes": splits["n_classes"],
        "sheet_counts": sheet_counts,
        "bbox_counts": bbox_counts,
        "pool": str(pool_dir),
        "samples_dir": str(samples_dir),
    }
    _write_reports(pool_dir, report)
    return report


def _write_reports(pool_dir: Path, report: dict) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / "verify_report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Classifier dataset verify report",
        "",
        f"**Ready:** {'YES' if report.get('ok') else 'NO'}",
        "",
        f"- pool: `{report.get('pool')}`",
        f"- train crops: {report.get('n_train', '?')}",
        f"- val crops: {report.get('n_val', '?')}",
        f"- classes: {report.get('n_classes', '?')}",
        "",
        "## Checks",
        "",
    ]
    for c in report.get("checks", []):
        mark = "PASS" if c.get("ok") else "FAIL"
        lines.append(f"- **{mark}** `{c['name']}` — {c.get('detail', '')}")
    lines += [
        "",
        "## RID split",
        "",
        "### Train",
    ]
    for r in report.get("train_rids", []):
        lines.append(f"- `{r}`")
    lines.append("")
    lines.append("### Val")
    for r in report.get("val_rids", []):
        lines.append(f"- `{r}`")
    lines += [
        "",
        "## Sample grids",
        "",
        f"See `{report.get('samples_dir', 'verify_samples')}/` "
        "(left=clean, right=speck+strikethrough).",
        "",
    ]
    (pool_dir / "verify_report.md").write_text("\n".join(lines))
    print((pool_dir / "verify_report.md").read_text())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    report = run_verify(
        rebuild_pool=args.rebuild_pool,
        min_per_class=args.min_per_class,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()

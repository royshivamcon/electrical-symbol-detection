"""Per-RID evaluation of a trained embedding checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from symbol_embed.config import Paths
from symbol_embed.dataset import (
    SymbolCropDataset,
    eval_transform,
    load_or_make_splits,
    make_splits,
)
from symbol_embed.train import extract_embeddings, load_model_from_ckpt, _knn_recall


def eval_rid(
    rid: str,
    ckpt_path: Path,
    *,
    split: str = "all",
    batch: int = 64,
    device: str | None = None,
    paths: Paths | None = None,
) -> dict:
    """Compute knn recall on a single RID's crops (requires --rid)."""
    paths = paths or Paths()
    crops_dir = paths.crops_dir(rid)
    if not (crops_dir / "manifest.jsonl").exists():
        raise FileNotFoundError(f"No crops for rid={rid} at {crops_dir}")

    splits = load_or_make_splits(crops_dir) if (crops_dir / "splits.json").exists() else make_splits(crops_dir)
    if split == "all":
        merged = dict(splits)
        merged["all"] = list(splits["train"]) + list(splits["val"])
        splits = merged
        use = "all"
    else:
        use = split

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_ckpt(ckpt_path, device=dev)
    ds = SymbolCropDataset(crops_dir, use, transform=eval_transform(), splits=splits)
    if len(ds) == 0:
        raise RuntimeError(f"empty dataset for rid={rid} split={use}")
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    emb, lab, img_paths = extract_embeddings(model, loader, dev)
    knn = _knn_recall(emb, lab)

    out = {
        "rid": rid,
        "ckpt": str(ckpt_path),
        "arm": model.arm,
        "split": use,
        "n": int(len(lab)),
        "n_classes_local": int(len(set(lab.tolist()))),
        "val_knn": knn,
    }
    out_dir = paths.eval_dir(rid)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(ckpt_path).parent.name
    (out_dir / f"{model.arm}_{stem}_metrics.json").write_text(json.dumps(out, indent=2))
    np.savez(out_dir / f"{model.arm}_{stem}_emb.npz", emb=emb, labels=lab, paths=np.array(img_paths))
    print(f"[eval] rid={rid[:8]} arm={model.arm} n={out['n']} knn={knn:.4f} → {out_dir}")
    return out


def _find_latest_ckpts(runs_root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for arm in ("pretrained", "contrastive", "arcface"):
        arm_dir = runs_root / arm
        if not arm_dir.exists():
            continue
        cands = sorted(arm_dir.glob("*/best.pt"), key=lambda p: p.stat().st_mtime)
        if cands:
            found[arm] = cands[-1]
    return found


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rid", required=True, help="Single request id to evaluate on")
    ap.add_argument("--split", default="all", choices=("train", "val", "all"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="arm=/path/to/best.pt (repeatable). Default: latest under runs/",
    )
    args = ap.parse_args(argv)

    paths = Paths()
    if args.ckpt:
        ckpt_map = {}
        for item in args.ckpt:
            if "=" not in item:
                raise SystemExit(f"--ckpt needs arm=path, got {item!r}")
            arm, path = item.split("=", 1)
            ckpt_map[arm] = Path(path)
    else:
        ckpt_map = _find_latest_ckpts(paths.runs_dir())
    if not ckpt_map:
        raise SystemExit(f"No checkpoints under {paths.runs_dir()}")

    results = []
    for arm, ckpt in sorted(ckpt_map.items()):
        results.append(
            eval_rid(
                args.rid,
                ckpt,
                split=args.split,
                batch=args.batch,
                device=args.device,
                paths=paths,
            )
        )
    summary = paths.eval_dir(args.rid) / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    print(f"[eval] summary → {summary}")


if __name__ == "__main__":
    main()

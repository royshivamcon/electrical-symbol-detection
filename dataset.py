"""Crop dataset + stratified / RID-holdout splits (single RID or pooled)."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from symbol_embed.config import VAL_RIDS, Paths


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CropRow:
    path: Path
    name: str
    label: int


def load_manifest(crops_dir: Path) -> list[dict]:
    man = crops_dir / "manifest.jsonl"
    if not man.exists():
        raise FileNotFoundError(f"missing {man}; run export_crops first")
    rows = []
    with open(man) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_row_path(crops_dir: Path, row: dict) -> Path:
    p = Path(row["path"])
    return p if p.is_absolute() else (crops_dir / p)


def load_rids_file(path: Path | str) -> list[str]:
    """Load one RID per line; blank lines and ``#`` comments ignored."""
    return [
        ln.strip()
        for ln in Path(path).read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def list_available_rids(paths: Paths | None = None) -> list[str]:
    paths = paths or Paths()
    if not paths.requests.exists():
        return []
    return sorted(
        p.name
        for p in paths.requests.iterdir()
        if p.is_dir() and (p / "worksheets_metadata.json").exists()
    )


def rids_with_crops(paths: Paths | None = None) -> list[str]:
    paths = paths or Paths()
    return [
        rid
        for rid in list_available_rids(paths)
        if (paths.crops_dir(rid) / "manifest.jsonl").exists()
    ]


def rids_with_quill_crops(paths: Paths | None = None) -> list[str]:
    paths = paths or Paths()
    return [
        rid
        for rid in list_available_rids(paths)
        if (paths.crops_quill_dir(rid) / "manifest.jsonl").exists()
    ]


def resolve_train_val_rids(
    *,
    train_rids: list[str] | None = None,
    val_rids: list[str] | None = None,
    paths: Paths | None = None,
) -> tuple[list[str], list[str]]:
    """Return (train_rids, val_rids) with zero overlap; defaults from config files."""
    paths = paths or Paths()
    if train_rids is None:
        train_rids = load_rids_file(paths.train_rids_file())
    if val_rids is None:
        if paths.val_rids_file().is_file():
            val_rids = load_rids_file(paths.val_rids_file())
        else:
            val_rids = list(VAL_RIDS)
    train_rids = list(train_rids)
    val_rids = list(val_rids)
    overlap = set(train_rids) & set(val_rids)
    if overlap:
        # Drop val RIDs from train list (val wins).
        train_rids = [r for r in train_rids if r not in overlap]
    if not train_rids:
        raise ValueError("empty train_rids after removing val overlap")
    if not val_rids:
        raise ValueError("empty val_rids")
    return train_rids, val_rids


def select_train_rids(
    *,
    rids: list[str] | None = None,
    n_rids: int = 10,
    seed: int = 0,
    paths: Paths | None = None,
    require_crops: bool = True,
) -> list[str]:
    """Pick RIDs for pooled training. Explicit ``rids`` wins; else take ``n_rids``."""
    paths = paths or Paths()
    if rids:
        out = list(rids)
        if require_crops:
            missing = [r for r in out if not (paths.crops_dir(r) / "manifest.jsonl").exists()]
            if missing:
                raise FileNotFoundError(
                    "Missing symbol_crops for: "
                    + ", ".join(m[:8] for m in missing)
                    + " — run export first"
                )
        return out

    pool = rids_with_crops(paths) if require_crops else list_available_rids(paths)
    if not pool:
        raise FileNotFoundError(
            "No RIDs with crops. Export first:\n"
            "  bash symbol_embed/run_export.sh --n-rids 10"
        )
    rng = random.Random(seed)
    if len(pool) <= n_rids:
        return pool
    return sorted(rng.sample(pool, n_rids))


def build_label_maps(
    rows: list[dict], *, min_per_class: int
) -> tuple[dict[str, int], list[dict]]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        name = (r.get("name") or "").strip()
        if name:
            counts[name] += 1
    keep = {n for n, c in counts.items() if c >= min_per_class}
    filtered = [r for r in rows if (r.get("name") or "").strip() in keep]
    names = sorted(keep)
    name_to_label = {n: i for i, n in enumerate(names)}
    return name_to_label, filtered


def pool_rid_manifests(
    rid_list: list[str],
    *,
    paths: Paths | None = None,
    crops_kind: str = "legacy",
) -> list[dict]:
    """Concatenate per-RID manifests with absolute image paths.

    ``crops_kind``: ``\"legacy\"`` → symbol_crops; ``\"quill\"`` → symbol_crops_quill.
    """
    paths = paths or Paths()
    rows: list[dict] = []
    for rid in rid_list:
        crops = (
            paths.crops_quill_dir(rid) if crops_kind == "quill" else paths.crops_dir(rid)
        )
        for r in load_manifest(crops):
            rr = dict(r)
            rr["rid"] = rid
            rr["path"] = str(_resolve_row_path(crops, r).resolve())
            rows.append(rr)
    return rows


def build_pooled_dataset(
    rid_list: list[str],
    *,
    out_dir: Path | None = None,
    min_per_class: int = 5,
    val_frac: float = 0.2,
    seed: int = 0,
    paths: Paths | None = None,
) -> Path:
    """Write pooled manifest + stratified splits under ``symbol_embed/data/pooled_*``."""
    paths = paths or Paths()
    rid_list = list(rid_list)
    if out_dir is None:
        out_dir = paths.pooled_dir(len(rid_list))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pool_rid_manifests(rid_list, paths=paths, crops_kind="legacy")
    man_path = out_dir / "manifest.jsonl"
    with open(man_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    (out_dir / "rids.json").write_text(
        json.dumps({"rids": rid_list, "n_rids": len(rid_list), "n_rows": len(rows)}, indent=2)
    )
    make_splits(
        out_dir,
        min_per_class=min_per_class,
        val_frac=val_frac,
        seed=seed,
        rid_list=rid_list,
    )
    print(f"[pool] {len(rid_list)} rids, {len(rows)} crops → {out_dir}")
    return out_dir


def make_rid_splits(
    crops_dir: Path,
    *,
    train_rids: list[str],
    val_rids: list[str],
    min_per_class: int = 5,
    seed: int = 0,
) -> dict:
    """Strict RID holdout: classes filtered on train only; no RID overlap."""
    crops_dir = Path(crops_dir)
    train_set = set(train_rids)
    val_set = set(val_rids)
    if train_set & val_set:
        raise ValueError(f"train/val RID overlap: {sorted(train_set & val_set)}")

    rows = load_manifest(crops_dir)
    train_rows = [r for r in rows if (r.get("rid") or "") in train_set]
    val_rows_all = [r for r in rows if (r.get("rid") or "") in val_set]

    name_to_label, train_filtered = build_label_maps(train_rows, min_per_class=min_per_class)
    keep = set(name_to_label)
    val_filtered = [
        r for r in val_rows_all if (r.get("name") or "").strip() in keep
    ]
    dropped_val_classes = sorted(
        {
            (r.get("name") or "").strip()
            for r in val_rows_all
            if (r.get("name") or "").strip() not in keep
        }
    )

    def _key(r: dict) -> str:
        return str(_resolve_row_path(crops_dir, r))

    train_paths = [_key(r) for r in train_filtered]
    val_paths = [_key(r) for r in val_filtered]

    per_rid_train = Counter((r.get("rid") or "") for r in train_filtered)
    per_rid_val = Counter((r.get("rid") or "") for r in val_filtered)

    split = {
        "split_mode": "rid_holdout",
        "train_rids": list(train_rids),
        "val_rids": list(val_rids),
        "rids": list(train_rids) + list(val_rids),
        "min_per_class": min_per_class,
        "seed": seed,
        "classes": list(name_to_label.keys()),
        "name_to_label": name_to_label,
        "train": train_paths,
        "val": val_paths,
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "n_classes": len(name_to_label),
        "per_rid_train": dict(per_rid_train),
        "per_rid_val": dict(per_rid_val),
        "dropped_val_classes": dropped_val_classes,
        "n_val_dropped": len(val_rows_all) - len(val_filtered),
    }
    (crops_dir / "splits.json").write_text(json.dumps(split, indent=2))

    stats = {
        "n_train": split["n_train"],
        "n_val": split["n_val"],
        "n_classes": split["n_classes"],
        "min_per_class": min_per_class,
        "train_rids": list(train_rids),
        "val_rids": list(val_rids),
        "per_rid_train": dict(per_rid_train),
        "per_rid_val": dict(per_rid_val),
        "dropped_val_classes": dropped_val_classes,
        "class_counts_train": dict(
            Counter((r.get("name") or "").strip() for r in train_filtered)
        ),
    }
    (crops_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (crops_dir / "classes.json").write_text(
        json.dumps(
            {
                "classes": list(name_to_label.keys()),
                "name_to_label": name_to_label,
                "n_classes": len(name_to_label),
            },
            indent=2,
        )
    )
    return split


def build_pooled_quill_dataset(
    *,
    train_rids: list[str] | None = None,
    val_rids: list[str] | None = None,
    out_dir: Path | None = None,
    min_per_class: int = 5,
    seed: int = 0,
    paths: Paths | None = None,
) -> Path:
    """Pool quill crops and write RID-holdout splits to ``pooled_10rids_quill``."""
    paths = paths or Paths()
    train_rids, val_rids = resolve_train_val_rids(
        train_rids=train_rids, val_rids=val_rids, paths=paths
    )
    all_rids = list(train_rids) + list(val_rids)
    missing = [
        r for r in all_rids if not (paths.crops_quill_dir(r) / "manifest.jsonl").exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing symbol_crops_quill for: "
            + ", ".join(m[:8] for m in missing)
            + " — run: bash symbol_embed/run_export_quill.sh"
        )

    out_dir = Path(out_dir or paths.pooled_quill_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pool_rid_manifests(all_rids, paths=paths, crops_kind="quill")
    with open(out_dir / "manifest.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    (out_dir / "rids.json").write_text(
        json.dumps(
            {
                "train_rids": train_rids,
                "val_rids": val_rids,
                "rids": all_rids,
                "n_train_rids": len(train_rids),
                "n_val_rids": len(val_rids),
                "n_rows": len(rows),
                "source": "quill_local",
                "zoom": 4,
            },
            indent=2,
        )
    )
    make_rid_splits(
        out_dir,
        train_rids=train_rids,
        val_rids=val_rids,
        min_per_class=min_per_class,
        seed=seed,
    )
    print(
        f"[pool_quill] train_rids={len(train_rids)} val_rids={len(val_rids)} "
        f"crops={len(rows)} → {out_dir}"
    )
    return out_dir


def make_splits(
    crops_dir: Path,
    *,
    min_per_class: int = 5,
    val_frac: float = 0.2,
    seed: int = 0,
    rid_list: list[str] | None = None,
) -> dict:
    rows = load_manifest(crops_dir)
    name_to_label, filtered = build_label_maps(rows, min_per_class=min_per_class)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in filtered:
        by_name[r["name"]].append(r)

    rng = random.Random(seed)
    train, val = [], []
    for _name, items in by_name.items():
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac))) if len(items) > 1 else 0
        if len(items) - n_val < 1:
            n_val = 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    # path keys: absolute for pooled, relative for single-rid layouts
    def _key(r: dict) -> str:
        return str(_resolve_row_path(crops_dir, r))

    single_rid = ""
    if crops_dir.name == "symbol_crops":
        single_rid = crops_dir.parent.name
    split = {
        "rid": single_rid,
        "rids": rid_list or ([single_rid] if single_rid else []),
        "min_per_class": min_per_class,
        "val_frac": val_frac,
        "seed": seed,
        "classes": list(name_to_label.keys()),
        "name_to_label": name_to_label,
        "train": [_key(r) for r in train],
        "val": [_key(r) for r in val],
        "n_train": len(train),
        "n_val": len(val),
        "n_classes": len(name_to_label),
    }
    (crops_dir / "splits.json").write_text(json.dumps(split, indent=2))
    return split


def load_or_make_splits(crops_dir: Path, **kwargs) -> dict:
    crops_dir = Path(crops_dir)
    path = crops_dir / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    rids_meta_path = crops_dir / "rids.json"
    if rids_meta_path.is_file():
        meta = json.loads(rids_meta_path.read_text())
        if meta.get("train_rids") and meta.get("val_rids"):
            return make_rid_splits(
                crops_dir,
                train_rids=list(meta["train_rids"]),
                val_rids=list(meta["val_rids"]),
                min_per_class=int(kwargs.get("min_per_class", 5)),
                seed=int(kwargs.get("seed", 0)),
            )
    return make_splits(crops_dir, **kwargs)


def train_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SymbolCropDataset(Dataset):
    """Single-view crop dataset keyed by splits.json paths."""

    def __init__(
        self,
        crops_dir: Path,
        split: str,
        *,
        transform=None,
        splits: dict | None = None,
        name_to_label: dict[str, int] | None = None,
    ) -> None:
        self.crops_dir = Path(crops_dir)
        self.splits = splits or load_or_make_splits(self.crops_dir)
        self.name_to_label = name_to_label or self.splits["name_to_label"]
        self.label_to_name = {int(v): k for k, v in self.name_to_label.items()}
        if self.name_to_label and isinstance(next(iter(self.name_to_label.values())), str):
            self.name_to_label = {k: int(v) for k, v in self.name_to_label.items()}
            self.label_to_name = {int(v): k for k, v in self.name_to_label.items()}

        path_set = {str(Path(p)) for p in self.splits[split]}
        rows = load_manifest(self.crops_dir)
        self.items: list[CropRow] = []
        for r in rows:
            abs_p = _resolve_row_path(self.crops_dir, r)
            if str(abs_p) not in path_set and r["path"] not in path_set:
                continue
            name = r["name"]
            if name not in self.name_to_label:
                continue
            self.items.append(
                CropRow(
                    path=abs_p,
                    name=name,
                    label=int(self.name_to_label[name]),
                )
            )
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        img = Image.open(row.path).convert("RGB")
        x = self.transform(img)
        return x, row.label, str(row.path)


class PositivePairDataset(Dataset):
    """Returns (anchor, positive, label) with same-class positive sampling."""

    def __init__(
        self,
        crops_dir: Path,
        *,
        transform=None,
        splits: dict | None = None,
        seed: int = 0,
        name_to_label: dict[str, int] | None = None,
    ) -> None:
        self.base = SymbolCropDataset(
            crops_dir,
            "train",
            transform=transform,
            splits=splits,
            name_to_label=name_to_label,
        )
        self.rng = random.Random(seed)
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(self.base.items):
            self.by_label[row.label].append(i)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_a, y, _path = self.base[idx]
        peers = self.by_label[y]
        if len(peers) > 1:
            j = idx
            while j == idx:
                j = peers[self.rng.randrange(len(peers))]
            x_p, _, _ = self.base[j]
        else:
            row = self.base.items[idx]
            img = Image.open(row.path).convert("RGB")
            x_p = self.base.transform(img)
        return x_a, x_p, y

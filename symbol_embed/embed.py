"""Extract embeddings for a checkpoint over crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from symbol_embed.dataset import SymbolCropDataset, eval_transform, load_or_make_splits
from symbol_embed.train import extract_embeddings, load_model_from_ckpt


def embed_crops(
    ckpt_path: Path,
    crops_dir: Path,
    *,
    split: str = "val",
    batch: int = 64,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Return (embeddings, labels, paths, meta) for ``split`` (or ``all``)."""
    crops_dir = Path(crops_dir)
    splits = load_or_make_splits(crops_dir)
    if split == "all":
        # merge train+val path lists into a synthetic split
        merged = dict(splits)
        merged["all"] = list(splits["train"]) + list(splits["val"])
        splits = merged
        use_split = "all"
    else:
        use_split = split

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_from_ckpt(ckpt_path, device=dev)
    ds = SymbolCropDataset(
        crops_dir, use_split, transform=eval_transform(), splits=splits
    )
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    emb, lab, paths = extract_embeddings(model, loader, dev)
    meta = {
        "arm": model.arm,
        "n_classes": len(splits["name_to_label"]),
        "label_to_name": {int(v): k for k, v in splits["name_to_label"].items()},
        "name_to_label": splits["name_to_label"],
    }
    return emb, lab, paths, meta

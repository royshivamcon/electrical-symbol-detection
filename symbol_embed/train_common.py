"""Shared val metrics, W&B, and AMP helpers for classifier training loops."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

from symbol_embed.config import WANDB_PROJECT

# Matcher-style ks used in earlier evaluate runs.
VAL_KS: tuple[int, ...] = (1, 2, 5, 50)


def gallery_topk_metrics(
    cost: np.ndarray,
    true_labels: list[int],
    gallery_labels: list[int],
    ks: tuple[int, ...] = VAL_KS,
) -> dict[int, dict[str, float]]:
    true = np.asarray(true_labels)
    gal = np.asarray(gallery_labels)
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        kk = min(int(k), cost.shape[1])
        if kk <= 0 or len(true) == 0:
            out[int(k)] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
            continue
        ranked = np.argsort(cost, axis=1)[:, :kk]
        accs, precs, recalls = [], [], []
        for i in range(len(true)):
            n_rel = int((gal == true[i]).sum())
            if n_rel <= 0:
                continue
            top = gal[ranked[i]]
            hits = int((top == true[i]).sum())
            accs.append(float(true[i] in top))
            precs.append(hits / kk)
            recalls.append(hits / n_rel)
        out[int(k)] = {
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
        }
    return out


def crop_topk_metrics(
    emb: np.ndarray,
    labels: np.ndarray,
    ks: tuple[int, ...] = VAL_KS,
) -> dict[int, dict[str, float]]:
    """Leave-one-out crop↔crop Acc/P/R @k (matcher ``topk_recall_precision``)."""
    n = len(labels)
    empty = {int(k): {"accuracy": 0.0, "precision": 0.0, "recall": 0.0} for k in ks}
    if n <= 1:
        return empty
    labels = np.asarray(labels)
    sim = emb @ emb.T
    dist = (1.0 - sim).astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)
    same_counts = np.array([(labels == labels[i]).sum() - 1 for i in range(n)])
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        accs, precs, recalls = [], [], []
        kk = min(int(k), n - 1)
        for i in range(n):
            n_same = int(same_counts[i])
            if n_same <= 0:
                continue
            top = order[i, :kk]
            hits = int((labels[top] == labels[i]).sum())
            accs.append(1.0 if hits > 0 else 0.0)
            precs.append(hits / kk)
            recalls.append(hits / n_same)
        out[int(k)] = {
            "accuracy": float(np.mean(accs)) if accs else 0.0,
            "precision": float(np.mean(precs)) if precs else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
        }
    return out


def class_prototypes(
    emb: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, list[int]]:
    protos, proto_labels = [], []
    for c in range(n_classes):
        mask = labels == c
        if not mask.any():
            continue
        m = emb[mask].mean(axis=0)
        n = np.linalg.norm(m) + 1e-8
        protos.append(m / n)
        proto_labels.append(c)
    if not protos:
        return np.zeros((0, emb.shape[1] if emb.ndim == 2 else 0), np.float32), []
    return np.stack(protos, 0).astype(np.float32), proto_labels


def prototype_ari(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    proto_emb: np.ndarray,
    proto_labels: list[int],
) -> float:
    """ARI between GT labels and nearest-prototype assignments."""
    if len(query_emb) == 0 or len(proto_emb) == 0:
        return 0.0
    cost = 1.0 - query_emb @ proto_emb.T
    pred = [proto_labels[int(i)] for i in np.argmin(cost, axis=1)]
    return float(adjusted_rand_score(query_labels.tolist(), pred))


def similarity_metrics(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    proto_emb: np.ndarray,
    proto_labels: list[int],
    ks: tuple[int, ...] = VAL_KS,
) -> dict[int, dict[str, float]]:
    empty = {int(k): {"accuracy": 0.0, "precision": 0.0, "recall": 0.0} for k in ks}
    if len(query_emb) == 0 or len(proto_emb) == 0:
        return empty
    cost = (1.0 - query_emb @ proto_emb.T).astype(np.float32)
    return gallery_topk_metrics(
        cost, query_labels.tolist(), proto_labels, ks=ks
    )


def evaluate_val(
    query_emb: np.ndarray,
    query_labels: np.ndarray,
    proto_emb: np.ndarray,
    proto_labels: list[int],
    *,
    ks: tuple[int, ...] = VAL_KS,
) -> dict[str, Any]:
    """Prototype Acc@k + ARI (template-style) + crop↔crop Acc@k."""
    proto = similarity_metrics(query_emb, query_labels, proto_emb, proto_labels, ks=ks)
    crop = crop_topk_metrics(query_emb, query_labels, ks=ks)
    ari = prototype_ari(query_emb, query_labels, proto_emb, proto_labels)
    return {
        "ari": ari,
        "proto": proto,
        "crop": crop,
        "acc1": float(proto.get(1, {}).get("accuracy", 0.0)),
    }


def format_metrics(ev: dict[str, Any], ks: tuple[int, ...] = VAL_KS) -> str:
    parts = [f"ARI={ev.get('ari', 0):.4f}"]
    proto = ev.get("proto") or {}
    for k in ks:
        t = proto.get(k, {})
        parts.append(f"proto@{k} Acc={t.get('accuracy', 0):.3f}")
    crop = ev.get("crop") or {}
    for k in ks:
        t = crop.get(k, {})
        parts.append(f"crop@{k} Acc={t.get('accuracy', 0):.3f}")
    return " | ".join(parts)


def metrics_to_wandb(ev: dict[str, Any], *, loss: float | None = None, epoch: int | None = None) -> dict[str, float]:
    """Flatten evaluate_val result into wandb scalar keys."""
    row: dict[str, float] = {}
    if loss is not None:
        row["train/loss"] = float(loss)
    if epoch is not None:
        row["epoch"] = float(epoch)
    row["val/ari"] = float(ev.get("ari", 0.0))
    for kind in ("proto", "crop"):
        block = ev.get(kind) or {}
        for k, t in block.items():
            row[f"val/{kind}_acc@{k}"] = float(t.get("accuracy", 0.0))
            row[f"val/{kind}_P@{k}"] = float(t.get("precision", 0.0))
            row[f"val/{kind}_R@{k}"] = float(t.get("recall", 0.0))
    # Aliases matching earlier matcher naming
    proto = ev.get("proto") or {}
    crop = ev.get("crop") or {}
    for k in VAL_KS:
        row[f"val/acc@{k}"] = float(proto.get(k, {}).get("accuracy", 0.0))
        row[f"val/topk_acc@{k}"] = float(crop.get(k, {}).get("accuracy", 0.0))
    return row


def setup_wandb(
    *,
    name: str,
    cfg: Any,
    extra: dict | None = None,
    mode: str = "online",
    init_timeout: int = 180,
) -> Any | None:
    mode = (mode or "online").lower()
    if mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb: disabled")
        return None

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_MODE"] = mode
    os.environ["WANDB_INIT_TIMEOUT"] = str(init_timeout)

    import wandb

    config: dict[str, Any] = {}
    if is_dataclass(cfg):
        config.update(asdict(cfg))
    elif isinstance(cfg, dict):
        config.update(cfg)
    if extra:
        config.update(extra)
    run = wandb.init(
        project=WANDB_PROJECT,
        name=name,
        config=config,
        reinit=True,
        settings=wandb.Settings(init_timeout=init_timeout),
    )
    print(f"wandb: {mode} → {WANDB_PROJECT}/{name}")
    return run


def wandb_log(row: dict[str, float], step: int | None = None) -> None:
    import wandb

    if wandb.run is None:
        return
    wandb.log(row, step=step)


def wandb_finish() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()


def amp_setup(amp: bool, device: torch.device):
    enabled = bool(amp) and device.type == "cuda"
    if not enabled:
        return False, torch.float32, torch.amp.GradScaler("cuda", enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    return True, dtype, scaler


@torch.inference_mode()
def extract_embeddings(
    embed_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    proj_dim: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    embs, labels = [], []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            z = embed_fn(x)
        embs.append(z.float().cpu().numpy())
        labels.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
    if not embs:
        return np.zeros((0, proj_dim), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(embs, 0), np.concatenate(labels, 0)


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_cuda_fast_path() -> None:
    """Fixed 224×224 crops — turn on cudnn autotune."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def loader_kwargs(
    device: torch.device,
    num_workers: int,
    *,
    shuffle: bool = False,
    drop_last: bool = False,
) -> dict:
    """Shared DataLoader knobs for high GPU feed rate."""
    kw: dict = {
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return kw


def subsample_for_prototypes(
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    max_per_class: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap samples/class before prototype mean — big speedup on 50k+ train sets."""
    if max_per_class <= 0 or len(labels) == 0:
        return emb, labels
    rng = np.random.RandomState(seed)
    keep: list[int] = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.extend(idx.tolist())
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    return emb[keep_arr], labels[keep_arr]

"""Shared training loop for full DINOv2 / ResNet50 on quill RID-holdout pool."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from symbol_embed.augment import build_eval_transform, build_train_transform
from symbol_embed.config import Paths
from symbol_embed.dataset import SymbolCropDataset, load_or_make_splits
from symbol_embed.losses import ArcFaceLoss
from symbol_embed.supcon import SupConLoss
from symbol_embed.train_common import (
    amp_setup,
    class_prototypes,
    enable_cuda_fast_path,
    evaluate_val,
    extract_embeddings,
    format_metrics,
    loader_kwargs,
    metrics_to_wandb,
    resolve_device,
    setup_wandb,
    subsample_for_prototypes,
    wandb_finish,
    wandb_log,
)


@dataclass
class BackboneTrainCfg:
    epochs: int = 20
    batch: int = 256
    embed_batch: int = 512
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 0
    num_workers: int = 16
    img_size: int = 224
    min_per_class: int = 5
    val_frac: float = 0.2
    proj_dim: int = 256
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    supcon_temperature: float = 0.07
    amp: bool = True
    device: str | None = None
    proto_max_per_class: int = 64
    wandb_mode: str = "online"
    wandb_init_timeout: int = 180


class TwoViewDataset(Dataset):
    def __init__(self, base: SymbolCropDataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = self.base.items[idx]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), self.transform(img), row.label


ModelFactory = Callable[..., torch.nn.Module]


def train_backbone_arm(
    arm: str,
    pool_dir: Path,
    cfg: BackboneTrainCfg,
    *,
    model_factory: ModelFactory,
    run_family: str,
    paths: Paths | None = None,
) -> Path:
    if arm not in ("supcon", "arcface"):
        raise ValueError(f"arm must be supcon|arcface, got {arm!r}")

    paths = paths or Paths()
    pool_dir = Path(pool_dir)
    splits = load_or_make_splits(
        pool_dir,
        min_per_class=cfg.min_per_class,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )
    n_classes = int(splits["n_classes"])
    device = resolve_device(cfg.device)
    enable_cuda_fast_path()
    torch.manual_seed(cfg.seed)

    run_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = paths.runs_dir() / f"{run_family}_{arm}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp, amp_dtype, scaler = amp_setup(cfg.amp, device)
    embed_batch = int(cfg.embed_batch or max(cfg.batch * 2, 512))
    setup_wandb(
        name=f"{run_family}_{arm}_{run_name}",
        cfg=cfg,
        extra={
            "arm": arm,
            "run_family": run_family,
            "pool": str(pool_dir),
            "n_classes": n_classes,
            "n_train": splits["n_train"],
            "n_val": splits["n_val"],
            "train_rids": splits.get("train_rids"),
            "val_rids": splits.get("val_rids"),
        },
        mode=cfg.wandb_mode,
        init_timeout=cfg.wandb_init_timeout,
    )
    print(
        f"[{run_family}/{arm}] classes={n_classes} "
        f"train={splits['n_train']} val={splits['n_val']} "
        f"batch={cfg.batch} embed_batch={embed_batch} "
        f"device={device} amp={'off' if not use_amp else amp_dtype} → {out_dir}"
    )
    if splits.get("train_rids"):
        print(
            f"  train_rids={len(splits['train_rids'])} "
            f"val_rids={len(splits.get('val_rids') or [])}"
        )

    tf_train = build_train_transform(cfg.img_size)
    tf_eval = build_eval_transform(cfg.img_size)
    train_single = SymbolCropDataset(pool_dir, "train", transform=tf_train, splits=splits)
    val_ds = SymbolCropDataset(pool_dir, "val", transform=tf_eval, splits=splits)
    train_eval_ds = SymbolCropDataset(pool_dir, "train", transform=tf_eval, splits=splits)
    train_ds: Dataset = (
        TwoViewDataset(train_single, tf_train) if arm == "supcon" else train_single
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        **loader_kwargs(device, cfg.num_workers, shuffle=True, drop_last=arm == "supcon"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=embed_batch,
        **loader_kwargs(device, cfg.num_workers),
    )
    train_proto_loader = DataLoader(
        train_eval_ds,
        batch_size=embed_batch,
        **loader_kwargs(device, cfg.num_workers),
    )

    model = model_factory(
        n_classes=n_classes,
        with_arcface=(arm == "arcface"),
        arcface_s=cfg.arcface_s,
        arcface_m=cfg.arcface_m,
        proj_dim=cfg.proj_dim,
    ).to(device)

    optim = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head),
        weight_decay=cfg.weight_decay,
    )
    criterion_arc = ArcFaceLoss()
    criterion_sc = SupConLoss(temperature=cfg.supcon_temperature)

    best_acc1 = -1.0
    best_ari = -1.0
    best_path = out_dir / "best.pt"
    history = []

    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            losses = []
            t0 = time.time()
            for batch in train_loader:
                optim.zero_grad(set_to_none=True)
                if arm == "supcon":
                    x1, x2, y = batch
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        z1 = model.embed(x1)
                        z2 = model.embed(x2)
                        loss = criterion_sc(
                            torch.cat([z1, z2], 0), torch.cat([y, y], 0)
                        )
                else:
                    x, y, _ = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        logits, _emb = model(x, y)
                        loss = criterion_arc(logits, y)

                if not torch.isfinite(loss):
                    print(f"[{run_family}/{arm}] skip non-finite loss")
                    continue
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optim.step()
                losses.append(float(loss.item()))

            train_emb, train_y = extract_embeddings(
                model.embed,
                train_proto_loader,
                device,
                amp=use_amp,
                amp_dtype=amp_dtype,
                proj_dim=cfg.proj_dim,
            )
            train_emb, train_y = subsample_for_prototypes(
                train_emb,
                train_y,
                max_per_class=cfg.proto_max_per_class,
                seed=cfg.seed + epoch,
            )
            val_emb, val_y = extract_embeddings(
                model.embed,
                val_loader,
                device,
                amp=use_amp,
                amp_dtype=amp_dtype,
                proj_dim=cfg.proj_dim,
            )
            proto_emb, proto_labels = class_prototypes(train_emb, train_y, n_classes)
            ev = evaluate_val(val_emb, val_y, proto_emb, proto_labels)
            mean_loss = float(np.mean(losses)) if losses else 0.0
            acc1 = float(ev["acc1"])
            ari = float(ev["ari"])
            history.append(
                {
                    "epoch": epoch,
                    "loss": mean_loss,
                    "ari": ari,
                    "metrics": {
                        "proto": {str(k): v for k, v in (ev.get("proto") or {}).items()},
                        "crop": {str(k): v for k, v in (ev.get("crop") or {}).items()},
                    },
                }
            )
            print(
                f"[{run_family}/{arm}] epoch {epoch}/{cfg.epochs} "
                f"loss={mean_loss:.4f} {format_metrics(ev)} "
                f"({time.time() - t0:.1f}s)"
            )
            wandb_log(metrics_to_wandb(ev, loss=mean_loss, epoch=epoch), step=epoch)

            if ari > best_ari:
                best_ari = ari
                best_acc1 = acc1
                torch.save(
                    {
                        "arm": arm,
                        "run_family": run_family,
                        "model": model.state_dict(),
                        "n_classes": n_classes,
                        "name_to_label": splits["name_to_label"],
                        "label_to_name": {
                            int(v): k for k, v in splits["name_to_label"].items()
                        },
                        "cfg": asdict(cfg),
                        "metrics": history[-1]["metrics"],
                        "ari": ari,
                        "acc1": acc1,
                        "epoch": epoch,
                        "pool": str(pool_dir),
                        "train_rids": splits.get("train_rids"),
                        "val_rids": splits.get("val_rids"),
                    },
                    best_path,
                )
                print(f"  ↑ best ARI={best_ari:.4f} Acc@1={best_acc1:.4f} → {best_path}")
    finally:
        wandb_finish()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[{run_family}/{arm}] done best ARI={best_ari:.4f} Acc@1={best_acc1:.4f} → {out_dir}")
    return out_dir

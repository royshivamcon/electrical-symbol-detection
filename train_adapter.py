"""Train DINOv2-Reg adapter with ArcFace or SupCon; val = ARI + Acc@1/2/5/50."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from symbol_embed.adapter_model import DinoAdapter, PROJ_DIM
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
class AdapterTrainCfg:
    epochs: int = 20
    batch: int = 768  # L40S ~46GB; SupCon is 2× views — drop to 512 if OOM
    embed_batch: int = 1024  # no-grad embed extraction can be larger
    lr: float = 1e-3
    weight_decay: float = 0.05
    seed: int = 0
    num_workers: int = 16  # match 16-vCPU host
    img_size: int = 224
    min_per_class: int = 5
    val_frac: float = 0.2
    proj_dim: int = PROJ_DIM
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    supcon_temperature: float = 0.07
    amp: bool = True  # bf16/fp16 autocast (faster + room for larger batches on L40S)
    device: str | None = None
    layers: str = "0,11"  # dual-layer patches; use "last" for prenorm baseline
    proto_max_per_class: int = 64  # subsample train embeds before prototype mean
    wandb_mode: str = "online"
    wandb_init_timeout: int = 180


class TwoViewDataset(Dataset):
    """Apply the same train transform twice for SupCon two-view batches."""

    def __init__(self, base: SymbolCropDataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = self.base.items[idx]
        img = Image.open(row.path).convert("RGB")
        return self.transform(img), self.transform(img), row.label


def train_adapter_arm(
    arm: str,
    pool_dir: Path,
    cfg: AdapterTrainCfg,
    *,
    paths: Paths | None = None,
    run_prefix: str | None = None,
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
    if run_prefix:
        folder = f"{run_prefix}_{arm}"
    else:
        layer_tag = "dual" if cfg.layers not in ("last", "prenorm", "") else "last"
        folder = f"adapter_{layer_tag}_{arm}"
    out_dir = paths.runs_dir() / folder / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp, amp_dtype, scaler = amp_setup(cfg.amp, device)
    embed_batch = int(cfg.embed_batch or max(cfg.batch, 1024))

    wb_name = f"{folder}_{run_name}"
    setup_wandb(
        name=wb_name,
        cfg=cfg,
        extra={
            "arm": arm,
            "run_family": folder,
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
        f"[adapter/{arm}] layers={cfg.layers!r} classes={n_classes} "
        f"train={splits['n_train']} val={splits['n_val']} "
        f"batch={cfg.batch} embed_batch={embed_batch} "
        f"device={device} amp={'off' if not use_amp else amp_dtype} → {out_dir}"
    )

    tf_train = build_train_transform(cfg.img_size)
    tf_eval = build_eval_transform(cfg.img_size)

    train_single = SymbolCropDataset(pool_dir, "train", transform=tf_train, splits=splits)
    val_ds = SymbolCropDataset(pool_dir, "val", transform=tf_eval, splits=splits)
    train_eval_ds = SymbolCropDataset(pool_dir, "train", transform=tf_eval, splits=splits)

    if arm == "supcon":
        train_ds: Dataset = TwoViewDataset(train_single, tf_train)
    else:
        train_ds = train_single

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

    model = DinoAdapter(
        n_classes=n_classes,
        with_arcface=(arm == "arcface"),
        arcface_s=cfg.arcface_s,
        arcface_m=cfg.arcface_m,
        layers=cfg.layers,
    ).to(device)

    params = [p for _, p in model.trainable_parameters()]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion_arc = ArcFaceLoss()
    criterion_sc = SupConLoss(temperature=cfg.supcon_temperature)

    best_acc1 = -1.0
    best_ari = -1.0
    best_path = out_dir / "best.pt"
    history = []

    try:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            model.backbone.eval()
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
                            torch.cat([z1, z2], dim=0), torch.cat([y, y], dim=0)
                        )
                else:
                    x, y, _paths = batch
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    with torch.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        logits, _emb = model(x, y)
                        loss = criterion_arc(logits, y)

                if not torch.isfinite(loss):
                    print(f"[adapter/{arm}] skip non-finite loss={loss.item()!r}")
                    continue
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
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
                f"[adapter/{arm}] epoch {epoch}/{cfg.epochs} "
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
                        "layers": cfg.layers,
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
    print(f"[adapter/{arm}] done best ARI={best_ari:.4f} Acc@1={best_acc1:.4f} → {out_dir}")
    return out_dir

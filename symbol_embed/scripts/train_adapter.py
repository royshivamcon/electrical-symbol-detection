#!/usr/bin/env python3
"""CLI: train DINOv2-Reg adapter (ArcFace vs SupCon).

Quill 10-RID RID-holdout pool (preferred):
  bash symbol_embed/run_export_quill.sh
  bash symbol_embed/run_verify_dataset.sh --rebuild-pool
  python -m symbol_embed.scripts.train_adapter --pool-quill --arm both --layers 0,11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.config import ADAPTER_ARMS, Paths
from symbol_embed.dataset import (
    build_pooled_dataset,
    build_pooled_quill_dataset,
    load_rids_file,
    select_train_rids,
)
from symbol_embed.train_adapter import AdapterTrainCfg, train_adapter_arm
from symbol_embed.verify_gate import require_verify_ok


def _resolve_rids(args: argparse.Namespace, paths: Paths) -> list[str]:
    if args.rids:
        explicit = [r.strip() for r in args.rids.split(",") if r.strip()]
        return select_train_rids(
            rids=explicit,
            n_rids=len(explicit),
            seed=args.seed,
            paths=paths,
            require_crops=True,
        )
    rids_path = Path(args.rids_file) if args.rids_file else paths.adapter_rids_file()
    if not rids_path.is_file():
        raise FileNotFoundError(
            f"Missing RID list {rids_path}. Pass --rids or create adapter_rids.txt."
        )
    explicit = load_rids_file(rids_path)
    if args.n_rids is not None:
        explicit = explicit[: args.n_rids]
    return select_train_rids(
        rids=explicit,
        n_rids=len(explicit),
        seed=args.seed,
        paths=paths,
        require_crops=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rids", default=None)
    ap.add_argument("--rids-file", default=None)
    ap.add_argument("--n-rids", type=int, default=None)
    ap.add_argument("--arm", choices=(*ADAPTER_ARMS, "both"), default="both")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--embed-batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--proto-max-per-class", type=int, default=64)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--layers", default="0,11")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rebuild-pool", action="store_true")
    ap.add_argument(
        "--pool-quill",
        action="store_true",
        help="Use pooled_10rids_quill with strict RID holdout (requires verify)",
    )
    ap.add_argument("--pool", default=None, help="Explicit pool directory")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    ap.add_argument("--wandb-init-timeout", type=int, default=180)
    args = ap.parse_args()

    paths = Paths()
    run_prefix = None

    if args.pool:
        pool_dir = Path(args.pool)
        run_prefix = "adapter_quill"
        require_verify_ok(pool_dir, skip=args.skip_verify, paths=paths)
    elif args.pool_quill:
        pool_dir = paths.pooled_quill_dir()
        require_verify_ok(pool_dir, skip=args.skip_verify, paths=paths)
        if args.rebuild_pool or not (pool_dir / "splits.json").exists():
            build_pooled_quill_dataset(
                out_dir=pool_dir,
                min_per_class=args.min_per_class,
                seed=args.seed,
                paths=paths,
            )
        run_prefix = "adapter_quill"
        print(f"[adapter] using quill RID-holdout pool {pool_dir}")
    else:
        rid_list = _resolve_rids(args, paths)
        print(f"[adapter] pooling {len(rid_list)} rids:")
        for r in rid_list:
            print(f"  - {r}")
        pool_dir = paths.pooled_dir(len(rid_list))
        if args.rebuild_pool or not (pool_dir / "manifest.jsonl").exists():
            build_pooled_dataset(
                rid_list,
                out_dir=pool_dir,
                min_per_class=args.min_per_class,
                seed=args.seed,
                paths=paths,
            )
        else:
            print(f"[adapter] reusing pool {pool_dir}")

    cfg = AdapterTrainCfg(
        epochs=args.epochs,
        batch=args.batch,
        embed_batch=args.embed_batch,
        lr=args.lr,
        num_workers=args.num_workers,
        min_per_class=args.min_per_class,
        seed=args.seed,
        amp=args.amp,
        device=args.device,
        layers=args.layers,
        proto_max_per_class=args.proto_max_per_class,
        wandb_mode=args.wandb_mode,
        wandb_init_timeout=args.wandb_init_timeout,
    )
    arms = list(ADAPTER_ARMS) if args.arm == "both" else [args.arm]
    for arm in arms:
        train_adapter_arm(arm, pool_dir, cfg, paths=paths, run_prefix=run_prefix)


if __name__ == "__main__":
    main()

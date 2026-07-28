#!/usr/bin/env python3
"""CLI: train on pooled multi-RID crops (default 10 RIDs).

Eval / t-SNE stay single-RID — see run_eval.sh / run_visualise.sh --rid …
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.config import ARMS, DEFAULT_N_RIDS, Paths, TrainCfg
from symbol_embed.dataset import build_pooled_dataset, select_train_rids
from symbol_embed.train import train_arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rids",
        default=None,
        help="Comma-separated RIDs to pool for training (overrides --n-rids)",
    )
    ap.add_argument(
        "--n-rids",
        type=int,
        default=DEFAULT_N_RIDS,
        help=f"How many RIDs to sample for pooled training (default {DEFAULT_N_RIDS})",
    )
    ap.add_argument(
        "--rid",
        default=None,
        help="Deprecated single-RID train; prefer --n-rids / --rids. Still accepted.",
    )
    ap.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-fresh", action="store_true")
    ap.add_argument("--wandb-init-timeout", type=int, default=180)
    ap.add_argument(
        "--rebuild-pool",
        action="store_true",
        help="Force rebuild of pooled manifest/splits",
    )
    args = ap.parse_args()

    paths = Paths()
    if args.rid and not args.rids:
        rid_list = [args.rid]
    else:
        explicit = [r.strip() for r in args.rids.split(",") if r.strip()] if args.rids else None
        rid_list = select_train_rids(
            rids=explicit,
            n_rids=args.n_rids,
            seed=args.seed,
            paths=paths,
            require_crops=True,
        )

    print(f"[train] pooling {len(rid_list)} rids:")
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
        print(f"[train] reusing pool {pool_dir} (pass --rebuild-pool to refresh)")

    cfg = TrainCfg(
        epochs=args.epochs,
        batch=args.batch,
        min_per_class=args.min_per_class,
        seed=args.seed,
        device=args.device,
        wandb_mode=args.wandb_mode,
        wandb_fresh=args.wandb_fresh,
        wandb_init_timeout=args.wandb_init_timeout,
    )
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        train_arm(arm, pool_dir, cfg, paths=paths)


if __name__ == "__main__":
    main()

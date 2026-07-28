"""Shared verify-before-train gate for quill classifier runs."""

from __future__ import annotations

import json
from pathlib import Path

from symbol_embed.config import Paths


def verify_report_paths(pool_dir: Path | None = None, paths: Paths | None = None) -> dict[str, Path]:
    paths = paths or Paths()
    pool_dir = Path(pool_dir or paths.pooled_quill_dir())
    return {
        "pool": pool_dir,
        "json": pool_dir / "verify_report.json",
        "md": pool_dir / "verify_report.md",
    }


def require_verify_ok(
    pool_dir: Path | None = None,
    *,
    skip: bool = False,
    paths: Paths | None = None,
) -> Path:
    """Raise unless verify_report.json exists and ``ok`` is true."""
    paths = paths or Paths()
    pool_dir = Path(pool_dir or paths.pooled_quill_dir())
    if skip:
        print(f"[verify] skipped (--skip-verify); pool={pool_dir}")
        return pool_dir
    report = pool_dir / "verify_report.json"
    if not report.is_file():
        raise SystemExit(
            f"Missing {report}\n"
            "Run first:\n"
            "  bash symbol_embed/run_export_quill.sh\n"
            "  bash symbol_embed/run_verify_dataset.sh\n"
            "Or pass --skip-verify to override."
        )
    data = json.loads(report.read_text())
    if not data.get("ok"):
        failed = [c["name"] for c in data.get("checks", []) if not c.get("ok")]
        raise SystemExit(
            f"Verify failed for {pool_dir}: {failed}\n"
            f"See {pool_dir / 'verify_report.md'}"
        )
    print(f"[verify] ok → {report}")
    return pool_dir

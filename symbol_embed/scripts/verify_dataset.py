#!/usr/bin/env python3
"""CLI: verify quill classifier pool before training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.verify_dataset import main

if __name__ == "__main__":
    main()

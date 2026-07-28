#!/usr/bin/env python3
"""CLI: t-SNE static + hover-thumbnail plots."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbol_embed.plot_tsne import main

if __name__ == "__main__":
    main()

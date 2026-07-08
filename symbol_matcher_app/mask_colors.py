"""Distinct per-mask colors for overlay compositing."""

from __future__ import annotations

import colorsys

_PHI = 0.618033988749895  # golden ratio — well-spaced hues


def mask_color_bgra(index: int, alpha: int = 170, sat: float = 0.85, val: float = 1.0) -> tuple[int, int, int, int]:
    """Return BGRA for mask ``index`` using golden-ratio hue spacing."""
    h = (int(index) * _PHI) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, sat, val)
    return int(b * 255), int(g * 255), int(r * 255), alpha

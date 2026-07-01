"""Detect grid lines via histogram jump analysis."""

import cv2
import numpy as np


def count_black_pixels(binary):
    """Count black pixels per row and per column on a 0/255 binary image."""
    black_per_row = np.sum(binary == 0, axis=1)
    black_per_col = np.sum(binary == 0, axis=0)
    return black_per_row, black_per_col


def find_jumps(black_per_row, black_per_col, h_percentile=93.9, v_percentile=99.2):
    """Locate row/column indices where the black-pixel count jumps sharply."""
    horizontal_diff = np.abs(np.diff(black_per_row))
    vertical_diff = np.abs(np.diff(black_per_col))

    h_threshold = np.percentile(horizontal_diff, h_percentile)
    v_threshold = np.percentile(vertical_diff, v_percentile)

    h_jumps = np.where(horizontal_diff > h_threshold)[0]
    v_jumps = np.where(vertical_diff > v_threshold)[0]
    return h_jumps, v_jumps


def draw_grid_lines(binary, h_jumps, v_jumps, color=(255, 255, 255), thickness=2):
    """Draw lines at the detected jump locations on a copy of the binary image."""
    marked = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    h, w = binary.shape

    for jump in h_jumps:
        cv2.line(marked, (0, int(jump)), (w, int(jump)), color, thickness)

    for jump in v_jumps:
        cv2.line(marked, (int(jump), 0), (int(jump), h), color, thickness)

    return marked

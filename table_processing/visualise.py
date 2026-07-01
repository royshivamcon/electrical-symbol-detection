"""Build a side-by-side merged visualisation: original | CCL | bbox.

Saves the merged image into table_processing/output/.
"""

import os

import cv2
import numpy as np
from PIL import Image

from .binarize import binarize, load_image
from .connected_components import apply_ccl
from .grid_detection import count_black_pixels, draw_grid_lines, find_jumps
from .legend_extraction import (
    build_legend_dict,
    find_first_major_gap,
    group_components_by_row,
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


def draw_bboxes(image, legend_dict, split_x):
    """Draw yellow symbol boxes, cyan text boxes, magenta split line."""
    visual = image.copy()
    h, w = visual.shape[:2]
    cv2.line(visual, (split_x, 0), (split_x, h), MAGENTA, 2)

    for row_data in legend_dict.values():
        if row_data[0] is not None:
            sx, sy, sw, sh = row_data[0]
            cv2.rectangle(visual, (sx, sy), (sx + sw, sy + sh), YELLOW, 2)
        if row_data[1] is not None:
            tx, ty, tw, th = row_data[1]
            cv2.rectangle(visual, (tx, ty), (tx + tw, ty + th), CYAN, 2)
    return visual


def merge_horizontally(images, pad_value=255):
    """Stack same-height RGB images side by side, white-padding shorter ones."""
    target_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < target_h:
            pad = np.full(
                (target_h - img.shape[0], img.shape[1], 3),
                pad_value,
                dtype=np.uint8,
            )
            img = np.vstack([img, pad])
        padded.append(img)
    return np.hstack(padded)


def build_visualisation(img_path, run_ocr=True):
    """Run the full pipeline and return (original, ccl_rgb, bbox_rgb) arrays."""
    _, img_array = load_image(img_path)

    binary = binarize(img_array)
    black_per_row, black_per_col = count_black_pixels(binary)
    h_jumps, v_jumps = find_jumps(black_per_row, black_per_col)
    marked = draw_grid_lines(binary, h_jumps, v_jumps)

    labeled, colored = apply_ccl(marked)
    ccl_rgb = (colored * 255).astype(np.uint8)

    image_width = marked.shape[1]
    split_x = find_first_major_gap(v_jumps, image_width=image_width)
    rows = group_components_by_row(labeled, split_x, image_width)
    legend_dict = build_legend_dict(rows, img_array, run_ocr=run_ocr)

    bbox_rgb = draw_bboxes(img_array, legend_dict, split_x)
    return img_array, ccl_rgb, bbox_rgb


def main(img_path, output_name=None, output_dir=OUTPUT_DIR, run_ocr=True):
    """Generate and save the merged visualisation."""
    os.makedirs(output_dir, exist_ok=True)

    if output_name is None:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        output_name = f"{stem}_merged.png"
    elif not os.path.splitext(output_name)[1]:
        output_name = f"{output_name}.png"

    original, ccl_rgb, bbox_rgb = build_visualisation(img_path, run_ocr=run_ocr)
    merged = merge_horizontally([original, ccl_rgb, bbox_rgb])

    out_path = os.path.join(output_dir, output_name)
    Image.fromarray(merged).save(out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("img_path", help="Path to the source image (e.g. img1.png)")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--no-ocr", action="store_true", help="Skip pytesseract OCR")
    args = parser.parse_args()

    saved = main(args.img_path, args.output_name, run_ocr=not args.no_ocr)
    print(f"Saved merged visualisation to {saved}")

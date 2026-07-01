"""Extract a two-column legend layout (symbol + text) from CCL output."""

import numpy as np
from PIL import Image
from skimage.measure import regionprops


def find_first_major_gap(
    v_jumps,
    image_width,
    min_gap_width=25,
    fallback_ratio=0.22,
):
    """Find the first wide horizontal gap in v_jumps; defines the symbol/text split."""
    sorted_v = np.sort(v_jumps)
    if len(sorted_v) < 2:
        return int(image_width * fallback_ratio)

    v_gaps = np.diff(sorted_v)
    large_gap_indices = np.where(v_gaps > min_gap_width)[0]
    if len(large_gap_indices) > 0:
        first_idx = large_gap_indices[0]
        return int(sorted_v[first_idx + 1])

    return int(image_width * fallback_ratio)


def group_components_by_row(labeled_array, split_x, image_width, row_tolerance=14):
    """Filter CCL components and group them into legend rows by vertical center."""
    components = []
    for prop in regionprops(labeled_array):
        y1, x1, y2, x2 = prop.bbox
        w = x2 - x1
        h = y2 - y1

        if w < 2 or h < 2 or w > image_width * 0.98:
            continue

        cx = x1 + (w / 2)
        cy = y1 + (h / 2)
        col_idx = 0 if cx < split_x else 1

        components.append({
            "bbox": (x1, y1, w, h),
            "cy": cy,
            "col_idx": col_idx,
        })

    components.sort(key=lambda c: c["cy"])
    if not components:
        return []

    rows = []
    current_row = [components[0]]
    for c in components[1:]:
        avg_cy = np.mean([item["cy"] for item in current_row])
        if abs(c["cy"] - avg_cy) < row_tolerance:
            current_row.append(c)
        else:
            rows.append(current_row)
            current_row = [c]
    rows.append(current_row)
    return rows


def shrink_wrap(bboxes):
    """Compute the smallest box that contains all given (x, y, w, h) boxes."""
    if not bboxes:
        return None
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def build_legend_dict(rows, image, run_ocr=True):
    """Turn grouped rows into {text_key: {0: symbol_bbox, 1: text_bbox}}."""
    legend_dict = {}
    for r_idx, row_items in enumerate(rows):
        row_data = {0: None, 1: None}
        col_groups = {0: [], 1: []}
        for item in row_items:
            col_groups[item["col_idx"]].append(item["bbox"])
        for col, bboxes in col_groups.items():
            row_data[col] = shrink_wrap(bboxes)

        text_key = ""
        if run_ocr and row_data[1] is not None:
            text_key = _ocr_text_box(image, row_data[1])

        if not text_key or row_data[0] is None:
            dict_key = f"Header_or_Section_Row_{r_idx}" if not text_key else text_key
        else:
            dict_key = text_key
        legend_dict[dict_key] = row_data
    return legend_dict


def _ocr_text_box(image, bbox):
    """OCR a single text-row crop; returns a cleaned string."""
    import pytesseract

    tx, ty, tw, th = bbox
    crop = image[ty:ty + th, tx:tx + tw]
    crop_gray = np.mean(crop, axis=2) if crop.ndim == 3 else crop
    crop_prepared = np.where(crop_gray < 180, 0, 255).astype(np.uint8)
    text = pytesseract.image_to_string(
        Image.fromarray(crop_prepared), config="--psm 7"
    ).strip()
    return " ".join(text.split())

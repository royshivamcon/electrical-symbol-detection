"""Detect and OCR tables in a PNG using img2table + tesseract.

Standalone module — does not depend on the rest of `table_processing`.

Outputs (under `table_processing/output_tables/` by default):
    <stem>_tables_marked.png   table + cell bboxes drawn on the source image
    <stem>_tables.xlsx         one sheet per detected table (via img2table)
    <stem>_tables.json         cell-level text + bbox dump
"""

import json
import os

import cv2
import numpy as np
from PIL import Image as PILImage

from img2table.document import Image as I2TImage
from img2table.ocr import TesseractOCR


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_tables")

TABLE_COLOR = (0, 0, 255)   # red outline for full tables (BGR)
CELL_COLOR = (0, 200, 0)    # green for cells (BGR)


def extract_tables(
    img_path,
    lang="eng",
    n_threads=1,
    implicit_rows=False,
    borderless_tables=False,
    min_confidence=50,
):
    """Run img2table extraction. Returns (img2table doc, list[Table])."""
    doc = I2TImage(src=img_path)
    ocr = TesseractOCR(n_threads=n_threads, lang=lang)
    tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=implicit_rows,
        borderless_tables=borderless_tables,
        min_confidence=min_confidence,
    )
    return doc, ocr, tables


def draw_table_overlay(img_bgr, tables):
    """Draw table outlines (red) and cell outlines (green) on a copy."""
    canvas = img_bgr.copy()
    for t_idx, table in enumerate(tables, start=1):
        bb = table.bbox
        cv2.rectangle(canvas, (bb.x1, bb.y1), (bb.x2, bb.y2), TABLE_COLOR, 3)
        cv2.putText(
            canvas,
            f"T{t_idx}",
            (bb.x1 + 4, bb.y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TABLE_COLOR,
            2,
        )
        for row in table.content.values():
            for cell in row:
                cb = cell.bbox
                cv2.rectangle(canvas, (cb.x1, cb.y1), (cb.x2, cb.y2), CELL_COLOR, 1)
    return canvas


def tables_to_records(tables):
    """Serialize tables to a JSON-friendly structure."""
    out = []
    for t_idx, table in enumerate(tables, start=1):
        bb = table.bbox
        rows = []
        for row_idx, row in table.content.items():
            rows.append([
                {
                    "bbox": [cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2],
                    "value": cell.value,
                }
                for cell in row
            ])
        out.append({
            "index": t_idx,
            "bbox": [bb.x1, bb.y1, bb.x2, bb.y2],
            "rows": rows,
        })
    return out


def main(
    img_path,
    output_dir=OUTPUT_DIR,
    lang="eng",
    implicit_rows=False,
    borderless_tables=False,
    min_confidence=50,
):
    """Detect tables, save marked PNG + Excel + JSON. Returns dict of paths."""
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]

    doc, ocr, tables = extract_tables(
        img_path,
        lang=lang,
        implicit_rows=implicit_rows,
        borderless_tables=borderless_tables,
        min_confidence=min_confidence,
    )

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    marked = draw_table_overlay(img_bgr, tables)

    marked_path = os.path.join(output_dir, f"{stem}_tables_marked.png")
    cv2.imwrite(marked_path, marked)

    xlsx_path = os.path.join(output_dir, f"{stem}_tables.xlsx")
    doc.to_xlsx(dest=xlsx_path, ocr=ocr,
                implicit_rows=implicit_rows,
                borderless_tables=borderless_tables,
                min_confidence=min_confidence)

    json_path = os.path.join(output_dir, f"{stem}_tables.json")
    with open(json_path, "w") as f:
        json.dump(tables_to_records(tables), f, indent=2)

    return {
        "marked_png": marked_path,
        "xlsx": xlsx_path,
        "json": json_path,
        "num_tables": len(tables),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("img_path", help="Source PNG (or other image format)")
    parser.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    parser.add_argument("--implicit-rows", action="store_true",
                        help="Infer rows even without horizontal lines")
    parser.add_argument("--borderless", action="store_true",
                        help="Also detect borderless tables")
    parser.add_argument("--min-confidence", type=int, default=50)
    args = parser.parse_args()

    result = main(
        args.img_path,
        lang=args.lang,
        implicit_rows=args.implicit_rows,
        borderless_tables=args.borderless,
        min_confidence=args.min_confidence,
    )
    print(f"Detected {result['num_tables']} table(s).")
    print(f"  marked PNG : {result['marked_png']}")
    print(f"  xlsx       : {result['xlsx']}")
    print(f"  json       : {result['json']}")

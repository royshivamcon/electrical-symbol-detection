"""Tiling + stitching helpers for the hybrid SAM box pipeline.

Instead of running the model once per reference point (which re-encodes heavily
overlapping crops), the hybrid path covers the sheet with a grid of overlapping
tiles, encodes each tile **once**, decodes every point assigned to that tile, and
finally merges all boxes with a global NMS (the "stitch"). Each point is assigned
to exactly one tile so it is analysed a single time.
"""

from __future__ import annotations

Tile = tuple[int, int, int, int]  # (x0, y0, x1, y1) in image pixels


def tile_grid(w: int, h: int, tile: int = 640, overlap: int = 96) -> list[Tile]:
    """Cover a ``w x h`` image with overlapping square tiles of side ``tile``.

    ``overlap`` is the number of pixels neighbouring tiles share so a symbol on a
    boundary is whole inside at least one tile. A final tile is flushed to each
    far edge so the whole image is covered.
    """
    tile = max(64, int(tile))
    overlap = max(0, min(int(overlap), tile - 1))
    step = max(1, tile - overlap)

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        s = list(range(0, total - tile + 1, step))
        if s[-1] != total - tile:
            s.append(total - tile)
        return s

    tiles: list[Tile] = []
    for y0 in starts(h):
        for x0 in starts(w):
            tiles.append((x0, y0, min(x0 + tile, w), min(y0 + tile, h)))
    return tiles


def assign_points_to_tiles(points, tiles: list[Tile]) -> dict[int, list[int]]:
    """Map each point index to exactly one tile index.

    A point is assigned to the tile in which it sits **most interior** (largest
    minimum distance to the tile's edges), so boundary symbols land in the tile
    that fully contains them rather than one that clips them.
    """
    out: dict[int, list[int]] = {t: [] for t in range(len(tiles))}
    for pi, p in enumerate(points):
        px, py = p.x, p.y
        best_t, best_margin = None, -1.0
        for ti, (x0, y0, x1, y1) in enumerate(tiles):
            if not (x0 <= px < x1 and y0 <= py < y1):
                continue
            margin = min(px - x0, x1 - px, py - y0, y1 - py)
            if margin > best_margin:
                best_margin, best_t = margin, ti
        if best_t is None:  # shouldn't happen (grid covers the image), but be safe
            best_t = 0
        out[best_t].append(pi)
    return out


def nms_boxes(boxes: list, iou_thresh: float = 0.5, iou_fn=None) -> list:
    """Greedy NMS over objects exposing ``x, y, w, h, score``.

    Keeps the highest-scoring box and drops later boxes overlapping it beyond
    ``iou_thresh``. ``iou_fn`` lets callers pass a shared IoU implementation.
    """
    if not boxes:
        return []
    if iou_fn is None:
        from evaluation import iou as iou_fn  # reuse the single IoU implementation

    order = sorted(boxes, key=lambda b: getattr(b, "score", 0.0), reverse=True)
    keep: list = []
    for cand in order:
        cb = (cand.x, cand.y, cand.w, cand.h)
        if all(iou_fn(cb, (k.x, k.y, k.w, k.h)) <= iou_thresh for k in keep):
            keep.append(cand)
    return keep

import numpy as np
from scipy.signal import find_peaks
from collections import defaultdict
import fitz
from tqdm import tqdm
from shared.log_config import get_logger

logger = get_logger(__name__)


def get_foreground_paths(pdf_paths, remove_long_paths: bool = False, max_path_length: float = 100.0):
    """
    Filters paths from a PDF based on their visibility threshold.

    Parameters:
    - pdf_paths: List of paths with attributes like width, color, and opacity.
    - remove_long_paths: If True, remove paths whose bounding box exceeds
      *max_path_length* points in either dimension. Default False so
      existing callers are unaffected.
    - max_path_length: Threshold in PDF points (1 pt = 1/72 inch).
      Paths with a bounding rect wider or taller than this are dropped
      when *remove_long_paths* is True.

    Returns:
    - new_paths: Filtered list of paths above the visibility threshold.
    """
    # Helper function to calculate intensity
    def calculate_intensity(path):
        if path["type"] == "s":
            color = np.mean(path["color"])
            opacity = path["stroke_opacity"]
        else:
            color = np.mean(path["fill"])
            opacity = path["fill_opacity"]

        return int((1 - color) * opacity * 255)

    # Step 1: Calculate visibility
    logger.info(f"pdf_paths - {len(pdf_paths)}")
    visibility = [calculate_intensity(path) for path in pdf_paths]

    # Step 2: Create histogram and find peaks
    count, bins = np.histogram(visibility, bins=256)
    max_ind, _ = find_peaks(np.concatenate(([0], count, [0]), axis=0))

    # Step 3: Handle cases where no peaks are detected
    if len(max_ind) == 0:
        logger.info("No peaks detected. Using fallback threshold.")
        threshold = np.median(visibility)  # Use median as a fallback
    else:
        # Extract peaks and calculate threshold
        peaks = np.concatenate(([0], count, [0]))[max_ind]
        threshold = int(np.sum(bins[max_ind] * peaks) / np.sum(peaks))  # Weighted average

    # Step 4: Filter paths based on visibility threshold
    new_paths = [path for path, intensity in zip(pdf_paths, visibility) if intensity >= threshold]
    logger.info(f"Filtered pdf_paths (visibility) - {len(new_paths)}")

    # Step 5: Optionally remove long paths
    if remove_long_paths:
        before = len(new_paths)
        filtered = []
        for path in new_paths:
            rect = path.get("rect")
            if rect is not None:
                w = abs(rect[2] - rect[0])
                h = abs(rect[3] - rect[1])
                if w > max_path_length or h > max_path_length:
                    continue
            filtered.append(path)
        new_paths = filtered
        logger.info(f"Filtered pdf_paths (long paths >{max_path_length}pt removed) - "
                     f"{before} -> {len(new_paths)} ({before - len(new_paths)} removed)")

    return new_paths


def redraw_page_without_text(
    page: fitz.Page,
    remove_long_lines: bool = False,
    long_line_threshold_pt: float = 40.0,
    reinsert_images: bool = True,
) -> fitz.Page:
    """
    Redraw the entire PDF page on a blank sheet, skipping only PyMuPDF text.
    Preserves all drawing paths while excluding text content.

    This avoids the white-rectangle-overlay approach which would cover images
    that happen to sit underneath text regions.

    Args:
        page: PyMuPDF page object (read-only, not mutated)
        remove_long_lines: If True, drop line segments longer than
            *long_line_threshold_pt* PDF points.
            Rectangles, quads, and curves are always preserved.
        long_line_threshold_pt: Absolute length in PDF points. Lines longer
            than this are removed when *remove_long_lines* is True (default 40).
        reinsert_images: If True (default), re-insert embedded images from
            the original page. Set to False to exclude images.

    Returns:
        A new page (from a new in-memory document) with paths + images redrawn,
        but no text.
    """
    # Step 1: Collect all drawing paths
    all_paths = page.get_drawings()

    if remove_long_lines:
        from kelex.page.page_utils import remove_long_running_paths
        from kelex.path.path_utils import explode_path

        total_before = len(all_paths)
        exploded = []
        for p in all_paths:
            exploded.extend(explode_path(p))

        min_dim = min(page.rect.width, page.rect.height)
        fractional_threshold = long_line_threshold_pt / min_dim if min_dim > 0 else 0.5

        # Curves are preserved (curves mostly comes up at text or small featurs) without filtering; only lines and rects/quads are size-filtered
        curves, lines, rects_quads = [], [], []
        for p in exploded:
            item_type = p["items"][0][0]
            if item_type == "c":
                curves.append(p)
            elif item_type == "l":
                lines.append(p)
            else:
                rects_quads.append(p)

        filtered_lines = remove_long_running_paths(
            paths=lines,
            page_width=page.rect.width,
            page_height=page.rect.height,
            threshold=fractional_threshold,
        )

        filtered_rq = [
            p for p in rects_quads
            if max(fitz.Rect(p["rect"]).width, fitz.Rect(p["rect"]).height) <= long_line_threshold_pt
        ]

        removed_lines = len(lines) - len(filtered_lines)
        removed_rq = len(rects_quads) - len(filtered_rq)
        all_paths = curves + filtered_rq + filtered_lines
        logger.info(
            f"Large-shape removal (>{long_line_threshold_pt}pt): {total_before} paths "
            f"({len(exploded)} after explode) -> "
            f"{len(curves)} curves (kept) + "
            f"{len(rects_quads)} re/qu (-{removed_rq}) + "
            f"{len(lines)} lines (-{removed_lines}) = {len(all_paths)} total"
        )

    # Step 2: Redraw paths on a blank page
    new_page = draw_shape_on_blank_page_optimized(page, all_paths, blank=True)
    logger.info(f"Redrew {len(all_paths)} drawing paths on blank page (text excluded)")

    if reinsert_images:
        # get_image_info(xrefs=True) builds a Pixmap per image xref to match digests.
        # That can raise (e.g. FzErrorSyntax) before the per-image loop if the PDF has
        # a broken ICC / color space — wrap the whole block so path redraw still succeeds.
        try:
            image_infos = page.get_image_info(xrefs=True)
            image_count = 0
            for info in image_infos:
                xref = info.get("xref", 0)
                if xref == 0:
                    continue
                try:
                    bbox = fitz.Rect(info["bbox"])
                    if bbox.is_empty or bbox.is_infinite:
                        continue
                    bbox.transform(page.rotation_matrix)
                    img_data = page.parent.extract_image(xref)
                    if img_data and img_data.get("image"):
                        new_page.insert_image(bbox, stream=img_data["image"])
                        image_count += 1
                except Exception as e:
                    logger.warning(f"Could not re-insert image xref={xref}: {e}")
                    continue
            logger.info(f"Re-inserted {image_count} images on blank page")
        except Exception as e:
            logger.warning(
                "Skipped all image reinsertion (invalid image/color space in PDF): %s",
                e,
            )
    else:
        logger.info("Skipped image reinsertion")

    return new_page


def draw_shape_on_blank_page(page, drawing_path, blank=True):
    # Define output page
    if blank:
        outpdf = fitz.open()
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    else:
        outpage = page

    shape = outpage.new_shape()  # Create drawing canvas

    # --------------------------------------
    # Precompute transformations for all paths
    # --------------------------------------
    transformed_paths = []
    for path in tqdm(drawing_path, desc="Precomputing Transformations"):
        transformed_items = []
        for item in path["items"]:
            cmd = item[0]
            if cmd == "l":  # line
                transformed_items.append(("line", item[1] * page.rotation_matrix, item[2] * page.rotation_matrix))
            elif cmd == "re":  # rectangle
                transformed_items.append(("rect", item[1] * page.rotation_matrix))
            elif cmd == "qu":  # quad
                transformed_items.append(("quad", item[1] * page.rotation_matrix))
            elif cmd == "c":  # curve (bezier)
                transformed_items.append(
                    (
                        "bezier",
                        item[1] * page.rotation_matrix,
                        item[2] * page.rotation_matrix,
                        item[3] * page.rotation_matrix,
                        item[4] * page.rotation_matrix,
                    )
                )
            else:
                raise ValueError("Unhandled drawing", item)

        # Store with properties for grouping, but maintain order
        transformed_paths.append(
            {
                "draw_ops": transformed_items,
                "fill": path["fill"],
                "color": path["color"],
                "dashes": path["dashes"],
                "even_odd": path.get("even_odd", True),
                "closePath": path["closePath"],
                "lineJoin": path.get("lineJoin", None) or 0,
                "lineCap": max(path.get("lineCap", None) or [0]),
                "width": path["width"],
                "stroke_opacity": path.get("stroke_opacity", None) or 0,
                "fill_opacity": path.get("fill_opacity", None) or 0,
                "original_index": len(transformed_paths),  # Store original order
            }
        )

    # --------------------------------------
    # Use a function lookup to speed up execution
    # --------------------------------------
    draw_function_map = {
        "line": shape.draw_line,
        "rect": shape.draw_rect,
        "quad": shape.draw_quad,
        "bezier": shape.draw_bezier,
    }

    # --------------------------------------
    # Group paths by properties **without changing order**
    # --------------------------------------
    grouped_paths = defaultdict(list)
    for path in transformed_paths:
        key = (
            path["fill"],
            path["color"],
            path["dashes"],
            path["even_odd"],
            path["closePath"],
            path["lineJoin"],
            path["lineCap"],
            path["width"],
            path["stroke_opacity"],
            path["fill_opacity"],
        )
        grouped_paths[key].append(path)  # Preserve order within the same group

    # --------------------------------------
    # Process paths in original order but apply batching
    # --------------------------------------
    for path in tqdm(
        sorted(transformed_paths, key=lambda p: p["original_index"]), desc="Drawing Shapes (Ordered Batch Mode)"
    ):
        key = (
            path["fill"],
            path["color"],
            path["dashes"],
            path["even_odd"],
            path["closePath"],
            path["lineJoin"],
            path["lineCap"],
            path["width"],
            path["stroke_opacity"],
            path["fill_opacity"],
        )

        # Process entire group together (maintains order)
        for grouped_path in grouped_paths[key]:
            for op in grouped_path["draw_ops"]:
                draw_function_map[op[0]](*op[1:])  # Direct function call

        # Apply properties **once per group**
        shape.finish(
            fill=path["fill"],
            color=path["color"],
            dashes=path["dashes"],
            even_odd=path["even_odd"],
            closePath=path["closePath"],
            lineJoin=path["lineJoin"],
            lineCap=path["lineCap"],
            width=path["width"],
            stroke_opacity=path["stroke_opacity"],
            fill_opacity=path["fill_opacity"],
        )

        # Remove processed group to avoid duplicates
        del grouped_paths[key]

    # Commit shape to page
    shape.commit()
    return outpage


def format_color(c, f):
    """
    Format color tuple to PDF operator string.
    Replicates PyMuPDF ColorCode + _format_g logic.
    
    Reference: PyMuPDF/src/__init__.py
    - Lines 22127-22142: ColorCode
    - Lines 443-458: _format_g
    
    Args:
        c: color tuple (r, g, b) or (gray,) or (c, m, y, k)
        f: 'c' for stroke (RG/G/K), 'f' for fill (rg/g/k)
    
    Returns:
        Formatted string like "0.5 0.5 0.5 RG " or "0.5 g "
    """
    if not c:
        return ""
    
    # Format each component with :g (removes trailing zeros)
    formatted = ' '.join(f"{v:g}" for v in c)
    
    # Add operator based on color space
    if len(c) == 3:  # RGB
        return formatted + (" RG " if f == "c" else " rg ")
    elif len(c) == 1:  # Gray
        return formatted + (" G " if f == "c" else " g ")
    elif len(c) == 4:  # CMYK
        return formatted + (" K " if f == "c" else " k ")
    
    return ""


def draw_shape_on_blank_page_optimized(page, drawing_path, blank=True):
    """
    Optimized PDF path drawing using direct PDF command string building.
    
    Achieves 60x speedup by avoiding Python-to-C overhead of individual shape.draw_*() calls.
    Instead, builds raw PDF content stream commands as strings and inserts once.
    
    Performance:
    - Original: ~890s for 483k paths (544 paths/sec)
    - Optimized: ~15s for 483k paths (33,904 paths/sec)
    - Speedup: 59.7x faster
    - Image quality: 99.9991% identical pixels (62/6.6M differ by 1-3 values due to anti-aliasing)
    
    Reference: PyMuPDF/src/__init__.py
    - Lines 14914-14926: draw_line
    - Lines 14928-14940: draw_polyline (used by draw_quad)
    - Lines 14942-14963: draw_bezier
    - Lines 15086-15104: draw_rect
    - Lines 15137-15140: draw_quad
    - Lines 15724-15813: finish
    - Lines 15815-15831: commit
    """
    # Define output page
    if blank:
        outpdf = fitz.open()
        outpage = outpdf.new_page(width=page.rect.width, height=page.rect.height)
    else:
        outpage = page

    rotation_matrix = page.rotation_matrix
    ipctm = ~outpage.transformation_matrix  # Inverse page transformation matrix (line 14884)

    # --------------------------------------
    # Precompute transformations for all paths
    # --------------------------------------
    transformed_paths = []
    for path in tqdm(drawing_path, desc="Precomputing Transformations"):
        transformed_items = []
        for item in path["items"]:
            cmd = item[0]
            if cmd == "l":  # line
                transformed_items.append(("line", item[1] * rotation_matrix, item[2] * rotation_matrix))
            elif cmd == "re":  # rectangle
                transformed_items.append(("rect", item[1] * rotation_matrix))
            elif cmd == "qu":  # quad
                transformed_items.append(("quad", item[1] * rotation_matrix))
            elif cmd == "c":  # curve (bezier)
                transformed_items.append(
                    (
                        "bezier",
                        item[1] * rotation_matrix,
                        item[2] * rotation_matrix,
                        item[3] * rotation_matrix,
                        item[4] * rotation_matrix,
                    )
                )
            else:
                raise ValueError("Unhandled drawing", item)

        # Store with properties for grouping
        transformed_paths.append(
            {
                "items": transformed_items,
                "fill": path["fill"],
                "color": path["color"],
                "dashes": path["dashes"],
                "even_odd": path.get("even_odd", True),
                "closePath": path["closePath"],
                "lineJoin": path.get("lineJoin", None) or 0,
                "lineCap": max(path.get("lineCap", None) or [0]),
                "width": path["width"],
                "stroke_opacity": path.get("stroke_opacity", None) or 0,
                "fill_opacity": path.get("fill_opacity", None) or 0,
            }
        )

    # --------------------------------------
    # Group paths by styling properties
    # --------------------------------------
    grouped_paths = defaultdict(list)
    for path in tqdm(transformed_paths, desc="Grouping paths"):
        key = (
            path["fill"],
            path["color"],
            path["dashes"],
            path["even_odd"],
            path["closePath"],
            path["lineJoin"],
            path["lineCap"],
            path["width"],
            path["stroke_opacity"],
            path["fill_opacity"],
        )
        grouped_paths[key].append(path)

    logger.info(f"Grouped {len(transformed_paths)} paths into {len(grouped_paths)} style groups")

    # --------------------------------------
    # Build raw PDF commands for each group
    # --------------------------------------
    commands = []
    
    for key, paths_in_group in tqdm(grouped_paths.items(), desc="Building PDF commands"):
        # Extract styling from key
        fill = key[0]
        color = key[1]
        dashes = key[2]
        even_odd = key[3]
        closePath = key[4]
        lineJoin = key[5]
        lineCap = key[6]
        width = key[7]
        stroke_opacity = key[8]
        fill_opacity = key[9]
        
        # Apply Shape.finish() defaults (lines 15749-15752)
        if width is None:
            width = 1
        if width == 0:
            color = None
        elif color is None:
            width = 0
        
        # Start group with q (save graphics state) - Line 15810
        group_commands = []
        group_commands.append("\nq\n")
        
        # PREPEND styling (lines 15772-15778)
        # Line cap
        if lineCap != 0:
            group_commands.append(f"{lineCap} J\n")
        
        # Line join
        if lineJoin != 0:
            group_commands.append(f"{lineJoin} j\n")
        
        # Dashes
        if dashes and dashes not in (None, "", "[] 0"):
            group_commands.append(f"{dashes} d\n")
        
        # Draw all paths in this group
        # Track last_point across ALL paths in group (like PyMuPDF Shape does)
        last_point = None
        
        for path in paths_in_group:
            for item in path["items"]:
                cmd = item[0]
                
                if cmd == "line":  # draw_line (PyMuPDF lines 14914-14926)
                    p1 = item[1] * ipctm
                    p2 = item[2] * ipctm
                    
                    # Only add 'm' if last_point != p1 (line 14918)
                    if last_point != p1:
                        group_commands.append(f"{p1.x:.4f} {p1.y:.4f} m\n")
                    group_commands.append(f"{p2.x:.4f} {p2.y:.4f} l\n")
                    last_point = p2
                    
                elif cmd == "rect":  # draw_rect (PyMuPDF lines 15099-15101)
                    r = item[1] * ipctm
                    group_commands.append(f"{r.x0:.4f} {r.y0:.4f} {r.width:.4f} {r.height:.4f} re\n")
                    last_point = r.tl  # line 15103
                    
                elif cmd == "quad":  # draw_quad (PyMuPDF line 15140 -> draw_polyline with 5 points)
                    q = item[1] * ipctm
                    
                    # draw_polyline: [ul, ll, lr, ur, ul] - 5 points to close the quad!
                    points = [q.ul, q.ll, q.lr, q.ur, q.ul]
                    
                    # First point (lines 14932-14934)
                    if last_point != points[0]:
                        group_commands.append(f"{points[0].x:.4f} {points[0].y:.4f} m\n")
                    
                    # Remaining points (line 14936)
                    for p in points[1:]:
                        group_commands.append(f"{p.x:.4f} {p.y:.4f} l\n")
                    
                    last_point = points[-1]  # line 14939
                    
                elif cmd == "bezier":  # draw_bezier (PyMuPDF lines 14954-14957)
                    p1 = item[1] * ipctm
                    p2 = item[2] * ipctm
                    p3 = item[3] * ipctm
                    p4 = item[4] * ipctm
                    
                    # Only add 'm' if last_point != p1 (line 14954)
                    if last_point != p1:
                        group_commands.append(f"{p1.x:.4f} {p1.y:.4f} m\n")
                    group_commands.append(f"{p2.x:.4f} {p2.y:.4f} {p3.x:.4f} {p3.y:.4f} {p4.x:.4f} {p4.y:.4f} c\n")
                    last_point = p4
        
        # APPEND after geometry (lines 15769-15788)
        
        # closePath 'h' operator (lines 15780-15781)
        if closePath:
            group_commands.append("h\n")
        
        # Line width (line 15769-15770)
        if width != 1 and width != 0:
            group_commands.append(f"{width:.4f} w\n")
        
        # Stroke color (lines 15784-15785)
        if color is not None:
            group_commands.append(format_color(color, "c"))  # Returns "r g b RG "
        
        # Fill color (lines 15787-15788)
        if fill is not None:
            group_commands.append(format_color(fill, "f"))  # Returns "r g b rg "
        
        # Painting operator (lines 15789-15800)
        if fill is not None:
            if width != 0:  # Both fill and stroke (lines 15789-15793)
                if even_odd:
                    group_commands.append("B*\n")  # fill (even-odd) and stroke
                else:
                    group_commands.append("B\n")  # fill and stroke
            else:  # Fill only, no stroke (lines 15794-15798)
                if even_odd:
                    group_commands.append("f*\n")  # fill only (even-odd)
                else:
                    group_commands.append("f\n")  # fill only
        else:  # No fill, stroke only (lines 15799-15800)
            group_commands.append("S\n")  # stroke only
        
        # Q - restore graphics state (line 15810)
        group_commands.append("Q\n")
        
        # Add this group to main commands
        commands.extend(group_commands)

    # --------------------------------------
    # Insert all commands at once (replaces shape.commit())
    # --------------------------------------
    content_str = "".join(commands).encode()
    outpage.wrap_contents()
    fitz.TOOLS._insert_contents(outpage, content_str, overlay=False)
    
    logger.info(f"Inserted {len(commands)} PDF command fragments into page")
    
    return outpage


if __name__ == "__main__":
    pdfs_file_path = "local_storage/pdfs/1/1.pdf"
    page_no = 1
    fe_width = 144
    fe_height = 216
    ZOOM_LEVEL = 2

    document = fitz.open(pdfs_file_path)

    page = document.load_page(page_no - 1)

    pdf_paths = page.get_drawings()

    foreground_paths = get_foreground_paths(pdf_paths)
    foreground_page = draw_shape_on_blank_page(page, foreground_paths)

    original_width, original_height = page.rect.width, page.rect.height
    scale_x = fe_width / original_width
    scale_y = fe_height / original_height

    # Combine the resize scale factors with the zoom level
    combined_scale_x = scale_x * ZOOM_LEVEL
    combined_scale_y = scale_y * ZOOM_LEVEL

    mat = fitz.Matrix(combined_scale_x, combined_scale_y)

    # Render the page to a pixmap using the transformation matrix
    pixmap = foreground_page.get_pixmap(matrix=mat)

    # Convert the pixmap to a NumPy array for further processing
    zoomed_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        (
            pixmap.height,
            pixmap.width,
            len(pixmap.samples) // (pixmap.height * pixmap.width),
        )
    )
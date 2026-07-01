from .binarize import load_image, to_grayscale, binarize
from .grid_detection import (
    count_black_pixels,
    find_jumps,
    draw_grid_lines,
)
from .connected_components import apply_ccl
from .legend_extraction import (
    find_first_major_gap,
    group_components_by_row,
    shrink_wrap,
    build_legend_dict,
)

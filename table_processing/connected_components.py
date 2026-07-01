"""Connected-component labeling on a grid-marked image."""

import numpy as np
from skimage import measure
from skimage.color import label2rgb


def apply_ccl(marked_img, threshold=200):
    """Label connected components on a grid-marked image.

    Returns the labeled array and a colored RGB visualization (float 0-1).
    """
    if len(marked_img.shape) == 3:
        gray = np.mean(marked_img, axis=2)
    else:
        gray = marked_img

    binary_for_ccl = (gray < threshold).astype(np.uint8)
    labeled_array = measure.label(binary_for_ccl)
    colored_labels = label2rgb(labeled_array, bg_label=0)
    return labeled_array, colored_labels

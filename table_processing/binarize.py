"""Image loading and binarization."""

import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path)
    return img, np.array(img.convert("RGB"))


def to_grayscale(img_array):
    if len(img_array.shape) == 3:
        return np.mean(img_array, axis=2)
    return img_array


def binarize(img_array, threshold=127):
    """Threshold to a 0/255 binary image."""
    gray = to_grayscale(img_array)
    return (gray > threshold).astype(np.uint8) * 255

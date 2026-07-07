"""
imaging.py
----------
Image loading and LAB-space mask generation utilities.
"""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image_rgb(path: str) -> np.ndarray:
    """Load an image from *path* and return it as an (H, W, 3) RGB array."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not open image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def mask_lab(
    img_rgb: np.ndarray,
    L_thresh: int = 140,
    a_thresh: int = 128,
    b_thresh: int = 155,
    dir_L: int = 1,
    dir_a: int = 1,
    dir_b: int = 1,
) -> np.ndarray:
    """
    Threshold an RGB image in LAB colour space and return a boolean mask.

    Parameters
    ----------
    img_rgb : (H, W, 3) ndarray
    L_thresh, a_thresh, b_thresh : int
        Per-channel threshold values.
    dir_L, dir_a, dir_b : {1, -1}
        1  → keep pixels *below* the threshold (<=).
        -1 → keep pixels *above* the threshold (>=).

    Returns
    -------
    mask : (H, W) bool ndarray
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(img_lab)

    def _apply(channel, thresh, direction):
        if direction == 1:
            return channel <= thresh
        if direction == -1:
            return channel >= thresh
        raise ValueError("direction must be 1 or -1")

    return _apply(L, L_thresh, dir_L) & _apply(a, a_thresh, dir_a) & _apply(b, b_thresh, dir_b)


def fill_mask_poly(mask: np.ndarray) -> np.ndarray:
    """Fill mask holes by drawing filled contour polygons. Returns uint8 mask."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(filled, contours, 255)
    return filled


def fill_mask_bin(mask: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """
    Fill mask holes via morphological closing followed by binary fill.

    Parameters
    ----------
    mask : (H, W) bool or uint8 ndarray
    kernel_size : int
        Diameter of the elliptical structuring element used for closing.

    Returns
    -------
    filled : (H, W) uint8 ndarray  (values 0 or 255)
    """
    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_filled = binary_fill_holes(mask_closed > 0)
    return (mask_filled * 255).astype(np.uint8)


def fill_mask_hull(mask: np.ndarray) -> np.ndarray:
    """Fill the convex hull of the largest contour in *mask*. Returns uint8 mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    mask_hull = np.zeros_like(mask)
    cv2.drawContours(mask_hull, [hull], -1, 255, -1)
    return mask_hull


def contour_largest(mask_u8: np.ndarray):
    """
    Isolate the largest connected component in *mask_u8*.

    Returns
    -------
    mask_largest : (H, W) uint8 ndarray
        Binary mask of the largest component (values 0 or 255).
    border : (H, W) uint8 ndarray
        Morphological gradient (edge image) of *mask_u8*.
    """
    kernel = np.ones((3, 3), np.uint8)
    border = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)

    num_labels, labels = cv2.connectedComponents(mask_u8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background

    largest = sizes.argmax()
    mask_largest = (labels == largest).astype(np.uint8) * 255
    return mask_largest, border


# ---------------------------------------------------------------------------
# Convenience: full segmentation pipeline
# ---------------------------------------------------------------------------

def segment_image(img_rgb: np.ndarray):
    """
    Run the standard segmentation pipeline on *img_rgb*.

    Steps
    -----
    1. Otsu threshold on the *a* and *b* LAB channels.
    2. Combine masks with AND.
    3. Fill holes (morphological closing + binary fill).
    4. Keep only the largest connected component.

    Returns
    -------
    img_lab   : (H, W, 3) ndarray  – LAB version of the input
    big_mask  : (H, W) uint8       – largest-component mask
    border    : (H, W) uint8       – edge image
    """
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    _, _, _ = cv2.split(img_lab)          # kept for symmetry; unpacked below
    L, a, b = cv2.split(img_lab)

    _, mask_a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = cv2.bitwise_and(mask_b, mask_a)

    filled_mask = fill_mask_bin(mascara)
    big_mask, border = contour_largest(filled_mask)
    return img_lab, big_mask, border


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def pixelate(img: np.ndarray, factor: int = 20) -> np.ndarray:
    """
    Pixelate *img* by downscaling by *factor* then upscaling with nearest-
    neighbour interpolation.
    """
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

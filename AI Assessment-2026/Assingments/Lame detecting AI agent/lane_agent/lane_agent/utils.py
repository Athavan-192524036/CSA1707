"""Small shared helpers for I/O and debugging visualization."""

from __future__ import annotations
import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_image(path: str, bgr: np.ndarray) -> None:
    cv2.imwrite(path, bgr)


def binary_to_bgr(binary: np.ndarray) -> np.ndarray:
    """Convert a single-channel 0/255 mask to a viewable 3-channel image."""
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def stack_debug_view(original_bgr: np.ndarray, binary_warped: np.ndarray) -> np.ndarray:
    """Side-by-side: annotated frame | bird's-eye binary mask (for debugging)."""
    h, w = original_bgr.shape[:2]
    warped_bgr = binary_to_bgr(binary_warped)
    warped_bgr = cv2.resize(warped_bgr, (w, h))
    return np.hstack([original_bgr, warped_bgr])

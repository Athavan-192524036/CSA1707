"""
Weather-robust preprocessing.

Road images captured in different weather/lighting conditions (bright sun,
overcast, dusk/night, rain, fog) vary hugely in contrast, brightness and
noise. These functions normalize a frame before feature extraction so the
same thresholds in `detector.py` work across conditions.

Techniques used:
    - Auto brightness estimation -> gamma correction (dark / night frames)
    - CLAHE (adaptive histogram equalization) -> low-contrast / foggy frames
    - Light denoising -> rain / sensor noise
"""

from __future__ import annotations
import cv2
import numpy as np


def estimate_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity in [0, 255]."""
    return float(np.mean(gray))


def auto_gamma_correct(bgr: np.ndarray, target_brightness: float = 130.0) -> np.ndarray:
    """
    Brighten dark/night frames or tone down overexposed frames by estimating
    a gamma value from current mean brightness and applying a LUT-based
    gamma correction (cheap and fast for video).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = estimate_brightness(gray)
    mean_brightness = max(mean_brightness, 1.0)  # avoid div by zero

    gamma = np.log(target_brightness / 255.0 + 1e-6) / np.log(mean_brightness / 255.0 + 1e-6)
    gamma = float(np.clip(gamma, 0.4, 2.5))

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(bgr, table)


def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization on the L channel of
    LAB color space. Restores contrast lost to fog/haze/overcast glare
    without blowing out color channels.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l_channel)
    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def auto_denoise(bgr: np.ndarray, noise_thresh: float = 12.0) -> np.ndarray:
    """
    Estimate noise level via the Laplacian variance of a downscaled patch;
    apply a light bilateral filter only if noise looks high (rain, sensor
    grain, low-light ISO noise). Skipped otherwise to preserve edge detail
    and keep runtime low.
    """
    small = cv2.resize(bgr, (bgr.shape[1] // 4 or 1, bgr.shape[0] // 4 or 1))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    noise_level = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())

    if noise_level > noise_thresh * 1000:
        return cv2.bilateralFilter(bgr, d=5, sigmaColor=50, sigmaSpace=50)
    return bgr


def preprocess_frame(bgr: np.ndarray, config) -> np.ndarray:
    """
    Full weather-adaptive preprocessing pipeline. Order matters: denoise
    first (raw sensor noise), then gamma (fix exposure), then CLAHE
    (restore local contrast for edge/color thresholding).
    """
    out = bgr

    if getattr(config, "enable_denoise_auto", True):
        out = auto_denoise(out)

    if getattr(config, "enable_gamma_auto", True):
        out = auto_gamma_correct(out)

    if getattr(config, "enable_clahe", True):
        out = apply_clahe(out)

    return out


def estimate_condition_confidence_penalty(bgr: np.ndarray) -> float:
    """
    Rough heuristic: very dark, very bright, or very low-contrast frames
    (fog/heavy rain/night) get a confidence penalty in [0, 0.5] that the
    agent subtracts from its detection confidence score. This lets the
    agent report "low confidence due to conditions" rather than silently
    producing a shaky lane estimate.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    brightness = estimate_brightness(gray)
    contrast = float(np.std(gray))

    penalty = 0.0
    if brightness < 60 or brightness > 200:
        penalty += 0.25
    if contrast < 35:
        penalty += 0.25

    return min(penalty, 0.5)

"""
Core lane detection algorithm (single-frame, stateless).

Pipeline:
    1. Weather-robust preprocessing (see preprocessing.py)
    2. Color threshold (white + yellow lane paint in HLS space)
    3. Gradient threshold (Sobel-x, catches edges color alone misses)
    4. Combine + region-of-interest mask
    5. Perspective warp to bird's-eye view
    6. Sliding-window search -> 2nd-order polynomial fit per lane line
    7. Curvature + vehicle offset computation
    8. Overlay drawing back onto the original frame

This is intentionally classical CV (no training data required) so it runs
out of the box. `LaneDetector.fit_lanes()` returns a `LaneResult` that the
higher-level `LaneDetectionAgent` (agent.py) consumes for temporal
smoothing and decision-making.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import cv2
import numpy as np

from .config import AgentConfig
from .preprocessing import preprocess_frame, estimate_condition_confidence_penalty


@dataclass
class LaneResult:
    success: bool
    left_fit: Optional[np.ndarray] = None       # polynomial coeffs (pixel space)
    right_fit: Optional[np.ndarray] = None
    left_fit_m: Optional[np.ndarray] = None      # polynomial coeffs (meter space)
    right_fit_m: Optional[np.ndarray] = None
    curvature_m: Optional[float] = None
    offset_m: Optional[float] = None
    confidence: float = 0.0
    left_pixel_count: int = 0
    right_pixel_count: int = 0
    binary_warped: Optional[np.ndarray] = None
    Minv: Optional[np.ndarray] = None


class LaneDetector:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    # ------------------------------------------------------------------ #
    # Thresholding
    # ------------------------------------------------------------------ #
    def color_threshold(self, bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

        white_mask = cv2.inRange(hls, np.array(cfg.white_hls_lower), np.array(cfg.white_hls_upper))
        yellow_mask = cv2.inRange(hls, np.array(cfg.yellow_hls_lower), np.array(cfg.yellow_hls_upper))

        return cv2.bitwise_or(white_mask, yellow_mask)

    def gradient_threshold(self, bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=cfg.sobel_kernel)
        abs_sobel = np.absolute(sobel_x)
        scaled = np.uint8(255 * abs_sobel / (np.max(abs_sobel) + 1e-6))

        mask = np.zeros_like(scaled)
        mask[(scaled >= cfg.sobel_thresh[0]) & (scaled <= cfg.sobel_thresh[1])] = 255
        return mask

    def region_of_interest(self, binary: np.ndarray) -> np.ndarray:
        h, w = binary.shape[:2]
        vertices = np.array([self.config.roi_vertices(w, h)], dtype=np.int32)
        mask = np.zeros_like(binary)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(binary, mask)

    def combined_binary(self, bgr: np.ndarray) -> np.ndarray:
        color_mask = self.color_threshold(bgr)
        grad_mask = self.gradient_threshold(bgr)
        combined = cv2.bitwise_or(color_mask, grad_mask)
        return self.region_of_interest(combined)

    # ------------------------------------------------------------------ #
    # Perspective transform
    # ------------------------------------------------------------------ #
    def warp_to_birdseye(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = binary.shape[:2]
        src, dst = self.config.perspective_points(w, h)
        src = np.float32(src)
        dst = np.float32(dst)

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(binary, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped, M, Minv

    # ------------------------------------------------------------------ #
    # Sliding window polynomial search
    # ------------------------------------------------------------------ #
    def sliding_window_search(self, binary_warped: np.ndarray):
        cfg = self.config
        h, w = binary_warped.shape[:2]

        histogram = np.sum(binary_warped[h // 2:, :], axis=0)
        midpoint = w // 2
        leftx_base = int(np.argmax(histogram[:midpoint])) if np.max(histogram[:midpoint]) > 0 else midpoint // 2
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint) if np.max(histogram[midpoint:]) > 0 else midpoint + midpoint // 2

        window_height = h // cfg.n_windows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds: List[np.ndarray] = []
        right_lane_inds: List[np.ndarray] = []

        for window in range(cfg.n_windows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_xleft_low = leftx_current - cfg.window_margin
            win_xleft_high = leftx_current + cfg.window_margin
            win_xright_low = rightx_current - cfg.window_margin
            win_xright_high = rightx_current + cfg.window_margin

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > cfg.min_pixels_recenter:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > cfg.min_pixels_recenter:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    @staticmethod
    def _safe_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 2, min_points: int = 30):
        if len(x) < min_points:
            return None
        try:
            return np.polyfit(y, x, deg)
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------ #
    # Full pipeline
    # ------------------------------------------------------------------ #
    def fit_lanes(self, bgr: np.ndarray) -> LaneResult:
        cfg = self.config
        h, w = bgr.shape[:2]

        pre = preprocess_frame(bgr, cfg)
        binary = self.combined_binary(pre)
        binary_warped, M, Minv = self.warp_to_birdseye(binary)

        leftx, lefty, rightx, righty = self.sliding_window_search(binary_warped)

        left_fit = self._safe_polyfit(leftx, lefty)
        right_fit = self._safe_polyfit(rightx, righty)

        if left_fit is None or right_fit is None:
            return LaneResult(success=False, confidence=0.0,
                               left_pixel_count=len(leftx), right_pixel_count=len(rightx),
                               binary_warped=binary_warped, Minv=Minv)

        left_fit_m = self._safe_polyfit(leftx * cfg.xm_per_pix, lefty * cfg.ym_per_pix) \
            if len(leftx) >= 30 else None
        right_fit_m = self._safe_polyfit(rightx * cfg.xm_per_pix, righty * cfg.ym_per_pix) \
            if len(rightx) >= 30 else None

        curvature_m = None
        offset_m = None
        if left_fit_m is not None and right_fit_m is not None:
            y_eval_m = h * cfg.ym_per_pix

            def curvature(fit_m):
                a, b, _ = fit_m
                return ((1 + (2 * a * y_eval_m + b) ** 2) ** 1.5) / (abs(2 * a) + 1e-6)

            curvature_m = float((curvature(left_fit_m) + curvature(right_fit_m)) / 2.0)

            left_x_bottom = np.polyval(left_fit, h - 1)
            right_x_bottom = np.polyval(right_fit, h - 1)
            lane_center_px = (left_x_bottom + right_x_bottom) / 2.0
            image_center_px = w / 2.0
            offset_m = float((image_center_px - lane_center_px) * cfg.xm_per_pix)

        # Confidence: based on pixel support for each lane line + weather penalty
        pixel_score = min(1.0, (len(leftx) + len(rightx)) / 4000.0)
        weather_penalty = estimate_condition_confidence_penalty(bgr)
        confidence = float(max(0.0, min(1.0, pixel_score - weather_penalty)))

        return LaneResult(
            success=True,
            left_fit=left_fit,
            right_fit=right_fit,
            left_fit_m=left_fit_m,
            right_fit_m=right_fit_m,
            curvature_m=curvature_m,
            offset_m=offset_m,
            confidence=confidence,
            left_pixel_count=len(leftx),
            right_pixel_count=len(rightx),
            binary_warped=binary_warped,
            Minv=Minv,
        )

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def draw_lane_overlay(self, bgr: np.ndarray, result: LaneResult) -> np.ndarray:
        if not result.success or result.Minv is None:
            return bgr

        h, w = bgr.shape[:2]
        ploty = np.linspace(0, h - 1, h)
        left_fitx = np.polyval(result.left_fit, ploty)
        right_fitx = np.polyval(result.right_fit, ploty)

        warp_zero = np.zeros((h, w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 200, 0))
        cv2.polylines(color_warp, np.int_([pts_left]), False, (0, 0, 255), 15)
        cv2.polylines(color_warp, np.int_([pts_right]), False, (255, 0, 0), 15)

        newwarp = cv2.warpPerspective(color_warp, result.Minv, (w, h))
        return cv2.addWeighted(bgr, 1.0, newwarp, 0.35, 0)

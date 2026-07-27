"""
Central configuration for the lane detection agent.

All tunable parameters live here so behavior can be adjusted without
touching the core algorithm code. Values are reasonable defaults for a
forward-facing dashcam-style camera at ~720p; adjust `roi_vertices_ratio`
and perspective points for your own camera setup.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class AgentConfig:
    # ---- Frame geometry ----
    # Region of interest as ratios of (width, height), so it scales to any
    # input resolution. Order: top-left, top-right, bottom-right, bottom-left.
    roi_vertices_ratio: Tuple[Tuple[float, float], ...] = (
        (0.44, 0.62),
        (0.56, 0.62),
        (0.95, 1.0),
        (0.05, 1.0),
    )

    # Perspective transform source/destination points (ratios of W,H) used
    # to warp the road into a bird's-eye view for polynomial lane fitting.
    perspective_src_ratio: Tuple[Tuple[float, float], ...] = (
        (0.44, 0.65),
        (0.56, 0.65),
        (0.90, 1.0),
        (0.10, 1.0),
    )
    perspective_dst_ratio: Tuple[Tuple[float, float], ...] = (
        (0.25, 0.0),
        (0.75, 0.0),
        (0.75, 1.0),
        (0.25, 1.0),
    )

    # ---- Color thresholds (HLS space) ----
    white_hls_lower: Tuple[int, int, int] = (0, 190, 0)
    white_hls_upper: Tuple[int, int, int] = (255, 255, 255)
    yellow_hls_lower: Tuple[int, int, int] = (15, 30, 90)
    yellow_hls_upper: Tuple[int, int, int] = (35, 204, 255)

    # ---- Gradient thresholds ----
    sobel_kernel: int = 5
    sobel_thresh: Tuple[int, int] = (20, 100)

    # ---- Sliding window search ----
    n_windows: int = 9
    window_margin: int = 80
    min_pixels_recenter: int = 50

    # ---- Temporal smoothing / agent decision logic ----
    smoothing_frames: int = 5          # rolling average window for polynomial coeffs
    max_lost_frames: int = 10          # frames tolerated with no detection before "LOST"
    departure_offset_thresh_m: float = 0.5   # meters from center -> WARNING
    departure_offset_critical_m: float = 0.9  # meters from center -> CRITICAL
    min_confidence: float = 0.35       # below this, treat frame as low-confidence

    # ---- Real-world scale (for curvature / offset in meters) ----
    # Defaults assume a ~3.7m lane width and ~30m of visible road in the
    # warped ROI, per common ADAS calibration references. Recalibrate for
    # your camera.
    xm_per_pix: float = 3.7 / 700
    ym_per_pix: float = 30 / 720

    # ---- Weather-adaptive preprocessing ----
    enable_clahe: bool = True
    enable_gamma_auto: bool = True
    enable_denoise_auto: bool = True

    def roi_vertices(self, width: int, height: int) -> List[Tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in self.roi_vertices_ratio]

    def perspective_points(self, width: int, height: int):
        src = [(x * width, y * height) for x, y in self.perspective_src_ratio]
        dst = [(x * width, y * height) for x, y in self.perspective_dst_ratio]
        return src, dst

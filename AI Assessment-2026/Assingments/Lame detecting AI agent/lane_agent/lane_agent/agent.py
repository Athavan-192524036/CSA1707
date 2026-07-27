"""
LaneDetectionAgent: the "agent" layer on top of the per-frame detector.

What makes this an agent rather than a bare CV pipeline:
    - It holds STATE across frames (a rolling buffer of recent polynomial
      fits), so it can smooth out per-frame jitter and survive a few bad
      frames (occlusion, glare, heavy rain) without losing the lane.
    - It makes DECISIONS, not just measurements: given the current lane
      offset and confidence, it emits a status (`OK`, `WARNING`,
      `CRITICAL`, `LOW_CONFIDENCE`, `LOST`) plus a human-readable message,
      which is what a downstream ADAS/warning system would consume.
    - It exposes a simple `process_frame()` call so it can be dropped into
      a video loop, a ROS node, or a REST endpoint unchanged.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Dict, Any
import numpy as np
import cv2

from .config import AgentConfig
from .detector import LaneDetector, LaneResult


@dataclass
class AgentStatus:
    state: str                 # "OK" | "WARNING" | "CRITICAL" | "LOW_CONFIDENCE" | "LOST"
    message: str
    offset_m: Optional[float]
    curvature_m: Optional[float]
    confidence: float
    frames_since_detection: int


class LaneDetectionAgent:
    """
    Usage:
        agent = LaneDetectionAgent()
        annotated_frame, status = agent.process_frame(frame_bgr)
        print(status.state, status.message)
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.detector = LaneDetector(self.config)

        self._left_fit_buffer: Deque[np.ndarray] = deque(maxlen=self.config.smoothing_frames)
        self._right_fit_buffer: Deque[np.ndarray] = deque(maxlen=self.config.smoothing_frames)
        self._frames_since_detection = 0
        self._last_result: Optional[LaneResult] = None
        self.frame_count = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear temporal state (call between unrelated video clips)."""
        self._left_fit_buffer.clear()
        self._right_fit_buffer.clear()
        self._frames_since_detection = 0
        self._last_result = None
        self.frame_count = 0

    # ------------------------------------------------------------------ #
    def _smoothed_fit(self, buffer: Deque[np.ndarray]) -> Optional[np.ndarray]:
        if not buffer:
            return None
        return np.mean(np.array(buffer), axis=0)

    def _decide(self, result: LaneResult) -> AgentStatus:
        cfg = self.config

        if not result.success:
            self._frames_since_detection += 1
            if self._frames_since_detection > cfg.max_lost_frames:
                return AgentStatus(
                    state="LOST",
                    message="Lane markings not detected for several frames. "
                            "Check camera view or road markings.",
                    offset_m=None, curvature_m=None, confidence=0.0,
                    frames_since_detection=self._frames_since_detection,
                )
            return AgentStatus(
                state="LOW_CONFIDENCE",
                message="Momentary loss of lane detection; using recent estimate.",
                offset_m=None, curvature_m=None, confidence=0.0,
                frames_since_detection=self._frames_since_detection,
            )

        self._frames_since_detection = 0

        if result.confidence < cfg.min_confidence:
            return AgentStatus(
                state="LOW_CONFIDENCE",
                message="Low detection confidence (likely poor visibility: "
                        "glare, fog, heavy rain, or worn markings).",
                offset_m=result.offset_m, curvature_m=result.curvature_m,
                confidence=result.confidence, frames_since_detection=0,
            )

        offset = result.offset_m or 0.0
        abs_offset = abs(offset)

        if abs_offset >= cfg.departure_offset_critical_m:
            side = "right" if offset < 0 else "left"
            return AgentStatus(
                state="CRITICAL",
                message=f"Lane departure imminent: drifting {side}, "
                        f"{abs_offset:.2f}m from center.",
                offset_m=offset, curvature_m=result.curvature_m,
                confidence=result.confidence, frames_since_detection=0,
            )

        if abs_offset >= cfg.departure_offset_thresh_m:
            side = "right" if offset < 0 else "left"
            return AgentStatus(
                state="WARNING",
                message=f"Drifting toward {side} lane edge "
                        f"({abs_offset:.2f}m from center).",
                offset_m=offset, curvature_m=result.curvature_m,
                confidence=result.confidence, frames_since_detection=0,
            )

        return AgentStatus(
            state="OK",
            message="Centered in lane.",
            offset_m=offset, curvature_m=result.curvature_m,
            confidence=result.confidence, frames_since_detection=0,
        )

    # ------------------------------------------------------------------ #
    def process_frame(self, bgr_frame: np.ndarray, draw: bool = True):
        """
        Run detection + temporal smoothing + decision logic on one frame.

        Returns:
            annotated_frame (np.ndarray): frame with lane overlay + HUD text
                                           (same as input if draw=False)
            status (AgentStatus): agent's decision for this frame
        """
        self.frame_count += 1
        result = self.detector.fit_lanes(bgr_frame)

        display_result = result
        if result.success:
            self._left_fit_buffer.append(result.left_fit)
            self._right_fit_buffer.append(result.right_fit)
        else:
            # Fall back to smoothed recent fit so the overlay doesn't
            # flicker away on a single bad frame.
            smoothed_left = self._smoothed_fit(self._left_fit_buffer)
            smoothed_right = self._smoothed_fit(self._right_fit_buffer)
            if smoothed_left is not None and smoothed_right is not None and self._last_result is not None:
                display_result = LaneResult(
                    success=True,
                    left_fit=smoothed_left,
                    right_fit=smoothed_right,
                    left_fit_m=self._last_result.left_fit_m,
                    right_fit_m=self._last_result.right_fit_m,
                    curvature_m=self._last_result.curvature_m,
                    offset_m=self._last_result.offset_m,
                    confidence=0.0,
                    Minv=result.Minv,
                )

        status = self._decide(result)
        if result.success:
            self._last_result = result

        annotated = bgr_frame
        if draw:
            annotated = self.detector.draw_lane_overlay(bgr_frame, display_result)
            annotated = self._draw_hud(annotated, status)

        return annotated, status

    # ------------------------------------------------------------------ #
    def _draw_hud(self, frame: np.ndarray, status: AgentStatus) -> np.ndarray:
        colors = {
            "OK": (0, 200, 0),
            "WARNING": (0, 165, 255),
            "CRITICAL": (0, 0, 255),
            "LOW_CONFIDENCE": (0, 255, 255),
            "LOST": (128, 128, 128),
        }
        color = colors.get(status.state, (255, 255, 255))

        out = frame.copy()
        cv2.rectangle(out, (0, 0), (420, 95), (0, 0, 0), thickness=-1)
        out = cv2.addWeighted(out, 0.45, frame, 0.55, 0)

        cv2.putText(out, f"STATUS: {status.state}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        offset_txt = f"{status.offset_m:.2f} m" if status.offset_m is not None else "N/A"
        curve_txt = f"{status.curvature_m:.0f} m" if status.curvature_m else "N/A"
        cv2.putText(out, f"Offset: {offset_txt}   Curvature: {curve_txt}", (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"Confidence: {status.confidence:.2f}", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return out

    # ------------------------------------------------------------------ #
    def to_dict(self, status: AgentStatus) -> Dict[str, Any]:
        """JSON-serializable telemetry, e.g. for logging or a REST API."""
        return {
            "frame": self.frame_count,
            "state": status.state,
            "message": status.message,
            "offset_m": status.offset_m,
            "curvature_m": status.curvature_m,
            "confidence": status.confidence,
            "frames_since_detection": status.frames_since_detection,
        }

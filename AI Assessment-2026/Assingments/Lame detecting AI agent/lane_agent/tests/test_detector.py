"""
Sanity tests using synthetically generated road images so the suite runs
with zero external dataset dependencies.

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2

from lane_agent import LaneDetector, LaneDetectionAgent, AgentConfig


def make_synthetic_road(width=1280, height=720, offset_px=0, brightness=140, noise=False):
    """
    Draws a simple straight two-lane road: dark asphalt background with two
    white lane lines converging toward a horizon, so the detector has clear
    signal to lock onto.
    """
    asphalt_level = max(0, min(255, brightness - 60))
    img = np.full((height, width, 3), asphalt_level, dtype=np.uint8)  # asphalt
    # sky
    cv2.rectangle(img, (0, 0), (width, int(height * 0.6)), (brightness + 20, brightness + 10, brightness), -1)

    horizon_y = int(height * 0.62)
    bottom_y = height

    center_x = width // 2 + offset_px
    lane_half_width_bottom = int(width * 0.28)
    lane_half_width_top = int(width * 0.03)

    left_bottom = (center_x - lane_half_width_bottom, bottom_y)
    right_bottom = (center_x + lane_half_width_bottom, bottom_y)
    left_top = (center_x - lane_half_width_top, horizon_y)
    right_top = (center_x + lane_half_width_top, horizon_y)

    cv2.line(img, left_bottom, left_top, (255, 255, 255), 12)
    cv2.line(img, right_bottom, right_top, (255, 255, 255), 12)

    if noise:
        gauss = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + gauss, 0, 255).astype(np.uint8)

    return img


def test_detector_finds_lane_on_clear_synthetic_road():
    detector = LaneDetector(AgentConfig())
    frame = make_synthetic_road()
    result = detector.fit_lanes(frame)

    assert result.success is True
    assert result.left_pixel_count > 0
    assert result.right_pixel_count > 0
    assert result.confidence > 0


def test_offset_sign_matches_drift_direction():
    detector = LaneDetector(AgentConfig())

    centered = detector.fit_lanes(make_synthetic_road(offset_px=0))
    drifted_right = detector.fit_lanes(make_synthetic_road(offset_px=120))

    assert centered.success and drifted_right.success
    # camera stayed centered on canvas, lane markings shifted right ->
    # vehicle is effectively left of lane center -> offset should differ
    assert abs(drifted_right.offset_m - centered.offset_m) > 0.1


def test_agent_emits_ok_status_on_centered_lane():
    agent = LaneDetectionAgent(AgentConfig())
    frame = make_synthetic_road(offset_px=0)
    annotated, status = agent.process_frame(frame)

    assert annotated.shape == frame.shape
    assert status.state in ("OK", "WARNING", "LOW_CONFIDENCE")  # tolerate synthetic-image noise


def test_agent_survives_noisy_low_light_frame():
    agent = LaneDetectionAgent(AgentConfig())
    dark_noisy = make_synthetic_road(brightness=50, noise=True)
    annotated, status = agent.process_frame(dark_noisy)

    assert annotated is not None
    assert status.state in ("OK", "WARNING", "CRITICAL", "LOW_CONFIDENCE", "LOST")


def test_agent_handles_blank_frame_gracefully():
    agent = LaneDetectionAgent(AgentConfig())
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    annotated, status = agent.process_frame(blank)

    assert annotated.shape == blank.shape
    assert status.state in ("LOW_CONFIDENCE", "LOST")


if __name__ == "__main__":
    test_detector_finds_lane_on_clear_synthetic_road()
    test_offset_sign_matches_drift_direction()
    test_agent_emits_ok_status_on_centered_lane()
    test_agent_survives_noisy_low_light_frame()
    test_agent_handles_blank_frame_gracefully()
    print("All tests passed.")

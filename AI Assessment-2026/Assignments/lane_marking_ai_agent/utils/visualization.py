"""
Visualization Utilities for Lane Detection
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def visualize_lanes(
    image: np.ndarray,
    lanes: List[Dict],
    weather_info: Optional[Dict] = None,
    timing_info: Optional[Dict] = None,
    lane_lost: bool = False
) -> np.ndarray:
    """Draw detected lanes on image with rich annotations."""
    vis = image.copy()
    h, w = vis.shape[:2]
    
    LANE_COLORS = {
        "solid_white": (255, 255, 255),
        "solid_yellow": (0, 255, 255),
        "dashed_white": (200, 200, 200),
        "dashed_yellow": (0, 200, 200),
        "double_solid": (0, 0, 255),
        "double_dashed": (255, 0, 255),
        "botts_dots": (0, 255, 0),
        "unknown": (128, 128, 128)
    }
    
    for lane in lanes:
        points = lane.get("points", [])
        if len(points) < 2:
            continue
            
        color = LANE_COLORS.get(lane["lane_type"], (0, 255, 0))
        confidence = lane.get("confidence", 0.0)
        thickness = max(2, int(4 * confidence))
        
        for i in range(len(points) - 1):
            cv2.line(vis, tuple(points[i]), tuple(points[i+1]), color, thickness)
        
        for pt in points[::5]:
            cv2.circle(vis, tuple(pt), 3, color, -1)
        
        if points:
            label = f"{lane['lane_id']}: {lane['lane_type']} ({confidence:.2f})"
            label_pos = (points[0][0], max(30, points[0][1] - 10))
            cv2.putText(vis, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if weather_info:
        panel_h = 80
        overlay = vis.copy()
        cv2.rectangle(overlay, (5, 5), (300, panel_h), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        weather_text = f"Weather: {weather_info['condition'].upper()}"
        cv2.putText(vis, weather_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        conf_text = f"Confidence: {weather_info['confidence']:.2%}"
        cv2.putText(vis, conf_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    if timing_info:
        fps = 1000.0 / timing_info['total_ms'] if timing_info['total_ms'] > 0 else 0
        time_text = f"{timing_info['total_ms']:.1f}ms ({fps:.1f} FPS)"
        text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(vis, time_text, (w - text_size[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if lane_lost:
        warning_text = "⚠ LANE LOST"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        cv2.rectangle(vis, (center_x - 20, center_y - 40), (center_x + text_size[0] + 20, center_y + 20), (0, 0, 255), -1)
        cv2.putText(vis, warning_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    draw_ego_lane_highlight(vis, lanes)
    return vis


def draw_ego_lane_highlight(image: np.ndarray, lanes: List[Dict]):
    """Highlight the ego-lane area between left and right lanes."""
    if len(lanes) < 2:
        return
    
    left_lane = None
    right_lane = None
    
    for lane in lanes:
        lane_id = lane.get("lane_id", -1)
        if lane_id == 1:
            left_lane = lane
        elif lane_id == 2:
            right_lane = lane
    
    if left_lane and right_lane and left_lane.get("points") and right_lane.get("points"):
        left_pts = np.array(left_lane["points"])
        right_pts = np.array(right_lane["points"])
        
        min_len = min(len(left_pts), len(right_pts))
        left_pts = left_pts[:min_len]
        right_pts = right_pts[:min_len]
        
        polygon = np.vstack([left_pts, right_pts[::-1]]).astype(np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
"""
Lane Detection AI Agent
========================

A weather-robust lane marking detection agent built on classical computer
vision (OpenCV). Designed to be modular so the core detector can later be
swapped for a deep-learning segmentation model (e.g. ENet / SCNN / U-Net)
trained on datasets like TuSimple or CULane, without changing the agent
interface.

Main entry points:
    - lane_agent.agent.LaneDetectionAgent : stateful agent, use this
    - lane_agent.detector.LaneDetector    : stateless single-frame detector
    - lane_agent.preprocessing            : weather-robust preprocessing
"""

from .agent import LaneDetectionAgent
from .detector import LaneDetector
from .config import AgentConfig

__all__ = ["LaneDetectionAgent", "LaneDetector", "AgentConfig"]
__version__ = "1.0.0"

#!/usr/bin/env python3
"""
Run the lane detection agent on a single image.

Usage:
    python run_image.py --input path/to/road.jpg --output out.jpg
    python run_image.py --input path/to/road.jpg --output out.jpg --debug
"""

import argparse
import sys

from lane_agent import LaneDetectionAgent, AgentConfig
from lane_agent.utils import read_image, save_image, stack_debug_view


def main():
    parser = argparse.ArgumentParser(description="Lane detection agent - single image")
    parser.add_argument("--input", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", default="output.jpg", help="Path to save annotated image")
    parser.add_argument("--debug", action="store_true", help="Also save a debug view with the bird's-eye binary mask")
    args = parser.parse_args()

    frame = read_image(args.input)

    agent = LaneDetectionAgent(AgentConfig())
    annotated, status = agent.process_frame(frame)

    save_image(args.output, annotated)
    print(f"Status: {status.state}")
    print(f"Message: {status.message}")
    print(f"Offset from center: {status.offset_m}")
    print(f"Curvature: {status.curvature_m}")
    print(f"Confidence: {status.confidence:.2f}")
    print(f"Saved annotated image to: {args.output}")

    if args.debug:
        result = agent.detector.fit_lanes(frame)
        if result.binary_warped is not None:
            debug_path = args.output.rsplit(".", 1)[0] + "_debug.jpg"
            debug_view = stack_debug_view(annotated, result.binary_warped)
            save_image(debug_path, debug_view)
            print(f"Saved debug view to: {debug_path}")


if __name__ == "__main__":
    sys.exit(main())

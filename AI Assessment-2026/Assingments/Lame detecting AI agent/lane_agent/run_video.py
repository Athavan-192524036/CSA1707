#!/usr/bin/env python3
"""
Run the lane detection agent on a video file or webcam stream.

Usage:
    python run_video.py --input path/to/drive.mp4 --output out.mp4
    python run_video.py --input 0                     # webcam, live preview window
    python run_video.py --input path/to/drive.mp4 --output out.mp4 --log telemetry.jsonl
"""

import argparse
import json
import sys

import cv2

from lane_agent import LaneDetectionAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(description="Lane detection agent - video/webcam")
    parser.add_argument("--input", "-i", required=True,
                         help="Path to input video, or '0' for default webcam")
    parser.add_argument("--output", "-o", default=None,
                         help="Path to save annotated output video (mp4). Omit to just preview.")
    parser.add_argument("--log", default=None,
                         help="Optional path to write per-frame telemetry as JSON Lines")
    parser.add_argument("--no-preview", action="store_true",
                         help="Don't open a live preview window (useful on headless servers)")
    args = parser.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source '{args.input}'", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    log_file = open(args.log, "w") if args.log else None

    agent = LaneDetectionAgent(AgentConfig())

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            annotated, status = agent.process_frame(frame)

            if writer is not None:
                writer.write(annotated)

            if log_file is not None:
                log_file.write(json.dumps(agent.to_dict(status)) + "\n")

            if not args.no_preview:
                cv2.imshow("Lane Detection Agent", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_idx % 30 == 0:
                print(f"frame {frame_idx}: {status.state} - {status.message}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames.")
    if args.output:
        print(f"Saved annotated video to: {args.output}")
    if args.log:
        print(f"Saved telemetry to: {args.log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

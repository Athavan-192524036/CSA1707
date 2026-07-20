"""
Inference Script for Weather-Aware Lane Detection
"""
import argparse
import time
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from models import build_model
from models.lane_head import LanePostProcessor


class LaneDetector:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda", half_precision: bool = False):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half_precision = half_precision and self.device.type == "cuda"
        
        self.model = build_model(config=self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        if self.half_precision:
            self.model = self.model.half()
        
        self.input_height = self.config["model"]["input_height"]
        self.input_width = self.config["model"]["input_width"]
        
        self.post_processor = LanePostProcessor(
            griding_num=self.config["model"]["griding_num"],
            cls_num_per_lane=self.config["model"]["cls_num_per_lane"],
            num_lanes=self.config["model"]["num_lanes"],
            num_classes=self.config["model"]["num_classes"],
            input_height=self.input_height,
            input_width=self.input_width,
            conf_threshold=self.config["inference"]["confidence_threshold"]
        )
        
        from models.lane_head import LaneTracker
        self.tracker = LaneTracker(
            num_lanes=self.config["model"]["num_lanes"],
            max_lost=self.config["inference"]["lane_lost_frames"]
        )
        self._warmup()
    
    def _warmup(self, num_iterations: int = 3):
        dummy_input = torch.randn(1, 3, self.input_height, self.input_width).to(self.device)
        if self.half_precision:
            dummy_input = dummy_input.half()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        print("Model warmed up.")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.shape[2] == 3 and image[0, 0, 0] > image[0, 0, 2]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(self.config["preprocessing"]["mean"])) / np.array(self.config["preprocessing"]["std"])
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.half_precision:
            tensor = tensor.half()
        return tensor
    
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> Dict:
        orig_h, orig_w = image.shape[:2]
        t0 = time.time()
        input_tensor = self.preprocess(image)
        preprocess_time = (time.time() - t0) * 1000
        
        t0 = time.time()
        outputs = self.model(input_tensor)
        inference_time = (time.time() - t0) * 1000
        
        t0 = time.time()
        detections = self.post_processor.process(outputs)
        if len(detections) == 1:
            tracked_lanes = self.tracker.update(detections[0])
        else:
            tracked_lanes = detections[0] if detections else []
        postprocess_time = (time.time() - t0) * 1000
        
        weather_probs = F.softmax(outputs["weather_logits"], dim=1)[0]
        weather_classes = ["clear", "rain", "fog", "snow", "night", "glare"]
        weather_idx = torch.argmax(weather_probs).item()
        weather_conf = weather_probs[weather_idx].item()
        
        for lane in tracked_lanes:
            if "points" in lane:
                lane["points"] = [
                    (int(x * orig_w / self.input_width), int(y * orig_h / self.input_height))
                    for x, y in lane["points"]
                ]
        
        total_time = preprocess_time + inference_time + postprocess_time
        
        return {
            "lanes": tracked_lanes,
            "weather": {
                "condition": weather_classes[weather_idx],
                "confidence": weather_conf,
                "all_probs": {cls: prob.item() for cls, prob in zip(weather_classes, weather_probs)}
            },
            "timing": {
                "preprocess_ms": preprocess_time,
                "inference_ms": inference_time,
                "postprocess_ms": postprocess_time,
                "total_ms": total_time
            },
            "lane_lost": self.tracker.is_lane_lost()
        }
    
    def visualize(self, image: np.ndarray, results: Dict, save_path: Optional[str] = None) -> np.ndarray:
        from utils.visualization import visualize_lanes
        vis_image = visualize_lanes(
            image, results["lanes"],
            weather_info=results.get("weather"),
            timing_info=results.get("timing"),
            lane_lost=results.get("lane_lost", False)
        )
        if save_path:
            cv2.imwrite(save_path, vis_image)
        return vis_image


def process_image(detector: LaneDetector, image_path: str, output_path: Optional[str] = None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    results = detector.detect(image)
    print(f"\nDetected {len(results['lanes'])} lanes")
    print(f"Weather: {results['weather']['condition']} ({results['weather']['confidence']:.2f})")
    print(f"Latency: {results['timing']['total_ms']:.1f}ms")
    for lane in results["lanes"]:
        print(f"  Lane {lane['lane_id']}: {lane['lane_type']} (conf: {lane['confidence']:.2f})")
    vis = detector.visualize(image, results, output_path)
    cv2.imshow("Lane Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector: LaneDetector, video_path: str, output_path: Optional[str] = None, display: bool = True):
    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count, total_time = 0, 0
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect(frame)
        total_time += results["timing"]["total_ms"]
        frame_count += 1
        vis = detector.visualize(frame, results)
        if writer:
            writer.write(vis)
        if display:
            cv2.imshow("Lane Detection", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        avg_latency = total_time / frame_count
        avg_fps = 1000 / avg_latency
        print(f"\nProcessed {frame_count} frames")
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")


def benchmark(detector: LaneDetector, num_runs: int = 100):
    dummy_image = np.random.randint(0, 255, (590, 1640, 3), dtype=np.uint8)
    for _ in range(10):
        detector.detect(dummy_image)
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        detector.detect(dummy_image)
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f"\nBenchmark Results ({num_runs} runs):")
    print(f"  Mean latency: {times.mean():.2f}ms")
    print(f"  Std dev: {times.std():.2f}ms")
    print(f"  Min: {times.min():.2f}ms")
    print(f"  Max: {times.max():.2f}ms")
    print(f"  P50: {np.percentile(times, 50):.2f}ms")
    print(f"  P95: {np.percentile(times, 95):.2f}ms")
    print(f"  P99: {np.percentile(times, 99):.2f}ms")
    print(f"  Mean FPS: {1000/times.mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Lane Detection Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    
    print("Loading model...")
    detector = LaneDetector(config_path=args.config, checkpoint_path=args.checkpoint, device=args.device, half_precision=args.half)
    
    if args.benchmark:
        benchmark(detector)
        return
    
    if args.input.lower() == "camera" or args.input.isdigit():
        video_path = int(args.input) if args.input.isdigit() else 0
        process_video(detector, video_path, args.output, not args.no_display)
    elif args.input.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(detector, args.input, args.output, not args.no_display)
    else:
        process_image(detector, args.input, args.output)


if __name__ == "__main__":
    main()
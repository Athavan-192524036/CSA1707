"""
Data Loading and Augmentation for Lane Detection
Supports TuSimple, CULane, LLAMAS, BDD100K, ACDC datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import random


class LaneDataset(Dataset):
    """
    Unified dataset for lane detection.
    Supports multiple formats and weather conditions.
    """
    def __init__(
        self,
        data_root: str,
        dataset_type: str = "tusimple",
        split: str = "train",
        input_size: Tuple[int, int] = (288, 800),
        griding_num: int = 100,
        cls_num_per_lane: int = 56,
        num_lanes: int = 4,
        augment: bool = True,
        weather_augment: bool = True
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.split = split
        self.input_height, self.input_width = input_size
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.num_lanes = num_lanes
        self.augment = augment
        self.weather_augment = weather_augment
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Weather condition mapping
        self.weather_map = {
            "clear": 0, "rain": 1, "fog": 2, 
            "snow": 3, "night": 4, "glare": 5
        }
        
        # Lane type mapping
        self.lane_type_map = {
            "solid_white": 0, "solid_yellow": 1,
            "dashed_white": 2, "dashed_yellow": 3,
            "double_solid": 4, "double_dashed": 5,
            "botts_dots": 6, "unknown": 7
        }
        
        # Basic transforms
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations based on type."""
        samples = []
        
        if self.dataset_type == "tusimple":
            samples = self._load_tusimple()
        elif self.dataset_type == "culane":
            samples = self._load_culane()
        elif self.dataset_type == "llamas":
            samples = self._load_llamas()
        elif self.dataset_type == "bdd100k":
            samples = self._load_bdd100k()
        elif self.dataset_type == "acdc":
            samples = self._load_acdc()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        return samples
    
    def _load_tusimple(self) -> List[Dict]:
        """Load TuSimple dataset annotations."""
        samples = []
        label_file = os.path.join(self.data_root, self.split, "label_data.json")
        
        if not os.path.exists(label_file):
            return samples
        
        with open(label_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                samples.append({
                    "image_path": os.path.join(self.data_root, data["raw_file"]),
                    "lanes": data["lanes"],
                    "h_samples": data["h_samples"],
                    "weather": "clear"  # TuSimple is mostly clear
                })
        
        return samples
    
    def _load_culane(self) -> List[Dict]:
        """Load CULane dataset annotations."""
        samples = []
        list_file = os.path.join(self.data_root, self.split, "list.txt")
        
        if not os.path.exists(list_file):
            return samples
        
        with open(list_file, 'r') as f:
            for line in f:
                img_path = line.strip()
                label_path = img_path.replace(".jpg", ".lines.txt")
                
                samples.append({
                    "image_path": os.path.join(self.data_root, img_path),
                    "label_path": os.path.join(self.data_root, label_path),
                    "weather": self._infer_weather_culane(img_path)
                })
        
        return samples
    
    def _load_llamas(self) -> List[Dict]:
        """Load LLAMAS dataset annotations."""
        samples = []
        # LLAMAS uses JSON annotations
        return samples
    
    def _load_bdd100k(self) -> List[Dict]:
        """Load BDD100K dataset annotations."""
        samples = []
        label_file = os.path.join(self.data_root, f"labels/lane_{self.split}.json")
        
        if not os.path.exists(label_file):
            return samples
        
        with open(label_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            samples.append({
                "image_path": os.path.join(self.data_root, "images", item["name"]),
                "labels": item.get("labels", []),
                "weather": item.get("attributes", {}).get("weather", "clear"),
                "timeofday": item.get("attributes", {}).get("timeofday", "daytime")
            })
        
        return samples
    
    def _load_acdc(self) -> List[Dict]:
        """Load ACDC (Adverse Conditions) dataset annotations."""
        samples = []
        # ACDC has fog, rain, night, snow splits
        for weather in ["fog", "rain", "night", "snow"]:
            weather_dir = os.path.join(self.data_root, weather, self.split)
            if not os.path.exists(weather_dir):
                continue
            
            for img_name in os.listdir(os.path.join(weather_dir, "rgb_images")):
                samples.append({
                    "image_path": os.path.join(weather_dir, "rgb_images", img_name),
                    "label_path": os.path.join(weather_dir, "gt_labels", img_name.replace(".png", "_gt.json")),
                    "weather": weather
                })
        
        return samples
    
    def _infer_weather_culane(self, path: str) -> str:
        """Infer weather condition from CULane path."""
        if "night" in path.lower():
            return "night"
        elif "shadow" in path.lower():
            return "clear"  # Shadow is still clear weather
        return "clear"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        
        # Load lane annotations
        lanes = self._load_lane_annotations(sample, orig_h, orig_w)
        
        # Get weather condition
        weather = sample.get("weather", "clear")
        if sample.get("timeofday") == "night":
            weather = "night"
        weather_label = self.weather_map.get(weather, 0)
        
        # Apply augmentations
        if self.augment:
            image, lanes = self._augment(image, lanes)
        
        if self.weather_augment:
            image = self._apply_weather_augmentation(image, weather)
        
        # Resize and normalize
        image = TF.resize(image, (self.input_height, self.input_width))
        image = TF.to_tensor(image)
        image = self.normalize(image)
        
        # Generate training targets
        targets = self._generate_targets(lanes, orig_h, orig_w)
        targets["weather_targets"] = torch.tensor(weather_label, dtype=torch.long)
        
        return {
            "image": image,
            **targets
        }
    
    def _load_lane_annotations(
        self, 
        sample: Dict, 
        orig_h: int, 
        orig_w: int
    ) -> List[Dict]:
        """Load and parse lane annotations."""
        lanes = []
        
        if self.dataset_type == "tusimple":
            for lane_points in sample.get("lanes", []):
                points = []
                for x, y in zip(lane_points, sample.get("h_samples", [])):
                    if x >= 0:
                        points.append((x, y))
                if points:
                    lanes.append({
                        "points": points,
                        "type": "unknown"
                    })
        
        elif self.dataset_type == "culane":
            label_path = sample.get("label_path")
            if label_path and os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        coords = list(map(float, line.strip().split()))
                        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        if points:
                            lanes.append({"points": points, "type": "unknown"})
        
        elif self.dataset_type == "bdd100k":
            for label in sample.get("labels", []):
                if label.get("category") == "lane":
                    poly = label.get("poly2d", [])
                    points = [(p[0], p[1]) for p in poly]
                    lane_type = label.get("attributes", {}).get("laneType", "unknown")
                    lanes.append({"points": points, "type": lane_type})
        
        return lanes
    
    def _augment(
        self, 
        image: Image.Image, 
        lanes: List[Dict]
    ) -> Tuple[Image.Image, List[Dict]]:
        """Apply geometric and photometric augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            w = image.width
            for lane in lanes:
                lane["points"] = [(w - x, y) for x, y in lane["points"]]
        
        # Color jitter
        color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        image = color_jitter(image)
        
        # Random perspective warp
        if random.random() > 0.5:
            image, lanes = self._perspective_warp(image, lanes)
        
        return image, lanes
    
    def _perspective_warp(
        self, 
        image: Image.Image, 
        lanes: List[Dict]
    ) -> Tuple[Image.Image, List[Dict]]:
        """Apply random perspective transformation."""
        w, h = image.size
        
        # Random perspective distortion
        margin = int(0.1 * w)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)]
        ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_np = np.array(image)
        warped = cv2.warpPerspective(img_np, matrix, (w, h))
        
        # Transform lane points
        for lane in lanes:
            points = np.array(lane["points"], dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points, matrix)
            lane["points"] = [(int(p[0][0]), int(p[0][1])) for p in transformed]
        
        return Image.fromarray(warped), lanes
    
    def _apply_weather_augmentation(self, image: Image.Image, current_weather: str) -> Image.Image:
        """Apply weather simulation augmentations."""
        img_np = np.array(image).astype(np.float32) / 255.0
        
        # Simulate different weather conditions
        weather_types = ["rain", "fog", "snow", "night", "glare"]
        
        if random.random() > 0.5:
            sim_weather = random.choice(weather_types)
            
            if sim_weather == "rain":
                img_np = self._add_rain(img_np)
            elif sim_weather == "fog":
                img_np = self._add_fog(img_np)
            elif sim_weather == "snow":
                img_np = self._add_snow(img_np)
            elif sim_weather == "night":
                img_np = self._add_night(img_np)
            elif sim_weather == "glare":
                img_np = self._add_glare(img_np)
        
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)
    
    def _add_rain(self, img: np.ndarray) -> np.ndarray:
        """Add rain streaks to image."""
        h, w = img.shape[:2]
        rain_layer = np.zeros_like(img)
        
        num_streaks = random.randint(500, 1500)
        for _ in range(num_streaks):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            length = random.randint(10, 30)
            thickness = random.randint(1, 2)
            
            cv2.line(rain_layer, (x, y), (x + random.randint(-2, 2), y + length), 
                    (0.8, 0.8, 0.8), thickness)
        
        # Motion blur for rain
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        return np.clip(img + rain_layer * 0.3, 0, 1)
    
    def _add_fog(self, img: np.ndarray) -> np.ndarray:
        """Add fog/haze effect."""
        h, w = img.shape[:2]
        
        # Atmospheric scattering model
        fog_color = np.array([0.8, 0.8, 0.8])
        transmission = np.exp(-np.linspace(0, 0.5, h).reshape(-1, 1, 1))
        transmission = np.repeat(transmission, w, axis=1)
        
        foggy = img * transmission + fog_color * (1 - transmission)
        return np.clip(foggy, 0, 1)
    
    def _add_snow(self, img: np.ndarray) -> np.ndarray:
        """Add snow particles."""
        h, w = img.shape[:2]
        snow_layer = np.zeros_like(img)
        
        num_particles = random.randint(200, 800)
        for _ in range(num_particles):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            size = random.randint(1, 3)
            cv2.circle(snow_layer, (x, y), size, (1.0, 1.0, 1.0), -1)
        
        return np.clip(img + snow_layer * 0.4, 0, 1)
    
    def _add_night(self, img: np.ndarray) -> np.ndarray:
        """Simulate night/low-light conditions."""
        # Reduce brightness
        dark = img * random.uniform(0.1, 0.4)
        
        # Add noise
        noise = np.random.normal(0, 0.02, img.shape)
        dark = dark + noise
        
        # Simulate headlight bloom
        h, w = img.shape[:2]
        bloom = np.zeros_like(img)
        cv2.circle(bloom, (w // 2, h // 2), w // 4, (0.3, 0.3, 0.2), -1)
        bloom = cv2.GaussianBlur(bloom, (51, 51), 0)
        
        return np.clip(dark + bloom, 0, 1)
    
    def _add_glare(self, img: np.ndarray) -> np.ndarray:
        """Add sun glare effect."""
        h, w = img.shape[:2]
        
        # Bright region
        glare = np.zeros_like(img)
        glare_center = (random.randint(0, w), random.randint(0, h // 2))
        cv2.circle(glare, glare_center, w // 3, (1.0, 1.0, 0.9), -1)
        glare = cv2.GaussianBlur(glare, (101, 101), 0)
        
        # Overexpose
        return np.clip(img + glare * 0.5, 0, 1)
    
    def _generate_targets(
        self, 
        lanes: List[Dict], 
        orig_h: int, 
        orig_w: int
    ) -> Dict[str, torch.Tensor]:
        """
        Generate training targets from lane annotations.
        
        Returns:
            lane_targets: [num_lanes, cls_num_per_lane] grid indices (-1 for no lane)
            lane_positions: [num_lanes, cls_num_per_lane] normalized positions
            lane_type_targets: [num_lanes] lane type indices
        """
        # Row anchors
        row_anchors = [
            int(orig_h * (0.42 + 0.58 * i / (self.cls_num_per_lane - 1)))
            for i in range(self.cls_num_per_lane)
        ]
        
        # Initialize targets
        lane_targets = torch.full((self.num_lanes, self.cls_num_per_lane), -1, dtype=torch.long)
        lane_positions = torch.full((self.num_lanes, self.cls_num_per_lane), -1.0, dtype=torch.float32)
        lane_type_targets = torch.zeros(self.num_lanes, dtype=torch.long)
        
        # Sort lanes by average x position (left to right)
        lanes_sorted = sorted(lanes, key=lambda l: np.mean([p[0] for p in l["points"]]) if l["points"] else 0)
        
        for lane_idx, lane in enumerate(lanes_sorted[:self.num_lanes]):
            points = lane["points"]
            if not points:
                continue
            
            # Fit polynomial to lane points
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            
            if len(x) < 2:
                continue
            
            # Fit x as function of y
            coeffs = np.polyfit(y, x, min(3, len(x) - 1))
            
            # Evaluate at row anchors
            for i, y_anchor in enumerate(row_anchors):
                if y_anchor < y.min() or y_anchor > y.max():
                    continue
                
                x_pred = np.polyval(coeffs, y_anchor)
                x_pred = np.clip(x_pred, 0, orig_w - 1)
                
                # Map to grid
                grid_idx = int(x_pred / orig_w * self.griding_num)
                grid_idx = np.clip(grid_idx, 0, self.griding_num - 1)
                
                lane_targets[lane_idx, i] = grid_idx
                lane_positions[lane_idx, i] = x_pred / orig_w
            
            # Lane type
            lane_type = lane.get("type", "unknown")
            lane_type_targets[lane_idx] = self.lane_type_map.get(lane_type, 7)
        
        return {
            "lane_targets": lane_targets,
            "lane_positions": lane_positions,
            "lane_type_targets": lane_type_targets
        }


def build_dataloader(
    data_root: str,
    dataset_type: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    config: Dict = None
) -> DataLoader:
    """Build data loader for training/evaluation."""
    
    dataset = LaneDataset(
        data_root=data_root,
        dataset_type=dataset_type,
        split=split,
        input_size=(config["model"]["input_height"], config["model"]["input_width"]),
        griding_num=config["model"]["griding_num"],
        cls_num_per_lane=config["model"]["cls_num_per_lane"],
        num_lanes=config["model"]["num_lanes"],
        augment=(split == "train"),
        weather_augment=(split == "train" and config["training"]["augmentation"]["weather_simulation"])
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
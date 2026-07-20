"""
Lane Detection Head
Hybrid approach: Row-wise classification with instance embedding.
Optimized for real-time performance on edge devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class LaneDetectionHead(nn.Module):
    """
    Lane detection head using row-wise classification with instance embedding.
    Based on UltraFast-LaneNet architecture with enhancements.
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_lanes: int = 4,
        num_classes: int = 8,  # Lane types
        griding_num: int = 100,
        cls_num_per_lane: int = 56,
        use_instance_embedding: bool = True,
        embedding_dim: int = 4
    ):
        super().__init__()
        self.num_lanes = num_lanes
        self.num_classes = num_classes
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.use_instance_embedding = use_instance_embedding
        self.embedding_dim = embedding_dim
        
        # Shared feature processing
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Row-wise classification head
        # For each lane, predict presence at each row anchor
        self.lane_cls = nn.Sequential(
            nn.Linear(in_channels // 2 * cls_num_per_lane, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, num_lanes * cls_num_per_lane * (griding_num + 1))
            # +1 for "no lane" class
        )
        
        # Lane type classification head
        self.lane_type_cls = nn.Sequential(
            nn.Linear(in_channels // 2 * cls_num_per_lane, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_lanes * num_classes)
        )
        
        # Instance embedding head (for clustering lanes)
        if use_instance_embedding:
            self.embedding_head = nn.Sequential(
                nn.Conv2d(in_channels // 2, embedding_dim, 1),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(in_channels // 2 * cls_num_per_lane, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_lanes)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, C, H, W] from FPN
        Returns:
            Dictionary containing:
                - lane_logits: [B, num_lanes, cls_num_per_lane, griding_num+1]
                - lane_types: [B, num_lanes, num_classes]
                - confidences: [B, num_lanes]
                - embeddings: [B, embedding_dim, H, W] (optional)
        """
        B, C, H, W = features.shape
        
        # Process features
        x = self.feature_conv(features)
        
        # Global average pooling for classification
        x_pooled = F.adaptive_avg_pool2d(x, (self.cls_num_per_lane, W // 4))
        x_flat = x_pooled.view(B, -1)
        
        # Lane presence classification
        lane_logits = self.lane_cls(x_flat)
        lane_logits = lane_logits.view(B, self.num_lanes, self.cls_num_per_lane, self.griding_num + 1)
        
        # Lane type classification
        lane_types = self.lane_type_cls(x_flat)
        lane_types = lane_types.view(B, self.num_lanes, self.num_classes)
        
        # Confidence scores
        confidences = torch.sigmoid(self.confidence_head(x_flat))
        
        outputs = {
            "lane_logits": lane_logits,
            "lane_types": lane_types,
            "confidences": confidences
        }
        
        # Instance embedding
        if self.use_instance_embedding:
            embeddings = self.embedding_head(x)
            outputs["embeddings"] = embeddings
            
        return outputs


class LanePostProcessor:
    """
    Post-process raw model outputs into lane curves.
    """
    def __init__(
        self,
        griding_num: int = 100,
        cls_num_per_lane: int = 56,
        num_lanes: int = 4,
        num_classes: int = 8,
        input_height: int = 288,
        input_width: int = 800,
        original_height: int = 590,
        original_width: int = 1640,
        conf_threshold: float = 0.6,
        min_lane_points: int = 10
    ):
        self.griding_num = griding_num
        self.cls_num_per_lane = cls_num_per_lane
        self.num_lanes = num_lanes
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.original_height = original_height
        self.original_width = original_width
        self.conf_threshold = conf_threshold
        self.min_lane_points = min_lane_points
        
        # Row anchors (vertical positions)
        self.row_anchors = self._get_row_anchors()
        
        # Lane type names
        self.lane_type_names = [
            "solid_white", "solid_yellow", "dashed_white", "dashed_yellow",
            "double_solid", "double_dashed", "botts_dots", "unknown"
        ]
        
    def _get_row_anchors(self) -> List[int]:
        """Generate row anchor positions."""
        # Distribute anchors from bottom to top
        return [
            int(self.original_height * (0.42 + 0.58 * i / (self.cls_num_per_lane - 1)))
            for i in range(self.cls_num_per_lane)
        ]
    
    def process(
        self, 
        outputs: Dict[str, torch.Tensor]
    ) -> List[Dict]:
        """
        Process model outputs into lane detections.
        
        Args:
            outputs: Model output dictionary
        Returns:
            List of lane dictionaries with points, type, and confidence
        """
        lane_logits = outputs["lane_logits"]
        lane_types = outputs["lane_types"]
        confidences = outputs["confidences"]
        
        B = lane_logits.shape[0]
        batch_results = []
        
        for b in range(B):
            lanes = []
            
            for lane_idx in range(self.num_lanes):
                # Check confidence
                if confidences[b, lane_idx].item() < self.conf_threshold:
                    continue
                
                # Get lane points from classification
                lane_points = self._get_lane_points(lane_logits[b, lane_idx])
                
                if len(lane_points) < self.min_lane_points:
                    continue
                
                # Get lane type
                lane_type_idx = torch.argmax(lane_types[b, lane_idx]).item()
                lane_type = self.lane_type_names[lane_type_idx]
                lane_type_conf = torch.softmax(lane_types[b, lane_idx], dim=0)[lane_type_idx].item()
                
                # Fit polynomial curve
                curve_params = self._fit_polynomial(lane_points)
                
                lanes.append({
                    "lane_id": lane_idx,
                    "points": lane_points,
                    "curve_params": curve_params,
                    "lane_type": lane_type,
                    "lane_type_confidence": lane_type_conf,
                    "confidence": confidences[b, lane_idx].item(),
                    "visibility": "visible" if len(lane_points) > self.cls_num_per_lane * 0.7 else "partial"
                })
            
            batch_results.append(lanes)
            
        return batch_results
    
    def _get_lane_points(self, lane_logit: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Extract lane points from row-wise classification output.
        
        Args:
            lane_logit: [cls_num_per_lane, griding_num+1]
        Returns:
            List of (x, y) points
        """
        points = []
        
        for i in range(self.cls_num_per_lane):
            # Get probability distribution over grid positions
            probs = F.softmax(lane_logit[i], dim=0)
            
            # Check if lane exists at this row (last class = no lane)
            if probs[-1] > 0.5:
                continue
            
            # Get expected position (soft argmax)
            grid_positions = torch.arange(self.griding_num, dtype=torch.float32, device=lane_logit.device)
            expected_pos = (probs[:-1] * grid_positions).sum()
            
            # Convert to original image coordinates
            x = int(expected_pos.item() / self.griding_num * self.original_width)
            y = self.row_anchors[i]
            
            points.append((x, y))
        
        return points
    
    def _fit_polynomial(self, points: List[Tuple[int, int]], degree: int = 3) -> np.ndarray:
        """
        Fit a polynomial curve to lane points.
        
        Args:
            points: List of (x, y) points
            degree: Polynomial degree
        Returns:
            Polynomial coefficients
        """
        if len(points) < degree + 1:
            degree = len(points) - 1
        
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Fit x as function of y (more stable for lanes)
        coeffs = np.polyfit(y, x, degree)
        
        return coeffs
    
    def get_lane_curve(
        self, 
        curve_params: np.ndarray, 
        y_start: int, 
        y_end: int, 
        num_points: int = 50
    ) -> List[Tuple[int, int]]:
        """
        Generate points along fitted lane curve.
        
        Args:
            curve_params: Polynomial coefficients
            y_start: Starting y coordinate
            y_end: Ending y coordinate
            num_points: Number of points to generate
        Returns:
            List of (x, y) points on curve
        """
        y_values = np.linspace(y_start, y_end, num_points)
        x_values = np.polyval(curve_params, y_values)
        
        return [(int(x), int(y)) for x, y in zip(x_values, y_values)]


class LaneTracker:
    """
    Frame-to-frame lane tracker using Kalman filter.
    Smooths lane detections and handles transient occlusions.
    """
    def __init__(
        self,
        num_lanes: int = 4,
        max_lost: int = 3,
        smoothing_factor: float = 0.8
    ):
        self.num_lanes = num_lanes
        self.max_lost = max_lost
        self.smoothing_factor = smoothing_factor
        
        # Track state for each lane
        self.tracks = {}
        self.lost_counts = {}
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: Current frame detections
        Returns:
            Smoothed and tracked lanes
        """
        # Match detections to existing tracks
        matched_tracks = {}
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            best_match = None
            best_dist = float('inf')
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                
                # Compute distance (simple center point distance)
                if track["points"] and det["points"]:
                    track_center = np.mean([p[0] for p in track["points"]])
                    det_center = np.mean([p[0] for p in det["points"]])
                    dist = abs(track_center - det_center)
                    
                    if dist < best_dist and dist < 100:  # Max matching distance
                        best_dist = dist
                        best_match = i
            
            if best_match is not None:
                matched_tracks[track_id] = detections[best_match]
                used_detections.add(best_match)
                self.lost_counts[track_id] = 0
            else:
                self.lost_counts[track_id] += 1
        
        # Add new detections as new tracks
        next_id = max(self.tracks.keys(), default=-1) + 1
        for i, det in enumerate(detections):
            if i not in used_detections:
                matched_tracks[next_id] = det
                self.lost_counts[next_id] = 0
                next_id += 1
        
        # Remove lost tracks
        self.tracks = {
            k: v for k, v in matched_tracks.items() 
            if self.lost_counts.get(k, 0) < self.max_lost
        }
        
        # Smooth tracks
        smoothed = []
        for track_id, track in self.tracks.items():
            if track_id in self.tracks and track_id in matched_tracks:
                # Apply exponential moving average
                smoothed_track = self._smooth_track(track, matched_tracks[track_id])
                smoothed.append(smoothed_track)
            else:
                smoothed.append(track)
        
        return smoothed
    
    def _smooth_track(self, prev_track: Dict, curr_track: Dict) -> Dict:
        """Apply temporal smoothing to track."""
        alpha = self.smoothing_factor
        
        smoothed = curr_track.copy()
        
        # Smooth confidence
        smoothed["confidence"] = (
            alpha * prev_track.get("confidence", 0) + 
            (1 - alpha) * curr_track["confidence"]
        )
        
        # Smooth curve parameters if available
        if "curve_params" in prev_track and "curve_params" in curr_track:
            smoothed["curve_params"] = (
                alpha * prev_track["curve_params"] + 
                (1 - alpha) * curr_track["curve_params"]
            )
        
        return smoothed
    
    def is_lane_lost(self) -> bool:
        """Check if all lanes are lost."""
        return len(self.tracks) == 0 or all(
            self.lost_counts.get(k, 0) >= self.max_lost 
            for k in self.tracks.keys()
        )
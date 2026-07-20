"""
Evaluation Metrics for Lane Detection
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_lane_metrics(
    predictions: List[Dict],
    targets: List[Dict],
    config: Dict
) -> Dict[str, float]:
    """
    Compute lane detection metrics.
    
    Metrics:
    - F1-score (primary)
    - Accuracy
    - False Positive Rate
    - False Negative Rate
    - Lane Type Accuracy
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    type_correct = 0
    type_total = 0
    
    for pred, target in zip(predictions, targets):
        pred_lanes = pred.get("lane_logits", None)
        target_lanes = target.get("lane_targets", None)
        
        if pred_lanes is None or target_lanes is None:
            continue
        
        # Compute IoU-based matching
        tp, fp, fn = _match_lanes(pred_lanes, target_lanes, config)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Lane type accuracy
        if "lane_types" in pred and "lane_type_targets" in target:
            pred_types = torch.argmax(pred["lane_types"], dim=-1)
            type_correct += (pred_types == target["lane_type_targets"]).sum().item()
            type_total += target["lane_type_targets"].numel()
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0
    fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
    
    type_acc = type_correct / type_total if type_total > 0 else 0
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": (total_tp) / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0,
        "fpr": fpr,
        "fnr": fnr,
        "type_accuracy": type_acc
    }


def _match_lanes(
    pred_logits: torch.Tensor,
    target_targets: torch.Tensor,
    config: Dict,
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Match predicted lanes to ground truth using IoU.
    
    Returns:
        tp, fp, fn counts
    """
    B, num_lanes, num_rows, num_grids = pred_logits.shape
    griding_num = config["model"]["griding_num"]
    
    tp, fp, fn = 0, 0, 0
    
    for b in range(B):
        pred_lanes = []
        target_lanes = []
        
        # Extract predicted lanes
        for lane_idx in range(num_lanes):
            lane_points = []
            for row_idx in range(num_rows):
                probs = torch.softmax(pred_logits[b, lane_idx, row_idx], dim=0)
                if probs[-1] < 0.5:  # Lane exists
                    grid_idx = torch.argmax(probs[:-1]).item()
                    lane_points.append(grid_idx)
                else:
                    lane_points.append(-1)
            pred_lanes.append(lane_points)
        
        # Extract target lanes
        for lane_idx in range(num_lanes):
            lane_points = []
            for row_idx in range(num_rows):
                val = target_targets[b, lane_idx, row_idx].item()
                lane_points.append(val if val >= 0 else -1)
            target_lanes.append(lane_points)
        
        # Match using IoU
        matched = set()
        for pred_idx, pred_lane in enumerate(pred_lanes):
            best_iou = 0
            best_target = -1
            
            for target_idx, target_lane in enumerate(target_lanes):
                if target_idx in matched:
                    continue
                
                iou = _compute_lane_iou(pred_lane, target_lane)
                if iou > best_iou:
                    best_iou = iou
                    best_target = target_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched.add(best_target)
            else:
                fp += 1
        
        fn += len(target_lanes) - len(matched)
    
    return tp, fp, fn


def _compute_lane_iou(lane1: List[int], lane2: List[int]) -> float:
    """Compute IoU between two lane representations."""
    # Convert to sets of valid points
    set1 = set((i, v) for i, v in enumerate(lane1) if v >= 0)
    set2 = set((i, v) for i, v in enumerate(lane2) if v >= 0)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def compute_endpoint_error(
    pred_points: List[Tuple[int, int]],
    gt_points: List[Tuple[int, int]],
    distances: List[int] = [30, 60, 90]
) -> Dict[str, float]:
    """
    Compute lateral displacement error at specific distances ahead.
    
    Args:
        pred_points: Predicted lane points [(x, y), ...]
        gt_points: Ground truth lane points [(x, y), ...]
        distances: Distances in meters to evaluate
    Returns:
        Dictionary of errors at each distance
    """
    errors = {}
    
    for dist in distances:
        # Find closest points at approximately this distance
        # Simplified: assume y corresponds to distance
        pred_at_dist = _find_point_at_distance(pred_points, dist)
        gt_at_dist = _find_point_at_distance(gt_points, dist)
        
        if pred_at_dist and gt_at_dist:
            error = abs(pred_at_dist[0] - gt_at_dist[0])
            errors[f"error_{dist}m"] = error
    
    return errors


def _find_point_at_distance(
    points: List[Tuple[int, int]], 
    target_dist: int
) -> Optional[Tuple[int, int]]:
    """Find point closest to target distance."""
    if not points:
        return None
    
    # Simple approximation: use y-coordinate as proxy for distance
    closest = min(points, key=lambda p: abs(p[1] - target_dist))
    return closest
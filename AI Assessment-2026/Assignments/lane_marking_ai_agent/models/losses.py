"""
Loss Functions for Lane Detection Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class LaneDetectionLoss(nn.Module):
    """
    Combined loss for lane detection:
    - Classification loss (cross-entropy for row-wise presence)
    - Localization loss (smooth L1 for precise position)
    - Weather classification loss
    - Instance embedding loss (discriminative loss)
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        weights = config["training"]["loss_weights"]
        self.w_cls = weights["lane_cls"]
        self.w_loc = weights["lane_loc"]
        self.w_weather = weights["weather_cls"]
        self.w_instance = weights["instance"]
        
        self.griding_num = config["model"]["griding_num"]
        self.cls_num_per_lane = config["model"]["cls_num_per_lane"]
        self.num_lanes = config["model"]["num_lanes"]
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model outputs
            targets: Ground truth annotations
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # 1. Lane classification loss (cross-entropy)
        lane_logits = predictions["lane_logits"]  # [B, num_lanes, cls_num_per_lane, griding_num+1]
        lane_targets = targets["lane_targets"]     # [B, num_lanes, cls_num_per_lane]
        
        # Reshape for cross-entropy
        B = lane_logits.shape[0]
        lane_logits_flat = lane_logits.view(-1, self.griding_num + 1)
        lane_targets_flat = lane_targets.view(-1).long()
        
        # Ignore invalid targets (-1)
        valid_mask = lane_targets_flat >= 0
        if valid_mask.sum() > 0:
            cls_loss = F.cross_entropy(
                lane_logits_flat[valid_mask],
                lane_targets_flat[valid_mask],
                reduction='mean'
            )
        else:
            cls_loss = torch.tensor(0.0, device=lane_logits.device)
        
        losses["lane_cls"] = cls_loss
        
        # 2. Localization loss (smooth L1 for expected position)
        if "lane_positions" in targets:
            pred_positions = self._get_expected_positions(lane_logits)
            true_positions = targets["lane_positions"]
            
            valid_pos_mask = true_positions >= 0
            if valid_pos_mask.sum() > 0:
                loc_loss = F.smooth_l1_loss(
                    pred_positions[valid_pos_mask],
                    true_positions[valid_pos_mask],
                    reduction='mean'
                )
            else:
                loc_loss = torch.tensor(0.0, device=lane_logits.device)
            
            losses["lane_loc"] = loc_loss
        else:
            losses["lane_loc"] = torch.tensor(0.0, device=lane_logits.device)
        
        # 3. Weather classification loss
        weather_logits = predictions["weather_logits"]
        weather_targets = targets["weather_targets"]
        
        weather_loss = F.cross_entropy(weather_logits, weather_targets)
        losses["weather_cls"] = weather_loss
        
        # 4. Lane type classification loss
        lane_types = predictions["lane_types"]
        lane_type_targets = targets["lane_type_targets"]
        
        type_loss = F.cross_entropy(
            lane_types.view(-1, lane_types.shape[-1]),
            lane_type_targets.view(-1),
            reduction='mean'
        )
        losses["lane_type"] = type_loss
        
        # 5. Instance embedding loss (discriminative)
        if "embeddings" in predictions and "instance_labels" in targets:
            instance_loss = self._discriminative_loss(
                predictions["embeddings"],
                targets["instance_labels"]
            )
            losses["instance"] = instance_loss
        else:
            losses["instance"] = torch.tensor(0.0, device=lane_logits.device)
        
        # Total loss
        total_loss = (
            self.w_cls * losses["lane_cls"] +
            self.w_loc * losses["lane_loc"] +
            self.w_weather * losses["weather_cls"] +
            0.5 * losses["lane_type"] +
            self.w_instance * losses["instance"]
        )
        
        losses["total"] = total_loss
        
        return losses
    
    def _get_expected_positions(self, lane_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute expected lane positions from classification logits.
        
        Args:
            lane_logits: [B, num_lanes, cls_num_per_lane, griding_num+1]
        Returns:
            positions: [B, num_lanes, cls_num_per_lane]
        """
        B, N, R, G = lane_logits.shape
        
        # Get probabilities (excluding "no lane" class)
        probs = F.softmax(lane_logits, dim=-1)[..., :-1]  # [B, N, R, griding_num]
        
        # Grid positions
        grid = torch.arange(self.griding_num, dtype=torch.float32, device=lane_logits.device)
        grid = grid.view(1, 1, 1, -1)
        
        # Expected position
        positions = (probs * grid).sum(dim=-1)  # [B, N, R]
        
        return positions
    
    def _discriminative_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        delta_v: float = 0.5,
        delta_d: float = 3.0
    ) -> torch.Tensor:
        """
        Discriminative loss for instance embedding.
        Pulls embeddings of same lane together, pushes different lanes apart.
        
        Args:
            embeddings: [B, D, H, W]
            labels: [B, H, W] instance labels
        """
        B, D, H, W = embeddings.shape
        
        # Reshape
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        labels = labels.reshape(-1)  # [B*H*W]
        
        # Get unique labels (excluding background -1)
        unique_labels = torch.unique(labels[labels >= 0])
        
        if len(unique_labels) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Variance term (pull)
        var_loss = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() == 0:
                continue
            
            cluster_embeddings = embeddings[mask]
            mean_embedding = cluster_embeddings.mean(dim=0)
            
            var = torch.norm(cluster_embeddings - mean_embedding, dim=1) - delta_v
            var = torch.clamp(var, min=0.0)
            var_loss += var.mean()
        
        var_loss /= len(unique_labels)
        
        # Distance term (push)
        dist_loss = 0.0
        num_pairs = 0
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                mask_i = labels == unique_labels[i]
                mask_j = labels == unique_labels[j]
                
                if mask_i.sum() == 0 or mask_j.sum() == 0:
                    continue
                
                mean_i = embeddings[mask_i].mean(dim=0)
                mean_j = embeddings[mask_j].mean(dim=0)
                
                dist = torch.norm(mean_i - mean_j)
                dist = torch.clamp(2 * delta_d - dist, min=0.0)
                dist_loss += dist ** 2
                num_pairs += 1
        
        if num_pairs > 0:
            dist_loss /= num_pairs
        
        # Regularization term
        reg_loss = 0.0
        for label in unique_labels:
            mask = labels == label
            if mask.sum() == 0:
                continue
            mean_embedding = embeddings[mask].mean(dim=0)
            reg_loss += torch.norm(mean_embedding)
        
        reg_loss /= len(unique_labels)
        
        return var_loss + dist_loss + 0.001 * reg_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in lane detection."""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
"""
Complete Weather-Aware Lane Detection Model
Integrates backbone, weather classifier, and lane detection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import yaml

from .backbone import WeatherConditionedBackbone, FeaturePyramidNetwork
from .weather_classifier import WeatherClassifier, WeatherAdaptivePreprocessor
from .lane_head import LaneDetectionHead, LanePostProcessor, LaneTracker


class WeatherAwareLaneDetector(nn.Module):
    """
    Complete weather-aware lane detection system.
    Two-stage: Weather classification -> Adaptive lane detection.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        model_cfg = config["model"]
        weather_cfg = config["weather"]
        
        # Weather classifier
        self.weather_classifier = WeatherClassifier(
            num_classes=weather_cfg["num_classes"],
            backbone_name=weather_cfg["classifier_backbone"],
            embedding_dim=model_cfg.get("weather_embedding_dim", 128)
        )
        
        # Weather-adaptive preprocessor
        self.weather_preprocessor = WeatherAdaptivePreprocessor()
        
        # Shared backbone with FiLM
        self.backbone = WeatherConditionedBackbone(
            backbone_name=model_cfg["backbone"],
            weather_dim=model_cfg.get("weather_embedding_dim", 128),
            pretrained=True
        )
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels=self.backbone.feature_channels,
            out_channels=256
        )
        
        # Lane detection head
        self.lane_head = LaneDetectionHead(
            in_channels=256,
            num_lanes=model_cfg["num_lanes"],
            num_classes=model_cfg["num_classes"],
            griding_num=model_cfg["griding_num"],
            cls_num_per_lane=model_cfg["cls_num_per_lane"],
            use_instance_embedding=model_cfg.get("use_instance_embedding", True)
        )
        
        # Post-processor
        self.post_processor = LanePostProcessor(
            griding_num=model_cfg["griding_num"],
            cls_num_per_lane=model_cfg["cls_num_per_lane"],
            num_lanes=model_cfg["num_lanes"],
            num_classes=model_cfg["num_classes"],
            input_height=model_cfg["input_height"],
            input_width=model_cfg["input_width"],
            conf_threshold=config["inference"]["confidence_threshold"]
        )
        
        # Tracker
        self.tracker = LaneTracker(
            num_lanes=model_cfg["num_lanes"],
            max_lost=config["inference"]["lane_lost_frames"]
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_weather: bool = False,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the complete model.
        
        Args:
            x: [B, 3, H, W] input image
            return_weather: Whether to return weather classification
            return_features: Whether to return intermediate features
        Returns:
            Dictionary with lane detections and optional weather info
        """
        B, C, H, W = x.shape
        
        # Stage 1: Weather classification
        weather_logits, weather_probs, weather_embedding = self.weather_classifier(x)
        
        # Stage 2: Weather-adaptive preprocessing
        x_enhanced = self.weather_preprocessor(x, weather_probs)
        
        # Stage 3: Feature extraction with FiLM conditioning
        features = self.backbone(x_enhanced, weather_embedding)
        
        # Stage 4: Feature pyramid
        fpn_features = self.fpn(features)
        
        # Use highest resolution feature map for lane detection
        lane_features = fpn_features["pyramid_0"]
        
        # Stage 5: Lane detection
        lane_outputs = self.lane_head(lane_features)
        
        outputs = {
            "lane_logits": lane_outputs["lane_logits"],
            "lane_types": lane_outputs["lane_types"],
            "confidences": lane_outputs["confidences"],
            "weather_probs": weather_probs,
            "weather_logits": weather_logits
        }
        
        if self.lane_head.use_instance_embedding:
            outputs["embeddings"] = lane_outputs["embeddings"]
        
        if return_weather:
            outputs["weather_embedding"] = weather_embedding
            
        if return_features:
            outputs["features"] = fpn_features
            
        return outputs
    
    def inference(
        self, 
        x: torch.Tensor,
        original_size: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Inference mode with post-processing and tracking.
        
        Args:
            x: [B, 3, H, W] input image (preprocessed)
            original_size: (H, W) original image size for coordinate mapping
        Returns:
            List of lane detection results per image in batch
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(x)
        
        # Post-process
        detections = self.post_processor.process(outputs)
        
        # Track across frames (for single image, just return detections)
        if len(detections) == 1:
            tracked = self.tracker.update(detections[0])
            return [tracked]
        
        return detections
    
    def get_weather_condition(self, x: torch.Tensor) -> Tuple[str, float]:
        """Get weather condition for input image."""
        return self.weather_classifier.get_weather_condition(x)
    
    def export_onnx(self, path: str, input_shape: Tuple[int, int, int, int] = (1, 3, 288, 800)):
        """Export model to ONNX format."""
        self.eval()
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["lane_logits", "lane_types", "confidences", "weather_probs"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "lane_logits": {0: "batch_size"},
                "lane_types": {0: "batch_size"},
                "confidences": {0: "batch_size"},
                "weather_probs": {0: "batch_size"}
            }
        )
        print(f"Model exported to ONNX: {path}")


def build_model(config_path: str = None, config: Dict = None) -> WeatherAwareLaneDetector:
    """
    Build model from configuration.
    
    Args:
        config_path: Path to YAML config file (optional if config provided)
        config: Config dictionary (optional if config_path provided)
    Returns:
        Initialized model
    """
    if config is None:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    model = WeatherAwareLaneDetector(config)
    return model
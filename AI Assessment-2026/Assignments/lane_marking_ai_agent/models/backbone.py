"""
Weather-Aware Backbone with FiLM (Feature-wise Linear Modulation)
Enables the network to adapt its feature representations based on weather conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation layer.
    Conditions feature maps on weather embeddings via learned scale and shift.
    """
    def __init__(self, num_features: int, weather_dim: int):
        super().__init__()
        self.num_features = num_features
        
        # FiLM generator: weather embedding -> (scale, shift)
        self.film_generator = nn.Sequential(
            nn.Linear(weather_dim, weather_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(weather_dim * 2, num_features * 2)  # *2 for scale and shift
        )
        
    def forward(self, features: torch.Tensor, weather_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] feature maps
            weather_embedding: [B, D] weather condition embedding
        Returns:
            modulated_features: [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Generate scale and shift parameters
        film_params = self.film_generator(weather_embedding)  # [B, C*2]
        scale, shift = film_params.chunk(2, dim=1)  # Each: [B, C]
        
        # Reshape for broadcasting
        scale = scale.view(B, C, 1, 1)
        shift = shift.view(B, C, 1, 1)
        
        # Apply FiLM: gamma * x + beta
        return features * (1 + scale) + shift


class WeatherConditionedBackbone(nn.Module):
    """
    Shared backbone with FiLM conditioning for weather-aware feature extraction.
    Supports EfficientNet-B3, ResNet-50, and ConvNeXt-Tiny.
    """
    def __init__(
        self,
        backbone_name: str = "efficientnet-b3",
        weather_dim: int = 128,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.weather_dim = weather_dim
        
        # Build backbone
        self.backbone, self.feature_channels, self.feature_layers = self._build_backbone(pretrained)
        
        # FiLM layers for each feature scale
        self.film_layers = nn.ModuleDict({
            f"scale_{i}": FiLMLayer(channels, weather_dim)
            for i, channels in enumerate(self.feature_channels)
        })
        
    def _build_backbone(self, pretrained: bool) -> Tuple[nn.Module, List[int], List[str]]:
        """Build the backbone network and extract feature channels."""
        
        if self.backbone_name == "efficientnet-b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            
            # Extract features at multiple scales
            features = backbone.features
            feature_channels = [24, 32, 48, 136, 384]  # After each block group
            feature_layers = ['0:2', '2:4', '4:6', '6:8', '8:']
            
            # Wrap in a custom forward to return multi-scale features
            backbone = EfficientNetFeatureExtractor(features, feature_layers)
            
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_channels = [64, 256, 512, 1024, 2048]
            backbone = ResNetFeatureExtractor(backbone)
            
        elif self.backbone_name == "convnext-tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            feature_channels = [96, 192, 384, 768]
            backbone = ConvNeXtFeatureExtractor(backbone.features)
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
            
        return backbone, feature_channels, feature_layers
    
    def forward(
        self, 
        x: torch.Tensor, 
        weather_embedding: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image
            weather_embedding: [B, D] weather condition embedding (optional)
        Returns:
            Dictionary of multi-scale features
        """
        # Extract multi-scale features
        features = self.backbone(x)  # Dict of features at different scales
        
        # Apply FiLM conditioning if weather embedding is provided
        if weather_embedding is not None:
            for i, (key, feat) in enumerate(features.items()):
                film_layer = self.film_layers[f"scale_{i}"]
                features[key] = film_layer(feat, weather_embedding)
                
        return features


class EfficientNetFeatureExtractor(nn.Module):
    """Extract multi-scale features from EfficientNet."""
    def __init__(self, features: nn.Module, feature_layers: List[str]):
        super().__init__()
        self.features = features
        self.feature_layers = feature_layers
        
        # Define extraction points
        self.layer_indices = [(0, 2), (2, 4), (4, 6), (6, 8), (8, len(features))]
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for i, (start, end) in enumerate(self.layer_indices):
            for j in range(start, end):
                x = self.features[j](x)
            features[f"stage_{i}"] = x
        return features


class ResNetFeatureExtractor(nn.Module):
    """Extract multi-scale features from ResNet."""
    def __init__(self, backbone: models.ResNet):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features["stage_0"] = x
        
        x = self.maxpool(x)
        x = self.layer1(x)
        features["stage_1"] = x
        
        x = self.layer2(x)
        features["stage_2"] = x
        
        x = self.layer3(x)
        features["stage_3"] = x
        
        x = self.layer4(x)
        features["stage_4"] = x
        
        return features


class ConvNeXtFeatureExtractor(nn.Module):
    """Extract multi-scale features from ConvNeXt."""
    def __init__(self, features: nn.Module):
        super().__init__()
        self.features = features
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for i in range(4):
            x = self.features[i * 3:(i + 1) * 3](x)
            features[f"stage_{i}"] = x
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    Combines features from different scales for robust lane detection.
    """
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels
        ])
        
        # Output convolutions
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels
        ])
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dictionary of multi-scale features
        Returns:
            Fused multi-scale features
        """
        # Convert dict to list maintaining order
        feature_list = [features[f"stage_{i}"] for i in range(len(features))]
        
        # Build pyramid from top-down
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, feature_list)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:], 
                mode='nearest'
            )
            laterals[i] = laterals[i] + upsampled
            
        # Apply output convolutions
        outputs = {}
        for i, (lateral, output_conv) in enumerate(zip(laterals, self.output_convs)):
            outputs[f"pyramid_{i}"] = output_conv(lateral)
            
        return outputs
"""
Lightweight Weather Condition Classifier
Classifies weather conditions to enable adaptive lane detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple


class WeatherClassifier(nn.Module):
    """
    Lightweight weather condition classifier.
    Classes: clear, rain, fog, snow, night, glare
    """
    def __init__(
        self,
        num_classes: int = 6,
        backbone_name: str = "mobilenet-v3-small",
        embedding_dim: int = 128,
        pretrained: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Build backbone
        self.backbone = self._build_backbone(backbone_name, pretrained)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Embedding head for FiLM conditioning
        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone.out_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
    def _build_backbone(self, name: str, pretrained: bool) -> nn.Module:
        """Build lightweight backbone for weather classification."""
        
        if name == "mobilenet-v3-small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            backbone = model.features
            backbone.out_features = 576  # Last channel of MobileNetV3-small
            
        elif name == "mobilenet-v3-large":
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.mobilenet_v3_large(weights=weights)
            backbone = model.features
            backbone.out_features = 960
            
        elif name == "efficientnet-b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            backbone = model.features
            backbone.out_features = 1280
            
        else:
            raise ValueError(f"Unsupported weather classifier backbone: {name}")
            
        # Add global pooling
        backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        backbone.flatten = nn.Flatten()
        
        return backbone
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W] input image
        Returns:
            logits: [B, num_classes] weather classification logits
            probs: [B, num_classes] weather classification probabilities
            embedding: [B, embedding_dim] weather embedding for FiLM
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        
        # Embedding for FiLM
        embedding = self.embedding_head(features)
        
        return logits, probs, embedding
    
    def get_weather_condition(self, x: torch.Tensor) -> Tuple[str, float]:
        """
        Get the predicted weather condition and confidence.
        
        Args:
            x: [B, 3, H, W] input image
        Returns:
            condition: Predicted weather condition name
            confidence: Prediction confidence
        """
        self.eval()
        with torch.no_grad():
            logits, probs, _ = self.forward(x)
            
        weather_classes = ["clear", "rain", "fog", "snow", "night", "glare"]
        pred_idx = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        
        return weather_classes[pred_idx.item()], confidence.item()


class WeatherAdaptivePreprocessor(nn.Module):
    """
    Weather-adaptive image preprocessing module.
    Applies different enhancement techniques based on detected weather.
    """
    def __init__(self):
        super().__init__()
        
        # Dehazing network (lightweight)
        self.dehaze_net = self._build_enhancement_net()
        
        # Denoising network
        self.denoise_net = self._build_enhancement_net()
        
        # Glare suppression
        self.glare_net = self._build_enhancement_net()
        
    def _build_enhancement_net(self) -> nn.Module:
        """Build a lightweight enhancement network."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        weather_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] input image
            weather_probs: [B, 6] weather condition probabilities
        Returns:
            enhanced: [B, 3, H, W] enhanced image
        """
        B, C, H, W = x.shape
        
        # Apply weather-specific enhancements
        # Fog enhancement
        fog_residual = self.dehaze_net(x)
        fog_enhanced = x + fog_residual * weather_probs[:, 2:3].view(B, 1, 1, 1)
        
        # Rain/denoising enhancement
        rain_residual = self.denoise_net(x)
        rain_enhanced = x + rain_residual * weather_probs[:, 1:2].view(B, 1, 1, 1)
        
        # Glare suppression
        glare_residual = self.glare_net(x)
        glare_enhanced = x + glare_residual * weather_probs[:, 5:6].view(B, 1, 1, 1)
        
        # Combine based on weather probabilities
        clear_weight = weather_probs[:, 0:1].view(B, 1, 1, 1)
        enhanced = (
            clear_weight * x +
            (1 - clear_weight) * (fog_enhanced + rain_enhanced + glare_enhanced) / 3
        )
        
        return torch.clamp(enhanced, 0, 1)
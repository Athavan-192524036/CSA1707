"""
Unit Tests for Lane Detection Model
"""
import pytest
import torch
import numpy as np

from models.backbone import WeatherConditionedBackbone, FiLMLayer, FeaturePyramidNetwork
from models.weather_classifier import WeatherClassifier
from models.lane_head import LaneDetectionHead, LanePostProcessor
from models.losses import LaneDetectionLoss


class TestFiLMLayer:
    def test_film_output_shape(self):
        film = FiLMLayer(num_features=64, weather_dim=128)
        features = torch.randn(2, 64, 32, 32)
        weather = torch.randn(2, 128)
        output = film(features, weather)
        assert output.shape == (2, 64, 32, 32)
    
    def test_film_conditioning(self):
        film = FiLMLayer(num_features=64, weather_dim=128)
        features = torch.randn(2, 64, 32, 32)
        weather1 = torch.randn(2, 128)
        weather2 = torch.randn(2, 128)
        out1 = film(features, weather1)
        out2 = film(features, weather2)
        assert not torch.allclose(out1, out2)


class TestWeatherClassifier:
    def test_classification_output(self):
        classifier = WeatherClassifier(num_classes=6, embedding_dim=128)
        x = torch.randn(2, 3, 224, 224)
        logits, probs, embedding = classifier(x)
        assert logits.shape == (2, 6)
        assert probs.shape == (2, 6)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        assert embedding.shape == (2, 128)


class TestLaneDetectionHead:
    def test_output_shapes(self):
        head = LaneDetectionHead(in_channels=256, num_lanes=4, num_classes=8, griding_num=100, cls_num_per_lane=56)
        features = torch.randn(2, 256, 36, 100)
        outputs = head(features)
        assert outputs["lane_logits"].shape == (2, 4, 56, 101)
        assert outputs["lane_types"].shape == (2, 4, 8)
        assert outputs["confidences"].shape == (2, 4)


class TestLossFunction:
    def test_loss_computation(self):
        config = {
            "model": {"griding_num": 100, "cls_num_per_lane": 56, "num_lanes": 4},
            "training": {"loss_weights": {"lane_cls": 1.0, "lane_loc": 0.5, "weather_cls": 0.3, "instance": 0.1}}
        }
        criterion = LaneDetectionLoss(config)
        predictions = {
            "lane_logits": torch.randn(2, 4, 56, 101),
            "lane_types": torch.randn(2, 4, 8),
            "weather_logits": torch.randn(2, 6),
            "embeddings": torch.randn(2, 4, 36, 100)
        }
        targets = {
            "lane_targets": torch.randint(0, 100, (2, 4, 56)),
            "lane_positions": torch.rand(2, 4, 56),
            "weather_targets": torch.randint(0, 6, (2,)),
            "lane_type_targets": torch.randint(0, 8, (2, 4)),
            "instance_labels": torch.randint(0, 4, (2, 36, 100))
        }
        losses = criterion(predictions, targets)
        assert "total" in losses
        assert losses["total"].item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
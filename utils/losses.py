from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

try:
    from torchvision.models import VGG16_Weights
except ImportError:  # pragma: no cover
    VGG16_Weights = None


def _rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    if image.size(1) == 1:
        return image
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class SobelEdgeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        luminance = _rgb_to_luminance(image)
        grad_x = F.conv2d(luminance, self.sobel_x, padding=1)
        grad_y = F.conv2d(luminance, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = self._gradient_magnitude(prediction)
        target_edges = self._gradient_magnitude(target)
        return F.l1_loss(pred_edges, target_edges)


class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers: Iterable[int] = (3, 8, 15)) -> None:
        super().__init__()
        if VGG16_Weights is None:
            raise RuntimeError("This torchvision build does not expose VGG16 weights. Disable perceptual loss.")

        try:
            backbone = models.vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Could not load VGG16 weights for perceptual loss. "
                "Disable `use_perceptual_loss` or pre-download the weights."
            ) from exc

        self.feature_layers = set(feature_layers)
        self.backbone = backbone[: max(self.feature_layers) + 1]
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def _extract_features(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        current = self._normalize(image)
        for index, layer in enumerate(self.backbone):
            current = layer(current)
            if index in self.feature_layers:
                features.append(current)
        return features

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self._extract_features(prediction)
        target_features = self._extract_features(target)
        loss = prediction.new_tensor(0.0)
        for pred_feature, target_feature in zip(pred_features, target_features):
            loss = loss + F.l1_loss(pred_feature, target_feature)
        return loss


class EnhancementLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        edge_weight: float = 0.2,
        perceptual_weight: float = 0.05,
        use_perceptual: bool = False,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        self.edge_loss = SobelEdgeLoss()
        self.perceptual_loss = PerceptualLoss() if use_perceptual else None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        l1_value = F.l1_loss(prediction, target)
        edge_value = self.edge_loss(prediction, target)
        perceptual_value = prediction.new_tensor(0.0)

        total = self.l1_weight * l1_value + self.edge_weight * edge_value
        if self.perceptual_loss is not None:
            perceptual_value = self.perceptual_loss(prediction, target)
            total = total + self.perceptual_weight * perceptual_value

        return total, {
            "total": total,
            "l1": l1_value,
            "edge": edge_value,
            "perceptual": perceptual_value,
        }

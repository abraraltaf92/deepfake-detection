"""Deepfake classifier model definitions.

Architectures:
    ResNetBinaryVideoClassifier    — frame-level ResNet-18, score-averaged across frames
    EfficientNetDeepfakeDetector   — frame-level EfficientNet-B4               (Phase 3)
    R3D18DeepfakeDetector          — clip-level 3D ResNet-18                  (Phase 3)
    ViTDeepfakeDetector            — frame-level timm ViT                      (Phase 3)
    R3D18RAFTDeepfakeDetector      — clip-level 3D ResNet-18 on 16 RAFT frames (Phase 3)

All models subclass DeepfakeClassifier which exposes head_parameters() and
backbone_parameters() so src.training.train_two_stage can do head-only
training in stage 1 and full fine-tune in stage 2 uniformly.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DeepfakeClassifier(nn.Module):
    """Base class. Subclasses must implement forward and expose head/backbone parameter groups."""

    @abstractmethod
    def head_parameters(self) -> Iterable[nn.Parameter]:
        ...

    @abstractmethod
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        ...


class ResNetBinaryVideoClassifier(DeepfakeClassifier):
    """ResNet-18 applied per frame; logits averaged across frames.

    Input:  (B, T, 3, H, W)
    Output: (B, 2) binary logits.

    Migrated from notebooks/04_model_baseline.ipynb so the baseline uses the
    same src.training / src.evaluation paths as the four advanced models.
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)                 # (B*T, F)
        logits = self.head(features)                # (B*T, 2)
        logits = logits.view(B, T, 2).mean(dim=1)   # (B, 2)
        return logits

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()


# =============================================================================
# Placeholders for Phase 3 — filled in during Tasks 16–19.
# Keeping the names here so type references from configs.experiments.py resolve
# even before Phase 3 implementation.
# =============================================================================
class EfficientNetDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 16")


class R3D18DeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 17")


class ViTDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 18")


class R3D18RAFTDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 19")

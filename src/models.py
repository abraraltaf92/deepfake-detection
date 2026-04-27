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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _dct_blockwise_magnitude(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    """Per-block 2D-FFT magnitude as an approximation of 2D-DCT magnitude.

    Splits ``x`` into non-overlapping ``block × block`` tiles, computes the
    2D-FFT of each tile, takes the absolute value (magnitude spectrum), and
    reassembles into a tensor with the same shape as the input. This is a
    fast PyTorch-native approximation of block-wise DCT-II magnitude — the
    two transforms differ in phase but not in the magnitude spectrum that
    GAN-fingerprint detectors care about.

    Args:
        x:     (B, C, H, W) float tensor. H and W must be divisible by ``block``.
        block: tile size, default 8 (matches F3-Net / FreqNet conventions).
    Returns:
        Magnitude map with the same shape as ``x`` (non-negative).
    """
    B, C, H, W = x.shape
    assert H % block == 0 and W % block == 0, (
        f"DCT block {block} does not divide H={H} or W={W}"
    )
    # Reshape to (B, C, H/b, b, W/b, b) so the last two dims are the block
    nh, nw = H // block, W // block
    blocks = x.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5)
    # blocks: (B, C, nh, nw, block, block) — last two dims are the spatial axes per tile
    spectrum = torch.fft.fft2(blocks)
    magnitude = spectrum.abs()
    # Reassemble back to (B, C, H, W) by inverting the permutation/view
    out = magnitude.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return out


class DeepfakeClassifier(nn.Module, ABC):
    """Base class. Subclasses must implement forward and expose head/backbone parameter groups."""

    @abstractmethod
    def head_parameters(self) -> list[nn.Parameter]:
        ...

    @abstractmethod
    def backbone_parameters(self) -> list[nn.Parameter]:
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
        features = self.backbone(x)                        # (B*T, F)
        features = features.view(B, T, -1).mean(dim=1)     # (B, F) — temporal avg pool BEFORE head
        logits = self.head(features)                       # (B, 2)
        return logits

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())


# =============================================================================
# Placeholders for Phase 3 — filled in during Tasks 16–19.
# Keeping the names here so type references from configs.experiments.py resolve
# even before Phase 3 implementation.
# =============================================================================
class EfficientNetDeepfakeDetector(DeepfakeClassifier):
    """EfficientNet-B4 per-frame classifier; features mean-pooled across frames, then head.

    Matches the mean-of-features + dropout + linear pattern used by the ResNet-18 baseline.
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b4(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)                            # (B*T, F)
        feats = feats.view(B, T, -1).mean(dim=1)            # (B, F) temporal pool BEFORE head
        return self.head(feats)                             # (B, 2)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())


class R3D18DeepfakeDetector(DeepfakeClassifier):
    """3D ResNet-18 from torchvision.models.video. Clip-level classifier.

    Input:  (B, T, C, H, W) — transposed internally to (B, C, T, H, W) for r3d_18.
    Output: (B, 2) binary logits.
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models.video as video_models
        weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
        backbone = video_models.r3d_18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        feats = self.backbone(x)   # (B, F)
        return self.head(feats)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())


class ViTDeepfakeDetector(DeepfakeClassifier):
    """timm ViT backbone, applied per-frame, feature-mean-pooled, then head."""

    def __init__(self, model_name: str = "vit_base_patch16_224",
                 dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1).mean(dim=1)      # temporal pool BEFORE head
        return self.head(feats)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())


class R3D18RAFTDeepfakeDetector(R3D18DeepfakeDetector):
    """Identical architecture to R3D18DeepfakeDetector.

    The difference is *what frames you feed it*: a DataLoader pointed at
    configs.paths.RAFT_FRAMES_ROOT supplies 16 RAFT-interpolated frames,
    whereas a DataLoader pointed at FRAMES_ROOT/MTCNN_FRAMES_ROOT supplies
    natively sampled frames. Kept as a separate class so leaderboard rows and
    configs.experiments entries can disambiguate.
    """
    pass


class FrequencyDeepfakeDetector(DeepfakeClassifier):
    """Frequency-domain deepfake detector — 6th member of the ensemble.

    Operates on per-block 2D-DCT (FFT-magnitude approximation) of MTCNN-cropped
    RGB frames. Motivated by F3-Net (Qian et al. 2020) and FreqNet (Tan et al.
    2024): GAN/diffusion generators leave periodic high-frequency artifacts
    that survive video compression, and these spectral signatures transfer
    better across datasets than spatial textures.

    Input:  (B, T, C=3, H, W) — same contract as ResNet18/EfficientNet/ViT
    Output: (B, 2) binary logits.
    """

    def __init__(self, dropout: float = 0.3, dct_block: int = 8) -> None:
        super().__init__()
        self.dct_block = dct_block

        # 4-conv CNN over DCT magnitudes — random init, no pretrained weights
        # because frequency-domain features have no ImageNet equivalent.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                               # 28 -> 1
            nn.Flatten(),                                          # (B*T, 256)
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) RGB, normalised by ImageNet mean/std (same as 2D models)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # DCT magnitudes per frame — non-negative, same shape as input
        x = _dct_blockwise_magnitude(x, block=self.dct_block)

        feats = self.backbone(x)                                   # (B*T, 256)
        feats = feats.view(B, T, -1).mean(dim=1)                   # (B, 256) — temporal mean-pool BEFORE head
        return self.head(feats)                                    # (B, 2)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

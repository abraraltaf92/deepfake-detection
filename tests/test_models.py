"""Validation script for src/models.py. Run with:
    .venv/bin/python tests/test_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import DeepfakeClassifier, ResNetBinaryVideoClassifier


def main() -> None:
    device = torch.device("cpu")  # CPU for the test — fast, deterministic

    # --- ResNet-18 binary video classifier ---
    model = ResNetBinaryVideoClassifier(dropout=0.3).to(device)
    assert isinstance(model, DeepfakeClassifier), "ResNet-18 must inherit from DeepfakeClassifier"
    # Input: (batch=2, num_frames=4, C=3, H=224, W=224)
    x = torch.randn(2, 4, 3, 224, 224, device=device)
    logits = model(x)
    assert logits.shape == (2, 2), f"expected (2, 2) logits, got {tuple(logits.shape)}"

    # --- param groups ---
    head_params = list(model.head_parameters())
    backbone_params = list(model.backbone_parameters())
    assert len(head_params) > 0, "head_parameters() returned empty"
    assert len(backbone_params) > 0, "backbone_parameters() returned empty"
    all_params = set(id(p) for p in model.parameters())
    partitioned = set(id(p) for p in head_params) | set(id(p) for p in backbone_params)
    assert all_params == partitioned, "head + backbone must partition all parameters"

    # --- EfficientNet-B4 ---
    from src.models import EfficientNetDeepfakeDetector
    m = EfficientNetDeepfakeDetector(dropout=0.3).to(device)
    assert isinstance(m, DeepfakeClassifier)
    x = torch.randn(2, 4, 3, 224, 224, device=device)
    logits = m(x)
    assert logits.shape == (2, 2)

    # --- R3D-18 ---
    from src.models import R3D18DeepfakeDetector
    m = R3D18DeepfakeDetector(dropout=0.3).to(device)
    # R3D expects (B, C, T, H, W) internally; the model's forward accepts (B, T, C, H, W) and transposes.
    x = torch.randn(2, 8, 3, 112, 112, device=device)
    logits = m(x)
    assert logits.shape == (2, 2)

    print("ok")


if __name__ == "__main__":
    main()

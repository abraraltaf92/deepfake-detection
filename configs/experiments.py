"""Per-model hyperparameter defaults. Override at notebook level for sweeps.

Usage in a notebook:
    from configs.experiments import EFFICIENTNET_B4_CONFIG
    CONFIG = {**EFFICIENTNET_B4_CONFIG, "batch_size": 8}   # one-line override
"""
from __future__ import annotations

BASE_CONFIG = dict(
    seed=42,
    num_frames=16,
    img_size=224,
    batch_size=16,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    weighted_sampler=True,
    augmentation=True,
    face_detector="mtcnn",
)

RESNET18_CONFIG = {
    **BASE_CONFIG,
    "model": "resnet18",
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 12,
    # The current baseline was trained pre-MTCNN — override here when re-using.
    "face_detector": "haar",
}

EFFICIENTNET_B4_CONFIG = {
    **BASE_CONFIG,
    "model": "efficientnet_b4",
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18",
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

VIT_CONFIG = {
    **BASE_CONFIG,
    "model": "vit_base_patch16_224",
    "lr_stage1": 5e-4,
    "lr_stage2": 5e-5,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_RAFT_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18_raft",
    "num_frames": 16,     # RAFT-interpolated, same count
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

ALL_CONFIGS = {
    "resnet18":              RESNET18_CONFIG,
    "efficientnet_b4":       EFFICIENTNET_B4_CONFIG,
    "r3d18":                 R3D18_CONFIG,
    "vit_base_patch16_224":  VIT_CONFIG,
    "r3d18_raft":            R3D18_RAFT_CONFIG,
}

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
    "train_mode": "single_stage",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 15,
    "scheduler_patience": 3,
    "scheduler_factor": 0.5,
    "weighted_sampler": False,  # baseline uses plain shuffle
    "augmentation": False,      # baseline uses normalize-only transforms
    "face_detector": "mtcnn",   # moved to MTCNN in Phase 2 — all 5 models share the same preprocessing
}

EFFICIENTNET_B4_CONFIG = {
    **BASE_CONFIG,
    "model": "efficientnet_b4",
    "train_mode": "two_stage",
    "batch_size": 8,         # 16 frames × batch=8 fits Colab L4 (22 GB) during stage-2 full fine-tune
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18",
    "train_mode": "two_stage",
    "batch_size": 8,         # 16 frames × batch=8 on L4
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

VIT_CONFIG = {
    **BASE_CONFIG,
    "model": "vit_base_patch16_224",
    "train_mode": "two_stage",
    "batch_size": 8,         # 16 frames × batch=8 on L4 — ViT attention is memory-heavy
    "lr_stage1": 5e-4,
    "lr_stage2": 5e-5,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_RAFT_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18_raft",
    "train_mode": "two_stage",
    "num_frames": 16,        # RAFT-interpolated, same count
    "batch_size": 8,
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

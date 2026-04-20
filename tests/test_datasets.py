"""Validation script for src/datasets.py. Run with:
    .venv/bin/python tests/test_datasets.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DeepfakeBinaryDataset, build_weighted_sampler, get_dataloaders
from tests.fixtures import make_tiny_split_df, make_fake_frames_root, tempdir


def main() -> None:
    tmp = tempdir()
    try:
        df = make_tiny_split_df(n_real=4, n_fake=8)  # imbalance for sampler test
        frames_root = make_fake_frames_root(tmp, df, num_frames=16, img_size=224)

        transform = T.Compose([T.ToTensor()])

        # --- DeepfakeBinaryDataset ---
        ds = DeepfakeBinaryDataset(df, frames_root=frames_root, transform=transform, num_frames=16)
        assert len(ds) == 12, f"len(ds) expected 12, got {len(ds)}"
        frames, label = ds[0]
        assert frames.shape == (16, 3, 224, 224), frames.shape
        assert isinstance(label, torch.Tensor), type(label)
        assert label.dtype == torch.long, label.dtype
        assert label.item() in (0, 1)

        # --- weighted sampler ---
        sampler = build_weighted_sampler(df)
        sampled_labels = [df.iloc[i]["binary_target"] for i in list(sampler)]
        # After weighted sampling, class distribution should be roughly balanced.
        # For 12 draws with true imbalance 4:8, weighted sampling should give roughly 6:6.
        n_real = sum(1 for y in sampled_labels if y == 0)
        assert 3 <= n_real <= 9, f"weighted sampler gave {n_real} reals out of 12 (expected ~6)"

        # --- get_dataloaders ---
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df=df, val_df=df.head(2), test_df=df.tail(2),
            frames_root=frames_root, train_transform=transform, eval_transform=transform,
            batch_size=2, num_frames=16, num_workers=0, weighted_sampler=False,
        )
        batch_frames, batch_labels = next(iter(train_loader))
        assert batch_frames.shape == (2, 16, 3, 224, 224), batch_frames.shape
        assert batch_labels.shape == (2,), batch_labels.shape

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

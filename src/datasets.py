"""Dataset classes and DataLoader construction for deepfake detection.

Factored out of 04_model_baseline.ipynb and the teammate's notebook so
both FF++ and Celeb-DF use the same pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DeepfakeBinaryDataset(Dataset):
    """Binary (real/fake) video-frame dataset.

    Each video contributes `num_frames` face crops stacked on dim 0.

    Directory layout expected at `frames_root`:
        <frames_root>/<binary_label>/<video_stem>/frame_XX.jpg

    Args:
        df: DataFrame with columns `file`, `binary_label`, `binary_target`.
        frames_root: Root directory of extracted face crops.
        transform: torchvision transform applied per-frame (e.g. ToTensor + Normalize).
        num_frames: Frames per video. Shorter videos are padded by repeating the last frame.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        frames_root: Path,
        transform=None,
        num_frames: int = 16,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.frames_root = Path(frames_root)
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.df)

    def _frame_dir(self, row: pd.Series) -> Path:
        video_stem = Path(row["file"]).stem
        return self.frames_root / row["binary_label"] / video_stem

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        frame_dir = self._frame_dir(row)
        frame_paths = sorted(frame_dir.glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found under {frame_dir}")

        # Uniform sample / pad to self.num_frames
        if len(frame_paths) >= self.num_frames:
            idxs = np.linspace(0, len(frame_paths) - 1, self.num_frames).astype(int)
            selected = [frame_paths[i] for i in idxs]
        else:
            selected = list(frame_paths) + [frame_paths[-1]] * (self.num_frames - len(frame_paths))

        frames = []
        for p in selected:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)
        frames_tensor = torch.stack(frames, dim=0)  # (num_frames, C, H, W)

        label = torch.tensor(int(row["binary_target"]), dtype=torch.long)
        return frames_tensor, label


def build_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Class-balanced sampler so each epoch sees roughly equal real/fake examples."""
    targets = df["binary_target"].to_numpy()
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    frames_root: Path,
    train_transform,
    eval_transform,
    batch_size: int,
    num_frames: int = 16,
    num_workers: int = 4,
    weighted_sampler: bool = True,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience constructor for the three canonical split loaders."""
    train_ds = DeepfakeBinaryDataset(train_df, frames_root, train_transform, num_frames)
    val_ds   = DeepfakeBinaryDataset(val_df,   frames_root, eval_transform,  num_frames)
    test_ds  = DeepfakeBinaryDataset(test_df,  frames_root, eval_transform,  num_frames)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    if weighted_sampler:
        train_loader = DataLoader(train_ds, sampler=build_weighted_sampler(train_df), **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader  = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

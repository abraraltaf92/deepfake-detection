"""Validation script for src/training.py. Run with:
    .venv/bin/python tests/test_training.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DeepfakeBinaryDataset
from src.models import ResNetBinaryVideoClassifier
from src.training import (
    pick_device, train_one_epoch, train_two_stage, save_checkpoint, load_checkpoint
)
from tests.fixtures import make_tiny_split_df, make_fake_frames_root, tempdir
from torch.utils.data import DataLoader


def main() -> None:
    # --- pick_device ---
    dev = pick_device()
    assert dev.type in {"cuda", "mps", "cpu"}, dev
    # On this Mac (MPS build available) expect mps
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        assert dev.type == "mps", f"Expected mps on this Mac, got {dev}"

    # --- tiny end-to-end train_two_stage ---
    tmp = tempdir()
    try:
        df = make_tiny_split_df(n_real=2, n_fake=2)
        frames_root = make_fake_frames_root(tmp, df, num_frames=4, img_size=64)  # tiny for speed
        tfm = T.Compose([T.ToTensor()])
        train_ds = DeepfakeBinaryDataset(df, frames_root, tfm, num_frames=4)
        train_loader = DataLoader(train_ds, batch_size=2, num_workers=0)
        val_loader   = DataLoader(train_ds, batch_size=2, num_workers=0)

        model = ResNetBinaryVideoClassifier()
        config = dict(
            lr_stage1=1e-3, lr_stage2=1e-4,
            epochs_stage1=1, epochs_stage2=1,
            checkpoint_dir=str(tmp / "ckpts"),
            run_id="test_run",
            class_weights=None,
        )
        best_state, history = train_two_stage(model, train_loader, val_loader, config, device=torch.device("cpu"))

        assert "stage1" in history and "stage2" in history
        for stage in ("stage1", "stage2"):
            for key in ("train_loss", "val_loss", "val_acc"):
                assert key in history[stage], f"missing {stage}/{key}"
                assert len(history[stage][key]) == 1, history[stage][key]
        assert isinstance(best_state, dict)

        # --- save / load round-trip ---
        ckpt_path = tmp / "roundtrip.pth"
        save_checkpoint(best_state, ckpt_path, meta={"test": True})
        model2 = ResNetBinaryVideoClassifier()
        meta = load_checkpoint(model2, ckpt_path)
        assert meta.get("test") is True

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

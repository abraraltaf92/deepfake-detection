"""Validation for src/visualization.py. Run with:
    .venv/bin/python tests/test_visualization.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization import (
    plot_roc_curves, plot_confusion_matrix, plot_training_history,
)
from tests.fixtures import tempdir


def main() -> None:
    tmp = tempdir()
    try:
        results = {
            "resnet18":       {"y_true": [0,1,0,1,1,0], "y_probs": [0.1,0.9,0.2,0.8,0.7,0.3]},
            "efficientnet":   {"y_true": [0,1,0,1,1,0], "y_probs": [0.2,0.7,0.1,0.9,0.8,0.2]},
        }
        fig = plot_roc_curves(results, title="Smoke")
        fig.savefig(tmp / "roc.png"); assert (tmp / "roc.png").exists()

        fig = plot_confusion_matrix([0,1,0,1,1,0], [0,1,1,1,0,0], labels=["real","fake"])
        fig.savefig(tmp / "cm.png"); assert (tmp / "cm.png").exists()

        history = {"stage1": {"train_loss":[1.0,0.5], "val_loss":[1.1,0.6], "val_acc":[0.5,0.7]},
                   "stage2": {"train_loss":[0.4,0.3], "val_loss":[0.5,0.4], "val_acc":[0.8,0.85]}}
        fig = plot_training_history(history)
        fig.savefig(tmp / "hist.png"); assert (tmp / "hist.png").exists()

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

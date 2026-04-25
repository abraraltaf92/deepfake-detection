"""Validation script for src/evaluation.py. Run with:
    .venv/bin/python tests/test_evaluation.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DeepfakeBinaryDataset
from src.models import ResNetBinaryVideoClassifier
from src.evaluation import evaluate, per_manipulation_breakdown, cross_dataset_eval
from tests.fixtures import make_tiny_split_df, make_fake_frames_root, tempdir, FAKE_CLASSES


def main() -> None:
    tmp = tempdir()
    try:
        df = make_tiny_split_df(n_real=4, n_fake=4)
        frames_root = make_fake_frames_root(tmp, df, num_frames=4, img_size=64)
        tfm = T.Compose([T.ToTensor()])
        ds = DeepfakeBinaryDataset(df, frames_root, tfm, num_frames=4)
        loader = DataLoader(ds, batch_size=2, num_workers=0)

        model = ResNetBinaryVideoClassifier()
        criterion = nn.CrossEntropyLoss()

        # --- evaluate returns the expected metric dict ---
        metrics = evaluate(model, loader, criterion, device=torch.device("cpu"))
        for k in ("accuracy", "precision", "recall", "f1", "auc",
                  "y_true", "y_pred", "y_probs", "confusion_matrix", "loss"):
            assert k in metrics, f"missing key: {k}"
        assert len(metrics["y_true"]) == 8
        assert len(metrics["y_pred"]) == 8
        assert len(metrics["y_probs"]) == 8

        # --- per-manipulation breakdown ---
        # Build a "test-like" df where each row has a source_class
        br = per_manipulation_breakdown(df, metrics["y_pred"], metrics["y_probs"], FAKE_CLASSES)
        # All 4 fake classes should appear in the breakdown
        assert set(br.keys()) == set(FAKE_CLASSES), br

        # --- cross_dataset_eval ---
        xd = cross_dataset_eval(model, {"ffpp_test": loader, "celebdf_test": loader}, criterion, device=torch.device("cpu"))
        assert set(xd.keys()) == {"ffpp_test", "celebdf_test"}
        assert "auc" in xd["ffpp_test"]

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

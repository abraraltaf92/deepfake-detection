"""Validation script for src/evaluation.py. Run with:
    .venv/bin/python tests/test_evaluation.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DeepfakeBinaryDataset
from src.models import ResNetBinaryVideoClassifier
from src.evaluation import evaluate, per_manipulation_breakdown, cross_dataset_eval
from tests.fixtures import make_tiny_split_df, make_fake_frames_root, tempdir, FAKE_CLASSES


def _build_synthetic_test_df(n_real: int, n_fake_per_class: int) -> pd.DataFrame:
    """Synthetic df mirroring FF++ test schema: 1 real class + 5 fake classes."""
    rows = []
    for _ in range(n_real):
        rows.append({"binary_target": 0, "source_class": "original"})
    for cls in FAKE_CLASSES:
        for _ in range(n_fake_per_class):
            rows.append({"binary_target": 1, "source_class": cls})
    return pd.DataFrame(rows)


def test_per_manipulation_varies_with_predictions() -> None:
    """When predictions DIFFER per manipulation, AUCs MUST differ.
    This catches regressions where the function silently returns the overall AUC.
    """
    np.random.seed(42)
    df = _build_synthetic_test_df(n_real=150, n_fake_per_class=150)

    # Per-class noise: model is excellent on Deepfakes, near-random on FaceShifter
    class_dist = {
        "original":       (0.05, 0.05),
        "Deepfakes":      (0.95, 0.05),
        "Face2Face":      (0.85, 0.15),
        "FaceSwap":       (0.75, 0.20),
        "NeuralTextures": (0.55, 0.25),
        "FaceShifter":    (0.40, 0.30),
    }
    y_probs = []
    for _, row in df.iterrows():
        mu, sigma = class_dist[row["source_class"]]
        y_probs.append(float(np.clip(np.random.normal(mu, sigma), 0, 1)))
    y_pred = [1 if p >= 0.5 else 0 for p in y_probs]

    out = per_manipulation_breakdown(df, y_pred, y_probs, FAKE_CLASSES)
    aucs = [out[m]["auc"] for m in FAKE_CLASSES]

    # Sanity: ordering matches the noise gradient.
    assert out["Deepfakes"]["auc"] > out["FaceShifter"]["auc"], (
        f"expected Deepfakes AUC > FaceShifter AUC; got {out}"
    )
    # Critical regression check: AUCs MUST differ across classes when predictions do.
    assert len(set(aucs)) > 1, (
        f"per_manipulation_breakdown returned identical AUCs across classes "
        f"despite class-varied predictions — likely a filter/alignment regression. "
        f"AUCs: {aucs}"
    )


def test_per_manipulation_identical_when_predictions_uniform() -> None:
    """When the model assigns the SAME confidence distribution to every fake class,
    per-class AUCs are mathematically identical. This is correct behavior, NOT a bug.
    Documenting this here so future readers understand the 06 notebook output.
    """
    df = _build_synthetic_test_df(n_real=150, n_fake_per_class=150)
    n_real = 150
    n_fake = 150 * len(FAKE_CLASSES)
    y_probs = [0.05] * n_real + [0.95] * n_fake
    y_pred  = [0]    * n_real + [1]    * n_fake

    out = per_manipulation_breakdown(df, y_pred, y_probs, FAKE_CLASSES)
    aucs = [out[m]["auc"] for m in FAKE_CLASSES]
    assert all(a == aucs[0] for a in aucs), aucs
    assert aucs[0] == 1.0, aucs


def test_per_manipulation_handles_missing_classes() -> None:
    """If a manipulation class has zero rows in df, breakdown should skip it
    gracefully (continue, don't raise)."""
    df = _build_synthetic_test_df(n_real=150, n_fake_per_class=150)
    df = df[df["source_class"] != "FaceShifter"].reset_index(drop=True)
    y_probs = [0.5] * len(df)
    y_pred  = [1]   * len(df)

    out = per_manipulation_breakdown(df, y_pred, y_probs, FAKE_CLASSES)
    assert "FaceShifter" not in out, out
    assert set(out.keys()) <= set(FAKE_CLASSES) - {"FaceShifter"}


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

        # --- per-manipulation breakdown (smoke test) ---
        # The tiny fixture cycles through FAKE_CLASSES, so only the first
        # n_fake values appear; assert the breakdown matches the actual
        # manipulations represented in the df.
        br = per_manipulation_breakdown(df, metrics["y_pred"], metrics["y_probs"], FAKE_CLASSES)
        expected = set(df[df["binary_target"] == 1]["source_class"].unique())
        assert set(br.keys()) == expected, (br, expected)

        # --- per-manipulation behavioral tests ---
        test_per_manipulation_varies_with_predictions()
        test_per_manipulation_identical_when_predictions_uniform()
        test_per_manipulation_handles_missing_classes()

        # --- cross_dataset_eval ---
        xd = cross_dataset_eval(model, {"ffpp_test": loader, "celebdf_test": loader}, criterion, device=torch.device("cpu"))
        assert set(xd.keys()) == {"ffpp_test", "celebdf_test"}
        assert "auc" in xd["ffpp_test"]

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

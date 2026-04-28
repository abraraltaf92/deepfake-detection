"""Evaluation utilities — metrics, per-manipulation breakdown, cross-dataset eval.

All functions return plain dicts that can be serialized straight to JSON.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Full evaluation pass. Returns accuracy, precision, recall, f1, auc + raw arrays."""
    model.to(device)
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    total_loss = 0.0
    n_batches = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                f"evaluate expects binary (B, 2) logits; got shape {tuple(logits.shape)}"
            )
        loss = criterion(logits, labels)
        total_loss += float(loss.cpu().item())
        n_batches += 1

        probs = torch.softmax(logits, dim=1)[:, 1]  # prob of "fake"
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); y_probs = np.asarray(y_probs)

    # AUC is undefined if only one class is present — emit None in that case.
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        warnings.warn(
            "evaluate: AUC undefined (only one class in y_true). "
            "Returning auc=None — check loader configuration.",
            RuntimeWarning, stacklevel=2,
        )
        auc = None

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       auc,
        "loss":      total_loss / max(n_batches, 1),
        "y_true":    y_true.tolist(),
        "y_pred":    y_pred.tolist(),
        "y_probs":   y_probs.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def per_manipulation_breakdown(
    df: pd.DataFrame,
    y_pred: list[int],
    y_probs: list[float],
    fake_classes: Iterable[str],
) -> dict:
    """For each fake manipulation type, compute AUC / F1 using that class + all reals.

    `df` must match the order of y_pred / y_probs — i.e. it's the test DataFrame
    that produced those predictions.

    Note on identical AUCs across manipulations:
        If a model assigns the SAME confidence distribution to every fake class
        (no per-class variation), all per-manip AUCs will be mathematically
        identical to the overall AUC. This is correct behavior, not a bug —
        it indicates the model doesn't discriminate between manipulation types.

    Returns dict mapping manipulation name → {auc, f1, n_videos, n_fakes,
    mean_prob_real, mean_prob_fake}. Manipulations with zero rows in `df` are
    skipped entirely (not returned with auc=None).
    """
    if len(y_pred) != len(df):
        raise ValueError(
            f"length mismatch: len(df)={len(df)} but len(y_pred)={len(y_pred)}. "
            "df must match the rows that produced y_pred / y_probs."
        )

    df = df.reset_index(drop=True).copy()
    df["_pred"]  = y_pred
    df["_probs"] = y_probs

    out = {}
    real_mask = df["source_class"] == "original"
    for manip in fake_classes:
        manip_mask = df["source_class"] == manip
        # Skip if this manipulation has zero rows — returning an entry with
        # auc=None just because reals exist would be misleading.
        n_fakes = int(manip_mask.sum())
        if n_fakes == 0:
            continue
        subset = df[real_mask | manip_mask]
        if subset.empty:
            continue
        y_true = subset["binary_target"].to_numpy()
        y_p    = subset["_pred"].to_numpy()
        y_s    = subset["_probs"].to_numpy()
        try:
            auc = float(roc_auc_score(y_true, y_s)) if len(set(y_true)) > 1 else None
        except ValueError:
            auc = None
        # Mean probability per ground-truth label inside the subset — surfaces
        # WHY two manips may produce identical AUCs (same per-class confidence).
        real_subset = subset[subset["binary_target"] == 0]
        fake_subset = subset[subset["binary_target"] == 1]
        out[manip] = {
            "auc":            auc,
            "f1":             float(f1_score(y_true, y_p, zero_division=0)),
            "n_videos":       int(len(subset)),
            "n_fakes":        n_fakes,
            "mean_prob_real": float(real_subset["_probs"].mean()) if len(real_subset) else None,
            "mean_prob_fake": float(fake_subset["_probs"].mean()) if len(fake_subset) else None,
        }
    return out


def cross_dataset_eval(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the same model on multiple named loaders. Returns dict keyed by loader name."""
    return {name: evaluate(model, loader, criterion, device) for name, loader in loaders.items()}

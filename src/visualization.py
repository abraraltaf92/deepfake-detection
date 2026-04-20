"""Plotting helpers: ROC overlay, confusion matrix, training history, Grad-CAM panel."""
from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix


def plot_roc_curves(results: dict, title: str = "ROC") -> plt.Figure:
    """Overlay ROC curves for multiple models.

    `results` is a dict: {model_name: {"y_true": [...], "y_probs": [...]}, ...}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_probs"])
        from sklearn.metrics import auc as _auc
        ax.plot(fpr, tpr, label=f"{name} (AUC={_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels: Iterable[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels), yticklabels=list(labels), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_training_history(history: dict) -> plt.Figure:
    """Two-row plot: losses on top, val accuracy on bottom, with stage boundary marked."""
    s1 = history["stage1"]; s2 = history["stage2"]
    e1 = len(s1["train_loss"])
    x_all = list(range(1, e1 + len(s2["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(x_all, s1["train_loss"] + s2["train_loss"], label="train loss")
    ax1.plot(x_all, s1["val_loss"]   + s2["val_loss"],   label="val loss")
    ax1.axvline(e1 + 0.5, color="gray", linestyle="--", alpha=0.6, label="stage boundary")
    ax1.set_ylabel("loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(x_all, s1["val_acc"] + s2["val_acc"], label="val acc", color="green")
    ax2.axvline(e1 + 0.5, color="gray", linestyle="--", alpha=0.6)
    ax2.set_ylabel("val accuracy"); ax2.set_xlabel("epoch (stage1 | stage2)")
    ax2.grid(alpha=0.3); ax2.legend()
    fig.tight_layout()
    return fig


def grad_cam_panel(
    model,
    sample_frames,  # torch.Tensor of shape (N, C, H, W) — single-frame images
    target_layer,
    class_idx: int = 1,
    denorm_mean: Optional[list] = None,
    denorm_std: Optional[list] = None,
) -> plt.Figure:
    """Grid of Grad-CAM overlays, one column per sample frame.

    Uses pytorch-grad-cam (from requirements.txt via grad-cam package).
    Callers are responsible for passing in the correct `target_layer` for the model.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)] * sample_frames.shape[0]

    grayscale_cam = cam(input_tensor=sample_frames, targets=targets)

    n = sample_frames.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1: axes = [axes]
    mean = np.array(denorm_mean) if denorm_mean else np.array([0.0, 0.0, 0.0])
    std  = np.array(denorm_std)  if denorm_std  else np.array([1.0, 1.0, 1.0])
    for i in range(n):
        img = sample_frames[i].permute(1, 2, 0).cpu().numpy()
        img = (img * std + mean).clip(0, 1)
        overlay = show_cam_on_image(img, grayscale_cam[i], use_rgb=True)
        axes[i].imshow(overlay); axes[i].axis("off")
    fig.tight_layout()
    return fig

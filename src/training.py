"""Training utilities — device selection, single-epoch loop, 2-stage fine-tune.

The 2-stage recipe (stage 1: head-only with frozen backbone; stage 2: full
fine-tune at lower LR) was duplicated four times in the teammate's notebook,
once per model. Here it lives once.

Mixed precision: enabled via GradScaler only on CUDA. MPS and CPU paths run
in full precision — GradScaler's MPS support lags CUDA.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def pick_device() -> torch.device:
    """Return the best available device in priority order: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """Run one training epoch. Returns mean loss over the epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(frames)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _quick_val(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    """Lightweight loss + accuracy on val set. For full metrics use src.evaluation.evaluate."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
        loss = criterion(logits, labels)
        total_loss += float(loss.cpu().item())
        n_batches += 1
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().cpu().item())
        total += int(labels.numel())
    return total_loss / max(n_batches, 1), correct / max(total, 1)


def _set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


def train_two_stage(
    model: "src.models.DeepfakeClassifier",
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: Optional[torch.device] = None,
) -> tuple[dict, dict]:
    """Two-stage transfer learning:
       stage 1: freeze backbone, train head only at lr_stage1.
       stage 2: unfreeze all, fine-tune at lr_stage2.

    Checkpoints per epoch to `config['checkpoint_dir']/<run_id>_stage{1|2}_epoch{N}.pth`
    so Colab disconnects don't lose work.

    Returns (best_state_dict_across_both_stages, history) where history is:
        { "stage1": {"train_loss": [...], "val_loss": [...], "val_acc": [...]},
          "stage2": {"train_loss": [...], "val_loss": [...], "val_acc": [...]} }
    """
    device = device or pick_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=config.get("class_weights"))
    if isinstance(criterion.weight, torch.Tensor):
        criterion.weight = criterion.weight.to(device)

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id = config["run_id"]

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {"stage1": {"train_loss": [], "val_loss": [], "val_acc": []},
               "stage2": {"train_loss": [], "val_loss": [], "val_acc": []}}
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    # ---- Stage 1: head only ----
    _set_requires_grad(model.backbone_parameters(), False)
    _set_requires_grad(model.head_parameters(), True)
    optimizer = torch.optim.Adam(model.head_parameters(), lr=config["lr_stage1"])
    for epoch in range(config["epochs_stage1"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = _quick_val(model, val_loader, criterion, device)
        history["stage1"]["train_loss"].append(train_loss)
        history["stage1"]["val_loss"].append(val_loss)
        history["stage1"]["val_acc"].append(val_acc)
        print(f"[stage1 epoch {epoch+1}/{config['epochs_stage1']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")
        save_checkpoint(model.state_dict(), ckpt_dir / f"{run_id}_stage1_epoch{epoch+1}.pth",
                        meta={"stage": 1, "epoch": epoch + 1, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # ---- Stage 2: full fine-tune ----
    _set_requires_grad(model.backbone_parameters(), True)
    _set_requires_grad(model.head_parameters(), True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_stage2"])
    for epoch in range(config["epochs_stage2"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = _quick_val(model, val_loader, criterion, device)
        history["stage2"]["train_loss"].append(train_loss)
        history["stage2"]["val_loss"].append(val_loss)
        history["stage2"]["val_acc"].append(val_acc)
        print(f"[stage2 epoch {epoch+1}/{config['epochs_stage2']}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")
        save_checkpoint(model.state_dict(), ckpt_dir / f"{run_id}_stage2_epoch{epoch+1}.pth",
                        meta={"stage": 2, "epoch": epoch + 1, "val_acc": val_acc})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    save_checkpoint(best_state, ckpt_dir / f"{run_id}_best.pth",
                    meta={"best_val_acc": best_val_acc})
    return best_state, history


def save_checkpoint(state_dict: dict, path: Path, meta: Optional[dict] = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "meta": meta or {}}, path)


def load_checkpoint(model: nn.Module, path: Path) -> dict:
    """Load weights into `model` in-place. Returns the meta dict stored alongside."""
    data = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(data["state_dict"])
    return data.get("meta", {})

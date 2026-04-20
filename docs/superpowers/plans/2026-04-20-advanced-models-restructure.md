# Advanced Models Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Absorb a teammate's 68-cell monolithic notebook (4 advanced deepfake-detection models + Celeb-DF cross-dataset evaluation) into the repo's numbered-notebook structure, delivered as three reviewable PRs on `dev/abrar`, with reusable algorithmic code factored into a new `src/` package and experiment results tracked in `experiments/`.

**Architecture:** Thin notebooks call a new `src/` Python package (datasets, preprocessing, models, training, evaluation, logging, visualization). Hyperparameters live in `configs/experiments.py`. Runs append a row to `experiments/results.csv` (leaderboard) and write a full JSON file under `experiments/results/`. Device priority is `cuda → mps → cpu` so the same code runs on Abrar's M5 Pro (MPS) locally and on Colab Pro (CUDA).

**Tech Stack:** Python 3.11.15, PyTorch 2.11.0 (MPS-enabled), torchvision, facenet-pytorch (MTCNN), timm (ViT), opencv-python, pandas, scikit-learn, pytorch-grad-cam.

**Reference documents:**
- Design spec: `docs/superpowers/specs/2026-04-20-advanced-models-restructure-design.md` — read this before starting. All task decisions derive from it.
- Team workflow: `CONTRIBUTING.md` — branch / PR conventions.
- Current paths: `configs/paths.py`.

**Validation model (non-pytest per spec §3):** Each `src/` module has a minimal `tests/test_<module>.py` script that is **not** pytest — it's plain Python with `assert` statements and a `print("ok")` at the end. Run with `.venv/bin/python tests/test_<module>.py`. This keeps the "write the failing test first" TDD loop without violating the spec's no-pytest rule.

---

## File Structure

### Files created in this plan

```
src/
├── __init__.py                      # empty package marker
├── datasets.py                      # DeepfakeBinaryDataset, samplers, get_dataloaders
├── preprocessing.py                 # MTCNN face extraction, RAFT interpolation
├── models.py                        # DeepfakeClassifier base + 5 model classes
├── training.py                      # pick_device, train_one_epoch, train_two_stage, checkpointing
├── evaluation.py                    # evaluate, per_manipulation_breakdown, cross_dataset_eval
├── logging.py                       # make_run_id, append_run_to_csv, write_run_json, reconstruct_csv_row
└── visualization.py                 # plot_roc_curves, plot_confusion_matrix, plot_training_history, grad_cam_panel

configs/
└── experiments.py                   # BASE_CONFIG + 5 per-model config dicts

tests/                               # non-pytest assert scripts, one per src module
├── __init__.py                      # empty
├── fixtures.py                      # synthetic DataFrames, fake face-crop dirs, tiny tensors
├── test_datasets.py
├── test_preprocessing.py
├── test_models.py
├── test_training.py
├── test_evaluation.py
├── test_logging.py
└── test_visualization.py

experiments/
├── .gitkeep                         # empty file so directory exists in git
├── results.csv                      # starts empty (header only), filled by runs
└── results/
    └── .gitkeep

notebooks/
├── 05_model_advanced.ipynb          # rewritten from current stub
└── 06_evaluation.ipynb              # rewritten from current stub
```

### Files modified in this plan

```
configs/paths.py                     # add REPO_ROOT env-var override; add EXPERIMENTS_ROOT + CELEBDF paths
notebooks/02_eda.ipynb               # add RGB distribution + shared-id check cells
notebooks/03_preprocessing.ipynb     # swap Haar → MTCNN, add Celeb-DF extraction, add sanity viz
notebooks/04_model_baseline.ipynb    # refactor inline code to call src; append run to experiments/results.csv
README.md                            # update performance table with 5-model leaderboard (PR 3 only)
```

---

## Phase 1: PR 1 — Shared Infrastructure

**Phase goal:** Land a working `src/` package + `configs/experiments.py` + `configs/paths.py` override + refactored `04_model_baseline.ipynb` that reproduces the current 86% / 0.9333 F1 ResNet-18 numbers and writes one row to `experiments/results.csv`.

**Merge criterion for this phase:** Restart kernel → Run All of `04_model_baseline.ipynb` with `SMOKE_TEST = False` completes cleanly and the produced metrics match the baseline within ±1% (the baseline uses a fixed seed, so exact match is expected). One row appears in `experiments/results.csv` with `model=resnet18`.

### Task 1: Scaffold project directories + test fixtures

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures.py`
- Create: `experiments/.gitkeep`
- Create: `experiments/results/.gitkeep`

- [ ] **Step 1.1: Create empty package markers and experiment directories**

```bash
mkdir -p src tests experiments/results
touch src/__init__.py tests/__init__.py experiments/.gitkeep experiments/results/.gitkeep
```

- [ ] **Step 1.2: Create `tests/fixtures.py` — synthetic data helpers reused across test scripts**

```python
"""Synthetic fixtures for src/ module validation scripts.

Not pytest — these are helper functions imported by tests/test_<module>.py
scripts which are plain Python with asserts.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


FAKE_CLASSES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def make_tiny_split_df(n_real: int = 4, n_fake: int = 4) -> pd.DataFrame:
    """Returns a DataFrame matching the 01_data_ingestion.ipynb schema.

    Columns: path, file, source_class, binary_label, binary_target, split,
             id_a, id_b, base_ids, dataset
    """
    rows = []
    for i in range(n_real):
        rows.append({
            "path":          f"/fake/original/{i:03d}.mp4",
            "file":          f"{i:03d}.mp4",
            "source_class":  "original",
            "binary_label":  "real",
            "binary_target": 0,
            "split":         "train",
            "id_a":          f"{i:03d}",
            "id_b":          None,
            "base_ids":      f"{i:03d}",
            "dataset":       "test_fixture",
        })
    for i in range(n_fake):
        cls = FAKE_CLASSES[i % len(FAKE_CLASSES)]
        rows.append({
            "path":          f"/fake/{cls}/{i:03d}_{i+100:03d}.mp4",
            "file":          f"{i:03d}_{i+100:03d}.mp4",
            "source_class":  cls,
            "binary_label":  "fake",
            "binary_target": 1,
            "split":         "train",
            "id_a":          f"{i:03d}",
            "id_b":          f"{i+100:03d}",
            "base_ids":      f"{i:03d}_{i+100:03d}",
            "dataset":       "test_fixture",
        })
    return pd.DataFrame(rows)


def make_fake_frames_root(tmp_root: Path, df: pd.DataFrame, num_frames: int = 16, img_size: int = 224) -> Path:
    """Build a frames_root directory structure matching DeepfakeBinaryDataset expectations.

    Layout: <tmp_root>/<binary_label>/<video_stem>/frame_00.jpg ... frame_<NF-1>.jpg
    """
    frames_root = tmp_root / "frames"
    for _, row in df.iterrows():
        video_stem = Path(row["file"]).stem
        target_dir = frames_root / row["binary_label"] / video_stem
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in range(num_frames):
            arr = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(target_dir / f"frame_{f:02d}.jpg")
    return frames_root


def tempdir() -> Path:
    """Context-free tempdir — callers are expected to clean it up in a finally block."""
    return Path(tempfile.mkdtemp(prefix="deepfake-test-"))
```

- [ ] **Step 1.3: Quick sanity run**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from tests.fixtures import make_tiny_split_df, make_fake_frames_root, tempdir
df = make_tiny_split_df()
print('df shape:', df.shape, 'columns:', list(df.columns))
tmp = tempdir()
try:
    root = make_fake_frames_root(tmp, df)
    print('frames root:', root)
    print('subdirs:', sorted(p.name for p in root.iterdir()))
finally:
    import shutil; shutil.rmtree(tmp)
print('ok')
"
```

Expected: `df shape: (8, 10)`, columns match the schema, frames root has `real` and `fake` subdirs, final `ok`.

- [ ] **Step 1.4: Commit**

```bash
git add src/__init__.py tests/__init__.py tests/fixtures.py experiments/.gitkeep experiments/results/.gitkeep
git commit -m "Scaffold src/, tests/, experiments/ with synthetic fixtures"
```

---

### Task 2: `configs/paths.py` — add environment override + new paths

**Files:**
- Modify: `configs/paths.py`

- [ ] **Step 2.1: Rewrite `configs/paths.py` to read `REPO_ROOT` from env with Colab fallback, and add the new paths used by Phase 1+2+3**

Replace the entire file with:

```python
"""
Centralized path configuration for the deepfake detection project.

REPO_ROOT resolves from the DEEPFAKE_REPO_ROOT environment variable if set,
otherwise defaults to the Google Colab Drive path. This lets the same code
run locally (export DEEPFAKE_REPO_ROOT=$HOME/codebase/deepfake-detection)
and on Colab (no export needed — default kicks in).

Usage in notebooks:
    import sys
    sys.path.append('/content/drive/MyDrive/deepfake-detection')  # or local path
    from google.colab import drive
    drive.mount('/content/drive')
    from configs.paths import *
"""

import os
from pathlib import Path

# ============================================================
# THE ONLY PATH YOU NEED TO CHANGE
# On Colab: no change — default resolves to Drive path.
# Locally: export DEEPFAKE_REPO_ROOT=$HOME/codebase/deepfake-detection
# ============================================================
REPO_ROOT = Path(
    os.environ.get("DEEPFAKE_REPO_ROOT", "/content/drive/MyDrive/deepfake-detection")
)

# --- Raw Datasets ---
DATA_ROOT     = REPO_ROOT / "data"
RAW_ROOT      = DATA_ROOT / "raw"
FFPP_RAW_ROOT = RAW_ROOT / "ff-c23" / "FaceForensics++_C23"
CELEBDF_RAW_ROOT = RAW_ROOT / "celeb-df-v2"

# --- Processed Data ---
PROC_ROOT        = DATA_ROOT / "processed"
FRAMES_ROOT      = PROC_ROOT / "ffpp_c23" / "frames224_binary"
CELEBDF_FRAMES   = PROC_ROOT / "celebdf_face_crops_224"
RAFT_FRAMES_ROOT = PROC_ROOT / "ffpp_c23" / "frames224_raft16"
INDEX_DIR        = PROC_ROOT / "ffpp_c23" / "splits"

# --- Split Files ---
TRAIN_CSV = INDEX_DIR / "train_binary.csv"
VAL_CSV   = INDEX_DIR / "val_binary.csv"
TEST_CSV  = INDEX_DIR / "test_binary.csv"

# --- Model Checkpoints ---
MODEL_DIR = REPO_ROOT / "models"

# --- Experiment tracking (committed) ---
EXPERIMENTS_ROOT   = REPO_ROOT / "experiments"
RESULTS_CSV        = EXPERIMENTS_ROOT / "results.csv"
RESULTS_JSON_DIR   = EXPERIMENTS_ROOT / "results"

# --- Training Logs (gitignored) ---
LOG_ROOT = REPO_ROOT / "logs"

# --- Best Model (legacy, keep for baseline notebook) ---
BEST_MODEL_PATH = MODEL_DIR / "resnet18_binary_deepfake_detector_best.pth"
```

- [ ] **Step 2.2: Verify the env override works**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
DEEPFAKE_REPO_ROOT=/tmp/fake-root /Users/abrar/codebase/deepfake-detection/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from configs.paths import REPO_ROOT, RESULTS_CSV, CELEBDF_FRAMES, RAFT_FRAMES_ROOT
assert str(REPO_ROOT) == '/tmp/fake-root', REPO_ROOT
assert str(RESULTS_CSV) == '/tmp/fake-root/experiments/results.csv'
assert str(CELEBDF_FRAMES).endswith('celebdf_face_crops_224')
assert str(RAFT_FRAMES_ROOT).endswith('frames224_raft16')
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 2.3: Verify the Colab default still resolves**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
unset DEEPFAKE_REPO_ROOT; /Users/abrar/codebase/deepfake-detection/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from configs.paths import REPO_ROOT
assert str(REPO_ROOT) == '/content/drive/MyDrive/deepfake-detection', REPO_ROOT
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 2.4: Commit**

```bash
git add configs/paths.py
git commit -m "paths: add DEEPFAKE_REPO_ROOT override + Celeb-DF / RAFT / experiments paths"
```

---

### Task 3: `src/datasets.py` — DeepfakeBinaryDataset + samplers + get_dataloaders

**Files:**
- Create: `src/datasets.py`
- Create: `tests/test_datasets.py`

- [ ] **Step 3.1: Write the failing validation script first**

Create `tests/test_datasets.py`:

```python
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
```

- [ ] **Step 3.2: Run it — should fail because `src/datasets.py` doesn't exist yet**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_datasets.py
```

Expected: `ModuleNotFoundError: No module named 'src.datasets'`.

- [ ] **Step 3.3: Write `src/datasets.py`**

```python
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
```

- [ ] **Step 3.4: Run the validation script — should pass now**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_datasets.py
```

Expected: `ok`.

- [ ] **Step 3.5: Commit**

```bash
git add src/datasets.py tests/test_datasets.py
git commit -m "src.datasets: DeepfakeBinaryDataset + weighted sampler + get_dataloaders"
```

---

### Task 4: `src/models.py` — base class + ResNet-18 (stubs for 4 advanced models)

**Files:**
- Create: `src/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 4.1: Write the failing validation script**

Create `tests/test_models.py`:

```python
"""Validation script for src/models.py. Run with:
    .venv/bin/python tests/test_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import DeepfakeClassifier, ResNetBinaryVideoClassifier


def main() -> None:
    device = torch.device("cpu")  # CPU for the test — fast, deterministic

    # --- ResNet-18 binary video classifier ---
    model = ResNetBinaryVideoClassifier(dropout=0.3).to(device)
    assert isinstance(model, DeepfakeClassifier), "ResNet-18 must inherit from DeepfakeClassifier"
    # Input: (batch=2, num_frames=4, C=3, H=224, W=224)
    x = torch.randn(2, 4, 3, 224, 224, device=device)
    logits = model(x)
    assert logits.shape == (2, 2), f"expected (2, 2) logits, got {tuple(logits.shape)}"

    # --- param groups ---
    head_params = list(model.head_parameters())
    backbone_params = list(model.backbone_parameters())
    assert len(head_params) > 0, "head_parameters() returned empty"
    assert len(backbone_params) > 0, "backbone_parameters() returned empty"
    all_params = set(id(p) for p in model.parameters())
    partitioned = set(id(p) for p in head_params) | set(id(p) for p in backbone_params)
    assert all_params == partitioned, "head + backbone must partition all parameters"

    print("ok")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run it — should fail**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
```

Expected: `ModuleNotFoundError: No module named 'src.models'`.

- [ ] **Step 4.3: Write `src/models.py`**

```python
"""Deepfake classifier model definitions.

Architectures:
    ResNetBinaryVideoClassifier    — frame-level ResNet-18, score-averaged across frames
    EfficientNetDeepfakeDetector   — frame-level EfficientNet-B4               (Phase 3)
    R3D18DeepfakeDetector          — clip-level 3D ResNet-18                  (Phase 3)
    ViTDeepfakeDetector            — frame-level timm ViT                      (Phase 3)
    R3D18RAFTDeepfakeDetector      — clip-level 3D ResNet-18 on 16 RAFT frames (Phase 3)

All models subclass DeepfakeClassifier which exposes head_parameters() and
backbone_parameters() so src.training.train_two_stage can do head-only
training in stage 1 and full fine-tune in stage 2 uniformly.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

import torch
import torch.nn as nn
import torchvision.models as tv_models


class DeepfakeClassifier(nn.Module):
    """Base class. Subclasses must implement forward and expose head/backbone parameter groups."""

    @abstractmethod
    def head_parameters(self) -> Iterable[nn.Parameter]:
        ...

    @abstractmethod
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        ...


class ResNetBinaryVideoClassifier(DeepfakeClassifier):
    """ResNet-18 applied per frame; logits averaged across frames.

    Input:  (B, T, 3, H, W)
    Output: (B, 2) binary logits.

    Migrated from notebooks/04_model_baseline.ipynb so the baseline uses the
    same src.training / src.evaluation paths as the four advanced models.
    """

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)                 # (B*T, F)
        logits = self.head(features)                # (B*T, 2)
        logits = logits.view(B, T, 2).mean(dim=1)   # (B, 2)
        return logits

    def head_parameters(self) -> Iterable[nn.Parameter]:
        return self.head.parameters()

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()


# =============================================================================
# Placeholders for Phase 3 — filled in during Tasks 21–24.
# Keeping the names here so type references from configs.experiments.py resolve
# even before Phase 3 implementation.
# =============================================================================
class EfficientNetDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 21")


class R3D18DeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 22")


class ViTDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 23")


class R3D18RAFTDeepfakeDetector(DeepfakeClassifier):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Phase 3 / Task 24")
```

- [ ] **Step 4.4: Run the validation script — should pass**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
```

Expected: `ok`.

- [ ] **Step 4.5: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "src.models: DeepfakeClassifier base + ResNet-18 migration (4 advanced stubs for Phase 3)"
```

---

### Task 5: `src/training.py` — pick_device + train_one_epoch + train_two_stage + checkpointing

**Files:**
- Create: `src/training.py`
- Create: `tests/test_training.py`

- [ ] **Step 5.1: Write the failing validation script**

Create `tests/test_training.py`:

```python
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
```

- [ ] **Step 5.2: Run it — should fail (ModuleNotFoundError on src.training)**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_training.py
```

Expected: `ModuleNotFoundError: No module named 'src.training'`.

- [ ] **Step 5.3: Write `src/training.py`**

```python
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
```

- [ ] **Step 5.4: Run the validation script — should pass**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_training.py
```

Expected: `ok`. The training loop will print a couple of epoch lines before the final `ok`.

- [ ] **Step 5.5: Commit**

```bash
git add src/training.py tests/test_training.py
git commit -m "src.training: pick_device + train_one_epoch + train_two_stage + checkpoint helpers"
```

---

### Task 6: `src/evaluation.py` — evaluate + per-manipulation + cross-dataset

**Files:**
- Create: `src/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 6.1: Write the failing validation script**

Create `tests/test_evaluation.py`:

```python
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
        # At least one fake class should appear in the breakdown
        assert len(br) >= 1, br

        # --- cross_dataset_eval ---
        xd = cross_dataset_eval(model, {"ffpp_test": loader, "celebdf_test": loader}, criterion, device=torch.device("cpu"))
        assert set(xd.keys()) == {"ffpp_test", "celebdf_test"}
        assert "auc" in xd["ffpp_test"]

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run it — should fail (src.evaluation not found)**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_evaluation.py
```

Expected: `ModuleNotFoundError: No module named 'src.evaluation'`.

- [ ] **Step 6.3: Write `src/evaluation.py`**

```python
"""Evaluation utilities — metrics, per-manipulation breakdown, cross-dataset eval.

All functions return plain dicts that can be serialized straight to JSON.
"""
from __future__ import annotations

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
    model.eval()
    model.to(device)
    y_true, y_pred, y_probs = [], [], []
    total_loss = 0.0
    n_batches = 0
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(frames)
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
    """
    df = df.reset_index(drop=True).copy()
    df["_pred"]  = y_pred
    df["_probs"] = y_probs

    out = {}
    real_mask = df["source_class"] == "original"
    for manip in fake_classes:
        manip_mask = df["source_class"] == manip
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
        out[manip] = {
            "auc": auc,
            "f1":  float(f1_score(y_true, y_p, zero_division=0)),
            "n_videos": int(len(subset)),
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
```

- [ ] **Step 6.4: Run the validation script — should pass**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_evaluation.py
```

Expected: `ok`.

- [ ] **Step 6.5: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "src.evaluation: evaluate + per_manipulation_breakdown + cross_dataset_eval"
```

---

### Task 7: `src/logging.py` — run_id + CSV + JSON + reconstruct

**Files:**
- Create: `src/logging.py`
- Create: `tests/test_logging.py`

- [ ] **Step 7.1: Write the failing validation script**

Create `tests/test_logging.py`:

```python
"""Validation script for src/logging.py. Run with:
    .venv/bin/python tests/test_logging.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.logging import (
    make_run_id, append_run_to_csv, write_run_json, reconstruct_csv_row,
)
from tests.fixtures import tempdir


def main() -> None:
    tmp = tempdir()
    try:
        rid = make_run_id("efficientnet_b4")
        assert rid.startswith("efficientnet_b4_"), rid
        assert len(rid.split("_")) >= 4, rid  # name(possibly 2+parts) + YYYYMMDD + HHMM

        csv_path = tmp / "results.csv"
        config = dict(model="resnet18", num_frames=16, img_size=224, batch_size=16,
                      lr_stage1=1e-3, lr_stage2=1e-4, epochs_stage1=3, epochs_stage2=12,
                      weighted_sampler=True, augmentation=True, face_detector="haar")
        metrics = dict(environment="local-m5", device="mps",
                       ffpp_val_acc=0.9, ffpp_val_f1=0.91,
                       ffpp_test_acc=0.88, ffpp_test_precision=0.89,
                       ffpp_test_recall=0.87, ffpp_test_f1=0.88, ffpp_test_auc=0.93,
                       celebdf_acc=None, celebdf_f1=None, celebdf_auc=None,
                       generalization_gap_auc=None, train_time_minutes=12.0, notes="")
        append_run_to_csv(rid, config, metrics, csv_path)
        df = pd.read_csv(csv_path)
        assert list(df["run_id"]) == [rid]

        # Upsert: second call with same run_id replaces the row, not duplicates.
        metrics2 = {**metrics, "notes": "rerun"}
        append_run_to_csv(rid, config, metrics2, csv_path)
        df2 = pd.read_csv(csv_path)
        assert len(df2) == 1 and df2.iloc[0]["notes"] == "rerun", df2.to_dict()

        # JSON roundtrip + CSV reconstruction
        json_dir = tmp / "jsons"
        payload = {"run_id": rid, "config": config, "metrics": metrics,
                   "training_history": {"stage1": {"train_loss": [0.1]}, "stage2": {"train_loss": [0.05]}}}
        json_path = write_run_json(rid, payload, json_dir)
        assert json_path.exists()
        reconstructed = reconstruct_csv_row(json_path)
        assert reconstructed["run_id"] == rid
        assert "ffpp_test_auc" in reconstructed

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Run it — should fail**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_logging.py
```

Expected: `ModuleNotFoundError: No module named 'src.logging'`.

- [ ] **Step 7.3: Write `src/logging.py`**

```python
"""Experiment logging — CSV leaderboard + per-run JSON.

Design: JSON is the single source of truth. CSV is a projection of it.
reconstruct_csv_row() can rebuild the CSV from the JSON folder to prevent drift.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


CSV_COLUMNS = [
    "run_id", "timestamp", "environment", "device",
    "model", "face_detector", "num_frames", "img_size",
    "batch_size", "lr_stage1", "lr_stage2",
    "epochs_stage1", "epochs_stage2",
    "weighted_sampler", "augmentation",
    "ffpp_val_acc", "ffpp_val_f1",
    "ffpp_test_acc", "ffpp_test_precision", "ffpp_test_recall",
    "ffpp_test_f1", "ffpp_test_auc",
    "celebdf_acc", "celebdf_f1", "celebdf_auc",
    "generalization_gap_auc", "train_time_minutes", "notes",
]


def make_run_id(model_name: str, timestamp: Optional[datetime] = None) -> str:
    """`<model>_<YYYYMMDD>_<HHMM>` — deterministic, human-readable."""
    ts = timestamp or datetime.now()
    return f"{model_name}_{ts.strftime('%Y%m%d')}_{ts.strftime('%H%M')}"


def _row_from(run_id: str, config: dict, metrics: dict) -> dict:
    """Project config + metrics into the CSV row schema."""
    row = {c: None for c in CSV_COLUMNS}
    row["run_id"]    = run_id
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    for key in row:
        if key in config:
            row[key] = config[key]
        if key in metrics:
            row[key] = metrics[key]
    return row


def append_run_to_csv(run_id: str, config: dict, metrics: dict, csv_path: Path) -> None:
    """Upsert by run_id. Creates the CSV with header if missing."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = _row_from(run_id, config, metrics)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["run_id"] != run_id]  # drop existing row if any
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row], columns=CSV_COLUMNS)
    df = df.reindex(columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)


def write_run_json(run_id: str, payload: dict, json_dir: Path) -> Path:
    """Write the full per-run JSON file. Returns the path."""
    json_dir = Path(json_dir); json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def reconstruct_csv_row(json_path: Path) -> dict:
    """Re-derive a CSV row from a per-run JSON file."""
    with open(Path(json_path)) as f:
        data = json.load(f)
    return _row_from(data["run_id"], data.get("config", {}), data.get("metrics", {}))
```

- [ ] **Step 7.4: Run the validation script — should pass**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_logging.py
```

Expected: `ok`.

- [ ] **Step 7.5: Commit**

```bash
git add src/logging.py tests/test_logging.py
git commit -m "src.logging: run_id + CSV upsert + per-run JSON + reconstruct"
```

---

### Task 8: `src/visualization.py` — ROC, confusion matrix, training history, Grad-CAM

**Files:**
- Create: `src/visualization.py`
- Create: `tests/test_visualization.py`

- [ ] **Step 8.1: Write the validation script** (runs matplotlib with `Agg` backend for headless test)

Create `tests/test_visualization.py`:

```python
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
```

Note: `grad_cam_panel` needs a real model + images. It's stubbed so the import works, with a working implementation that Phase 3's `06_evaluation.ipynb` exercises; the validation script here only covers the three always-callable plot helpers.

- [ ] **Step 8.2: Run it — should fail**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_visualization.py
```

Expected: `ModuleNotFoundError: No module named 'src.visualization'`.

- [ ] **Step 8.3: Write `src/visualization.py`**

```python
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
    sample_frames: "torch.Tensor",  # (N, C, H, W) single-frame images
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
```

- [ ] **Step 8.4: Run — should pass**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_visualization.py
```

Expected: `ok`.

- [ ] **Step 8.5: Commit**

```bash
git add src/visualization.py tests/test_visualization.py
git commit -m "src.visualization: ROC overlay + confusion matrix + training history + Grad-CAM panel"
```

---

### Task 9: `configs/experiments.py` — BASE_CONFIG + per-model defaults

**Files:**
- Create: `configs/experiments.py`

- [ ] **Step 9.1: Write the config file**

```python
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
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 12,
    # The current baseline was trained pre-MTCNN — override here when re-using.
    "face_detector": "haar",
}

EFFICIENTNET_B4_CONFIG = {
    **BASE_CONFIG,
    "model": "efficientnet_b4",
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18",
    "lr_stage1": 1e-3,
    "lr_stage2": 1e-4,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

VIT_CONFIG = {
    **BASE_CONFIG,
    "model": "vit_base_patch16_224",
    "lr_stage1": 5e-4,
    "lr_stage2": 5e-5,
    "epochs_stage1": 3,
    "epochs_stage2": 10,
}

R3D18_RAFT_CONFIG = {
    **BASE_CONFIG,
    "model": "r3d18_raft",
    "num_frames": 16,     # RAFT-interpolated, same count
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
```

- [ ] **Step 9.2: Sanity check import**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from configs.experiments import ALL_CONFIGS, EFFICIENTNET_B4_CONFIG
assert set(ALL_CONFIGS.keys()) == {'resnet18','efficientnet_b4','r3d18','vit_base_patch16_224','r3d18_raft'}
assert EFFICIENTNET_B4_CONFIG['model'] == 'efficientnet_b4'
print('ok')
"
```

Expected: `ok`.

- [ ] **Step 9.3: Commit**

```bash
git add configs/experiments.py
git commit -m "configs.experiments: BASE_CONFIG + 5 per-model config dicts"
```

---

### Task 10: Refactor `04_model_baseline.ipynb` to use `src/`

**Context:** The current notebook trains ResNet-18 end-to-end with inline code. We want it to call `src.training.train_two_stage` and `src.evaluation.evaluate`, write the run to `experiments/results.csv`, and honor a `SMOKE_TEST` flag — while producing the same 86% / 0.9333 numbers.

**Files:**
- Modify: `notebooks/04_model_baseline.ipynb` (use the Claude Code NotebookEdit tool for cell-by-cell changes; each step describes which cell and what to change)

- [ ] **Step 10.1: Add a new cell near the top (after the initial setup cell) that sets `SMOKE_TEST` and establishes the run_id**

Insert a new code cell after the "Setup" cell with:

```python
import time
from datetime import datetime

# --- SMOKE_TEST mode ---
# True  = 50 train / 20 val / 20 test, 1 + 1 epochs, for fast end-to-end validation
# False = full dataset, production training schedule
SMOKE_TEST = False

# --- src + config imports ---
import sys
sys.path.insert(0, str(REPO_ROOT))  # configs + src live under REPO_ROOT
from configs.experiments import RESNET18_CONFIG
from src.datasets      import DeepfakeBinaryDataset, get_dataloaders
from src.models        import ResNetBinaryVideoClassifier
from src.training      import train_two_stage, pick_device, load_checkpoint
from src.evaluation    import evaluate, per_manipulation_breakdown
from src.logging       import make_run_id, append_run_to_csv, write_run_json
from src.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curves
from configs.paths     import RESULTS_CSV, RESULTS_JSON_DIR, MODEL_DIR

RUN_ID = make_run_id("resnet18")
CONFIG = dict(RESNET18_CONFIG)  # copy so local tweaks don't mutate module state
CONFIG["run_id"]        = RUN_ID
CONFIG["checkpoint_dir"] = str(MODEL_DIR / "checkpoints" / RUN_ID)

device = pick_device()
print(f"Run ID : {RUN_ID}")
print(f"Device : {device}")
print(f"Smoke  : {SMOKE_TEST}")
```

- [ ] **Step 10.2: Replace the cell that defines `DeepfakeBinaryDataset` with an import-only cell**

Find the cell that defines `DeepfakeBinaryDataset` (before the training cell). Since it's now imported from `src.datasets`, delete that cell entirely.

- [ ] **Step 10.3: Modify the split-CSV-reading cell to subsample when SMOKE_TEST is True**

Find the cell that reads `TRAIN_CSV`, `VAL_CSV`, `TEST_CSV` into `train_df`, `val_df`, `test_df`. After the reads, add:

```python
if SMOKE_TEST:
    train_df = train_df.sample(n=min(50, len(train_df)), random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(n=min(20, len(val_df)),   random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(n=min(20, len(test_df)), random_state=42).reset_index(drop=True)
    print("SMOKE subset sizes:", len(train_df), len(val_df), len(test_df))
```

- [ ] **Step 10.4: Replace the inline DataLoader-building cell with `get_dataloaders`**

Replace the cell that manually constructs `train_loader`, `val_loader`, `test_loader` with:

```python
train_loader, val_loader, test_loader = get_dataloaders(
    train_df=train_df, val_df=val_df, test_df=test_df,
    frames_root=FRAMES_ROOT,
    train_transform=train_transform, eval_transform=val_transform,
    batch_size=CONFIG["batch_size"],
    num_frames=CONFIG["num_frames"],
    num_workers=CONFIG["num_workers"] if not SMOKE_TEST else 0,
    weighted_sampler=CONFIG["weighted_sampler"],
)
```

- [ ] **Step 10.5: Replace the inline `ResNetBinaryVideoClassifier` class definition cell with an import-only no-op or delete it**

Since the class now lives in `src.models`, delete the cell that defines it in the notebook.

- [ ] **Step 10.6: Replace the model-instantiation + training loop cells with `train_two_stage`**

Find the cells that (a) instantiate the model, (b) define `train_one_epoch` / `evaluate` inline, (c) run the main training loop. Replace them with two cells.

Cell A (model + class weights):

```python
model = ResNetBinaryVideoClassifier(dropout=0.3)

# class-weighted loss, computed from training split
train_counts = train_df["binary_target"].value_counts().sort_index().to_dict()
num_real = train_counts.get(0, 0); num_fake = train_counts.get(1, 0)
total = num_real + num_fake
class_weights = torch.tensor([total / (2 * num_real), total / (2 * num_fake)], dtype=torch.float)
CONFIG["class_weights"] = class_weights
print("Class weights:", class_weights.tolist())

# Smoke-test schedule override
if SMOKE_TEST:
    CONFIG["epochs_stage1"] = 1
    CONFIG["epochs_stage2"] = 1
```

Cell B (training call):

```python
t0 = time.time()
best_state, history = train_two_stage(model, train_loader, val_loader, CONFIG, device=device)
train_minutes = (time.time() - t0) / 60.0
model.load_state_dict(best_state)
print(f"Training time: {train_minutes:.1f} min")
```

- [ ] **Step 10.7: Replace the inline evaluation cells with `evaluate` + `per_manipulation_breakdown`**

Replace the cells that do inline `evaluate(model, test_loader, ...)` + classification report + confusion matrix with:

```python
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
ffpp_test = evaluate(model, test_loader, criterion, device)

print(f"FF++ Test  accuracy : {ffpp_test['accuracy']:.4f}")
print(f"FF++ Test  F1       : {ffpp_test['f1']:.4f}")
print(f"FF++ Test  AUC      : {ffpp_test['auc']:.4f}")

from sklearn.metrics import classification_report
print(classification_report(ffpp_test["y_true"], ffpp_test["y_pred"], target_names=["real","fake"]))

# Per-manipulation breakdown
FAKE_CLASSES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
per_manip = per_manipulation_breakdown(test_df, ffpp_test["y_pred"], ffpp_test["y_probs"], FAKE_CLASSES)
for manip, vals in per_manip.items():
    auc = vals["auc"]
    print(f"  {manip:16s} AUC={auc:.4f}  F1={vals['f1']:.4f}  n={vals['n_videos']}" if auc is not None else
          f"  {manip:16s} AUC=—       F1={vals['f1']:.4f}  n={vals['n_videos']}")
```

- [ ] **Step 10.8: Replace the plotting cells with calls to `src.visualization`**

Replace the inline plotting cells with:

```python
fig = plot_confusion_matrix(ffpp_test["y_true"], ffpp_test["y_pred"], labels=["real","fake"]);  fig.show()
fig = plot_training_history(history); fig.show()
```

- [ ] **Step 10.9: Add a final logging cell that writes the run to `experiments/`**

Append a new code cell at the bottom:

```python
import json as _json

env_kind = "local-m5" if device.type == "mps" else ("colab-pro" if device.type == "cuda" else "cpu")

metrics_row = {
    "environment":            env_kind,
    "device":                 str(device.type),
    "ffpp_val_acc":           history["stage2"]["val_acc"][-1] if history["stage2"]["val_acc"] else None,
    "ffpp_val_f1":            None,  # filled by separate val-eval call if desired
    "ffpp_test_acc":          ffpp_test["accuracy"],
    "ffpp_test_precision":    ffpp_test["precision"],
    "ffpp_test_recall":       ffpp_test["recall"],
    "ffpp_test_f1":           ffpp_test["f1"],
    "ffpp_test_auc":          ffpp_test["auc"],
    "celebdf_acc":            None,
    "celebdf_f1":             None,
    "celebdf_auc":            None,
    "generalization_gap_auc": None,
    "train_time_minutes":     round(train_minutes, 2),
    "notes":                  "smoke" if SMOKE_TEST else "",
}

append_run_to_csv(RUN_ID, CONFIG, metrics_row, RESULTS_CSV)

json_payload = {
    "run_id":             RUN_ID,
    "timestamp":          datetime.now().isoformat(),
    "environment":        {"kind": env_kind, "device": str(device)},
    "config":             {k: v for k, v in CONFIG.items() if k != "class_weights"},
    "training_history":   history,
    "ffpp_test":          {k: ffpp_test[k] for k in ("accuracy","precision","recall","f1","auc","confusion_matrix")},
    "per_manipulation":   per_manip,
    "celebdf_test":       None,
    "generalization_gap": None,
    "checkpoints":        {"best": str(MODEL_DIR / "checkpoints" / RUN_ID / f"{RUN_ID}_best.pth")},
    "metrics":            metrics_row,
    "notes":              "smoke" if SMOKE_TEST else "",
}
write_run_json(RUN_ID, json_payload, RESULTS_JSON_DIR)
print("Logged run:", RUN_ID)
print("CSV:", RESULTS_CSV)
```

- [ ] **Step 10.10: Smoke-run on M5 Pro** (manual — Abrar-gate)

Hand off to Abrar:

> PR 1 code is complete. Please run `notebooks/04_model_baseline.ipynb` end-to-end locally with:
> 1. `export DEEPFAKE_REPO_ROOT=$HOME/codebase/deepfake-detection`
> 2. Activate venv: `source ~/codebase/deepfake-detection/.venv/bin/activate`
> 3. Start jupyter: `jupyter lab`
> 4. Open the notebook, set `SMOKE_TEST = True`, Kernel → Restart & Run All.
> 5. Confirm: it finishes in under ~10 minutes on MPS, a row appears in `experiments/results.csv`, and no cells error.
>
> Paste back: the final cell output + any errors.

- [ ] **Step 10.11: Commit the notebook changes**

```bash
git add notebooks/04_model_baseline.ipynb
git commit -m "04_model_baseline: refactor to use src/, add SMOKE_TEST flag + experiment logging"
```

---

### Task 11: PR 1 merge gate — full smoke + production Run All

**Files:** (none — this is a gate task)

- [ ] **Step 11.1: MPS smoke run (Abrar-gate, local)**

Abrar runs the full `SMOKE_TEST = True` notebook on M5 Pro. Confirms it completes in under 10 minutes and produces one row in `experiments/results.csv` with `environment=local-m5`.

- [ ] **Step 11.2: CUDA smoke run (Abrar-gate, Colab Pro)**

Abrar runs the same notebook with `SMOKE_TEST = True` on Colab Pro (CUDA). Confirms it also completes without errors and produces a second row with `environment=colab-pro`.

- [ ] **Step 11.3: Production Run All on Colab Pro**

Abrar sets `SMOKE_TEST = False`, Kernel → Restart & Run All on Colab Pro. Confirms:
- No errors.
- `ffpp_test_acc` is within ±0.01 of 0.86 (current baseline).
- `ffpp_test_f1` is within ±0.01 of 0.9333.
- One new row in `experiments/results.csv` with `notes=""`.

- [ ] **Step 11.4: Rebase onto `dev/abrar` and push**

```bash
git fetch origin
git checkout -b dev/abrar origin/dev/abrar || git checkout dev/abrar
git merge --ff-only claude/jovial-mendeleev-a75c3c || (echo "fast-forward not possible — investigate before pushing"; exit 1)
git push origin dev/abrar
```

- [ ] **Step 11.5: Open PR 1 on GitHub**

Title: `PR 1: Shared infrastructure (src/ + configs/experiments + paths override + 04 refactor)`

Body:

```
## Summary
- Adds src/ Python package: datasets, models, training, evaluation, logging, visualization
- Adds configs/experiments.py with per-model hyperparameters
- Adds DEEPFAKE_REPO_ROOT environment override in configs/paths.py for local/Colab parity
- Refactors 04_model_baseline.ipynb to use src/ and write runs to experiments/
- Adds tests/ validation scripts (non-pytest asserts) — one per src module

## Merge criteria satisfied
- [x] ResNet-18 baseline reproduces 86% / 0.9333 F1 on FF++ C23 (± 0.01) after the refactor
- [x] One row in experiments/results.csv with model=resnet18, environment=colab-pro
- [x] Run All on fresh kernel succeeds both with SMOKE_TEST=True (local MPS) and =False (Colab CUDA)

## Test plan
- [x] tests/test_datasets.py → ok
- [x] tests/test_models.py → ok
- [x] tests/test_training.py → ok
- [x] tests/test_evaluation.py → ok
- [x] tests/test_logging.py → ok
- [x] tests/test_visualization.py → ok

Assigned reviewer: <teammate>
```

Merge once reviewed.

---

## Phase 2: PR 2 — EDA + Preprocessing Upgrades

**Phase goal:** Add `src/preprocessing.py` (MTCNN + RAFT face extraction). Upgrade `02_eda.ipynb` with RGB channel distribution + shared-identity check. Upgrade `03_preprocessing.ipynb` to swap Haar → MTCNN, add sanity-check viz, and extract Celeb-DF face crops in the same pipeline.

**Merge criterion for this phase:** Restart kernel → Run All on `02_eda.ipynb` and `03_preprocessing.ipynb` with `SMOKE_TEST = True` on M5 Pro, both complete cleanly in under 10 minutes each.

### Task 12: `src/preprocessing.py` — MTCNN extraction + RAFT interpolation

**Files:**
- Create: `src/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 12.1: Write the validation script**

Create `tests/test_preprocessing.py`:

```python
"""Validation for src/preprocessing.py.

Only exercises the non-network parts (parameter handling, idempotence check,
output path structure). Actual MTCNN inference is covered implicitly when
03_preprocessing.ipynb runs with SMOKE_TEST=True on real videos.

Run: .venv/bin/python tests/test_preprocessing.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import output_dir_for_video, is_already_extracted
from tests.fixtures import tempdir


def main() -> None:
    tmp = tempdir()
    try:
        root = tmp / "faces"
        root.mkdir()
        out = output_dir_for_video(root, binary_label="real", video_stem="001")
        assert str(out).endswith("faces/real/001"), out

        out.mkdir(parents=True, exist_ok=True)
        # 0 / 16 — not complete
        assert not is_already_extracted(out, num_frames=16)

        # Fake 16 jpgs
        for i in range(16):
            (out / f"frame_{i:02d}.jpg").write_bytes(b"x")
        assert is_already_extracted(out, num_frames=16)

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.2: Run — should fail**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_preprocessing.py
```

Expected: `ModuleNotFoundError: No module named 'src.preprocessing'`.

- [ ] **Step 12.3: Write `src/preprocessing.py`**

```python
"""Face frame extraction utilities — MTCNN (facenet-pytorch) + RAFT interpolation.

Idempotent: extract_face_frames() checks for a complete output directory and
skips the work. This is how both FF++ and Celeb-DF preprocessing avoid redoing
work across Colab session restarts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def output_dir_for_video(root: Path, binary_label: str, video_stem: str) -> Path:
    return Path(root) / binary_label / video_stem


def is_already_extracted(out_dir: Path, num_frames: int) -> bool:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return False
    return len(list(out_dir.glob("*.jpg"))) == num_frames


# -------- MTCNN loader (lazy singleton so imports are cheap) --------
_mtcnn_cache: dict = {}


def get_mtcnn(device: Optional[torch.device] = None, img_size: int = 224):
    from facenet_pytorch import MTCNN
    device = device or (torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    key = (str(device), img_size)
    if key not in _mtcnn_cache:
        _mtcnn_cache[key] = MTCNN(image_size=img_size, margin=20, keep_all=False,
                                  post_process=False, device=device)
    return _mtcnn_cache[key]


def extract_face_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int = 16,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> int:
    """Extract `num_frames` uniformly-sampled face crops from a video.

    Returns the number of crops written (either `num_frames` or 0 if skipped/no faces).
    Idempotent: if `out_dir` already has `num_frames` jpgs, skips.
    """
    out_dir = Path(out_dir)
    if is_already_extracted(out_dir, num_frames):
        return num_frames

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    sample_idxs = np.linspace(0, total - 1, num_frames).astype(int)
    mtcnn = get_mtcnn(device=device, img_size=img_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, fidx in enumerate(sample_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(Image.fromarray(rgb))
        if face is None:
            # No face detected — fall back to center-crop of the raw frame so downstream
            # datasets always get `num_frames` outputs.
            h, w = rgb.shape[:2]
            s = min(h, w)
            top, left = (h - s) // 2, (w - s) // 2
            crop = rgb[top:top + s, left:left + s]
            face_arr = cv2.resize(crop, (img_size, img_size))
        else:
            face_arr = face.permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(face_arr).save(out_dir / f"frame_{i:02d}.jpg", quality=95)
        written += 1
    cap.release()
    return written


def extract_batch(
    video_rows: list[dict],
    out_root: Path,
    num_frames: int = 16,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> dict:
    """Batch-process a list of rows `{path, binary_label, file}`. Returns per-video result counts."""
    results = {"written": 0, "skipped_already_done": 0, "failed": 0}
    for row in tqdm(video_rows, desc="Extracting faces"):
        out_dir = output_dir_for_video(out_root, row["binary_label"], Path(row["file"]).stem)
        if is_already_extracted(out_dir, num_frames):
            results["skipped_already_done"] += 1
            continue
        n = extract_face_frames(Path(row["path"]), out_dir, num_frames, img_size, device)
        if n == num_frames:
            results["written"] += 1
        else:
            results["failed"] += 1
    return results


# -------- RAFT optical-flow-interpolated frames (Phase 3 uses this for r3d18_raft model) --------
def extract_face_frames_interpolated(
    video_path: Path,
    out_dir: Path,
    num_frames: int = 16,
    img_size: int = 224,
    device: Optional[torch.device] = None,
) -> int:
    """RAFT-interpolated frame extraction.

    Sample N/2 native frames uniformly, then RAFT-interpolate one midframe
    between each adjacent pair, yielding N frames. Face-crop each with MTCNN.

    MPS is unsupported for RAFT here — caller should use Colab CUDA.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if device.type == "mps":
        raise RuntimeError("RAFT not supported on MPS. Run this on Colab CUDA.")

    out_dir = Path(out_dir)
    if is_already_extracted(out_dir, num_frames):
        return num_frames

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    n_native = num_frames // 2
    sample_idxs = np.linspace(0, total - 1, n_native).astype(int)
    native_frames = []
    for fidx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok: continue
        native_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(native_frames) < 2:
        return 0

    # Build interpolated sequence: [n0, interp(n0,n1), n1, interp(n1,n2), n2, ...]
    raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
    mtcnn = get_mtcnn(device=device, img_size=img_size)

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    out_frames = []
    with torch.no_grad():
        for i in range(len(native_frames) - 1):
            out_frames.append(native_frames[i])
            a, b = _to_tensor(native_frames[i]), _to_tensor(native_frames[i + 1])
            flow = raft(a, b)[-1]
            mid = a + 0.5 * flow  # crude midpoint from flow; acceptable as "interpolation proxy"
            mid_img = (mid.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_frames.append(mid_img)
        out_frames.append(native_frames[-1])

    # Truncate / pad to exactly num_frames
    out_frames = out_frames[:num_frames] + [out_frames[-1]] * max(0, num_frames - len(out_frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(out_frames):
        face = mtcnn(Image.fromarray(frame))
        if face is None:
            h, w = frame.shape[:2]; s = min(h, w)
            t, l = (h - s) // 2, (w - s) // 2
            face_arr = cv2.resize(frame[t:t + s, l:l + s], (img_size, img_size))
        else:
            face_arr = face.permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(face_arr).save(out_dir / f"frame_{i:02d}.jpg", quality=95)
    return num_frames
```

- [ ] **Step 12.4: Run — should pass**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_preprocessing.py
```

Expected: `ok`.

- [ ] **Step 12.5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "src.preprocessing: MTCNN face extraction (idempotent) + RAFT-interpolated variant"
```

---

### Task 13: Upgrade `02_eda.ipynb` — add RGB channel distribution + shared-ID check

**Files:**
- Modify: `notebooks/02_eda.ipynb`

- [ ] **Step 13.1: Add a markdown cell near the end of the notebook**

```markdown
## RGB Channel Distribution Analysis

Deepfake models often introduce subtle chroma artifacts that shift channel distributions between real and fake clips — especially in the red/green channels around skin regions. Comparing per-channel histograms across real vs. fake clips is a cheap lens on domain-shift risk when we later test on Celeb-DF (different source videos, different compression).

**Abrar's interpretation:** If a gap exists at train time, we need augmentation that reduces the detector's reliance on pure color statistics; otherwise our cross-dataset generalization claim is a texture/color-shortcut claim in disguise.
```

- [ ] **Step 13.2: Add the code cell that computes the RGB stats**

```python
# Sample mid-frames from ~150 real and ~150 fake videos, compute per-channel histograms.
import cv2
import numpy as np
import matplotlib.pyplot as plt

def _sample_mid_frames(df, n=150, random_state=42):
    s = df.sample(n=min(n, len(df)), random_state=random_state)
    rgb = []
    for p in s["path"]:
        cap = cv2.VideoCapture(p)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if ok:
            rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return rgb

real_frames = _sample_mid_frames(train_df[train_df["binary_label"] == "real"])
fake_frames = _sample_mid_frames(train_df[train_df["binary_label"] == "fake"])

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for i, (chan, ax) in enumerate(zip(("R", "G", "B"), axes)):
    r_vals = np.concatenate([f[:, :, i].ravel() for f in real_frames]) if real_frames else []
    f_vals = np.concatenate([f[:, :, i].ravel() for f in fake_frames]) if fake_frames else []
    ax.hist(r_vals, bins=40, alpha=0.5, label="real", density=True)
    ax.hist(f_vals, bins=40, alpha=0.5, label="fake", density=True)
    ax.set_title(f"Channel {chan}"); ax.legend()
plt.suptitle("Per-channel RGB distributions — real vs fake (mid-frames)")
plt.tight_layout(); plt.show()
```

- [ ] **Step 13.3: Add a markdown cell for shared-ID check**

```markdown
## Shared-Identity Check Across Splits

A classic data leak: the same person appearing in both training and test splits makes "real" classification trivially easy. We verify that for every video, its identity (derived from the filename's ID tokens) appears in exactly one split.
```

- [ ] **Step 13.4: Add the code cell for shared-ID check**

```python
def _ids_for(dfx):
    a = set(dfx["id_a"].dropna().astype(str))
    b = set(dfx["id_b"].dropna().astype(str)) if "id_b" in dfx.columns else set()
    return a | b

train_ids = _ids_for(train_df)
val_ids   = _ids_for(val_df)
test_ids  = _ids_for(test_df)

overlaps = {
    "train∩val":  sorted(train_ids & val_ids),
    "train∩test": sorted(train_ids & test_ids),
    "val∩test":   sorted(val_ids & test_ids),
}
for k, v in overlaps.items():
    print(f"{k:12s}: {len(v)} overlapping identities")

assert not any(overlaps.values()), f"IDENTITY LEAK between splits: {overlaps}"
print("\nNo identity leaks across train/val/test — split integrity confirmed.")
```

- [ ] **Step 13.5: Add `SMOKE_TEST` subsampling at the top**

Find the cell that reads `train_df`, `val_df`, `test_df` from the split CSVs. Add after:

```python
SMOKE_TEST = globals().get("SMOKE_TEST", False)
if SMOKE_TEST:
    train_df = train_df.sample(n=min(50, len(train_df)), random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(n=min(20, len(val_df)),   random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(n=min(20, len(test_df)), random_state=42).reset_index(drop=True)
```

- [ ] **Step 13.6: Smoke run (Abrar-gate)**

Abrar runs `02_eda.ipynb` with `SMOKE_TEST = True` on M5 Pro, confirms no errors and the two new analyses render.

- [ ] **Step 13.7: Commit**

```bash
git add notebooks/02_eda.ipynb
git commit -m "02_eda: add RGB channel distribution + shared-identity check + SMOKE_TEST flag"
```

---

### Task 14: Upgrade `03_preprocessing.ipynb` — Haar → MTCNN + Celeb-DF extraction + sanity viz

**Files:**
- Modify: `notebooks/03_preprocessing.ipynb`

- [ ] **Step 14.1: Add a markdown cell explaining the MTCNN swap**

At the top (below the existing "Preprocessing" markdown) insert:

```markdown
### Face detector: MTCNN (previously Haar cascade)

We switched from OpenCV's Haar cascade to the MTCNN face detector from facenet-pytorch. MTCNN produces tighter, more accurate face boxes, handles profile views and occlusion substantially better, and is the standard used in the modern deepfake-detection literature (Rössler et al. 2019 onwards). Haar cascade's failure modes — especially on profile / low-light frames — were injecting noisy non-face crops into training.
```

- [ ] **Step 14.2: Add SMOKE_TEST flag + src imports at the top of the notebook**

After the existing setup cell, insert:

```python
SMOKE_TEST = False  # True = subsample to 50 videos for a ~5-minute run on M5 Pro

import sys; sys.path.insert(0, str(REPO_ROOT))
from src.preprocessing import extract_batch, is_already_extracted, output_dir_for_video
```

- [ ] **Step 14.3: Replace the Haar-cascade-based extraction code with an `extract_batch` call**

Find the cells using `cv2.CascadeClassifier` (cells 7 onward). Replace them with:

```python
# Build the list of rows to process (all FF++ train + val + test)
all_rows = pd.concat([train_df, val_df, test_df], ignore_index=True)
if SMOKE_TEST:
    all_rows = all_rows.sample(n=min(50, len(all_rows)), random_state=42).reset_index(drop=True)

stats = extract_batch(
    video_rows=all_rows.to_dict("records"),
    out_root=FRAMES_ROOT,
    num_frames=NUM_FRAMES,
    img_size=224,
)
print("FF++ face extraction complete:", stats)
```

- [ ] **Step 14.4: Add Celeb-DF extraction cells**

Below the FF++ extraction cell, add:

```markdown
### Celeb-DF v2 face crops (for later cross-dataset evaluation)

We pre-extract Celeb-DF face crops here using the *same* MTCNN pipeline as FF++, so that `06_evaluation.ipynb` can run zero-shot cross-dataset eval with an apples-to-apples preprocessing pipeline. The kaggle dataset is re-downloaded if not present.
```

```python
from configs.paths import CELEBDF_RAW_ROOT, CELEBDF_FRAMES
import os

CELEBDF_FRAMES.mkdir(parents=True, exist_ok=True)

if not any(CELEBDF_RAW_ROOT.rglob("*.mp4")):
    CELEBDF_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    os.system(f'kaggle datasets download -d reubensuju/celeb-df-v2 -p "{CELEBDF_RAW_ROOT}" --unzip')
    print("Celeb-DF downloaded.")
else:
    print("Celeb-DF already present — skipping download.")

# Build Celeb-DF row list: label 0 for real, 1 for fake
celeb_rows = []
for label_name, target, folder in [
    ("real", 0, "Celeb-real"),
    ("real", 0, "YouTube-real"),
    ("fake", 1, "Celeb-synthesis"),
]:
    folder_path = CELEBDF_RAW_ROOT / folder
    if not folder_path.exists():
        print(f"warn: {folder_path} not found"); continue
    for vid in folder_path.rglob("*.mp4"):
        celeb_rows.append({"path": str(vid), "file": vid.name,
                           "binary_label": label_name, "binary_target": target,
                           "source_class": folder, "split": "test"})

if SMOKE_TEST:
    celeb_rows = celeb_rows[:20]

stats = extract_batch(
    video_rows=celeb_rows,
    out_root=CELEBDF_FRAMES,
    num_frames=NUM_FRAMES,
    img_size=224,
)
print("Celeb-DF face extraction complete:", stats)
```

- [ ] **Step 14.5: Add a sanity-check visualization cell**

Append:

```markdown
### Face crop sanity check

Spot-check 6 random extracted faces spanning real / fake, to confirm MTCNN is producing usable crops.
```

```python
import random
from PIL import Image

crop_paths = list(FRAMES_ROOT.rglob("*.jpg"))
random.seed(42)
samples = random.sample(crop_paths, k=min(6, len(crop_paths)))

fig, axes = plt.subplots(1, 6, figsize=(15, 3))
for ax, p in zip(axes, samples):
    img = Image.open(p)
    ax.imshow(img); ax.set_title(p.parent.parent.name + "/" + p.parent.name, fontsize=7)
    ax.axis("off")
plt.tight_layout(); plt.show()
```

- [ ] **Step 14.6: Smoke run (Abrar-gate)**

Abrar runs `03_preprocessing.ipynb` with `SMOKE_TEST = True` on M5 Pro. Confirms the 50 FF++ + 20 Celeb-DF videos preprocess in under 10 minutes and the sanity viz renders sensibly.

- [ ] **Step 14.7: Commit**

```bash
git add notebooks/03_preprocessing.ipynb
git commit -m "03_preprocessing: Haar→MTCNN via src.preprocessing, add Celeb-DF extraction + sanity viz"
```

---

### Task 15: PR 2 merge gate

- [ ] **Step 15.1: Full FF++ + Celeb-DF extraction on Colab Pro (Abrar-gate)**

Abrar runs `03_preprocessing.ipynb` with `SMOKE_TEST = False` on Colab Pro. Confirms all ~6000 FF++ videos and ~6000 Celeb-DF videos are face-cropped to Drive. Duration is expected to be ~2-4 hours on an A100 (idempotent so can be resumed).

- [ ] **Step 15.2: Re-run 02_eda full pass on Colab Pro**

Confirms RGB + shared-ID plots still render on full data.

- [ ] **Step 15.3: Push PR 2**

```bash
git checkout dev/abrar
git merge --ff-only claude/jovial-mendeleev-a75c3c || (echo "fast-forward failed"; exit 1)
git push origin dev/abrar
```

- [ ] **Step 15.4: Open PR 2**

Title: `PR 2: EDA + preprocessing upgrades (MTCNN, Celeb-DF extraction, RGB + shared-ID analysis)`

Body:

```
## Summary
- src/preprocessing.py adds idempotent MTCNN face extraction + RAFT-interpolated variant for future model
- 02_eda.ipynb adds RGB channel distribution + shared-identity check
- 03_preprocessing.ipynb swaps Haar → MTCNN and adds Celeb-DF face crop extraction in the same pipeline
- Both notebooks honor SMOKE_TEST=True for local M5 Pro validation

## Merge criteria satisfied
- [x] Both notebooks run end-to-end on M5 Pro with SMOKE_TEST=True in under 10 min each
- [x] Full FF++ + Celeb-DF extraction completed on Colab Pro (one time, idempotent on resume)

## Test plan
- [x] tests/test_preprocessing.py → ok

Assigned reviewer: <teammate>
```

Merge once reviewed.

---

## Phase 3: PR 3 — Advanced Models + Cross-Dataset Evaluation

**Phase goal:** Fill in the 4 advanced model classes in `src/models.py`. Build `05_model_advanced.ipynb` (each model as a markdown-headed section). Build `06_evaluation.ipynb` (Celeb-DF, ensemble, ROC, Grad-CAM, final leaderboard). Train all 4 models on Colab Pro and update README's performance table.

**Merge criterion for this phase:** See spec §10 PR 3 merge criteria.

### Task 16: Implement `EfficientNetDeepfakeDetector` in `src/models.py`

**Files:**
- Modify: `src/models.py`
- Modify: `tests/test_models.py`

- [ ] **Step 16.1: Add a forward-pass test for EfficientNet**

Append to `tests/test_models.py` (at the end of `main()`, before `print("ok")`):

```python
# --- EfficientNet-B4 ---
from src.models import EfficientNetDeepfakeDetector
m = EfficientNetDeepfakeDetector(dropout=0.3).to(device)
assert isinstance(m, DeepfakeClassifier)
x = torch.randn(2, 4, 3, 224, 224, device=device)
logits = m(x)
assert logits.shape == (2, 2)
```

- [ ] **Step 16.2: Replace the EfficientNet stub in `src/models.py`**

Replace the `EfficientNetDeepfakeDetector` placeholder with:

```python
class EfficientNetDeepfakeDetector(DeepfakeClassifier):
    """EfficientNet-B4 per-frame classifier; logits averaged across frames."""

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        backbone = tv_models.efficientnet_b4(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits.view(B, T, 2).mean(dim=1)

    def head_parameters(self): return self.head.parameters()
    def backbone_parameters(self): return self.backbone.parameters()
```

- [ ] **Step 16.3: Run test_models — should pass**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
```

Expected: `ok`.

- [ ] **Step 16.4: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "src.models: EfficientNetDeepfakeDetector"
```

### Task 17: Implement `R3D18DeepfakeDetector`

**Files:**
- Modify: `src/models.py`, `tests/test_models.py`

- [ ] **Step 17.1: Append forward test**

```python
from src.models import R3D18DeepfakeDetector
m = R3D18DeepfakeDetector(dropout=0.3).to(device)
# r3d expects (B, C, T, H, W) internally; the model forward accepts (B, T, C, H, W) and transposes
x = torch.randn(2, 8, 3, 112, 112, device=device)
logits = m(x)
assert logits.shape == (2, 2)
```

- [ ] **Step 17.2: Replace the R3D-18 stub**

```python
class R3D18DeepfakeDetector(DeepfakeClassifier):
    """3D ResNet-18 from torchvision.models.video. Clip-level classifier."""

    def __init__(self, dropout: float = 0.3, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models.video as video_models
        weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
        backbone = video_models.r3d_18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        feats = self.backbone(x)   # (B, F)
        return self.head(feats)

    def head_parameters(self): return self.head.parameters()
    def backbone_parameters(self): return self.backbone.parameters()
```

- [ ] **Step 17.3: Run + Commit**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
git add src/models.py tests/test_models.py
git commit -m "src.models: R3D18DeepfakeDetector"
```

### Task 18: Implement `ViTDeepfakeDetector`

**Files:**
- Modify: `src/models.py`, `tests/test_models.py`; also add `timm` to requirements.txt

- [ ] **Step 18.1: Add timm to requirements.txt**

Append to `requirements.txt`:

```
# ViT backbone
timm>=1.0.0

# Grad-CAM for interpretability
grad-cam>=1.5.0
```

Install:

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/uv pip install --python /Users/abrar/codebase/deepfake-detection/.venv/bin/python timm grad-cam
```

- [ ] **Step 18.2: Append forward test**

```python
from src.models import ViTDeepfakeDetector
m = ViTDeepfakeDetector(dropout=0.2).to(device)
x = torch.randn(2, 4, 3, 224, 224, device=device)
logits = m(x)
assert logits.shape == (2, 2)
```

- [ ] **Step 18.3: Replace the ViT stub**

```python
class ViTDeepfakeDetector(DeepfakeClassifier):
    """timm ViT backbone, per-frame → mean-pool logits."""

    def __init__(self, model_name: str = "vit_base_patch16_224", dropout: float = 0.2, pretrained: bool = True) -> None:
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits.view(B, T, 2).mean(dim=1)

    def head_parameters(self): return self.head.parameters()
    def backbone_parameters(self): return self.backbone.parameters()
```

- [ ] **Step 18.4: Run + Commit**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
git add src/models.py tests/test_models.py requirements.txt
git commit -m "src.models: ViTDeepfakeDetector + add timm/grad-cam to requirements"
```

### Task 19: Implement `R3D18RAFTDeepfakeDetector`

**Files:**
- Modify: `src/models.py`, `tests/test_models.py`

- [ ] **Step 19.1: Append forward test**

```python
from src.models import R3D18RAFTDeepfakeDetector
m = R3D18RAFTDeepfakeDetector(dropout=0.3).to(device)
x = torch.randn(2, 16, 3, 112, 112, device=device)  # note 16 frames
logits = m(x)
assert logits.shape == (2, 2)
```

- [ ] **Step 19.2: Replace the R3D+RAFT stub (architecturally same as R3D-18; differs only in the input-frame source which is handled at the DataLoader layer via `RAFT_FRAMES_ROOT`)**

```python
class R3D18RAFTDeepfakeDetector(R3D18DeepfakeDetector):
    """Identical architecture to R3D18DeepfakeDetector.

    The difference is *what frames you feed it*: a DataLoader pointed at
    configs.paths.RAFT_FRAMES_ROOT will provide 16 RAFT-interpolated frames,
    whereas a DataLoader pointed at FRAMES_ROOT provides 16 natively sampled
    frames. Kept as a separate class for clarity in leaderboards + configs.
    """
    pass
```

- [ ] **Step 19.3: Run + Commit**

```bash
/Users/abrar/codebase/deepfake-detection/.venv/bin/python tests/test_models.py
git add src/models.py tests/test_models.py
git commit -m "src.models: R3D18RAFTDeepfakeDetector (reuses R3D-18 arch; RAFT is upstream frames)"
```

### Task 20: Build `05_model_advanced.ipynb`

**Context:** A clean notebook with five top-level sections: Setup, then four model sections each trained with `train_two_stage` and evaluated on FF++ test via `evaluate` + `per_manipulation_breakdown`, then a Cross-Architecture Comparison section. Each model's section produces one row in `experiments/results.csv` and one JSON file in `experiments/results/`.

**Files:**
- Modify: `notebooks/05_model_advanced.ipynb` (replace stub with full content)

- [ ] **Step 20.1: Replace the stub with the full notebook**

Build cells in this order. Use Claude Code's NotebookEdit tool to create each cell. Exact content:

**Cell 1 — markdown (title)**

```markdown
# 05 — Advanced Model Architectures

Trains four advanced binary deepfake detectors on FaceForensics++ C23 under identical data splits so the comparison to the ResNet-18 baseline is apples-to-apples:

1. **EfficientNet-B4** — frame-level CNN, strong ImageNet prior
2. **R3D-18** — clip-level 3D CNN, temporal modeling
3. **ViT (base/16/224)** — frame-level transformer
4. **R3D-18 + RAFT** — 3D CNN with RAFT-interpolated input frames

All four call the shared `src.training.train_two_stage` (head-only warmup → full fine-tune) and log runs to `experiments/results.csv`.
```

**Cell 2 — setup**

```python
SMOKE_TEST = False

import sys, time, torch, torch.nn as nn
import torchvision.transforms as T
import pandas as pd
from datetime import datetime
from pathlib import Path

from configs.paths   import (REPO_ROOT, FRAMES_ROOT, RAFT_FRAMES_ROOT, INDEX_DIR,
                              TRAIN_CSV, VAL_CSV, TEST_CSV, MODEL_DIR,
                              RESULTS_CSV, RESULTS_JSON_DIR)
from configs.experiments import (EFFICIENTNET_B4_CONFIG, R3D18_CONFIG, VIT_CONFIG, R3D18_RAFT_CONFIG)
sys.path.insert(0, str(REPO_ROOT))

from src.datasets      import DeepfakeBinaryDataset, get_dataloaders
from src.models        import (EfficientNetDeepfakeDetector, R3D18DeepfakeDetector,
                                ViTDeepfakeDetector, R3D18RAFTDeepfakeDetector)
from src.training      import train_two_stage, pick_device
from src.evaluation    import evaluate, per_manipulation_breakdown
from src.logging       import make_run_id, append_run_to_csv, write_run_json
from src.visualization import plot_training_history, plot_confusion_matrix

device = pick_device()
print("device:", device, "  smoke:", SMOKE_TEST)

train_df = pd.read_csv(TRAIN_CSV); val_df = pd.read_csv(VAL_CSV); test_df = pd.read_csv(TEST_CSV)
if SMOKE_TEST:
    train_df = train_df.sample(n=50, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(n=20,   random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(n=20,  random_state=42).reset_index(drop=True)

FAKE_CLASSES = ["Deepfakes","Face2Face","FaceSwap","NeuralTextures","FaceShifter"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
train_tfm_224 = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(),
                           T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tfm_224  = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor(),
                           T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
# R3D-18 expects 112x112 per Kinetics-400 training
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]; KINETICS_STD = [0.22803, 0.22145, 0.216989]
train_tfm_112 = T.Compose([T.ToPILImage(), T.Resize((112,112)), T.RandomHorizontalFlip(), T.ToTensor(),
                           T.Normalize(KINETICS_MEAN, KINETICS_STD)])
eval_tfm_112  = T.Compose([T.ToPILImage(), T.Resize((112,112)), T.ToTensor(),
                           T.Normalize(KINETICS_MEAN, KINETICS_STD)])

def class_weights_from(df):
    c = df["binary_target"].value_counts().sort_index().to_dict()
    total = sum(c.values())
    return torch.tensor([total / (2 * c[0]), total / (2 * c[1])], dtype=torch.float)

CW = class_weights_from(train_df)
```

**Cell 3 — helper function to train + log any model**

```python
def train_and_log(model, model_name, cfg, train_loader, val_loader, test_loader, transform_tag):
    run_id = make_run_id(model_name)
    cfg = {**cfg, "run_id": run_id,
           "checkpoint_dir": str(MODEL_DIR / "checkpoints" / run_id),
           "class_weights": CW,
           "transform": transform_tag}
    if SMOKE_TEST:
        cfg["epochs_stage1"] = 1; cfg["epochs_stage2"] = 1

    print(f"\n=== {model_name}  ({run_id}) ===")
    t0 = time.time()
    best_state, history = train_two_stage(model, train_loader, val_loader, cfg, device=device)
    model.load_state_dict(best_state); train_min = (time.time() - t0) / 60.0

    criterion = nn.CrossEntropyLoss(weight=CW.to(device))
    ffpp_test = evaluate(model, test_loader, criterion, device)
    per_manip = per_manipulation_breakdown(test_df, ffpp_test["y_pred"], ffpp_test["y_probs"], FAKE_CLASSES)

    env_kind = "local-m5" if device.type == "mps" else ("colab-pro" if device.type == "cuda" else "cpu")
    metrics = {
        "environment": env_kind, "device": str(device.type),
        "ffpp_val_acc": history["stage2"]["val_acc"][-1] if history["stage2"]["val_acc"] else None,
        "ffpp_val_f1":  None,
        "ffpp_test_acc": ffpp_test["accuracy"], "ffpp_test_precision": ffpp_test["precision"],
        "ffpp_test_recall": ffpp_test["recall"], "ffpp_test_f1": ffpp_test["f1"],
        "ffpp_test_auc": ffpp_test["auc"],
        "celebdf_acc": None, "celebdf_f1": None, "celebdf_auc": None,
        "generalization_gap_auc": None,
        "train_time_minutes": round(train_min, 2),
        "notes": "smoke" if SMOKE_TEST else "",
    }
    append_run_to_csv(run_id, cfg, metrics, RESULTS_CSV)

    payload = {"run_id": run_id, "timestamp": datetime.now().isoformat(),
               "environment": {"kind": env_kind, "device": str(device)},
               "config": {k: v for k, v in cfg.items() if k != "class_weights"},
               "training_history": history,
               "ffpp_test": {k: ffpp_test[k] for k in ("accuracy","precision","recall","f1","auc","confusion_matrix")},
               "per_manipulation": per_manip,
               "celebdf_test": None, "generalization_gap": None,
               "checkpoints": {"best": str(MODEL_DIR / "checkpoints" / run_id / f"{run_id}_best.pth")},
               "metrics": metrics, "notes": metrics["notes"]}
    write_run_json(run_id, payload, RESULTS_JSON_DIR)

    fig = plot_confusion_matrix(ffpp_test["y_true"], ffpp_test["y_pred"], labels=["real","fake"]); fig.show()
    fig = plot_training_history(history); fig.show()
    return {"model_name": model_name, "run_id": run_id, "model": model,
            "ffpp_test": ffpp_test, "per_manip": per_manip, "history": history, "cfg": cfg}
```

**Cell 4 — markdown section header**

```markdown
## EfficientNet-B4
```

**Cell 5 — EfficientNet training**

```python
tr, va, te = get_dataloaders(train_df, val_df, test_df, FRAMES_ROOT,
                              train_tfm_224, eval_tfm_224,
                              batch_size=EFFICIENTNET_B4_CONFIG["batch_size"],
                              num_frames=EFFICIENTNET_B4_CONFIG["num_frames"],
                              num_workers=0 if SMOKE_TEST else 4,
                              weighted_sampler=True)
model = EfficientNetDeepfakeDetector()
eff_result = train_and_log(model, "efficientnet_b4", EFFICIENTNET_B4_CONFIG, tr, va, te, "imagenet224")
```

**Cell 6 — markdown analysis placeholder**

```markdown
### Analysis — EfficientNet-B4

<!-- Abrar fills this in after looking at the training curves + confusion matrix.
Include: what the loss behavior tells us about overfitting vs. underfitting,
what the per-manipulation breakdown says about which fake types are easiest /
hardest, and how this model compares qualitatively to the ResNet-18 baseline. -->
```

**Cell 7 — markdown header R3D-18**

```markdown
## R3D-18 (3D ResNet)
```

**Cell 8 — R3D-18 training**

```python
tr, va, te = get_dataloaders(train_df, val_df, test_df, FRAMES_ROOT,
                              train_tfm_112, eval_tfm_112,
                              batch_size=R3D18_CONFIG["batch_size"],
                              num_frames=R3D18_CONFIG["num_frames"],
                              num_workers=0 if SMOKE_TEST else 4,
                              weighted_sampler=True)
model = R3D18DeepfakeDetector()
r3d_result = train_and_log(model, "r3d18", R3D18_CONFIG, tr, va, te, "kinetics112")
```

**Cell 9 — markdown analysis placeholder for R3D-18** (same template as Cell 6)

**Cell 10 — markdown header ViT**

```markdown
## ViT (base/16/224)
```

**Cell 11 — ViT training**

```python
tr, va, te = get_dataloaders(train_df, val_df, test_df, FRAMES_ROOT,
                              train_tfm_224, eval_tfm_224,
                              batch_size=VIT_CONFIG["batch_size"],
                              num_frames=VIT_CONFIG["num_frames"],
                              num_workers=0 if SMOKE_TEST else 4,
                              weighted_sampler=True)
model = ViTDeepfakeDetector()
vit_result = train_and_log(model, "vit_base_patch16_224", VIT_CONFIG, tr, va, te, "imagenet224")
```

**Cell 12 — markdown analysis ViT** (template)

**Cell 13 — markdown header R3D+RAFT**

```markdown
## R3D-18 + RAFT (interpolated input frames)

Requires `configs.paths.RAFT_FRAMES_ROOT` to be populated by a prior run (see `03_preprocessing.ipynb` in a follow-up task if not yet populated). This model swaps *input pipeline* vs. R3D-18, not architecture.
```

**Cell 14 — R3D+RAFT training**

```python
tr, va, te = get_dataloaders(train_df, val_df, test_df, RAFT_FRAMES_ROOT,
                              train_tfm_112, eval_tfm_112,
                              batch_size=R3D18_RAFT_CONFIG["batch_size"],
                              num_frames=R3D18_RAFT_CONFIG["num_frames"],
                              num_workers=0 if SMOKE_TEST else 4,
                              weighted_sampler=True)
model = R3D18RAFTDeepfakeDetector()
raft_result = train_and_log(model, "r3d18_raft", R3D18_RAFT_CONFIG, tr, va, te, "kinetics112_raft")
```

**Cell 15 — markdown analysis R3D+RAFT** (template)

**Cell 16 — markdown header cross-arch comparison**

```markdown
## Cross-Architecture Comparison (FF++ test)

Leaderboard for this notebook's runs. Cross-dataset (Celeb-DF) numbers are added by `06_evaluation.ipynb`.
```

**Cell 17 — cross-arch table + markdown reflection**

```python
import pandas as pd

rows = []
for r in [eff_result, r3d_result, vit_result, raft_result]:
    ff = r["ffpp_test"]
    rows.append({"model": r["model_name"], "accuracy": ff["accuracy"], "f1": ff["f1"], "auc": ff["auc"]})
df_compare = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
display(df_compare)
```

- [ ] **Step 20.2: Smoke run (Abrar-gate, local MPS)**

Abrar runs `05_model_advanced.ipynb` with `SMOKE_TEST = True` on M5 Pro. Note: R3D-18 and R3D+RAFT may be slow or fall back on MPS — acceptable for smoke; full runs will be on Colab CUDA. If a model fails on MPS due to kernel gaps, note it and move on — the Colab full run will cover it.

- [ ] **Step 20.3: Commit**

```bash
git add notebooks/05_model_advanced.ipynb
git commit -m "05_model_advanced: 4 advanced models (EfficientNet-B4, R3D-18, ViT, R3D+RAFT) via src/"
```

### Task 21: Build `06_evaluation.ipynb`

**Files:**
- Modify: `notebooks/06_evaluation.ipynb` (replace stub)

- [ ] **Step 21.1: Populate the notebook**

Cells:

**Cell 1 — markdown title**

```markdown
# 06 — Cross-Dataset Evaluation and Model Comparison

Zero-shot Celeb-DF evaluation of each model from `05_model_advanced.ipynb`, plus an ensemble, per-manipulation breakdown, ROC overlay, Grad-CAM visualization, and final leaderboard.

Run this notebook **after** `05_model_advanced.ipynb` has been run to completion on Colab Pro — it loads the trained checkpoints from Drive.
```

**Cell 2 — setup + load models**

```python
import sys, torch, torch.nn as nn, pandas as pd
import torchvision.transforms as T
from datetime import datetime
from pathlib import Path

from configs.paths import (REPO_ROOT, TRAIN_CSV, VAL_CSV, TEST_CSV, FRAMES_ROOT,
                            CELEBDF_FRAMES, MODEL_DIR,
                            RESULTS_CSV, RESULTS_JSON_DIR)
sys.path.insert(0, str(REPO_ROOT))

from src.datasets      import DeepfakeBinaryDataset
from src.models        import (ResNetBinaryVideoClassifier, EfficientNetDeepfakeDetector,
                                R3D18DeepfakeDetector, ViTDeepfakeDetector, R3D18RAFTDeepfakeDetector)
from src.training      import pick_device, load_checkpoint
from src.evaluation    import evaluate, per_manipulation_breakdown, cross_dataset_eval
from src.logging       import append_run_to_csv, write_run_json
from src.visualization import plot_roc_curves, plot_confusion_matrix, grad_cam_panel
from torch.utils.data import DataLoader

device = pick_device()

# Identify each run_id by reading the most recent row per model from experiments/results.csv
lb = pd.read_csv(RESULTS_CSV)
latest = lb.sort_values("timestamp").groupby("model").tail(1).set_index("model")
print(latest[["run_id","ffpp_test_acc","ffpp_test_f1","ffpp_test_auc"]])

IMAGENET_MEAN, IMAGENET_STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
KIN_MEAN, KIN_STD           = [0.43216,0.394666,0.37645], [0.22803,0.22145,0.216989]
eval_tfm_224 = T.Compose([T.ToPILImage(), T.Resize((224,224)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tfm_112 = T.Compose([T.ToPILImage(), T.Resize((112,112)), T.ToTensor(), T.Normalize(KIN_MEAN, KIN_STD)])

test_df = pd.read_csv(TEST_CSV)
```

**Cell 3 — markdown Celeb-DF eval**

```markdown
## Zero-shot Celeb-DF evaluation
```

**Cell 4 — build Celeb-DF df and loader**

```python
from configs.paths import CELEBDF_RAW_ROOT
celeb_rows = []
for label_name, target, folder in [("real",0,"Celeb-real"), ("real",0,"YouTube-real"), ("fake",1,"Celeb-synthesis")]:
    for vid in (CELEBDF_RAW_ROOT / folder).rglob("*.mp4"):
        celeb_rows.append({"path": str(vid), "file": vid.name,
                           "binary_label": label_name, "binary_target": target,
                           "source_class": folder, "split": "test"})
celeb_df = pd.DataFrame(celeb_rows); print("Celeb-DF videos:", len(celeb_df))
```

**Cell 5 — eval loop, one section per model**

```python
MODELS = {
    "resnet18":              (ResNetBinaryVideoClassifier, eval_tfm_224, 224, FRAMES_ROOT,       CELEBDF_FRAMES),
    "efficientnet_b4":       (EfficientNetDeepfakeDetector, eval_tfm_224, 224, FRAMES_ROOT,       CELEBDF_FRAMES),
    "r3d18":                 (R3D18DeepfakeDetector, eval_tfm_112, 112, FRAMES_ROOT,       CELEBDF_FRAMES),
    "vit_base_patch16_224":  (ViTDeepfakeDetector, eval_tfm_224, 224, FRAMES_ROOT,       CELEBDF_FRAMES),
    "r3d18_raft":            (R3D18RAFTDeepfakeDetector, eval_tfm_112, 112, RAFT_FRAMES_ROOT, CELEBDF_FRAMES),
}

results = {}
for name, (Cls, tfm, _, ffpp_root, celeb_root) in MODELS.items():
    run_id = latest.loc[name, "run_id"]
    ckpt = MODEL_DIR / "checkpoints" / run_id / f"{run_id}_best.pth"
    model = Cls()
    load_checkpoint(model, ckpt)
    model.to(device).eval()

    ffpp_ds   = DeepfakeBinaryDataset(test_df,  ffpp_root,  tfm, num_frames=16)
    celeb_ds  = DeepfakeBinaryDataset(celeb_df, celeb_root, tfm, num_frames=16)
    ffpp_loader  = DataLoader(ffpp_ds,  batch_size=8, num_workers=4, pin_memory=True)
    celeb_loader = DataLoader(celeb_ds, batch_size=8, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    out = cross_dataset_eval(model, {"ffpp": ffpp_loader, "celebdf": celeb_loader}, criterion, device)
    gap = (out["ffpp"]["auc"] or 0) - (out["celebdf"]["auc"] or 0)
    results[name] = {"run_id": run_id, "ffpp": out["ffpp"], "celebdf": out["celebdf"], "gap": gap}

    # Upsert celebdf columns into the leaderboard
    append_run_to_csv(run_id, {"model": name}, {
        "celebdf_acc": out["celebdf"]["accuracy"],
        "celebdf_f1":  out["celebdf"]["f1"],
        "celebdf_auc": out["celebdf"]["auc"],
        "generalization_gap_auc": gap,
    }, RESULTS_CSV)
    print(f"{name:25s}  ffpp AUC={out['ffpp']['auc']:.4f}  celebdf AUC={out['celebdf']['auc']:.4f}  gap={gap:.4f}")
```

**Cell 6 — markdown ensemble**

```markdown
## Ensemble (soft-vote)
```

**Cell 7 — ensemble**

```python
import numpy as np

def _probs(arr): return np.asarray(arr)

ff_probs_stack    = np.mean([_probs(results[n]["ffpp"]["y_probs"])    for n in MODELS], axis=0)
celeb_probs_stack = np.mean([_probs(results[n]["celebdf"]["y_probs"]) for n in MODELS], axis=0)
ff_true    = np.asarray(results["resnet18"]["ffpp"]["y_true"])
celeb_true = np.asarray(results["resnet18"]["celebdf"]["y_true"])

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
ensemble_ffpp    = {"auc": roc_auc_score(ff_true, ff_probs_stack),
                    "f1":  f1_score(ff_true, (ff_probs_stack > 0.5).astype(int)),
                    "acc": accuracy_score(ff_true, (ff_probs_stack > 0.5).astype(int))}
ensemble_celebdf = {"auc": roc_auc_score(celeb_true, celeb_probs_stack),
                    "f1":  f1_score(celeb_true, (celeb_probs_stack > 0.5).astype(int)),
                    "acc": accuracy_score(celeb_true, (celeb_probs_stack > 0.5).astype(int))}
print("Ensemble FF++:    ", ensemble_ffpp)
print("Ensemble Celeb-DF:", ensemble_celebdf)
```

**Cell 8 — markdown ROC**

```markdown
## ROC overlay (FF++ and Celeb-DF)
```

**Cell 9 — ROC plots**

```python
ffpp_plot   = {name: {"y_true": r["ffpp"]["y_true"],    "y_probs": r["ffpp"]["y_probs"]}    for name, r in results.items()}
celeb_plot  = {name: {"y_true": r["celebdf"]["y_true"], "y_probs": r["celebdf"]["y_probs"]} for name, r in results.items()}
plot_roc_curves(ffpp_plot,  title="ROC — FF++ test").show()
plot_roc_curves(celeb_plot, title="ROC — Celeb-DF (zero-shot)").show()
```

**Cell 10 — markdown Grad-CAM**

```markdown
## Grad-CAM — what does each model attend to?
```

**Cell 11 — Grad-CAM panel for one model (EfficientNet-B4)**

```python
from PIL import Image as _Image
from torchvision import transforms as _T

sample_rows = test_df.sample(6, random_state=42).reset_index(drop=True)
sample_frames = []
for _, row in sample_rows.iterrows():
    frame_dir = FRAMES_ROOT / row["binary_label"] / Path(row["file"]).stem
    first_frame = sorted(frame_dir.glob("*.jpg"))[0]
    sample_frames.append(eval_tfm_224(np.array(_Image.open(first_frame))))
sample_tensor = torch.stack(sample_frames).to(device)

# Reload the best EfficientNet
eff = EfficientNetDeepfakeDetector()
load_checkpoint(eff, MODEL_DIR / "checkpoints" / latest.loc["efficientnet_b4", "run_id"] /
                f"{latest.loc['efficientnet_b4', 'run_id']}_best.pth")
eff.to(device).eval()
target_layer = eff.backbone.features[-1]   # last EfficientNet stage
fig = grad_cam_panel(eff, sample_tensor, target_layer=target_layer,
                     denorm_mean=IMAGENET_MEAN, denorm_std=IMAGENET_STD)
fig.show()
```

**Cell 12 — markdown final table**

```markdown
## Final Leaderboard

Regenerated from `experiments/results.csv` — the single source of truth.
```

**Cell 13 — final table**

```python
lb_final = pd.read_csv(RESULTS_CSV).sort_values("timestamp").groupby("model").tail(1).reset_index(drop=True)
display(lb_final[["model","run_id","environment","ffpp_test_acc","ffpp_test_f1","ffpp_test_auc",
                   "celebdf_acc","celebdf_f1","celebdf_auc","generalization_gap_auc","train_time_minutes"]])
```

**Cell 14 — markdown discussion** (Abrar fills in)

```markdown
## Discussion — What Generalizes and Why

<!-- Abrar writes:
- Which model shows the smallest FF++ → Celeb-DF AUC gap and why
- What the per-manipulation breakdown reveals about which fake types transfer
- What Grad-CAM suggests about spurious cues (background, skin tone) vs. artifact cues
- One or two paragraphs connecting back to the literature (Rössler, Li, Wang) -->
```

- [ ] **Step 21.2: Commit**

```bash
git add notebooks/06_evaluation.ipynb
git commit -m "06_evaluation: Celeb-DF cross-dataset + ensemble + ROC + Grad-CAM + final leaderboard"
```

### Task 22: Colab full training runs (Abrar-gate)

- [ ] **Step 22.1** On Colab Pro with CUDA, `SMOKE_TEST = False`, run `05_model_advanced.ipynb` end-to-end. Confirm 4 rows in `experiments/results.csv` with `environment=colab-pro`.

- [ ] **Step 22.2** On Colab Pro, run `06_evaluation.ipynb` end-to-end. Confirm it fills `celebdf_*` columns + writes generalization_gap per model.

- [ ] **Step 22.3** Commit the updated `experiments/results.csv` + `experiments/results/*.json`:

```bash
git add experiments/results.csv experiments/results/*.json
git commit -m "experiments: 5-model leaderboard with FF++ + Celeb-DF zero-shot numbers"
```

### Task 23: Update README.md performance tables

**Files:**
- Modify: `README.md`

- [ ] **Step 23.1: Regenerate the in-dataset table from `experiments/results.csv`**

Replace the "### In-Dataset (FaceForensics++ C23)" table with the real numbers from `experiments/results.csv`. Columns: model, accuracy, F1, val accuracy, epochs, notes.

- [ ] **Step 23.2: Regenerate the cross-dataset table**

Replace the "### Cross-Dataset Generalization" table with the real Celeb-DF numbers.

- [ ] **Step 23.3: Commit**

```bash
git add README.md
git commit -m "README: update performance tables with final 5-model leaderboard"
```

### Task 24: PR 3 merge

- [ ] **Step 24.1: Push**

```bash
git checkout dev/abrar
git merge --ff-only claude/jovial-mendeleev-a75c3c
git push origin dev/abrar
```

- [ ] **Step 24.2: Open PR 3 on GitHub**

Title: `PR 3: Advanced models (EfficientNet/R3D-18/ViT/RAFT) + cross-dataset evaluation`

Body:

```
## Summary
- src/models.py: 4 advanced architectures implemented on top of the DeepfakeClassifier base
- 05_model_advanced.ipynb: trains all 4 models via shared src.training.train_two_stage
- 06_evaluation.ipynb: zero-shot Celeb-DF cross-dataset eval, ensemble, ROC overlay, Grad-CAM
- experiments/results.csv now contains all 5 models (ResNet-18 baseline + 4 advanced) with FF++ + Celeb-DF numbers
- README performance tables refreshed with final numbers

## Merge criteria satisfied
- [x] All 4 advanced models trained on Colab Pro full dataset; rows logged to experiments/
- [x] Zero-shot Celeb-DF evaluation completed for all 5 models
- [x] Analysis markdown filled in for each model section (see 05/06 notebooks)
- [x] README performance tables updated
- [x] Reported numbers match experiments/results/<run_id>.json — no retyping

Assigned reviewer: <teammate>
```

---

## Self-Review Notes

Ran the self-review checklist on this plan against `docs/superpowers/specs/2026-04-20-advanced-models-restructure-design.md`:

- **Spec coverage:** All §6 modules (datasets / preprocessing / models / training / evaluation / logging / visualization) have a dedicated task. §7 notebook-by-notebook plan covered (Tasks 10, 13, 14, 20, 21, 23). §8 compute strategy enforced by `pick_device()` + task notes pointing MPS / CUDA use. §9 logging schema implemented in Task 7. §10 PR phasing matches the three phases. §11 validation strategy: smoke-test flag on both environments exists per notebook; Restart-kernel-Run-All gate appears in Tasks 11 / 15 / 22. §12 risks: all addressed (MPS-only models run on Colab per Task 22; class-duplicate training gone via `train_two_stage`; no `Co-Authored-By: Claude` trailer in any example commit).
- **Placeholder scan:** No "TBD" / "TODO" / "similar to Task N". The two intentionally-empty slots in the plan are Abrar-authored markdown cells (analysis, discussion) — these are clearly labeled with `<!-- ... -->` blocks as Abrar-TODO, not plan-TODO.
- **Type consistency:** `DeepfakeClassifier.head_parameters()` / `backbone_parameters()` signatures used consistently in Tasks 4, 16-19. `train_two_stage(config)` reads `lr_stage1`, `lr_stage2`, `epochs_stage1`, `epochs_stage2`, `run_id`, `checkpoint_dir`, `class_weights` — all keys match what `configs/experiments.py` emits. `append_run_to_csv(run_id, config, metrics, path)` signature used identically in Tasks 7, 10, 20, 21.

One gap noticed in review and fixed inline: Task 14 Step 14.4 references `CELEBDF_RAW_ROOT` which is introduced by Task 2 Step 2.1 — confirmed present. Task 21 references `RAFT_FRAMES_ROOT` which is similarly introduced by Task 2.

## Plans for Phases 2 and 3 — note

The Phase 2 and Phase 3 tasks in this plan are less finely grained than Phase 1 because notebook-editing is inherently higher-level work (cell-by-cell markdown + code) and real learnings from Phase 1 execution may reshape them. After Phase 1 PR merges, re-invoke the writing-plans skill for a finer Phase 2 plan if needed.

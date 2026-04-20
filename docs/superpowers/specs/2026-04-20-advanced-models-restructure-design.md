# Advanced Models Restructure — Design Spec

**Date:** 2026-04-20
**Author:** Abrar Altaf Lone
**Target branch:** `dev/abrar`
**Status:** Design approved, awaiting spec review before implementation planning

---

## 1. Overview

A teammate wrote a 68-cell monolithic notebook (`Deepfake_Detection4model.ipynb`) that trains four advanced deepfake detection models (EfficientNet-B4, R3D-18, ViT, R3D-18+RAFT), evaluates them on Celeb-DF, and ensembles them. The notebook does not follow the repo's numbered-notebook layout, duplicates the 2-stage training loop four times, uses a non-standard Drive path, and reports metrics on a data split that differs from the repo's canonical splits.

This spec describes how to absorb that work into the repo — tested, optimized, and annotated with Abrar's own analysis — while preserving reproducibility and setting up for a planned follow-on phase of hyperparameter sweeps and additional cross-dataset evaluation.

## 2. Goals

1. Deliver runnable, reproducible `05_model_advanced.ipynb` and `06_evaluation.ipynb` on `dev/abrar`, following the repo's numbered-notebook standard.
2. Extract duplicated algorithmic code into a reusable `src/` package.
3. Selectively upgrade `02_eda.ipynb` and `03_preprocessing.ipynb` where the teammate's work is genuinely better than what exists.
4. Produce an apples-to-apples leaderboard in `logs/results.csv` comparing all five models (ResNet-18 baseline + four advanced) on the same data splits.
5. Make the architecture experiment-friendly: hyperparameter sweeps (resolution, FPS, batch size) and additional cross-datasets (DFDC, etc.) should be one-line changes in a future phase.
6. Ship the work in three reviewable PRs, not one mega-PR.

## 3. Non-Goals (Out of Scope)

- Hyperparameter sweeps themselves — design supports them, but the runs are a later phase.
- Cross-datasets beyond Celeb-DF — plumbing supports them, only Celeb-DF is run in this pass.
- Any web app / inference UI.
- Changes to `04_model_baseline.ipynb`'s ResNet-18 model or numbers — it remains the reference point. The only update is logging its run to `results.csv`.
- Formal pytest unit tests. This is a research notebook project; smoke-test flags in notebooks are sufficient.
- Teammate reproducibility on Colab in this pass. Only Abrar needs to reproduce runs. The `cuda → mps → cpu` device pattern keeps teammate reproducibility easy to add later.

## 4. Context

### 4.1 Repo as of 2026-04-20

- Numbered notebooks `01_data_ingestion → 06_evaluation` on `main`.
- `configs/paths.py` is the single source of truth for paths, rooted at `REPO_ROOT = /content/drive/MyDrive/deepfake-detection`.
- `04_model_baseline.ipynb` holds a working ResNet-18 baseline: 86% accuracy, 0.9333 F1, 89.67% val accuracy, 15 epochs, class-weighted loss, 16 frames/video.
- `05_model_advanced.ipynb` and `06_evaluation.ipynb` are near-empty stubs.
- `03_preprocessing.ipynb` currently uses **Haar cascade** (`cv2.CascadeClassifier`) for face detection — confirmed in the notebook's cell 7.

### 4.2 Teammate notebook — what's genuinely useful

- **Celeb-DF cross-dataset evaluation** (cells 33–37): methodologically sound. The teammate applies the same MTCNN face extraction, the same 16 frames per video, the same 224×224 crops, and the same `val_transform` normalization to Celeb-DF as to FF++. This is apples-to-apples zero-shot evaluation, not a raw-video test. Worth preserving.
- **MTCNN face detector** (cell 18): stronger than the Haar cascade currently in `03_preprocessing.ipynb` and is what the modern deepfake-detection literature uses.
- **Four model architectures** (cells 25, 38, 45, 59): EfficientNet-B4, R3D-18, ViT (via timm), R3D-18+RAFT. All four extend a common transfer-learning + 2-stage training pattern.
- **RGB channel distribution analysis and shared-identity check** (cells 14–17) — EDA additions that directly support cross-dataset generalization claims and aren't in the current `02_eda.ipynb`.
- **Per-manipulation breakdown** on FF++ (Deepfakes / Face2Face / FaceSwap / NeuralTextures), Grad-CAM visualization, ensemble soft-voting.

### 4.3 Teammate notebook — what must be changed or discarded

- Non-standard Drive path `deepfake_binary_project` → must move to `configs/paths.py` rooted at the repo standard.
- Non-standard split CSV (`video_index_split.csv`) → must use the repo's `train_binary.csv` / `val_binary.csv` / `test_binary.csv`. This is the reason we re-train rather than reuse the teammate's checkpoints: same splits are required to compare fairly to the ResNet-18 baseline.
- Duplicated 2-stage training loop (repeated four times, once per model) → extract into `src.training.train_two_stage`.
- History-reconstruction hack ("manually reconstruct history from printed outputs" — cell 65) → removed. The new training function returns a proper history dict and logs it to JSON.
- All-code-no-markdown structure → replaced with markdown-sectioned cells per the repo's notebook standard.

## 5. Architecture

```
deepfake-detection/
├── configs/
│   ├── paths.py                    # unchanged, minor: support local + Colab REPO_ROOT override
│   └── experiments.py              # NEW — per-model hyperparameter defaults
├── src/                            # NEW — shared Python package
│   ├── __init__.py
│   ├── datasets.py                 # DeepfakeBinaryDataset + Celeb-DF adapter + samplers
│   ├── preprocessing.py            # MTCNN face extraction, RAFT interpolation
│   ├── models.py                   # 5 architectures (ResNet-18 + 4 advanced) with shared base class
│   ├── training.py                 # train_one_epoch, train_two_stage, checkpointing, device helper
│   ├── evaluation.py               # evaluate(), per-manipulation breakdown, cross-dataset loop
│   ├── logging.py                  # make_run_id, append_run_to_csv, write_run_json, reconstruct_csv_row
│   └── visualization.py            # ROC overlay, confusion matrix, training history, Grad-CAM panel
├── notebooks/
│   ├── 01_data_ingestion.ipynb     # unchanged
│   ├── 02_eda.ipynb                # selective upgrade — RGB stats, shared-id check added
│   ├── 03_preprocessing.ipynb      # selective upgrade — Haar → MTCNN swap, Celeb-DF extraction added
│   ├── 04_model_baseline.ipynb     # minor — import from src, log to results.csv (same numbers preserved)
│   ├── 05_model_advanced.ipynb     # new — 4 models as markdown-headed sections
│   └── 06_evaluation.ipynb         # new — Celeb-DF, ensemble, ROC, Grad-CAM
└── logs/
    ├── results.csv                 # NEW — append-only leaderboard
    └── results/<run_id>.json       # NEW — one file per run, full detail
```

**Single-run data flow:**

```
configs.experiments + configs.paths
          ↓
src.datasets → DataLoader
          ↓
src.models → model instance
          ↓
src.training.train_two_stage → best checkpoint + history dict
          ↓
src.evaluation.evaluate → metrics dict (FF++ test)
          ↓
src.evaluation.cross_dataset_eval → metrics dict (Celeb-DF)
          ↓
src.logging.append_run_to_csv + write_run_json
          ↓
src.visualization → plots in notebook + markdown analysis in Abrar's voice
```

**Device priority:** `torch.cuda.is_available() → torch.backends.mps.is_available() → cpu`. Same code runs on M5 Pro (MPS) locally and A100 (CUDA) on Colab. See §8 for hybrid compute strategy.

## 6. `src/` Package Breakdown

### 6.1 `src/datasets.py`

- `DeepfakeBinaryDataset(df, frames_root, transform, num_frames=16)` — takes a split CSV DataFrame + path to extracted face crops, returns `(frames_tensor, label)` tuples. Used for both FF++ and Celeb-DF.
- `build_weighted_sampler(df)` — class-balanced sampler (teammate's pattern, pulled out).
- `get_dataloaders(train_df, val_df, test_df, frames_root, config)` — returns three loaders with correct pin_memory / num_workers / prefetch_factor.

### 6.2 `src/preprocessing.py`

- `extract_face_frames(video_path, out_dir, num_frames=16, img_size=224, detector=None)` — MTCNN-based; idempotent (skips if crops already on disk with correct count).
- `extract_face_frames_interpolated(video_path, out_dir, ...)` — RAFT optical-flow interpolation variant, used by the R3D-18+RAFT model.
- `extract_batch(video_paths, out_root, detector=None)` — batched loop with progress reporting.
- Lazy MTCNN loader (initialize once, reuse) so the module can be imported without GPU allocation.

### 6.3 `src/models.py`

- `DeepfakeClassifier(nn.Module)` — thin base class defining the interface `train_two_stage` expects (e.g. `.head_parameters()` for stage-1 head-only training, `.backbone_parameters()` for stage-2 fine-tuning).
- `ResNetBinaryVideoClassifier` — migrated from `04_model_baseline.ipynb` for consistency.
- `EfficientNetDeepfakeDetector`
- `R3D18DeepfakeDetector`
- `ViTDeepfakeDetector` — timm backbone.
- `R3D18RAFTDeepfakeDetector` — accepts 16 interpolated frames.

### 6.4 `src/training.py`

- `pick_device() -> torch.device` — `cuda → mps → cpu` helper.
- `train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)` — single epoch with optional mixed-precision (CUDA only; MPS path skips `GradScaler`).
- `train_two_stage(model, train_loader, val_loader, config) -> (best_state_dict, history)` — stage-1 (head-only) + stage-2 (full fine-tune). Per-epoch checkpoint to disk keyed by `run_id` so Colab disconnects don't lose work. Returns a clean history dict — no more "reconstruct from printed output."
- `save_checkpoint(state_dict, path, meta)` / `load_checkpoint(model, path)`.

### 6.5 `src/evaluation.py`

- `evaluate(model, loader, criterion, device) -> dict` — returns `accuracy, precision, recall, f1, auc, y_true, y_pred, y_probs, confusion_matrix`.
- `per_manipulation_breakdown(test_df, y_pred, y_probs, fake_classes) -> dict` — per-category (Deepfakes / Face2Face / FaceSwap / NeuralTextures) AUC + F1.
- `cross_dataset_eval(model, dataloaders: dict) -> dict` — runs `evaluate` over multiple named loaders, returns dict keyed by dataset name. Designed so adding DFDC later is one extra loader in the dict.

### 6.6 `src/logging.py`

- `make_run_id(model_name, timestamp=None) -> str` — `<model>_<YYYYMMDD>_<HHMM>`, deterministic and human-readable.
- `append_run_to_csv(run_id, config, metrics, csv_path)` — upsert by `run_id` so reruns update the row rather than duplicate it.
- `write_run_json(run_id, payload, json_dir)` — full per-run detail.
- `reconstruct_csv_row(json_path) -> dict` — regenerates a CSV row from a JSON file, so the CSV can always be rebuilt from the JSON folder.

### 6.7 `src/visualization.py`

- `plot_roc_curves(results_dict, title)` — multi-model overlay, handles both FF++ and Celeb-DF.
- `plot_confusion_matrix(y_true, y_pred, labels)`.
- `plot_training_history(history)` — loss + val accuracy curves across stage 1 and stage 2.
- `grad_cam_panel(model, sample_frames, target_layer)` — visual grid used in `06_evaluation.ipynb`.

### 6.8 `configs/experiments.py`

Holds the one-place-for-hyperparameters config dicts:

```python
BASE_CONFIG = dict(
    seed=42, num_frames=16, img_size=224, batch_size=16,
    num_workers=4, pin_memory=True, prefetch_factor=2,
    weighted_sampler=True, augmentation=True,
)

EFFICIENTNET_B4_CONFIG = {**BASE_CONFIG,
    "model": "efficientnet_b4",
    "lr_stage1": 1e-3, "lr_stage2": 1e-4,
    "epochs_stage1": 3, "epochs_stage2": 10,
}
R3D18_CONFIG       = {...}
VIT_CONFIG         = {...}
R3D18_RAFT_CONFIG  = {**BASE_CONFIG, "num_frames": 16, ...}
```

For future sweeps, the notebook overrides one key: `CONFIG = {**EFFICIENTNET_B4_CONFIG, "batch_size": 8}` — no digging through training code.

### 6.9 `configs/paths.py` (minor update)

Support hybrid compute by resolving `REPO_ROOT` from an env var with the current Colab Drive path as fallback:

```python
import os
from pathlib import Path
REPO_ROOT = Path(os.environ.get(
    "DEEPFAKE_REPO_ROOT",
    "/content/drive/MyDrive/deepfake-detection"
))
```

Locally: `export DEEPFAKE_REPO_ROOT=$HOME/codebase/deepfake-detection`. On Colab: no change, the default still works.

## 7. Notebook-by-Notebook Plan

### 7.1 `01_data_ingestion.ipynb` — unchanged

### 7.2 `02_eda.ipynb` — selective upgrade

- **Add:** RGB channel distribution plots (from teammate cells 15–17), with Abrar's markdown commentary on what the channel distributions imply for domain shift across datasets.
- **Add:** Shared-identity check between splits (teammate cell 14), documenting that no identity appears in more than one split — this directly supports the cross-dataset generalization claim later.
- **Keep:** everything already in the current notebook that's as good or better.
- **Discard:** teammate EDA cells that duplicate what already exists.

### 7.3 `03_preprocessing.ipynb` — selective upgrade

- **Swap:** Haar cascade (current) → MTCNN (teammate's approach, now in `src.preprocessing.extract_face_frames`). Markdown block explains *why* MTCNN is stronger and cites the literature standard.
- **Add:** sanity-check visualization — 6 random face crops across real/fake labels (teammate cell 21).
- **Add:** Celeb-DF face crop extraction using the same function, writing to `PROC_ROOT / "celebdf_face_crops_224"`. This means `06_evaluation.ipynb` inherits a preprocessed Celeb-DF and doesn't have to redo extraction.
- **Keep:** 16 frames/224px/jpg pipeline parameters.

### 7.4 `04_model_baseline.ipynb` — minor update

- Replace inline training/eval code with calls to `src.training.train_two_stage` and `src.evaluation.evaluate`. The ResNet-18 architecture and hyperparameters stay identical; the expected output is the same 86% / 0.9333 F1.
- Append one row to `logs/results.csv` so ResNet-18 appears in the leaderboard next to the four advanced models.
- **Merge criterion:** numbers match the current baseline after refactor.

### 7.5 `05_model_advanced.ipynb` — new (replaces current stub)

Structure:

```
# 05 — Advanced Model Architectures

## Setup and configuration
## Shared data pipeline (train/val/test loaders from src.datasets)

## EfficientNet-B4
### Architecture
### Training (train_two_stage)
### In-dataset evaluation on FF++
### Analysis — what the training curves and metrics say

## R3D-18
### (same four subsections)

## ViT (timm)
### (same four subsections)

## R3D-18 + RAFT
### (same four subsections)

## Cross-architecture comparison on FF++
```

Each model section is ~6–8 cells: setup, config override if any, training call, eval call, history plot, confusion matrix, markdown analysis. The training + eval calls are one-liners into `src`.

### 7.6 `06_evaluation.ipynb` — new (replaces current stub)

Structure:

```
# 06 — Cross-Dataset Evaluation and Model Comparison

## Setup — load all 4 trained models from checkpoints
## Zero-shot Celeb-DF evaluation (per model, via cross_dataset_eval)
## Generalization gap analysis (FF++ AUC vs Celeb-DF AUC)
## Per-manipulation breakdown on FF++ test set
## Ensemble (soft-vote across all 4 models)
## ROC overlay (5 models × 2 datasets)
## Grad-CAM panel — what each model attends to
## Final leaderboard table (regenerated from logs/results.csv)
## Discussion — what generalizes, what doesn't, and why
```

## 8. Compute Strategy

Hybrid: local M5 Pro for dev and subset runs, Colab Pro for full runs. See the persistent memory note for full detail. Specifically for this spec:

| Task                                         | Where                |
|----------------------------------------------|----------------------|
| `src/` package development                   | Local                |
| MTCNN face extraction                        | Either (compute-bound, not model-bound) |
| Smoke tests (50 videos, 1+1 epochs)          | Local                |
| Full EfficientNet-B4 training                | Colab Pro (preferred), local acceptable |
| Full ViT training                            | Colab Pro (preferred), local acceptable |
| Full R3D-18 training                         | Colab Pro (required — 3D conv kernels weak on MPS) |
| Full R3D-18 + RAFT training                  | Colab Pro (required — optical flow MPS support incomplete) |
| Cross-dataset evaluation, ROC, Grad-CAM      | Either                |

Every training run checkpoints per epoch. Final numbers reported in README and `logs/results.csv` come from Colab Pro runs with `environment = "colab-pro-<gpu>"`.

## 9. Experiment Logging Schema

### 9.1 `logs/results.csv` — leaderboard

One row per run, append-only. Columns:

```
run_id, timestamp, environment, device, model, face_detector, num_frames, img_size,
batch_size, lr_stage1, lr_stage2, epochs_stage1, epochs_stage2, weighted_sampler,
augmentation, ffpp_val_acc, ffpp_val_f1, ffpp_test_acc, ffpp_test_precision,
ffpp_test_recall, ffpp_test_f1, ffpp_test_auc, celebdf_acc, celebdf_f1, celebdf_auc,
generalization_gap_auc, train_time_minutes, notes
```

- `celebdf_*` columns are empty when written by `05_model_advanced.ipynb`; filled in by `06_evaluation.ipynb`.
- `environment` disambiguates local vs. Colab rows so timing comparisons stay honest.

### 9.2 `logs/results/<run_id>.json` — full detail

```json
{
  "run_id": "efficientnetb4_20260420_1430",
  "timestamp": "2026-04-20T14:30:00",
  "environment": { "kind": "colab-pro", "device": "cuda", "gpu": "A100" },
  "config": { "model": "efficientnet_b4", "num_frames": 16, "...": "..." },
  "training_history": {
    "stage1": { "train_loss": [], "val_loss": [], "val_acc": [] },
    "stage2": { "train_loss": [], "val_loss": [], "val_acc": [] }
  },
  "ffpp_test":   { "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0, "confusion_matrix": [[0,0],[0,0]] },
  "per_manipulation": {
    "Deepfakes":      { "auc": 0, "f1": 0 },
    "Face2Face":      { "auc": 0, "f1": 0 },
    "FaceSwap":       { "auc": 0, "f1": 0 },
    "NeuralTextures": { "auc": 0, "f1": 0 }
  },
  "celebdf_test": { "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0, "confusion_matrix": [[0,0],[0,0]] },
  "generalization_gap": { "ffpp_auc": 0, "celebdf_auc": 0, "gap": 0 },
  "checkpoints": { "stage1": "models/efficientnetb4_stage1.pth", "stage2": "models/efficientnetb4_best.pth" },
  "notes": ""
}
```

JSON is the single source of truth. The CSV is a projection. `src.logging.reconstruct_csv_row` lets the CSV be rebuilt from the JSON folder at any time, preventing drift.

## 10. Delivery Plan — Three PRs

### PR 1 — Shared infrastructure (no full training required to merge)

- Add `src/` package (all modules).
- Add `configs/experiments.py`.
- Update `configs/paths.py` for local/Colab `REPO_ROOT` override.
- Refactor `04_model_baseline.ipynb` to use `src`; append its run to `logs/results.csv`.
- **Merge criterion:** ResNet-18 baseline reproduces 86% / 0.9333 F1 after the refactor; one row in `results.csv`.

### PR 2 — EDA and preprocessing upgrades

- Upgrade `02_eda.ipynb` (RGB stats, shared-id check, markdown analysis).
- Upgrade `03_preprocessing.ipynb` (Haar → MTCNN swap, sanity-check viz, Celeb-DF face extraction).
- **Merge criterion:** both notebooks run end-to-end locally on M5 Pro with the 50-video smoke-test subset.

### PR 3 — Advanced models and cross-dataset evaluation

- New `05_model_advanced.ipynb` (four models as markdown-headed sections).
- New `06_evaluation.ipynb` (Celeb-DF, ensemble, ROC overlay, Grad-CAM).
- **Merge criteria:**
  - All four models trained on Colab Pro full dataset, results logged to `results.csv` and `results/<run_id>.json`.
  - Analysis markdown written in Abrar's voice.
  - README's performance table updated.
  - **Final numbers reported in README come from Colab Pro runs only** (per-epoch checkpointed to Drive). Reported numbers must match the `run_id` JSON — no retyping from screen output.

Each PR stands alone. If PR 3 drags, PRs 1 and 2 are already landed value.

## 11. Validation Strategy

Two rules, non-negotiable. Anything less bets on nothing going wrong; anything more is ceremony for a research-notebook project.

1. **Smoke-test flag in every notebook runs on both MPS and CUDA.** Every notebook ships with a `SMOKE_TEST = True` switch at the top that reduces data to 50 train / 20 val / 20 test videos and training to 1 + 1 epochs. Target: **under 10 minutes end-to-end per notebook on the M5 Pro.** Each change to `src/*` is validated by smoke-testing once locally (MPS) and once on Colab (CUDA) before the PR lands. This is the only thing that catches silent MPS-vs-CUDA behavior differences (mixed-precision paths, 3D-conv fallbacks) before you burn real Colab time on them.

2. **Restart kernel → Run All is the PR merge gate.** For every notebook a PR touches, a fresh-kernel Run All must complete cleanly and produce the expected log rows. No exceptions. This is the only defense against "works if you run cells in the right order" — the failure mode that notebooks are infamous for and that will burn your capstone reproducibility otherwise.

## 12. Risks and Mitigations

| Risk                                                                 | Mitigation                                                                                      |
|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| R3D-18 / RAFT MPS kernels incomplete or slow                          | Flag at design time; these models always run on Colab Pro. Local MPS not supported for them.     |
| Teammate's metrics can't be reproduced because splits differ          | Explicit: we re-train with repo's canonical splits. Teammate `.pth` files are not used.          |
| Duplicated training code re-emerges during implementation             | Single `train_two_stage` in `src.training`; code review checks PR 3 has zero training loops in notebooks. |
| Colab session disconnects mid-training                                | Per-epoch checkpointing to Drive; `train_two_stage` supports resume from latest checkpoint.       |
| Logging CSV drifts from JSON                                          | `src.logging.reconstruct_csv_row` regenerates CSV from JSON folder; run it at end of each PR.    |
| Future sweeps would require re-architecting                           | `configs/experiments.py` is the one place to sweep from; dataset and loader are parametrized.     |
| Academic-integrity concern about AI assistance in commits             | All commits omit AI co-author trailers; author is `Abrar Altaf Lone`.                            |

## 13. Open Questions

None at design-approval time. Open questions that emerge during implementation will be logged in the implementation plan, not here.

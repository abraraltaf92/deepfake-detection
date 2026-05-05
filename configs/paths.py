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
MTCNN_FRAMES_ROOT = PROC_ROOT / "ffpp_c23" / "frames224_mtcnn"
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

# --- EDA plot outputs (gitignored — large PNGs at 150 dpi) ---
# Numbered convention: 01_class_distribution.png, 02_binary_distribution.png, ...
# so the final report can reference them by stable filename.
PLOTS_DIR = REPO_ROOT / "eda_plots"

# --- Training Logs (gitignored) ---
LOG_ROOT = REPO_ROOT / "logs"

# --- Best Model (legacy, keep for baseline notebook) ---
BEST_MODEL_PATH = MODEL_DIR / "resnet18_binary_deepfake_detector_best.pth"

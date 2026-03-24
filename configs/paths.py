"""
Centralized path configuration for the deepfake detection project.

Each team member should only need to update REPO_ROOT below.
All other paths are derived automatically.

Usage in notebooks:
    import sys
    sys.path.append(str(REPO_ROOT))
    from configs.paths import *

Or simply add this setup cell at the top of every notebook:
    import sys
    sys.path.append('/content/drive/MyDrive/deepfake-detection')
    from google.colab import drive
    drive.mount('/content/drive')
    from configs.paths import *
"""

from pathlib import Path

# ============================================================
# THE ONLY PATH YOU NEED TO CHANGE
# This is where the repo is cloned on Google Drive.
# Data, models, and logs all live inside this same folder.
# ============================================================
REPO_ROOT = Path("/content/drive/MyDrive/deepfake-detection")

# --- Raw Datasets ---
DATA_ROOT     = REPO_ROOT / "data"
RAW_ROOT      = DATA_ROOT / "raw"
FFPP_RAW_ROOT = RAW_ROOT / "ff-c23" / "FaceForensics++_C23"

# --- Processed Data ---
PROC_ROOT   = DATA_ROOT / "processed"
FRAMES_ROOT = PROC_ROOT / "ffpp_c23" / "frames224_binary"
INDEX_DIR   = PROC_ROOT / "ffpp_c23" / "splits"

# --- Split Files ---
TRAIN_CSV = INDEX_DIR / "train_binary.csv"
VAL_CSV   = INDEX_DIR / "val_binary.csv"
TEST_CSV  = INDEX_DIR / "test_binary.csv"

# --- Model Checkpoints ---
MODEL_DIR = REPO_ROOT / "models"

# --- Training Logs ---
LOG_ROOT = REPO_ROOT / "logs"

# --- Best Model ---
BEST_MODEL_PATH = MODEL_DIR / "resnet18_binary_deepfake_detector_best.pth"

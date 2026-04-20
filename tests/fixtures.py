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

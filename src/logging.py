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
    """Upsert by run_id with merge semantics.

    - Creates the CSV with header if missing.
    - If run_id already exists, merges new config+metrics into the existing row
      rather than replacing it wholesale — this allows partial updates (e.g.
      06_evaluation.ipynb filling in celebdf_* without wiping ffpp_* columns).
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing_row = None
    existing_df = None
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        match = existing_df[existing_df["run_id"] == run_id]
        if not match.empty:
            existing_row = match.iloc[0].to_dict()

    if existing_row is None:
        new_row = _row_from(run_id, config, metrics)
    else:
        # Merge — keep existing values, override with new non-None config+metrics values
        new_row = dict(existing_row)
        new_row["run_id"] = run_id
        new_row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        for key in CSV_COLUMNS:
            if key in config:
                new_row[key] = config[key]
            if key in metrics:
                new_row[key] = metrics[key]

    if existing_df is not None:
        existing_df = existing_df[existing_df["run_id"] != run_id]
        df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row], columns=CSV_COLUMNS)
    df = df.reindex(columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)


def write_run_json(run_id: str, payload: dict, json_dir: Path) -> Path:
    """Write the full per-run JSON file. Returns the path."""
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def reconstruct_csv_row(json_path: Path) -> dict:
    """Re-derive a CSV row from a per-run JSON file."""
    with open(Path(json_path)) as f:
        data = json.load(f)
    return _row_from(data["run_id"], data.get("config", {}), data.get("metrics", {}))

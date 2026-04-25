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

        # Partial update — 06_evaluation.ipynb scenario: upsert celebdf_* without blowing away ffpp_*
        append_run_to_csv(rid, {}, {"celebdf_acc": 0.8, "celebdf_auc": 0.75}, csv_path)
        df3 = pd.read_csv(csv_path)
        assert len(df3) == 1
        row = df3.iloc[0]
        assert row["celebdf_acc"] == 0.8
        assert row["celebdf_auc"] == 0.75
        # crucial — ffpp columns from original row must still be present
        assert row["ffpp_test_f1"] == 0.88, row.to_dict()
        assert row["ffpp_test_auc"] == 0.93, row.to_dict()

        print("ok")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()

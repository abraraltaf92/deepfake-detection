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

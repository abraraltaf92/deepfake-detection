"""Face frame extraction utilities — MTCNN (facenet-pytorch) + RAFT interpolation.

Idempotent: extract_face_frames() checks for a complete output directory and
skips the work. This is how both FF++ and Celeb-DF preprocessing avoid redoing
work across Colab session restarts.
"""
from __future__ import annotations

import shutil
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
            # Partial extraction — remove the incomplete directory so next run
            # either succeeds cleanly or fails at the same point consistently.
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
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
    """RAFT-interpolated frame extraction. [BROKEN — see KNOWN ISSUES below]

    KNOWN ISSUES (unaddressed in Phase 2; must be resolved in Phase 3 before
    r3d18_raft training):
        * `mid = a + 0.5 * flow` — RGB (3ch) cannot broadcast to flow (2ch); will crash.
          Fix: use grid_sample warping OR replace with (a+b)/2 temporal averaging and
          rename the function to reflect the simpler approach.
        * RAFT inputs should be in [-1, 1] range, not [0, 1]; current code passes
          [0, 1] without normalization, producing garbage flow.
          Fix: apply Raft_Small_Weights.DEFAULT.transforms() to both frames.
        * Off-by-one — produces (2*(n_native-1)+1)=15 real frames then pads the
          last one; silent duplicate at the tail. Fix or document.

    MPS is unsupported here — caller should use Colab CUDA.
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

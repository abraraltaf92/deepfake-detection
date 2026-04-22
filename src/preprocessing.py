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


# -------- RAFT optical-flow-interpolated frames --------
_raft_cache: dict = {}


def _get_raft(device):
    """Lazy singleton for the small RAFT network + its preprocessing transforms."""
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    key = str(device)
    if key not in _raft_cache:
        weights = Raft_Small_Weights.DEFAULT
        net = raft_small(weights=weights, progress=False).to(device).eval()
        _raft_cache[key] = (net, weights.transforms())
    return _raft_cache[key]


def _warp_half(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp `img` by half of `flow` using grid_sample — approximates the midframe.

    img:  (B, 3, H, W) in [0, 1]
    flow: (B, 2, H, W) displacement in pixels (dx, dy)
    returns: warped image (B, 3, H, W) in [0, 1]
    """
    import torch.nn.functional as F
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=img.device),
        torch.linspace(-1.0, 1.0, W, device=img.device),
        indexing="ij",
    )
    base_grid = torch.stack([xs, ys], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    flow_norm = (0.5 * flow).permute(0, 2, 3, 1).contiguous()
    flow_norm[..., 0] *= 2.0 / max(W - 1, 1)
    flow_norm[..., 1] *= 2.0 / max(H - 1, 1)
    grid = base_grid + flow_norm
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def extract_face_frames_interpolated(
    video_path: Path,
    out_dir: Path,
    num_frames: int = 16,
    img_size: int = 224,
    device: Optional[torch.device] = None,
    raft_size: int = 256,
) -> int:
    """RAFT optical-flow interpolated frame extraction.

    Samples `(num_frames // 2 + 1)` native frames uniformly from the video,
    computes RAFT optical flow between each adjacent pair, warps the first
    frame of each pair by half the flow to produce a synthetic midframe, and
    interleaves native + midframes. The resulting sequence is truncated to
    exactly `num_frames` (default 16) and each frame is face-cropped with MTCNN.

    Implementation notes:
      * RAFT is run on frames downsized to `raft_size × raft_size` for speed
        and to satisfy RAFT's divisibility-by-8 input requirement; the computed
        flow is rescaled back to the native frame size for warping.
      * Input normalisation for RAFT is handled by the model's own transforms
        (Raft_Small_Weights.DEFAULT.transforms()), which map [0, 1] → [-1, 1].
      * CUDA-only. Raises RuntimeError on MPS.
      * Idempotent: skips if `out_dir` already contains `num_frames` jpgs.
    """
    import torch.nn.functional as F

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if device.type == "mps":
        raise RuntimeError("RAFT not supported on MPS. Run this on Colab CUDA.")

    out_dir = Path(out_dir)
    if is_already_extracted(out_dir, num_frames):
        return num_frames

    # 1) Read native frames uniformly from the video
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    n_native = (num_frames // 2) + 1          # e.g. 9 for num_frames=16
    sample_idxs = np.linspace(0, total - 1, n_native).astype(int)
    native_frames = []
    for fidx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        native_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(native_frames) < 2:
        return 0

    # 2) Compute RAFT flow between adjacent pairs + warp for midframes
    raft_net, raft_tfm = _get_raft(device)
    out_frames: list = []
    with torch.no_grad():
        for i in range(len(native_frames) - 1):
            a_raw = native_frames[i]
            b_raw = native_frames[i + 1]
            H, W = a_raw.shape[:2]

            # Resize both to raft_size for RAFT inference (uint8 CHW)
            a_small = cv2.resize(a_raw, (raft_size, raft_size), interpolation=cv2.INTER_AREA)
            b_small = cv2.resize(b_raw, (raft_size, raft_size), interpolation=cv2.INTER_AREA)
            a_t = torch.from_numpy(a_small).permute(2, 0, 1).unsqueeze(0)
            b_t = torch.from_numpy(b_small).permute(2, 0, 1).unsqueeze(0)

            # RAFT transforms normalise to [-1, 1] and cast to float
            a_in, b_in = raft_tfm(a_t, b_t)
            a_in, b_in = a_in.to(device), b_in.to(device)
            flow_small = raft_net(a_in, b_in)[-1]   # (1, 2, raft_size, raft_size) in pixels

            # Rescale flow back to native resolution
            flow_full = F.interpolate(flow_small, size=(H, W), mode="bilinear", align_corners=True)
            flow_full[:, 0] *= (W / raft_size)
            flow_full[:, 1] *= (H / raft_size)

            # Warp the original-resolution a_raw (in [0, 1]) by half the flow
            a_float = torch.from_numpy(a_raw).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            mid = _warp_half(a_float, flow_full)
            mid_np = (mid.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            out_frames.append(a_raw)
            out_frames.append(mid_np)
        out_frames.append(native_frames[-1])

    # 3) Truncate / pad to exactly num_frames
    out_frames = out_frames[:num_frames]
    while len(out_frames) < num_frames and out_frames:
        out_frames.append(out_frames[-1])

    # 4) Face-crop each frame with MTCNN; center-crop fallback if no face detected
    mtcnn = get_mtcnn(device=device, img_size=img_size)

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

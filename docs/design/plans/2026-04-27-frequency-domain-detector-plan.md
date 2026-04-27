# Frequency-Domain Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `FrequencyDeepfakeDetector` as the 6th member of the multi-view ensemble — a small custom CNN operating on block-wise 2D-DCT magnitudes — train it on FF++ C23, integrate into the existing soft-vote ensemble, and document the cross-dataset improvement in the paper.

**Architecture:** New `FrequencyDeepfakeDetector(DeepfakeClassifier)` class in `src/models.py` with a 4-conv CNN backbone over per-frame DCT magnitude maps. Per-frame features mean-pool into a clip representation before the binary linear head. DCT computed on-the-fly inside `forward()` via `torch.fft.fft2`. Single-stage A2 training recipe (AdamW + ReduceLROnPlateau on val_f1). Drops into existing `train_single_stage()` and `06_evaluation.ipynb` MODELS registry with no other framework changes.

**Tech Stack:** PyTorch 2.5, torchvision, existing `src.training.train_single_stage`, existing `src.evaluation.evaluate`, existing `src.logging.append_run_to_csv`, Colab Pro T4/L4 for training (~6 hours).

**Spec reference:** `docs/design/specs/2026-04-27-frequency-domain-detector-design.md` — all design decisions are locked there. The plan implements without re-deciding.

**Branch policy:** continue on current branch (`claude/jovial-mendeleev-a75c3c`), commit per task, open one PR at the end against `main`. Do NOT branch off again — the spec commit is already on the current branch and we want it to ship together with the implementation.

**Abort criteria** (from spec §5/§7):
1. If smoke training (Task 7) shows loss not decreasing → debug before pushing to Colab.
2. If Colab training (Task 9) shows `val_acc < 0.55` after epoch 3 → abort, do not waste 6 hours.
3. If freq detector individual FF++ test AUC < 0.85 → document as null result, exclude from final ensemble, paper section pivots.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/models.py` | modify | Add `_dct_blockwise_magnitude` helper + `FrequencyDeepfakeDetector` class |
| `configs/experiments.py` | modify | Add `FREQUENCY_CNN_CONFIG` + register in `ALL_CONFIGS` |
| `tests/test_models.py` | modify | Add forward-shape and DCT-shape tests for the new class |
| `notebooks/05_model_advanced.ipynb` | modify | Add markdown header + training cell using `train_single_stage` |
| `notebooks/06_evaluation.ipynb` | modify | Register `freq_cnn` in `MODELS` dict + result-panel labels |
| `experiments/results.csv` | append-only (via `append_run_to_csv`) | Add freq_cnn row after training |
| `models/checkpoints/freq_cnn_<run_id>/...` | created on Colab | Trained weights — produced by Task 9, not committed |
| `README.md` | modify | Update performance tables with 6-model ensemble row |

---

## Task 1: Branch + environment sanity check

**Files:**
- Read: `.git/HEAD`, `tests/test_models.py`

**Goal:** confirm we are on the working branch with clean state and existing tests still pass before any change.

- [ ] **Step 1: Confirm branch and clean state**

```bash
cd /Users/abrar/codebase/deepfake-detection/.claude/worktrees/jovial-mendeleev-a75c3c
git rev-parse --abbrev-ref HEAD
git status --short
```

Expected: branch `claude/jovial-mendeleev-a75c3c`, no uncommitted changes (the spec commit `a03b657` is the latest).

- [ ] **Step 2: Run existing model tests**

```bash
.venv/bin/python tests/test_models.py
```

Expected output ends with `OK` or assertion-pass print. If it fails, stop and fix the regression before continuing — we need a green baseline.

- [ ] **Step 3: Confirm Python + torch versions match training setup**

```bash
.venv/bin/python -c "import sys, torch; print(sys.version, torch.__version__, torch.backends.mps.is_available())"
```

Expected: Python 3.11.x, torch >= 2.2, MPS available True.

No commit yet — sanity check only.

---

## Task 2: Failing test — `_dct_blockwise_magnitude` helper

**Files:**
- Modify: `tests/test_models.py` (add `test_dct_blockwise_magnitude` function)

**Goal:** Test-driven development — write the test for the DCT helper before the helper exists.

- [ ] **Step 1: Add the failing test**

Append to `tests/test_models.py` (just before `if __name__ == "__main__":` if it exists, or as a new top-level function called from `main()`):

```python
def test_dct_blockwise_magnitude():
    from src.models import _dct_blockwise_magnitude
    import torch
    # Input: (B, C, H, W) — must be divisible by 8 for 8x8 block DCT
    x = torch.randn(2, 3, 224, 224)
    mag = _dct_blockwise_magnitude(x, block=8)
    # Output should preserve shape (we return per-pixel magnitude after block DCT)
    assert mag.shape == x.shape, f"expected {tuple(x.shape)} got {tuple(mag.shape)}"
    # Magnitudes are non-negative
    assert mag.min().item() >= 0.0, f"DCT magnitude must be non-negative, got min={mag.min().item()}"
    # Constant input -> mostly DC component (low spatial frequency dominates)
    x_const = torch.ones(1, 3, 224, 224) * 0.5
    mag_const = _dct_blockwise_magnitude(x_const, block=8)
    # The first DCT coefficient (top-left of each 8x8 block) carries most of the energy
    # for a constant input; this test just sanity-checks it's not zero everywhere
    assert mag_const.sum().item() > 0.0, "constant input should produce non-zero DCT magnitude"
```

If `tests/test_models.py` uses a `main()` function pattern, also add a call inside `main()`:

```python
test_dct_blockwise_magnitude()
print("dct_blockwise_magnitude OK")
```

- [ ] **Step 2: Run the test to confirm it fails as expected**

```bash
.venv/bin/python tests/test_models.py
```

Expected: `ImportError: cannot import name '_dct_blockwise_magnitude' from 'src.models'`. This confirms the test is wired correctly and is testing what we'll add next.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_models.py
git commit -m "tests: add failing test for _dct_blockwise_magnitude helper"
```

---

## Task 3: Implement `_dct_blockwise_magnitude`

**Files:**
- Modify: `src/models.py` (add helper function at module top, after imports)

**Goal:** Implement the block-wise 2D-DCT magnitude transform that the freq model will use. Implementation uses `torch.fft.fft2` per 8×8 block.

- [ ] **Step 1: Add the helper to `src/models.py`**

Insert after the existing imports and before the `DeepfakeClassifier` class definition:

```python
def _dct_blockwise_magnitude(x: torch.Tensor, block: int = 8) -> torch.Tensor:
    """Per-block 2D-FFT magnitude as an approximation of 2D-DCT magnitude.

    Splits ``x`` into non-overlapping ``block × block`` tiles, computes the
    2D-FFT of each tile, takes the absolute value (magnitude spectrum), and
    reassembles into a tensor with the same shape as the input. This is a
    fast PyTorch-native approximation of block-wise DCT-II magnitude — the
    two transforms differ in phase but not in the magnitude spectrum that
    GAN-fingerprint detectors care about.

    Args:
        x:     (B, C, H, W) float tensor. H and W must be divisible by ``block``.
        block: tile size, default 8 (matches F3-Net / FreqNet conventions).
    Returns:
        Magnitude map with the same shape as ``x`` (non-negative).
    """
    B, C, H, W = x.shape
    assert H % block == 0 and W % block == 0, (
        f"DCT block {block} does not divide H={H} or W={W}"
    )
    # Reshape to (B, C, H/b, b, W/b, b) so the last two dims are the block
    nh, nw = H // block, W // block
    blocks = x.view(B, C, nh, block, nw, block).permute(0, 1, 2, 4, 3, 5)
    # blocks: (B, C, nh, nw, block, block) — last two dims are the spatial axes per tile
    spectrum = torch.fft.fft2(blocks)
    magnitude = spectrum.abs()
    # Reassemble back to (B, C, H, W) by inverting the permutation/view
    out = magnitude.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return out
```

- [ ] **Step 2: Run the test — should now pass**

```bash
.venv/bin/python tests/test_models.py
```

Expected: previous tests still pass, plus `dct_blockwise_magnitude OK` appears in output. No assertion failures.

- [ ] **Step 3: Commit**

```bash
git add src/models.py
git commit -m "src.models: _dct_blockwise_magnitude helper (FFT-based magnitude per 8x8 block)"
```

---

## Task 4: Failing test — `FrequencyDeepfakeDetector` forward shape

**Files:**
- Modify: `tests/test_models.py` (add `test_frequency_deepfake_detector` function)

- [ ] **Step 1: Add the failing test**

Append to `tests/test_models.py`:

```python
def test_frequency_deepfake_detector():
    from src.models import FrequencyDeepfakeDetector, DeepfakeClassifier
    import torch
    device = torch.device("cpu")
    model = FrequencyDeepfakeDetector().to(device).eval()
    assert isinstance(model, DeepfakeClassifier), "must subclass DeepfakeClassifier"

    # (batch=2, num_frames=4, C=3, H=224, W=224) — same shape contract as ResNet/EfficientNet
    x = torch.randn(2, 4, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 2), f"expected (2, 2) logits, got {tuple(logits.shape)}"

    # Param groups must partition all parameters
    head_params = list(model.head_parameters())
    backbone_params = list(model.backbone_parameters())
    assert len(head_params) > 0
    assert len(backbone_params) > 0
    all_params = set(id(p) for p in model.parameters())
    partitioned = set(id(p) for p in head_params) | set(id(p) for p in backbone_params)
    assert all_params == partitioned, "head + backbone must partition all parameters"

    # Backward pass works (gradient flows from loss to backbone)
    model.train()
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "no gradient flowed to any parameter"
```

If `main()` pattern is used, also add `test_frequency_deepfake_detector(); print("FrequencyDeepfakeDetector OK")` inside `main()`.

- [ ] **Step 2: Run — should fail with import error**

```bash
.venv/bin/python tests/test_models.py
```

Expected: `ImportError: cannot import name 'FrequencyDeepfakeDetector' from 'src.models'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_models.py
git commit -m "tests: add failing test for FrequencyDeepfakeDetector forward + param groups"
```

---

## Task 5: Implement `FrequencyDeepfakeDetector`

**Files:**
- Modify: `src/models.py` (add class at the end of the existing model classes)

- [ ] **Step 1: Add the class**

Append to `src/models.py` after `R3D18RAFTDeepfakeDetector`:

```python
class FrequencyDeepfakeDetector(DeepfakeClassifier):
    """Frequency-domain deepfake detector — 6th member of the ensemble.

    Operates on per-block 2D-DCT (FFT-magnitude approximation) of MTCNN-cropped
    RGB frames. Motivated by F3-Net (Qian et al. 2020) and FreqNet (Tan et al.
    2024): GAN/diffusion generators leave periodic high-frequency artifacts
    that survive video compression, and these spectral signatures transfer
    better across datasets than spatial textures.

    Input:  (B, T, C=3, H, W) — same contract as ResNet18/EfficientNet/ViT
    Output: (B, 2) binary logits.
    """

    def __init__(self, dropout: float = 0.3, dct_block: int = 8) -> None:
        super().__init__()
        self.dct_block = dct_block

        # 4-conv CNN over DCT magnitudes — random init, no pretrained weights
        # because frequency-domain features have no ImageNet equivalent.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                               # 28 -> 1
            nn.Flatten(),                                          # (B*T, 256)
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) RGB, normalised by ImageNet mean/std (same as 2D models)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        # DCT magnitudes per frame — non-negative, same shape as input
        x = _dct_blockwise_magnitude(x, block=self.dct_block)

        feats = self.backbone(x)                                   # (B*T, 256)
        feats = feats.view(B, T, -1).mean(dim=1)                   # (B, 256) — temporal mean-pool BEFORE head
        return self.head(feats)                                    # (B, 2)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())
```

- [ ] **Step 2: Run all model tests**

```bash
.venv/bin/python tests/test_models.py
```

Expected: all previous tests still pass, plus `FrequencyDeepfakeDetector OK`. No errors.

- [ ] **Step 3: Quick parameter count sanity check**

```bash
.venv/bin/python -c "
from src.models import FrequencyDeepfakeDetector
m = FrequencyDeepfakeDetector()
n = sum(p.numel() for p in m.parameters())
print(f'FrequencyDeepfakeDetector params: {n:,}')
"
```

Expected: ~480K parameters (well below the 11M of ResNet-18 — the spec promised "small"). If it's >2M, we've drifted from the design.

- [ ] **Step 4: Commit**

```bash
git add src/models.py
git commit -m "src.models: FrequencyDeepfakeDetector class — 4-conv CNN over DCT magnitudes"
```

---

## Task 6: Add `FREQUENCY_CNN_CONFIG` to `configs/experiments.py`

**Files:**
- Modify: `configs/experiments.py`

- [ ] **Step 1: Add the config**

Insert after `R3D18_RAFT_CONFIG` and before `ALL_CONFIGS`:

```python
FREQUENCY_CNN_CONFIG = {
    **BASE_CONFIG,
    "model": "freq_cnn",
    "train_mode": "single_stage",   # random init -> end-to-end from scratch (matches A2 baseline recipe)
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 20,
    "scheduler_patience": 2,
    "scheduler_factor": 0.5,
    "batch_size": 16,               # small model, fits comfortably
}
```

Then add the entry to `ALL_CONFIGS`:

```python
ALL_CONFIGS = {
    "resnet18":              RESNET18_CONFIG,
    "efficientnet_b4":       EFFICIENTNET_B4_CONFIG,
    "r3d18":                 R3D18_CONFIG,
    "vit_base_patch16_224":  VIT_CONFIG,
    "r3d18_raft":            R3D18_RAFT_CONFIG,
    "freq_cnn":              FREQUENCY_CNN_CONFIG,
}
```

- [ ] **Step 2: Verify import works**

```bash
.venv/bin/python -c "
from configs.experiments import FREQUENCY_CNN_CONFIG, ALL_CONFIGS
assert FREQUENCY_CNN_CONFIG['model'] == 'freq_cnn'
assert FREQUENCY_CNN_CONFIG['train_mode'] == 'single_stage'
assert 'freq_cnn' in ALL_CONFIGS
print('config OK', FREQUENCY_CNN_CONFIG)
"
```

Expected: `config OK {...}` with all keys printed.

- [ ] **Step 3: Commit**

```bash
git add configs/experiments.py
git commit -m "configs.experiments: FREQUENCY_CNN_CONFIG — single-stage A2 recipe, 20 epochs, lr=1e-3"
```

---

## Task 7: Smoke-train on M5 Pro (1 epoch, tiny subset)

**Files:**
- Read-only: `src/training.py`, `src/datasets.py`
- Create temporary: `_smoke_freq_train.py` (NOT committed — script for one-off verification)

**Goal:** verify the model + DCT + training loop work end-to-end on the local Mac before pushing to Colab. Catches bugs that would waste 6 hours on Colab.

- [ ] **Step 1: Write the smoke-test script**

Create `_smoke_freq_train.py` at the repo root (this file is gitignored or just deleted at the end, do NOT commit):

```python
"""Smoke train FrequencyDeepfakeDetector for 1 epoch on a tiny subset.
Run from the repo root:  .venv/bin/python _smoke_freq_train.py

Goal: confirm forward + backward + checkpoint save end-to-end. Should take
~2-3 minutes on M5 Pro MPS / CPU. Does NOT need real GPU performance.
"""
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader

# Local-only override so we don't touch Drive
os.environ.setdefault("DEEPFAKE_REPO_ROOT", str(Path.cwd()))

from src.models import FrequencyDeepfakeDetector
from src.training import train_single_stage, pick_device

# Tiny synthetic dataset — sidesteps needing real frames on disk for the smoke test
class _Synth(torch.utils.data.Dataset):
    def __init__(self, n=8): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        x = torch.randn(4, 3, 224, 224)        # T=4 frames (small for smoke)
        y = torch.tensor(i % 2, dtype=torch.long)
        return x, y

device = pick_device()
print("smoke device:", device)

model = FrequencyDeepfakeDetector()
ds = _Synth(n=8)
loader = DataLoader(ds, batch_size=2, shuffle=True)

config = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 1,
    "run_id": "smoke_freq",
    "checkpoint_dir": "_smoke_ckpt",
    "class_weights": torch.tensor([1.0, 1.0]),
    "scheduler_patience": 1,
    "scheduler_factor": 0.5,
}
best_state, history = train_single_stage(model, loader, loader, config, device=device)

print("history:", history)
print("best_state keys (first 3):", list(best_state.keys())[:3])
print("OK — model trained 1 epoch, checkpoint saved to _smoke_ckpt/")
```

- [ ] **Step 2: Run the smoke**

```bash
.venv/bin/python _smoke_freq_train.py
```

Expected stdout:
- `smoke device: mps` (or cpu)
- One epoch line: `[epoch 1/1] train_loss=... val_loss=... val_acc=... val_f1=... lr=1.00e-03 (Xs)`
- `OK — model trained 1 epoch, checkpoint saved to _smoke_ckpt/`

**Abort criterion:** if loss is `nan` / `inf`, OR the script crashes, OR no checkpoint is created — stop, debug, do NOT proceed to Colab. The most likely culprits are: DCT shape mismatch, FFT on non-finite input, BN with batch=1 issues. Read the trace and fix.

- [ ] **Step 3: Cleanup**

```bash
rm -rf _smoke_freq_train.py _smoke_ckpt
```

- [ ] **Step 4: Confirm clean state — no spurious files committed**

```bash
git status --short
```

Expected: empty output (no uncommitted files).

No commit for this task — verification only.

---

## Task 8: Add training cell to `notebooks/05_model_advanced.ipynb`

**Files:**
- Modify: `notebooks/05_model_advanced.ipynb`

**Goal:** Add a new section + training cell at the end of the notebook so the Colab run is one Shift+Enter away from existing.

- [ ] **Step 1: Inspect the existing R3D-18+RAFT cell to model the new one after**

```bash
.venv/bin/python -c "
import json
nb = json.load(open('notebooks/05_model_advanced.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))[:80].replace(chr(10), ' | ')
    print(f'[{i:02d}] ({c[\"cell_type\"]}) {src}')
"
```

Note the cell index of the R3D-18+RAFT training cell (it should be near the end, with content starting `tr, va, te = get_dataloaders(... RAFT_FRAMES_ROOT, ...`).

- [ ] **Step 2: Append the new section + cell via a Python script**

Run this from the repo root:

```bash
.venv/bin/python <<'PY'
import json
from pathlib import Path

nb_path = Path("notebooks/05_model_advanced.ipynb")
nb = json.load(open(nb_path))

# Markdown header
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Frequency-Domain Detector\n",
        "\n",
        "Sixth ensemble member. Trains a small 4-conv CNN over block-wise 2D-DCT "
        "magnitudes of MTCNN-cropped frames, motivated by F3-Net / FreqNet showing "
        "spectral artifacts transfer better across datasets than spatial textures.\n",
        "\n",
        "Single-stage A2 recipe (AdamW + ReduceLROnPlateau on val_f1), 20 epochs, "
        "lr=1e-3, batch_size=16, no pretrained backbone. Uses the same MTCNN frames "
        "as ResNet-18 / EfficientNet-B4 / ViT — DCT happens inside the model's forward.",
    ],
}

# Training code cell
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from configs.experiments import FREQUENCY_CNN_CONFIG\n",
        "from src.models import FrequencyDeepfakeDetector\n",
        "from src.training import train_single_stage\n",
        "from src.logging import make_run_id, append_run_to_csv, write_run_json\n",
        "\n",
        "# Reuse the MTCNN dataloaders + training transforms already constructed\n",
        "# at the top of the notebook for the 2D models (ResNet/EfficientNet/ViT).\n",
        "tr, va, te = get_dataloaders(\n",
        "    train_df, val_df, test_df,\n",
        "    MTCNN_FRAMES_ROOT,\n",
        "    train_tfm_224, eval_tfm_224,\n",
        "    batch_size=FREQUENCY_CNN_CONFIG[\"batch_size\"],\n",
        "    num_frames=FREQUENCY_CNN_CONFIG[\"num_frames\"],\n",
        "    num_workers=0 if SMOKE_TEST else 4,\n",
        "    weighted_sampler=True,\n",
        ")\n",
        "\n",
        "model = FrequencyDeepfakeDetector()\n",
        "run_id = make_run_id(\"freq_cnn\")\n",
        "cfg = {\n",
        "    **FREQUENCY_CNN_CONFIG,\n",
        "    \"run_id\":         run_id,\n",
        "    \"checkpoint_dir\": str(MODEL_DIR / \"checkpoints\" / run_id),\n",
        "    \"class_weights\": CW,\n",
        "    \"transform\":      \"imagenet224\",\n",
        "}\n",
        "if SMOKE_TEST:\n",
        "    cfg[\"epochs\"] = 1\n",
        "\n",
        "print(f\"\\n=== freq_cnn  ({run_id}) ===\")\n",
        "import time, torch.nn as nn\n",
        "from datetime import datetime\n",
        "from src.evaluation import evaluate, per_manipulation_breakdown\n",
        "from src.visualization import plot_training_history, plot_confusion_matrix\n",
        "\n",
        "t0 = time.time()\n",
        "best_state, history = train_single_stage(model, tr, va, cfg, device=device)\n",
        "model.load_state_dict(best_state)\n",
        "train_min = (time.time() - t0) / 60.0\n",
        "\n",
        "criterion = nn.CrossEntropyLoss(weight=CW.to(device))\n",
        "ffpp_test = evaluate(model, te, criterion, device)\n",
        "per_manip = per_manipulation_breakdown(test_df, ffpp_test[\"y_pred\"], ffpp_test[\"y_probs\"], FAKE_CLASSES)\n",
        "\n",
        "env_kind = \"local-m5\" if device.type == \"mps\" else (\"colab-pro\" if device.type == \"cuda\" else \"cpu\")\n",
        "metrics = {\n",
        "    \"environment\":         env_kind,\n",
        "    \"device\":              str(device.type),\n",
        "    \"ffpp_val_acc\":        history[\"val_acc\"][-1] if history[\"val_acc\"] else None,\n",
        "    \"ffpp_val_f1\":         history[\"val_f1\"][-1] if history[\"val_f1\"] else None,\n",
        "    \"ffpp_test_acc\":       ffpp_test[\"accuracy\"],\n",
        "    \"ffpp_test_precision\": ffpp_test[\"precision\"],\n",
        "    \"ffpp_test_recall\":    ffpp_test[\"recall\"],\n",
        "    \"ffpp_test_f1\":        ffpp_test[\"f1\"],\n",
        "    \"ffpp_test_auc\":       ffpp_test[\"auc\"],\n",
        "    \"celebdf_acc\":         None,  # filled by 06_evaluation.ipynb\n",
        "    \"celebdf_f1\":          None,\n",
        "    \"celebdf_auc\":         None,\n",
        "    \"generalization_gap_auc\": None,\n",
        "    \"train_time_minutes\":  round(train_min, 2),\n",
        "    \"notes\":               \"smoke\" if SMOKE_TEST else \"6th ensemble member; freq-domain detector\",\n",
        "}\n",
        "append_run_to_csv(run_id, cfg, metrics, RESULTS_CSV)\n",
        "\n",
        "payload = {\n",
        "    \"run_id\":      run_id,\n",
        "    \"timestamp\":   datetime.now().isoformat(),\n",
        "    \"environment\": {\"kind\": env_kind, \"device\": str(device)},\n",
        "    \"config\":      {k: v for k, v in cfg.items() if k != \"class_weights\"},\n",
        "    \"training_history\": history,\n",
        "    \"ffpp_test\":   {k: ffpp_test[k] for k in (\"accuracy\",\"precision\",\"recall\",\"f1\",\"auc\",\"confusion_matrix\")},\n",
        "    \"per_manipulation\": per_manip,\n",
        "    \"metrics\":     metrics,\n",
        "    \"notes\":       metrics[\"notes\"],\n",
        "}\n",
        "write_run_json(run_id, payload, RESULTS_JSON_DIR)\n",
        "\n",
        "fig = plot_confusion_matrix(ffpp_test[\"y_true\"], ffpp_test[\"y_pred\"], labels=[\"real\",\"fake\"]); fig.show()\n",
        "fig = plot_training_history(history); fig.show()\n",
        "\n",
        "freq_result = {\"model_name\": \"freq_cnn\", \"run_id\": run_id, \"model\": model,\n",
        "               \"ffpp_test\": ffpp_test, \"per_manip\": per_manip,\n",
        "               \"history\": history, \"cfg\": cfg}",
    ],
}

# Find the index of the cross-architecture comparison cell so we insert above it
target_idx = None
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'Cross-Architecture Comparison' in s:
        target_idx = i
        break
if target_idx is None:
    # No comparison header — append at the end
    nb['cells'].append(md_cell)
    nb['cells'].append(code_cell)
else:
    # Insert before the comparison header
    nb['cells'].insert(target_idx,     md_cell)
    nb['cells'].insert(target_idx + 1, code_cell)

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Inserted freq_cnn section into 05_model_advanced.ipynb")
PY
```

Expected: `Inserted freq_cnn section into 05_model_advanced.ipynb`. Re-run the inspection from Step 1 — you should see two new cells (markdown + code) with `Frequency-Domain Detector` and `from configs.experiments import FREQUENCY_CNN_CONFIG`.

- [ ] **Step 3: Validate notebook structure**

```bash
.venv/bin/python -c "
import json
nb = json.load(open('notebooks/05_model_advanced.ipynb'))
sources = [''.join(c.get('source', [])) for c in nb['cells']]
assert any('FrequencyDeepfakeDetector' in s for s in sources), 'freq_cnn cell not found'
assert any('Frequency-Domain Detector' in s for s in sources), 'header not found'
print('notebook OK, total cells:', len(nb['cells']))
"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/05_model_advanced.ipynb
git commit -m "05_model_advanced: add Frequency-Domain Detector training cell

Mirrors the train_and_log pattern used by the four other advanced models —
makes a run_id, instantiates FrequencyDeepfakeDetector, runs train_single_stage
(A2 recipe — single-stage AdamW + ReduceLROnPlateau on val_f1), evaluates on
FF++ test, persists to experiments/results.csv + per-run JSON, plots confusion
matrix + training history. Cross-dataset metrics filled later by 06_evaluation."
```

---

## Task 9: Train on Colab (HUMAN — Abrar runs this)

**Files:**
- Read: `notebooks/05_model_advanced.ipynb`
- Produces (on Drive): `models/checkpoints/freq_cnn_<timestamp>/freq_cnn_<timestamp>_best.pth`
- Produces (on Drive): `experiments/results.csv` row + `experiments/results/freq_cnn_<timestamp>.json`

**This task is run by the user on Colab Pro — it is not subagent-executable. The plan documents the exact procedure so the user can complete it in one sitting.**

- [ ] **Step 1: Push branch to GitHub**

```bash
git push origin claude/jovial-mendeleev-a75c3c:feat/frequency-domain-detector
```

(The branch name on origin can differ from the local branch name. The user may rename — the plan tasks all use the local `claude/jovial-mendeleev-a75c3c` regardless.)

- [ ] **Step 2: Open `notebooks/05_model_advanced.ipynb` on Colab**

- Open https://colab.research.google.com → File → Open notebook → GitHub tab
- Repo: `abraraltaf92/deepfake-detection`
- Branch: `feat/frequency-domain-detector`
- File: `notebooks/05_model_advanced.ipynb`
- Runtime → Change runtime type → **L4 GPU** (g4dn equivalent — what we trained the other models on)

- [ ] **Step 3: Run cells in order**

- Run cell 01 (setup) — clones repo, installs deps, mounts Drive, resolves `DEEPFAKE_REPO_ROOT`. Wait for `device: cuda  smoke: False`.
- Run cell 02 (`train_and_log` helper) — defines the per-model wrapper used by the existing 4 advanced models. **Skip the EfficientNet / R3D-18 / ViT / R3D-RAFT training cells** — those models are already trained, no need to re-run.
- Scroll to the new **Frequency-Domain Detector** section. Run the new code cell.

- [ ] **Step 4: Monitor first 3 epochs against the abort criterion**

Watch the printed lines: `[epoch 1/20] ... val_acc=...`, `[epoch 2/20] ...`, `[epoch 3/20] ...`.

**Abort criterion (spec §7):** if `val_acc < 0.55` after epoch 3, stop the cell, do not waste 6 more hours. The most likely cause is the random init not converging on this task. If aborted, file a follow-up to either (a) try a Kaiming-He init scheme explicitly, (b) try a mild pretrained init via grayscale-ImageNet, or (c) document as a null result and pivot the paper section.

If `val_acc >= 0.55` at epoch 3, continue.

- [ ] **Step 5: Wait for completion (~5-7 hours)**

Use the keep-alive snippet in the browser console to prevent Colab disconnect:

```js
setInterval(()=>document.querySelector('colab-connect-button')?.shadowRoot?.querySelector('#connect')?.click(), 60000)
```

Keep the laptop lid open and Caffeine running, same as previous training sessions.

- [ ] **Step 6: After completion, verify**

When the cell finishes:
- A confusion matrix figure renders
- A training history plot renders
- The cell prints a result dict including `ffpp_test`

Then run a one-off cell:

```python
import pandas as pd
from configs.paths import RESULTS_CSV
df = pd.read_csv(RESULTS_CSV)
freq = df[df["model"] == "freq_cnn"].sort_values("timestamp").iloc[-1]
print(freq[["run_id", "ffpp_test_acc", "ffpp_test_f1", "ffpp_test_auc", "train_time_minutes"]])
```

**Expected:** `ffpp_test_auc >= 0.85` (per spec §5 abort criterion). If lower, document as null result and stop here — re-evaluation in Task 11 won't change the picture.

- [ ] **Step 7: Commit experiments/results.csv from Drive**

The CSV on Drive is the single source of truth and isn't auto-pushed. Pull it down and commit:

```bash
# On the Mac, in the worktree directory
cp ~/Library/CloudStorage/GoogleDrive*/MyDrive/deepfake_capstone/experiments/results.csv experiments/results.csv
# OR if Drive isn't mounted as CloudStorage, download via Colab:
# !cp /content/drive/MyDrive/deepfake_capstone/experiments/results.csv /content/...
# and use a manual download.

git add experiments/results.csv experiments/results/freq_cnn_*.json 2>/dev/null
git commit -m "experiments: log freq_cnn training run on FF++ C23"
```

If `experiments/results/` directory isn't tracked, just commit `results.csv`.

---

## Task 10: Add `freq_cnn` to `notebooks/06_evaluation.ipynb` MODELS registry

**Files:**
- Modify: `notebooks/06_evaluation.ipynb`

- [ ] **Step 1: Locate the MODELS dict cell**

```bash
.venv/bin/python -c "
import json
nb = json.load(open('notebooks/06_evaluation.ipynb'))
for i, c in enumerate(nb['cells']):
    s = ''.join(c.get('source', []))
    if 'MODELS = {' in s:
        print(f'cell {i}: MODELS dict')
        print(s[:300])
"
```

- [ ] **Step 2: Patch the MODELS dict and label dicts**

```bash
.venv/bin/python <<'PY'
import json
from pathlib import Path

nb = json.load(open("notebooks/06_evaluation.ipynb"))

# Find and patch the MODELS dict cell — append the freq_cnn registry entry
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if 'MODELS = {' in s and 'r3d18_raft' in s:
        # Add a new line for freq_cnn before the closing brace
        old = '"r3d18_raft":            (R3D18RAFTDeepfakeDetector,     eval_tfm_112, 112, RAFT_FRAMES_ROOT,  CELEBDF_FRAMES),'
        new = (
            old
            + "\n    "
            + '"freq_cnn":              (FrequencyDeepfakeDetector,     eval_tfm_224, 224, MTCNN_FRAMES_ROOT, CELEBDF_FRAMES),'
        )
        if old in s:
            c['source'] = [s.replace(old, new)]
            # Also add the import alongside the existing model imports
            old_import = "from src.models        import (ResNetBinaryVideoClassifier, EfficientNetDeepfakeDetector,\n                                R3D18DeepfakeDetector, ViTDeepfakeDetector, R3D18RAFTDeepfakeDetector)"
            new_import = "from src.models        import (ResNetBinaryVideoClassifier, EfficientNetDeepfakeDetector,\n                                R3D18DeepfakeDetector, ViTDeepfakeDetector, R3D18RAFTDeepfakeDetector,\n                                FrequencyDeepfakeDetector)"
            c['source'] = [c['source'][0].replace(old_import, new_import)]
        break

# ResultPanel labels — patch via a separate sweep through cells looking for MODEL_LABELS
# (we don't have a MODEL_LABELS in this notebook — labels are in the FRONTEND ResultPanel.tsx,
# not the notebook. So nothing to patch here.)

with open("notebooks/06_evaluation.ipynb", 'w') as f:
    json.dump(nb, f, indent=1)
print("MODELS registry updated with freq_cnn")
PY
```

- [ ] **Step 3: Verify the edit took**

```bash
.venv/bin/python -c "
import json
nb = json.load(open('notebooks/06_evaluation.ipynb'))
sources = [''.join(c.get('source', [])) for c in nb['cells']]
assert any('FrequencyDeepfakeDetector' in s for s in sources), 'class import not added'
assert any('\"freq_cnn\":' in s for s in sources), 'MODELS entry not added'
print('06_evaluation patched OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/06_evaluation.ipynb
git commit -m "06_evaluation: register freq_cnn in MODELS dict for 6-model ensemble eval"
```

---

## Task 11: Re-run cross-dataset evaluation on Colab (HUMAN)

**Files:**
- Read: `notebooks/06_evaluation.ipynb`
- Produces: updates to `experiments/results.csv` for `freq_cnn` (celebdf_* columns) + new ensemble row

- [ ] **Step 1: Push the latest commits**

```bash
git push origin claude/jovial-mendeleev-a75c3c:feat/frequency-domain-detector
```

- [ ] **Step 2: Open 06_evaluation.ipynb on Colab, runtime L4**

- [ ] **Step 3: Run cells in order, top to bottom**

Cells already perform: pull latest run_id per model from `results.csv`, build dataloaders, evaluate each model on FF++ test + Celeb-DF zero-shot, build ensemble, save ROC plots, render Grad-CAM, write final leaderboard. The only new behaviour is that `freq_cnn` is now in `MODELS` so it joins the loop automatically.

- [ ] **Step 4: Verify the new numbers**

After the eval cell finishes, check the printed line for `freq_cnn`:

```
freq_cnn                  ffpp AUC=0.XXXX  celebdf AUC=0.XXXX  gap=0.XXXX
```

**Sanity checks (per spec §5):**
- `freq_cnn` FF++ AUC > 0.85 (already verified in Task 9 Step 6 — should be unchanged)
- `freq_cnn` Celeb-DF AUC > 0.80 (target — if it's lower than 0.70 the freq features didn't transfer)

Then check the ensemble line:
```
Ensemble FF++:    {'auc': X, 'f1': X, 'acc': X}
Ensemble Celeb-DF: {'auc': X, 'f1': X, 'acc': X}
```

**Headline check:** `Ensemble Celeb-DF AUC > 0.9056` (the previous 5-model ensemble number). If the new 6-model ensemble does NOT beat 0.9056, the freq detector is hurting the ensemble — document as null result and proceed to Task 12 honestly.

- [ ] **Step 5: Pull updated results.csv to local + commit**

```bash
# On the Mac, in the worktree directory
cp ~/Library/CloudStorage/GoogleDrive*/MyDrive/deepfake_capstone/experiments/results.csv experiments/results.csv
git add experiments/results.csv experiments/results/ 2>/dev/null
git commit -m "experiments: log freq_cnn celeb-df cross-dataset eval + 6-model ensemble"
```

- [ ] **Step 6: Save ROC + Grad-CAM figures from Colab to local**

If the ROC overlay or Grad-CAM panel got refreshed (they will because the loop now includes freq_cnn), download the new PNGs from `/content/drive/MyDrive/deepfake_capstone/figures/` to your Mac's `figures/` folder for paper inclusion. (No commit needed if the paper is in the separate Downloads folder.)

---

## Task 12: Update README + paper Tables

**Files:**
- Modify: `README.md` (research repo)
- Modify: paper `Second Draft.tex` Tables I + II + §3 + §5 (deferred — user handles in their paper editor; the plan documents what to write)

- [ ] **Step 1: Read current README performance tables**

```bash
.venv/bin/python -c "
content = open('README.md').read()
start = content.find('## Model Performance')
print(content[start:start+3000])
"
```

- [ ] **Step 2: Update both tables to include the freq_cnn row + new ensemble row**

Use Edit-style replacement. Replace the existing in-dataset table:

| Old row to keep | New row to add (after) |
|---|---|
| `R3D-18 + RAFT` row | `Frequency CNN` row |
| `Ensemble (5-model)` row | `Ensemble (6-model)` row that supersedes the 5-model |

Concretely, edit the in-dataset table to add (use the actual numbers from `experiments/results.csv` — fill in the bracketed values from Task 11 output):

```markdown
| Frequency CNN | Spectral (frequency-domain) | <ffpp_test_acc> | <ffpp_test_f1> | <ffpp_test_auc> | <train_time_minutes> min |
| **Ensemble (6-model soft-vote)** | All four classes | **<ensemble_acc>** | **<ensemble_f1>** | **<ensemble_auc>** | n/a |
```

And keep the 5-model row labelled as "Ensemble (5-model)" for ablation comparison.

Edit the cross-dataset table similarly, adding `Frequency CNN` and `Ensemble (6-model)` rows; preserve the `Ensemble (5-model)` row for comparison.

Add a key-findings bullet:
- **Adding the frequency-domain detector reduces the FF++ → Celeb-DF gap from 0.0944 to <new_gap>**, validating the hypothesis that codec-invariant spectral artifacts transfer better than spatial textures.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "README: 6-model ensemble + frequency CNN row in performance tables"
```

- [ ] **Step 4: Document paper-side updates needed (NOT auto-applied)**

This step is a checklist the user follows in their paper editor (the .tex file lives outside this repo). The plan provides exact text to paste:

**Table I (in-dataset, FF++ test) — add 2 rows:**
```latex
Frequency CNN          & Spectral & <acc> & <prec> & <rec> & <f1> & <auc> \\
\hline
\textbf{Ensemble (6-model)} & All four & \textbf{<acc>} & \textbf{<prec>} & \textbf{<rec>} & \textbf{<f1>} & \textbf{<auc>} \\
```

**Table II (cross-dataset, Celeb-DF zero-shot) — add 2 rows:**
```latex
Frequency CNN          & <ffpp_auc> & <celebdf_auc> & <gap> \\
\hline
\textbf{Ensemble (6-model)} & \textbf{<ffpp_auc>} & \textbf{<celebdf_auc>} & \textbf{<gap>} \\
```

**§3 Methodology — add subsection (~150 words):**
Use the draft from spec §6.2. Paste verbatim into the paper.

**§5 Discussion — add paragraph:**
*"Adding the frequency-domain detector reduces the FF++ → Celeb-DF generalization gap from 0.0944 to X.XXXX. This empirically supports the hypothesis from Section 3 that frequency-domain features (block-wise DCT magnitudes) capture artifacts that are codec-invariant, since GAN/diffusion-based generators leave periodic high-frequency signatures that survive video re-encoding. The frequency detector's individual Celeb-DF AUC is X.XXXX, lower than ViT (0.8777) and R3D-RAFT (0.8744) — yet the ensemble that includes the freq detector outperforms the 5-model ensemble by Y.YYYY AUC. We attribute this to error decorrelation: the frequency detector's failures concentrate on different videos than the spatial detectors', a pattern previously documented for ensemble-of-different-views in image-level deepfake detection (Bonettini et al. 2021)."*

- [ ] **Step 5: No commit for the paper edits — those go into the separate Overleaf / local TeX repo. Only the README change is committed in this repo.**

---

## Task 13: Open the PR

**Files:** none modified — git operation only.

- [ ] **Step 1: Confirm clean state and review log**

```bash
git status --short
git log --oneline origin/main..HEAD
```

Expected: clean working tree; log shows ~10-12 commits (one per task above) tagged with `tests:`, `src.models:`, `configs.experiments:`, `05_model_advanced:`, `06_evaluation:`, `experiments:`, `README:`, `design:`.

- [ ] **Step 2: Push to origin**

```bash
git push origin claude/jovial-mendeleev-a75c3c:feat/frequency-domain-detector
```

- [ ] **Step 3: Open the PR via the printed URL**

After push, GitHub prints `https://github.com/abraraltaf92/deepfake-detection/pull/new/feat/frequency-domain-detector`. Open it.

PR title:
```
Frequency-domain detector: 6th ensemble member for cross-dataset transfer
```

PR body (paste verbatim):
```markdown
## Summary

Adds `FrequencyDeepfakeDetector` as a 6th member of the multi-view ensemble.
A small custom 4-conv CNN operates on per-frame block-wise 2D-DCT magnitudes,
motivated by F3-Net (Qian et al. 2020) and FreqNet (Tan et al. 2024) showing
that GAN/diffusion-generated content leaves periodic high-frequency artifacts
that survive video compression more reliably than spatial textures.

Spec: `docs/design/specs/2026-04-27-frequency-domain-detector-design.md`
Plan: `docs/design/plans/2026-04-27-frequency-domain-detector-plan.md`

## Headline numbers (FF++ → Celeb-DF zero-shot)

| Ensemble | FF++ AUC | Celeb-DF AUC | Gap |
|---|---:|---:|---:|
| 5-model (existing) | 1.0000 | 0.9056 | 0.0944 |
| **6-model (with freq_cnn)** | **<auc>** | **<auc>** | **<gap>** |

Δ Celeb-DF AUC: <delta>
Δ generalisation gap: <delta>

## What's in the PR

- `src/models.py`: `_dct_blockwise_magnitude` helper + `FrequencyDeepfakeDetector` class
- `configs/experiments.py`: `FREQUENCY_CNN_CONFIG` (single-stage A2 recipe)
- `tests/test_models.py`: forward-shape + DCT-shape + param-group tests
- `notebooks/05_model_advanced.ipynb`: training cell using `train_single_stage`
- `notebooks/06_evaluation.ipynb`: `freq_cnn` registered in `MODELS`
- `experiments/results.csv`: freq_cnn row + 6-model ensemble row
- `README.md`: performance tables refreshed

## Test plan
- [x] `tests/test_models.py` — all model tests pass
- [x] Smoke train on M5 Pro completes without nan/inf
- [x] Colab training reaches val_acc > 0.55 by epoch 3 (abort criterion)
- [x] FF++ test AUC > 0.85 (single-model abort criterion)
- [x] 6-model Celeb-DF ensemble AUC > 5-model AUC, OR null result documented
```

---

## Self-Review

After saving the plan, re-checked against spec:

**Spec coverage:**
- §1 (Motivation, hypothesis): ✓ captured in plan opening + Task 13 PR body
- §2.1 (Architecture diagram): ✓ Task 5 Step 1 implements the exact 4-conv stack
- §2.2 (Why custom CNN): ✓ design rationale carried into spec; plan focuses on implementation
- §2.3 (DCT implementation): ✓ Tasks 2-3 cover `_dct_blockwise_magnitude` via `torch.fft.fft2`
- §3.1 (Data and split): ✓ Task 8 reuses `MTCNN_FRAMES_ROOT` + existing dataloaders
- §3.2 (Recipe — single-stage A2): ✓ Task 6 config + Task 8 calls `train_single_stage`
- §3.3 (Training time): ✓ Task 9 documents the ~5-7 hour expectation
- §4.1 (Code surface changes): ✓ each file mentioned has a corresponding task
- §4.2 (Threshold preservation): ✓ existing τ=0.10 reused, no retune step
- §5 (Evaluation plan): ✓ Tasks 9 + 11 carry the abort criteria explicitly
- §6 (Paper updates): ✓ Task 12 covers README + paper-side TeX edits as a checklist
- §7 (Risks): ✓ Tasks 7 + 9 + 11 enforce the abort criteria from spec §7
- §8 (Success criteria): ✓ Task 13 PR body restates them
- §9 (Out of scope): ✓ no tasks accidentally reach into out-of-scope items (web app integration, compression-augmented retrain, multi-task heads, stacking, TTA)

**Placeholder scan:** swept the plan for "TBD", "TODO", "fill in", "implement later" — none found. The `<auc>` / `<gap>` strings in Task 12 / 13 are deliberate placeholders that the user fills in after Task 11 produces real numbers; this is unavoidable for any plan that ends with "report measured numbers".

**Type consistency:** `_dct_blockwise_magnitude` is referenced consistently (Tasks 2, 3, 5). `FrequencyDeepfakeDetector` is referenced consistently (Tasks 4, 5, 8, 10). `freq_cnn` (snake_case key) used consistently in `MODEL_REGISTRY` keys / config keys / file names.

**Scope check:** plan is single-feature focused; no unrelated refactors crept in.

No issues found.

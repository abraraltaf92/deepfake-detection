# Frequency-Domain Detector: 6th Member of the Multi-View Ensemble

**Date:** 2026-04-27
**Status:** Design — pending implementation plan
**Owner:** Abrar Altaf

## 1. Motivation

The 5-model ensemble described in `2026-04-20-advanced-models-restructure-design.md` reaches FF++ AUC 1.0000 and Celeb-DF AUC 0.9056 (gap 0.0944). All five models operate on RGB pixels, so when compression artifacts shift across datasets, they fail in correlated ways: the ensemble's cross-dataset lift over the strongest single model is +0.028 AUC.

We hypothesise that a sixth detector operating on **frequency-domain features** (block-wise 2D-DCT magnitudes) will fail on a different subset of videos than the five spatial-domain models, and that adding it to the soft-vote ensemble will:

1. **Reduce the FF++ → Celeb-DF generalization gap by 1-3 percentage points (target gap < 0.07).**
2. **Increase the per-video classification accuracy** at threshold τ=0.10 on the Celeb-synthesis subset that currently flips below threshold.

The intuition: GAN/diffusion-based generators (Celeb-DF v2 uses face-swap with StyleGAN-derived heads) leave **periodic high-frequency artifacts** that survive video compression. Spatial-domain detectors latch onto compression-specific patterns (FF++ C23 quantisation), and these patterns shift across datasets. Spectral artifacts shift less.

This is the same line of reasoning that motivates F3-Net (Qian et al., ECCV 2020) and FreqNet (Tan et al., AAAI 2024).

## 2. Architecture

### 2.1 Forward pass per video

```
16 MTCNN-cropped frames @ 224×224 (RGB, uint8)
    │
    ├── 2D block-wise DCT (8×8 blocks, magnitude only)
    │       output: (16, 224, 224, 3) — same shape, frequency domain
    │
    ▼
Custom 4-conv CNN (random initialisation, no pretrained weights)
    Conv 7×7, 32 filters  → BN → ReLU → MaxPool 2×2     (224 → 112)
    Conv 5×5, 64 filters  → BN → ReLU → MaxPool 2×2     (112 → 56)
    Conv 3×3, 128 filters → BN → ReLU → MaxPool 2×2     (56  → 28)
    Conv 3×3, 256 filters → BN → ReLU → AdaptiveAvgPool → 256-d feature
    │
    ├── Per-frame feature → mean over 16 frames → 256-d clip feature
    │
    ▼
Linear(256 → 2) — binary logits
```

### 2.2 Why a custom CNN, not ResNet-18 / EfficientNet?

The user-facing concern is that the existing ensemble already contains ResNet-18 and EfficientNet-B4 backbones, so reusing one of them risks correlated feature learning. While the **input modality** (DCT magnitudes vs RGB pixels) is the dominant decorrelation lever, architectural diversity adds a second axis. A small purpose-built CNN:

- Gives architectural decorrelation (4-conv block stack vs ResNet residual blocks vs EfficientNet MBConv blocks vs ViT patch attention vs R3D 3D convs)
- Avoids waste — pretrained weights from ImageNet have no semantic meaning in DCT-magnitude space and would be discarded anyway
- Is small (~500K params vs ResNet-18's 11M), so trains in 5-7 hours on Colab Pro instead of 8-10 hours
- Mirrors the architecture style of F3-Net's frequency branch, anchoring the choice in deepfake detection literature

### 2.3 DCT implementation

The DCT is computed inside `forward()`, on-the-fly per batch — no disk re-extraction is needed. Implementation uses PyTorch's `torch.fft.fft2` to compute the 2D Fast Fourier Transform per 8×8 block, then takes the magnitude. (A true DCT-II is preferred but the difference for the magnitude spectrum is negligible for this application; if profiling shows the FFT-based approximation hurts, we can swap to a torch-native DCT-II implementation in ~50 lines.)

For the spec, the DCT module is treated as a black box that takes `(B, T, 3, H, W)` RGB and returns `(B, T, 3, H, W)` magnitude maps in frequency domain. The CNN consumes these magnitudes as if they were images.

## 3. Training Pipeline

### 3.1 Data and split

Identical to the existing four advanced models:

- **Source:** FaceForensics++ C23 video corpus
- **Frames:** 16 MTCNN-cropped frames @ 224×224 per video, already extracted to `MTCNN_FRAMES_ROOT` on Drive
- **Split:** identity-component-based 70/15/15 (`train_binary.csv` / `val_binary.csv` / `test_binary.csv`)
- **Augmentation:** the same `train_tfm_224` used for ResNet-18 / EfficientNet-B4 / ViT (RandomHorizontalFlip + ColorJitter + Normalize). The DCT happens *after* RGB augmentation, so augmentation diversity carries through to frequency space.

### 3.2 Recipe — single-stage A2

Two-stage (head warmup → full fine-tune) is appropriate when there is a pretrained backbone to freeze. With random initialisation, single-stage end-to-end training is cleaner.

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimizer | AdamW | Same as ResNet-18 baseline |
| Initial LR | 1e-3 | Higher than the 1e-4 fine-tune LR — fresh-init network needs aggressive early steps |
| Weight decay | 1e-4 | Standard |
| Batch size | 16 | Small model, fits comfortably; matches ResNet-18 baseline |
| Epochs | 20 | More than two-stage fine-tune (10) since starting from scratch |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 2 on val_loss) | Same as A2 baseline recipe |
| Loss | Class-weighted CE | `w_c = N_train / (2 · N_c)` per existing convention |
| Mixed precision | Yes (CUDA `torch.amp`) | Speeds up on T4/L4 |

Best checkpoint selection: highest `val_f1` (matches `train_single_stage`'s default).

### 3.3 Training time and compute

Estimated **5-7 hours** on Colab Pro T4/L4. Smaller backbone than EfficientNet-B4 (which took 148 min) but more epochs (20 vs 13), so net wall-clock is similar. We log to `experiments/results.csv` via the existing `append_run_to_csv` upsert.

## 4. Integration with the Existing Ensemble

### 4.1 Code surface changes

**`src/models.py`**
- Add class `FrequencyDeepfakeDetector(DeepfakeClassifier)`
- Add `_DCTBlockwise` helper module (or simple function) for the frequency transform

**`configs/experiments.py`**
- Add `FREQUENCY_CNN_CONFIG` dict
- Append `"freq_cnn": FREQUENCY_CNN_CONFIG` to `ALL_CONFIGS`

**`notebooks/05_model_advanced.ipynb`**
- Add a markdown header `## Frequency-Domain Detector`
- Add a training cell that calls `train_single_stage` (not `train_two_stage`) — this is the only model that uses the single-stage path among the advanced ones
- Add a brief analysis cell

**`notebooks/06_evaluation.ipynb`**
- Add `freq_cnn` to the `MODELS` registry dict (key, class, transform, frames_root)
- The ensemble code already iterates over `available = [n for n in MODELS if n in results]` and averages probabilities — the soft vote trivially extends to 6 models with no other change
- The `MODEL_LABELS` and `ANOMALY_CLASS` lookups in the result panel get one new entry — `freq_cnn → "Frequency CNN"` / `frequency-domain`

**`backend/app/models.py` (web app, optional)**
- If we decide to surface freq_cnn in the live demo, add the class + registry entry. **For this spec, the demo is unchanged** — the new model is paper-only. Web-app integration is deferred.

### 4.2 Threshold

Decision threshold τ stays at **0.10** — the value already val-tuned for the 5-model ensemble in `2026-04-25-...`. The freq detector enters the soft-vote with equal weight 1/6, so the ensemble probability distribution shifts only slightly. We re-validate on FF++ val to confirm the bimodal margin is preserved; if the freq detector materially shifts the distribution we re-tune τ on val and document the new value.

## 5. Evaluation Plan

After training completes, run evaluations in this order. **Do not invoke any inferences against `id21_id19_0009`-style demo videos for hyperparameter selection** — that would constitute leakage from cross-dataset onto our val-tuning loop.

1. **Per-model validation** — confirm the freq detector trained successfully:
   - Target: FF++ test AUC > 0.95 (ensemble members all sit at 0.999+, but a freq-only model is allowed to be lower since we rely on decorrelation)
   - Target: Celeb-DF v2 zero-shot AUC > 0.85
   - **Abort criterion:** if FF++ test AUC < 0.85, the model didn't learn — debug the training before proceeding to ensemble work

2. **6-model ensemble on FF++ test** — should remain near 1.000 (saturated)

3. **6-model ensemble on Celeb-DF v2 (zero-shot)** — **the headline number**
   - Target: 0.93+ (current 5-model: 0.9056)
   - Stretch: 0.95
   - Documented either way; null result is informative

4. **New generalization gap**: Δ AUC = FF++ test AUC − Celeb-DF AUC
   - Current 5-model gap: 0.0944
   - Target: < 0.07

5. **Per-video flip count**: of the six previously-tested Celeb-synthesis cases (id21_id19_0009, id42_id40_0001, id3_id9_0003, id0_id1_0007, id53_id57_0005, id8_id3_0003), count how many cross threshold 0.10 with the freq detector added. **This is the user-facing demo improvement**.

## 6. Paper Updates

### 6.1 Tables

**Table I (FF++ test)** — add two rows:
- `FrequencyCNN` — single-row metrics
- `Ensemble (6-model soft-vote)` — replaces the existing 5-model ensemble row

**Table II (cross-dataset Celeb-DF v2 zero-shot)** — add same two rows.

### 6.2 Methodology section

New ~150-word subsection in §3:

> **3.X Frequency-domain detector.** We add a sixth detector that operates on block-wise 2D-DCT magnitude maps rather than RGB pixels, motivated by literature showing that GAN-generated content leaves spectral fingerprints that survive video compression more reliably than spatial-domain artifacts (Qian et al. 2020; Tan et al. 2024). Architecturally, a small four-convolution CNN (~500K parameters, randomly initialised) consumes the DCT magnitudes per frame and pools features over the 16-frame clip, mirroring the per-frame mean-pool pattern used by our spatial detectors. We train end-to-end for 20 epochs with the single-stage A2 recipe (AdamW, ReduceLROnPlateau, class-weighted cross-entropy) on the same FaceForensics++ C23 split as the other detectors.

### 6.3 Results / Discussion

- New paragraph in §4 reporting the individual freq detector numbers and the 6-model ensemble numbers
- New paragraph in §5 on **architectural decorrelation** — interpret the ensemble lift (or null result) through the lens of error decorrelation: "the frequency detector's failures concentrate on different videos than the spatial detectors', which is what allows the soft-vote average to recover correct verdicts that any single detector misses."

## 7. Risks and Aborts

| Risk | Likelihood | Mitigation / Abort |
|---|---|---|
| Random init doesn't converge — val_acc plateaus low | Medium | Monitor first 3 epochs; abort if val_acc < 0.55 after epoch 3 |
| Individual model AUC < 0.7 on Celeb-DF | Medium-low | Document as null result; exclude from final ensemble; paper section pivots to "frequency features alone are insufficient" |
| Decorrelation hypothesis fails — errors correlated | Low-medium | Ensemble lift smaller than +3-5%; still publishable as ablation |
| FFT-based DCT approximation distorts the spectral signature | Low | Swap to torch-native DCT-II if results suggest the approximation is hurting (~50 lines of code) |
| DCT computation slows training significantly | Low-medium | Profile first; pre-extract DCT to disk if it bottlenecks |

## 8. Success Criteria

The work ships when **all** of the following are true:

1. `FrequencyDeepfakeDetector` trains to convergence (val AUC > 0.85)
2. 6-model ensemble Celeb-DF AUC > 0.92 OR null result is cleanly documented
3. `experiments/results.csv` has rows for both `freq_cnn_<run_id>` and `ensemble_6_<timestamp>`
4. Paper Tables I-II include the new rows
5. Paper §3 has the new subsection
6. Paper §5 has the discussion paragraph
7. Web-app behaviour at `https://deepfakedetection.xyz` is unchanged (deferred for a follow-up phase)

## 9. Out of Scope (for this spec)

- Live demo integration of the freq detector — deferred to a follow-up
- Compression-augmented retraining of the existing 5 models — Option B from earlier brainstorm, deferred
- Multi-task auxiliary heads — deferred
- Stacking ensemble — deferred
- Test-time augmentation — deferred
- F3-Net's frequency-aware attention modules — we use a simpler 4-conv CNN for this iteration

These are all documented in `docs/design/specs/...` for future work; this spec is about the single addition.

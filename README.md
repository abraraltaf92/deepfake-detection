# Robust Deepfake Detection Using Learning-Based Computer Vision Methods

Binary deepfake detection with a focus on building robust models that **generalize across multiple datasets**. The goal is not just high accuracy on FaceForensics++, but strong cross-dataset performance on unseen data sources such as CelebDF, DFDC, and others — ensuring real-world applicability for content moderation.

**Course:** CS 668 — Analytics Capstone
**Instructor:** Dr. Krishna Bathula

---

## Team Members

| Member | Focus Area |
|--------|------------|
| Abrar Altaf Lone | TBD |
| Sajan Singh Shergill | TBD |
| Muhammad Abdullah Zahid | TBD |
| Krishna Kirit Maniyar | TBD |

---

## Current Phase

**Phase 4 complete: Production deployment of v2 ensemble**

Five complementary detectors targeting three anomaly classes (spatial, global-consistency, temporal-motion), aggregated via equal-weight soft-vote with a calibrated threshold (τ = 0.15). All models trained on FaceForensics++ C23 with a two-stage transfer-learning recipe (10 frozen-backbone warmup epochs + 10 end-to-end fine-tune epochs), zero-shot evaluated on Celeb-DF v2. Weights are published on Hugging Face Hub (`abraraltaf92/deepfake-detection-models`, tag `v2-2026-05-07`) and served live at [api.deepfakedetection.xyz](https://api.deepfakedetection.xyz) via a containerised FastAPI backend.

---

## Progress Log

| Date | Milestone | Status |
|------|-----------|--------|
| 02-15-2026 | Project setup and dataset acquisition | ✅ Done |
| 02-23-2026 | EDA and data analysis complete | ✅ Done |
| 03-05-2026 | Preprocessing pipeline (frame extraction, face detection, sampling) | ✅ Done |
| 03-15-2026 | Baseline ResNet-18 binary classifier trained | ✅ Done |
| 03-23-2026 | GitHub repo setup with version control workflow | ✅ Done |
| 04-21-2026 | ResNet-18 baseline retrained on full MTCNN frames (A2 single-stage recipe) | ✅ Done |
| 04-22-2026 | EfficientNet-B4, R3D-18, ViT-B/16 trained on FF++ C23 | ✅ Done |
| 04-23-2026 | RAFT optical-flow extraction + R3D-18+RAFT trained | ✅ Done |
| 04-24-2026 | Cross-dataset Celeb-DF v2 zero-shot eval + 5-model ensemble + Grad-CAM | ✅ Done |
| 04-28-2026 | Frequency-domain detector experiment (block-wise 2D-DCT CNN) — negative result | ✅ Done |
| 05-05-2026 | v2 ensemble retrained — full two-stage recipe across all 5 architectures | ✅ Done |
| 05-07-2026 | v2 weights published to Hugging Face Hub + deployed to AWS production backend | ✅ Done |

---

## Model Performance

### In-Dataset (FaceForensics++ C23 test, 900 videos)

| Model | Anomaly Class | Accuracy | F1 Score | AUC | Train Time |
|-------|---------------|---------:|---------:|----:|-----------:|
| ResNet-18 (baseline) | Spatial | 0.9878 | 0.9926 | 0.9999 | 79.3 min |
| EfficientNet-B4 | Spatial (high-cap) | **1.0000** | **1.0000** | **1.0000** | 106.7 min |
| R3D-18 | Temporal-motion | 0.9889 | 0.9933 | 0.9995 | 174.6 min |
| ViT-B/16 | Global consistency | 0.9889 | 0.9933 | 0.9998 | 196.4 min |
| R3D-18 + RAFT | Motion-flow | 0.9878 | 0.9926 | 0.9996 | 164.1 min |
| **Ensemble (5-model soft-vote, τ=0.15)** | All three | **1.0000** | **1.0000** | **1.0000** | n/a |

All 5 architectures saturate FF++ test (AUC ≥ 0.999). In-dataset metrics do not differentiate models — see cross-dataset table below for the real story.

### Cross-Dataset Generalization (Zero-Shot Celeb-DF v2, 6,528 videos)

| Model | FF++ AUC | Celeb-DF AUC | Δ Generalization Gap |
|-------|---------:|-------------:|---------------------:|
| EfficientNet-B4 | 1.0000 | 0.8109 | 0.1891 |
| R3D-18 | 0.9995 | 0.8383 | 0.1612 |
| ResNet-18 | 0.9999 | 0.8427 | 0.1572 |
| ViT-B/16 | 0.9998 | 0.8475 | 0.1523 |
| R3D-18 + RAFT | 0.9996 | 0.8775 | 0.1221 |
| **Ensemble (5-model, τ=0.15)** | **1.0000** | **0.8851** | **0.1149** |

**Key findings:**

- **EfficientNet-B4 ranks first in-dataset, last cross-dataset** — high capacity overfits FF++ compression artifacts.
- **Anomaly class predicts transferability:** spatial-only models drop hardest; motion-flow (R3D+RAFT) is the strongest individual cross-dataset detector and global-consistency (ViT) is competitive.
- **The 5-model ensemble shrinks the generalization gap by 26.9%** over the ResNet-18 baseline (0.1572 → 0.1149), validating the multi-view anomaly-detection framing. The ensemble also improves on the strongest single model (R3D-18+RAFT, gap 0.1221) by another 5.9%.
- **Grad-CAM** on EfficientNet-B4's final convolutional stage confirms attention concentrates on manipulation-prone facial regions (eye band, nose-cheek transitions, lip corners) rather than spurious background cues.

See `06_evaluation.ipynb` and `experiments/results.csv` for full reproducible numbers.

### Negative Result — Frequency-Domain Detector

We tested whether a frequency-domain detector operating on block-wise 2D-DCT magnitudes would transfer better across datasets than spatial detectors. Hypothesis: GAN/VAE forgeries leave characteristic high-frequency footprints that should be invariant to dataset-specific RGB statistics.

**Result: hypothesis rejected.**

The standalone Freq-CNN was retained from the v1 evaluation (no v2 retrain — its negative result already validated the design decision to exclude it). Standalone metrics under v2 evaluation:

| Configuration | FF++ AUC | Celeb-DF AUC | Δ Gap |
|---|---:|---:|---:|
| FrequencyDeepfakeDetector (standalone, v2 eval) | 0.9817 | 0.6913 | **0.2905** |
| 5-model ensemble (without freq_cnn, v2) | 1.0000 | 0.8851 | 0.1149 |

The randomly-initialized 480K-parameter CNN has the **largest** generalization gap of any model in the study (0.2905 vs the ensemble's 0.1149). The original v1 6-model ensemble (with freq_cnn included) regressed Celeb-DF AUC from 0.9056 → 0.8935 — the same conclusion holds under v2 numbers and the model is intentionally excluded from the production ensemble.

**Mechanism analysis.** Every other model in the ensemble carried a transferable inductive bias from pretraining (ImageNet for 2D, Kinetics-400 for 3D). The frequency-domain detector started from random initialization and had to learn from FF++ alone. What it learned was the **codec signature** (H.264 CRF=23 8×8 DCT block-quantization patterns) rather than the **forgery signature** (generation-process artifacts). When the codec changed (Celeb-DF uses different encoders), the model collapsed.

**Implication for future work.** Frequency-domain methods for cross-dataset deepfake detection require either (a) pretrained backbones operating on frequency representations, or (b) explicit codec-invariance regularization during training. Random-init small CNNs on DCT inputs are insufficient.

See `docs/design/specs/2026-04-27-frequency-domain-detector-design.md` for the full hypothesis and experimental design, and `notebooks/05_model_advanced.ipynb` (frequency CNN cell) + `notebooks/06_evaluation.ipynb` for the data.

---

## Production Deployment

The trained ensemble runs as a live demo at **[deepfakedetection.xyz](https://deepfakedetection.xyz)**.

| Component | Stack | Notes |
|---|---|---|
| Frontend | Next.js 14 + Tailwind, deployed on Vercel | Repo: [deepfake-detection-app](https://github.com/abraraltaf92/deepfake-detection-app) |
| Backend | FastAPI + PyTorch, Docker on AWS EC2 (CUDA) | `api.deepfakedetection.xyz` |
| Model weights | Hugging Face Hub | [`abraraltaf92/deepfake-detection-models`](https://huggingface.co/abraraltaf92/deepfake-detection-models), tag `v2-2026-05-07` |
| Inference | All 5 models loaded once at FastAPI startup, shared across requests |
| Threshold | τ = 0.15 (calibrated on Celeb-DF v2 zero-shot validation) |

A user uploads a face video; the backend extracts 16 MTCNN-cropped frames + 16 RAFT optical-flow-interpolated frames, runs all 5 models, soft-vote averages the per-frame fake-class probabilities, applies τ = 0.15, and returns a verdict + per-model breakdown + a Grad-CAM heatmap.

---

## Quick Start

1. **Clone this repo** to your Google Drive:
   ```bash
   # In Google Colab:
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/<your-username>/deepfake-detection.git /content/drive/MyDrive/deepfake-detection
   ```

2. **Download datasets** and place them in `data/raw/` inside the repo folder:
   ```
   deepfake-detection/
   └── data/
       └── raw/
           └── ff-c23/FaceForensics++_C23/   ← put FF++ dataset here
   ```
   - [FaceForensics++ C23](https://github.com/ondyari/FaceForensics)
   - [CelebDF v2](https://github.com/yuezunli/celeb-deepfakeforensics)

3. **Update `REPO_ROOT`** in `configs/paths.py` if your Drive path differs (default: `/content/drive/MyDrive/deepfake-detection`)

4. **Open notebooks in Google Colab** in numbered order:
   ```
   01_data_ingestion.ipynb → 02_eda.ipynb → 03_preprocessing.ipynb →
   04_model_baseline.ipynb → 05_model_advanced.ipynb → 06_evaluation.ipynb
   ```

5. **See [CONTRIBUTING.md](CONTRIBUTING.md)** for team workflow and how to commit changes

---

## Key Resources

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [CelebDF v2 Dataset](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC Dataset](https://ai.meta.com/datasets/dfdc/)
- [Project Presentation (Midterm)](docs/references.md)
- [Research Papers & References](docs/references.md)

---

## Repository Structure

```
deepfake-detection/
├── README.md                          # Project dashboard (this file)
├── CONTRIBUTING.md                    # Team workflow guide
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excluded files (data, models, checkpoints)
├── configs/
│   ├── paths.py                       # Centralized path configuration
│   └── experiments.py                 # Per-model hyperparameter configs
├── src/                               # Reusable training + eval library
│   ├── datasets.py                    # DeepfakeBinaryDataset + dataloader builders
│   ├── models.py                      # 5 model classes (ResNet-18, EfficientNet-B4, R3D-18, ViT-B/16, R3D-18+RAFT)
│   ├── preprocessing.py               # MTCNN face extraction + RAFT optical-flow interpolation
│   ├── training.py                    # Single-stage and two-stage trainers, checkpoint I/O
│   ├── evaluation.py                  # In/cross-dataset eval + per-manipulation breakdown
│   ├── visualization.py               # Training curves, confusion matrices, ROC, Grad-CAM
│   └── logging.py                     # Run-id generation + experiments/results.csv schema
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Dataset discovery, indexing, identity-based splitting
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_preprocessing.ipynb         # MTCNN face extraction (FF++ + Celeb-DF) + sanity viz
│   ├── 04_model_baseline.ipynb        # ResNet-18 baseline (single-stage A2 recipe)
│   ├── 05_model_advanced.ipynb        # 4 advanced architectures (two-stage warmup → fine-tune)
│   └── 06_evaluation.ipynb            # Cross-dataset Celeb-DF eval + 5-model ensemble + Grad-CAM
├── experiments/
│   ├── results.csv                    # Single-source-of-truth leaderboard (28 cols, all runs)
│   └── results/                       # Per-run JSON with full provenance
├── tests/                             # pytest suite for src/ modules
└── docs/
    ├── design/                        # Architecture diagrams + paper sections
    └── references.md                  # Research papers and resources
```

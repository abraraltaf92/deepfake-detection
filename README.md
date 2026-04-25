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

**Phase 3 complete: Multi-View Anomaly Detection Ensemble**

Five complementary detectors targeting three anomaly classes (spatial, global-consistency, temporal-motion), aggregated via equal-weight soft-vote. All models trained on FaceForensics++ C23, zero-shot evaluated on Celeb-DF v2.

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

---

## Model Performance

### In-Dataset (FaceForensics++ C23 test, 900 videos)

| Model | Anomaly Class | Accuracy | F1 Score | AUC | Train Time |
|-------|---------------|---------:|---------:|----:|-----------:|
| ResNet-18 (baseline) | Spatial | 0.9989 | 0.9993 | 0.9999 | — |
| EfficientNet-B4 | Spatial (high-cap) | 0.9944 | 0.9967 | **1.0000** | 148.8 min |
| R3D-18 | Temporal-motion | 0.9756 | 0.9852 | 0.9991 | 35.9 min |
| ViT-B/16 | Global consistency | 0.9700 | 0.9817 | 0.9992 | 72.1 min |
| R3D-18 + RAFT | Motion-flow | 0.9833 | 0.9899 | 0.9993 | 130.1 min |
| **Ensemble (5-model soft-vote)** | All three | **0.9989** | **0.9993** | **1.0000** | n/a |

All 5 architectures saturate FF++ test (AUC ≥ 0.999). In-dataset metrics do not differentiate models — see cross-dataset table below for the real story.

### Cross-Dataset Generalization (Zero-Shot Celeb-DF v2, 6,528 videos)

| Model | FF++ AUC | Celeb-DF AUC | Δ Generalization Gap |
|-------|---------:|-------------:|---------------------:|
| EfficientNet-B4 | 1.0000 | 0.8173 | 0.1827 |
| ResNet-18 | 0.9999 | 0.8209 | 0.1790 |
| R3D-18 | 0.9991 | 0.8413 | 0.1577 |
| R3D-18 + RAFT | 0.9993 | 0.8744 | 0.1249 |
| ViT-B/16 | 0.9992 | 0.8777 | 0.1216 |
| **Ensemble (5-model)** | **1.0000** | **0.9056** | **0.0944** |

**Key findings:**

- **EfficientNet-B4 ranks first in-dataset, last cross-dataset** — high capacity overfits FF++ compression artifacts.
- **Anomaly class predicts transferability:** spatial-only models drop hardest; motion-flow (R3D+RAFT) and global-consistency (ViT) transfer best.
- **The 5-model ensemble shrinks the generalization gap by 22%** (0.1216 → 0.0944) over the strongest single model, validating the multi-view anomaly-detection framing.
- **Grad-CAM** on EfficientNet-B4's final convolutional stage confirms attention concentrates on manipulation-prone facial regions (eye band, nose-cheek transitions, lip corners) rather than spurious background cues.

See `06_evaluation.ipynb` and `experiments/results.csv` for full reproducible numbers.

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

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

**Phase 2: Advanced Model Architectures**

- Exploring 3D CNN + LSTM for temporal feature learning
- Investigating attention mechanisms and Vision Transformers (ViT)
- Target: improve cross-dataset generalization beyond baseline ResNet-18

---

## Weekly Goals

- [ ] Implement 3D CNN with temporal pooling
- [ ] Research ViT-based approaches for frame-level deepfake detection
- [ ] Begin cross-dataset evaluation pipeline (train on FF++, test on CelebDF)
- [ ] Document model experiment results

---

## Progress Log

| Date | Milestone | Status |
|------|-----------|--------|
| 02-15-2026 | Project setup and dataset acquisition | ✅ Done |
| 02-23-2026 | EDA and data analysis complete | ✅ Done |
| 03-05-2026 | Preprocessing pipeline (frame extraction, face detection, sampling) | ✅ Done |
| 03-15-2026 | Baseline ResNet-18 binary classifier trained | ✅ Done |
| 03-23-2026 | GitHub repo setup with version control workflow | ✅ Done |
| XX-XX-2026 | Advanced model exploration (3D CNN, LSTM, ViT) | 🔄 In Progress |
| XX-XX-2026 | Cross-dataset generalization evaluation | ⬚ Upcoming |

---

## Model Performance

### In-Dataset (FaceForensics++ C23)

| Model | Accuracy | F1 Score | Val Accuracy | Epochs | Notes |
|-------|----------|----------|--------------|--------|-------|
| ResNet-18 (baseline) | 86% | 0.9333 | 89.67% | 15 | Class-weighted loss, 16 frames/video |
| 3D CNN + LSTM | — | — | — | — | In progress |
| ViT-based | — | — | — | — | Planned |

### Cross-Dataset Generalization

| Model | Trained On | Tested On | Accuracy | F1 Score | Notes |
|-------|------------|-----------|----------|----------|-------|
| ResNet-18 (baseline) | FF++ C23 | CelebDF | — | — | Pending |
| ResNet-18 (baseline) | FF++ C23 | DFDC | — | — | Pending |

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
│   └── paths.py                       # Centralized path configuration
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # Dataset discovery, indexing, splitting
│   ├── 02_eda.ipynb                   # Exploratory data analysis
│   ├── 03_preprocessing.ipynb         # Frame extraction, face detection, tensor prep
│   ├── 04_model_baseline.ipynb        # ResNet-18 baseline model
│   ├── 05_model_advanced.ipynb        # Advanced architectures (3D CNN, LSTM, ViT)
│   └── 06_evaluation.ipynb            # Cross-dataset evaluation & comparison
└── docs/
    └── references.md                  # Research papers and resources
```

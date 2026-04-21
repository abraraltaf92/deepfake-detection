# Project Architecture

End-to-end pipeline for binary deepfake detection: raw videos → preprocessing → five independently-trained models → ensemble → cross-dataset evaluation → explainability.

```mermaid
flowchart TB
    %% ============ DATASETS ============
    dataset(["Datasets<br/>FaceForensics++ C23 · 6,000 videos<br/>Celeb-DF v2 · 6,529 videos"])

    %% ============ PREPROCESSING ============
    subgraph prep ["Preprocessing · src/preprocessing.py"]
        direction TB
        split["Uniform frame sampling<br/>16 frames / video"]
        mtcnn["Face detection: MTCNN<br/>(facenet-pytorch)"]
        crop["Face crop · 224 × 224 px<br/>Center-crop fallback if no face"]
        split --> mtcnn --> crop
        aug["Train-time augmentation<br/>HorizontalFlip · ColorJitter · Normalize<br/>(baseline: Normalize only)"]
        crop --> aug
    end

    %% ============ MODELS ============
    subgraph train ["Model Training · 5 independent models"]
        direction TB
        subgraph baseline_block ["Baseline"]
            resnet["<b>ResNet-18</b><br/>Single-stage · AdamW + ReduceLROnPlateau<br/>15 epochs · shuffle · no aug<br/>Frame features → mean-pool → head"]
        end
        subgraph advanced_block ["Advanced (two-stage: head warmup → full fine-tune)"]
            direction LR
            eff["<b>EfficientNet-B4</b><br/>16 frames · 224 × 224<br/>Frame features<br/>Mean-pool"]
            r3d["<b>R3D-18</b><br/>16 frames · 112 × 112<br/>3D convolutions<br/>Clip features"]
            vit["<b>ViT-B/16</b><br/>16 frames · 224 × 224<br/>Patch attention<br/>Mean-pool"]
            raft["<b>R3D-18 + RAFT</b><br/>16 RAFT-interpolated frames<br/>Motion features<br/>Optical flow"]
        end
    end

    %% ============ ENSEMBLE ============
    subgraph ensemble ["Ensemble · 06_evaluation.ipynb"]
        softvote["Soft-vote<br/>Equal-weight mean of per-video fake probabilities"]
        threshold["Threshold = 0.5 (fixed)"]
        softvote --> threshold
    end

    %% ============ EVALUATION ============
    subgraph eval ["Evaluation"]
        direction TB
        in_ds["<b>In-dataset</b> · FF++ test<br/>Accuracy · Precision · Recall · F1 · AUC<br/>Per-manipulation breakdown<br/>(Deepfakes · Face2Face · FaceSwap · NeuralTextures · FaceShifter)<br/>Confusion matrix · Training curves"]
        cross_ds["<b>Cross-dataset</b> · Celeb-DF v2<br/>Zero-shot evaluation<br/>Generalization gap = FF++ AUC − Celeb-DF AUC<br/>ROC overlay across all models"]
        explain["<b>Explainability</b> · Grad-CAM<br/>EfficientNet-B4 final conv stage<br/>Heat-map overlays on sample FF++ frames"]
    end

    %% ============ OUTPUT ============
    decide(["Real / Fake prediction<br/>Logged to experiments/results.csv<br/>+ per-run JSON in experiments/results/"])

    %% ============ FLOW ============
    dataset --> prep
    prep --> resnet
    prep --> eff
    prep --> r3d
    prep --> vit
    prep --> raft
    resnet --> ensemble
    eff --> ensemble
    r3d --> ensemble
    vit --> ensemble
    raft --> ensemble
    ensemble --> eval
    eval --> decide

    %% ============ STYLING ============
    classDef dataNode fill:#fef3c7,stroke:#b45309,stroke-width:2px
    classDef modelNode fill:#dbeafe,stroke:#1e40af,stroke-width:1.5px
    classDef evalNode fill:#dcfce7,stroke:#15803d,stroke-width:1.5px
    class dataset,decide dataNode
    class resnet,eff,r3d,vit,raft modelNode
    class in_ds,cross_ds,explain evalNode
```

## Notes

- **Experiment tracking**: every training and evaluation run appends a row to `experiments/results.csv` and writes a per-run JSON payload to `experiments/results/<run_id>.json`. The CSV is the leaderboard; the JSON is the full provenance record (config, training history, per-manipulation metrics, checkpoint path).

- **Device priority**: all model and training code uses `src.training.pick_device()` which resolves `cuda → mps → cpu`. The same notebooks run on Google Colab Pro (CUDA) and on local Apple Silicon (MPS) without changes.

- **R3D-18 + RAFT**: the interpolation step requires CUDA (MPS is unsupported for `torchvision.models.optical_flow.raft`). Known issues in the current interpolation math are documented in the `extract_face_frames_interpolated` docstring in `src/preprocessing.py`.

- **Baseline vs advanced training recipes differ deliberately**. ResNet-18 uses single-stage AdamW + `ReduceLROnPlateau` with no augmentation — this matches the historical recipe that produced the repo's reference numbers. The four advanced models use two-stage (head warmup → full fine-tune) with `WeightedRandomSampler` and train-time augmentation. Evaluation is identical across all five models, so leaderboard comparisons are apples-to-apples at eval time even though training dynamics differ.

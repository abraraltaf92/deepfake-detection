# Solution Architecture

End-to-end workflow from raw video to real/fake prediction. This is the "how the system is wired together" view — see `model_architectures.md` for layer-level detail and equations.

```mermaid
flowchart TB
    %% ==== dataset ====
    D(["Datasets<br/>FF++ C23 · 6 000<br/>Celeb-DF v2 · 6 529"])

    %% ==== preprocessing ====
    subgraph P ["Preprocessing"]
        direction TB
        P1["16 uniformly-sampled frames per video"]
        P2["MTCNN face detection"]
        P3["Crop 224 × 224<br/>center-crop fallback"]
        P4["Augmentation<br/>HFlip · ColorJitter · Normalize"]
        P1 --> P2 --> P3 --> P4
    end

    %% ==== models (parallel, independent) ====
    M1(["ResNet-18<br/>single-stage<br/>baseline"])
    M2(["EfficientNet-B4<br/>two-stage"])
    M3(["R3D-18<br/>two-stage"])
    M4(["ViT-B/16<br/>two-stage"])
    M5(["R3D-18 + RAFT<br/>two-stage"])

    %% ==== ensemble ====
    subgraph E ["Ensemble"]
        direction TB
        E1["Soft-vote: mean of 5 fake-probabilities"]
        E2["Threshold = 0.5"]
        E1 --> E2
    end

    %% ==== evaluation ====
    subgraph V ["Evaluation"]
        direction TB
        V1["In-dataset · FF++ test<br/>Acc · Precision · Recall · F1 · AUC<br/>per-manipulation breakdown"]
        V2["Cross-dataset · Celeb-DF v2<br/>zero-shot<br/>generalization gap"]
        V3["Grad-CAM · EfficientNet-B4<br/>explainability"]
    end

    %% ==== output ====
    O(["Real / Fake prediction<br/>→ experiments/results.csv<br/>→ experiments/results/&lt;run_id&gt;.json"])

    %% ==== flow ====
    D --> P
    P --> M1
    P --> M2
    P --> M3
    P --> M4
    P --> M5
    M1 --> E
    M2 --> E
    M3 --> E
    M4 --> E
    M5 --> E
    E --> V
    V --> O

    %% ==== styling ====
    classDef dataNode fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#1a1a1a
    classDef modelNode fill:#dbeafe,stroke:#1e40af,stroke-width:1.5px,color:#1a1a1a
    classDef evalNode fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#1a1a1a
    class D,O dataNode
    class M1,M2,M3,M4,M5 modelNode
    class V1,V2,V3 evalNode
```

## Key notes

- **Parallel, not sequential.** Each model trains and predicts independently on the same preprocessed data. No model ever sees another model's weights or intermediate features. The ensemble combines **predictions only**.
- **Baseline vs advanced** have deliberately different training recipes; evaluation is identical across all five so the leaderboard comparison is apples-to-apples.
- **Device priority** is `cuda → mps → cpu` via `src.training.pick_device()`, so the same notebooks run unchanged on Colab (A100) and local Apple Silicon (MPS).
- **Experiment tracking**: every run writes a row to `experiments/results.csv` (leaderboard) and a JSON file to `experiments/results/` (full provenance).

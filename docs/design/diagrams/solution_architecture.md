# Solution Architecture

End-to-end workflow from raw video to real/fake prediction, framed as multi-view anomaly detection. Each model targets a different class of deepfake anomaly; the ensemble aggregates their predictions into a single soft-vote score.

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

    %% ==== models as anomaly detectors ====
    M0["ANOMALY DETECTORS · 5 complementary views"]
    M1(["ResNet-18<br/>baseline<br/>spatial anomalies<br/>(texture · edges)"])
    M2(["EfficientNet-B4<br/>spatial anomalies<br/>(high-capacity SE blocks)"])
    M3(["R3D-18<br/>temporal-motion<br/>anomalies (3D conv)"])
    M4(["ViT-B/16<br/>global consistency<br/>anomalies (attention)"])
    M5(["R3D-18 + RAFT<br/>motion-flow<br/>anomalies (RAFT interp)"])

    %% ==== ensemble ====
    subgraph E ["Multi-View Ensemble"]
        direction TB
        E1["Soft-vote: mean of 5 anomaly scores"]
        E2["Threshold = 0.5"]
        E1 --> E2
    end

    %% ==== evaluation ====
    subgraph V ["Evaluation"]
        direction TB
        V1["In-dataset · FF++ test<br/>Acc · Precision · Recall · F1 · AUC<br/>per-manipulation anomaly breakdown"]
        V2["Cross-dataset · Celeb-DF v2<br/>zero-shot<br/>generalization gap"]
        V3["Grad-CAM · anomaly localisation<br/>EfficientNet-B4"]
    end

    %% ==== output ====
    O(["Real / Fake prediction<br/>→ experiments/results.csv<br/>→ experiments/results/&lt;run_id&gt;.json"])

    %% ==== flow ====
    D --> P
    P --> M0
    M0 --> M1
    M0 --> M2
    M0 --> M3
    M0 --> M4
    M0 --> M5
    M1 --> E
    M2 --> E
    M3 --> E
    M4 --> E
    M5 --> E
    E --> V
    V --> O

    %% ==== styling ====
    classDef dataNode fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#1a1a1a
    classDef headerNode fill:#fde2e8,stroke:#be185d,stroke-width:2px,color:#1a1a1a
    classDef modelNode fill:#dbeafe,stroke:#1e40af,stroke-width:1.5px,color:#1a1a1a
    classDef evalNode fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#1a1a1a
    class D,O dataNode
    class M0 headerNode
    class M1,M2,M3,M4,M5 modelNode
    class V1,V2,V3 evalNode
```

## Key notes

- **Anomaly detection framing.** Real faces occupy a learnable manifold; every deepfake manipulation introduces deviations from it. Each of the five models learns a *different class* of anomaly signature. The ensemble aggregates these complementary views.

| Model | Anomaly class targeted | Architectural mechanism |
|---|---|---|
| ResNet-18 | Spatial (local texture, edges) | 4 residual stages with 2D convs |
| EfficientNet-B4 | Spatial (high-capacity) | 7 MBConv stages with squeeze-excitation |
| R3D-18 | Temporal-motion | 3D convolutions over (T, H, W) |
| ViT-B/16 | Global consistency | Self-attention over 14 × 14 patches |
| R3D-18 + RAFT | Motion-flow | 3D convs on RAFT-interpolated frames |

- **Ensemble as multi-view anomaly detector.** Because the five detectors operate on different feature classes, their errors are partially decorrelated. A deepfake that evades per-frame detection by matching natural texture statistics may still be caught by the 3D-conv model because its motion is off, and vice versa.
- **Parallel, not sequential.** Each model trains and predicts independently on the same preprocessed data. No model ever sees another model's weights or intermediate features.
- **Device priority** is `cuda → mps → cpu` via `src.training.pick_device()`. Same notebooks run unchanged on Colab (A100/L4) and local Apple Silicon (MPS).
- **Experiment tracking**: every run writes a row to `experiments/results.csv` (leaderboard) and a JSON file to `experiments/results/` (full provenance).

# 🏗️ System Architecture

## Overview

The system implements a **3-component hybrid architecture** for industrial machine audio anomaly detection, evaluated on the DCASE 2020 Task 2 benchmark.

```
Raw Audio (.wav)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Feature Extraction                           │
│                                                                     │
│  ┌────────────────────────┐     ┌────────────────────────────────┐  │
│  │ AST Embedding (768-dim)│     │ Classical Features (955-dim)   │  │
│  │                        │     │                                │  │
│  │ • MIT pretrained AST   │     │ • Mel-spectrogram stats (640)  │  │
│  │ • CLS token output     │     │ • MFCCs + deltas (100)         │  │
│  │ • Captures semantics   │     │ • Spectral features (150+)     │  │
│  │                        │     │ • Temporal features (ZCR, RMS) │  │
│  └────────────┬───────────┘     └───────────────┬────────────────┘  │
│               │                                 │                   │
│               └──────────────┬──────────────────┘                   │
│                              ▼                                       │
│               Concatenate → 1723-dim Hybrid Features                │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HybridAnomalyDetector                            │
│                                                                     │
│  StandardScaler → [ GMM | OCSVM | XGBoost | Logistic Regression ]  │
│                                                                     │
│  Best methods per machine:                                          │
│  • Pump/Slider/Valve/ToyCar/ToyConveyor: GMM-16 or GMM-8           │
│  • Fan: One-Class SVM                                               │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                      Anomaly Score [0, 1]
                    (higher = more anomalous)
```

---

## Component Descriptions

### 1. Audio Spectrogram Transformer (AST)
- **Model**: `MIT/ast-finetuned-audioset-10-10-0.4593`
- **Output**: 768-dimensional CLS token embedding
- **Strengths**: Captures high-level semantic audio patterns
- **Limitations**: Domain mismatch with industrial machines

### 2. Classical Feature Extractor
- **Output**: 955-dimensional vector
- **Components**:
  - Mel-spectrogram statistics: 128 bands × 5 stats = 640 features
  - MFCCs (n=20) + delta + delta-delta = 100 features
  - Spectral centroid/bandwidth/rolloff = 12 features
  - Spectral contrast (7 bands × 2) = 14 features
  - ZCR + RMS = 8 features
  - Chroma (12 bins × 2) = 24 features
  - Tonnetz (6 × 2) = 12 features
  - Padding to exactly 955 dimensions

### 3. Hybrid Ensemble Detector
- **Input**: 1723-dim concatenated features
- **Preprocessing**: `StandardScaler` normalization
- **Supported Methods**:
  | Method | Type | Notes |
  |--------|------|-------|
  | GMM | Unsupervised | Best overall; n_components=8 or 16 |
  | OCSVM | Unsupervised | Best for Fan |
  | XGBoost | Supervised | Requires synthetic anomalies |
  | Logistic Regression | Supervised | Fast baseline |

---

## Data Flow

```
1. Load .wav files from DCASE directory structure
2. Resample to 16 kHz, trim/pad to 10 seconds
3. Extract AST embeddings (768-dim) via Hugging Face Transformers
4. Extract classical features (955-dim) via librosa
5. Concatenate → 1723-dim feature matrix
6. Fit StandardScaler on training (normal) data
7. Fit anomaly detector on scaled training features
8. At inference: score → negate log-likelihood (GMM) or decision function
9. Evaluate with AUC on labeled test set
```

---

## Model Architecture Details

### GMM Parameters (Best Configuration)
```python
GaussianMixture(
    n_components=16,       # Pump (0.874 AUC)
    covariance_type="full",
    reg_covar=1e-3,        # Numerical stability
    random_state=42,
)
```

### OCSVM Parameters (Best for Fan)
```python
OneClassSVM(
    kernel="rbf",
    nu=0.1,
    gamma="scale",
)
```

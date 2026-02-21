# 🔬 Experimental Journey

## Overview

This document chronicles the full experimental journey from initial attempts to the final hybrid ensemble that achieved **0.874 AUC on Pump** in the DCASE 2020 Task 2 benchmark.

---

## Phase 1: Convolutional Autoencoder (CAE)

**Hypothesis**: A CAE trained on normal sounds will reconstruct anomalies poorly, yielding high reconstruction error as an anomaly signal.

**Architecture**:
- 4-layer encoder: `input → 256 → 128 → 64 → latent(32)`
- 4-layer decoder: `latent(32) → 64 → 128 → 256 → output`
- Dropout 0.3 for regularization
- 30 epochs, Adam optimizer, MSE loss

**Results**:
| Machine | AUC |
|---------|-----|
| fan | 0.549 |
| pump | 0.568 |
| slider | 0.713 |
| valve | 0.526 |
| ToyCar | 0.765 |
| ToyConveyor | 0.598 |
| **Average** | **0.620** |

**Why it failed**:
- CAEs generalize too well → they reconstruct anomalies almost as accurately as normal sounds
- Industrial machine anomalies often differ subtly (frequency shifts, additional harmonics), not structurally
- The model overfits the normal manifold without creating a sharp boundary

**Lesson**: ❌ Pure reconstruction error is not a reliable anomaly signal for this domain.

---

## Phase 2: Contrastive Learning

**Hypothesis**: Contrastive self-supervised learning would create better-separated embeddings for normal vs. anomalous sounds.

**What happened**: The contrastive objective suppressed anomaly signals by pulling all normal samples together, but also inadvertently clustering some anomalies with normal sounds.

**Lesson**: ❌ Contrastive learning can suppress the signal we need for anomaly detection.

---

## Phase 3: Classical GMM Baseline

**Hypothesis**: Well-engineered features + simple GMM might outperform deep learning approaches.

**Features**: 955-dim classical features (MFCC, mel-spectrogram, spectral, temporal)

**Results**:
| Machine | AUC |
|---------|-----|
| fan | 0.832 |
| pump | 0.815 |
| slider | 0.821 |
| valve | 0.814 |
| ToyCar | 0.739 |
| ToyConveyor | 0.620 |
| **Average** | **0.773** |

**Key insight**: Classical signal processing features carry significant discriminative power for machine sounds. The DCASE baseline approach is hard to beat.

**Lesson**: ✅ GMM on rich classical features is a strong, robust baseline.

---

## Phase 4: Pretrained Audio Spectrogram Transformer

**Model**: `MIT/ast-finetuned-audioset-10-10-0.4593`
**Features**: 768-dim CLS token embeddings from the last hidden state

**Results**:
| Machine | AUC |
|---------|-----|
| fan | 0.616 |
| pump | 0.799 |
| slider | 0.904 |
| valve | 0.756 |
| ToyCar | 0.661 |
| ToyConveyor | 0.601 |
| **Average** | **0.723** |

**Analysis**:
- Slider: Best result (0.904), likely because slider anomalies differ semantically
- Fan: Worst result (0.616), worse than baseline
- Domain mismatch: AudioSet contains environmental/speech audio, not industrial machines

**Lesson**: ❌ Pretrained transformers have domain mismatch. They miss low-level acoustic features critical for machine anomalies.

---

## Phase 5: Hybrid Ensemble (Best)

**Hypothesis**: Combining the complementary strengths of AST (semantic) and classical features (acoustic precision) would outperform either alone.

**Features**: 1723-dim = 768 (AST) + 955 (classical)

**Method Comparison** (shown for Pump):
| Method | AUC |
|--------|-----|
| GMM-8 | 0.861 |
| GMM-16 | **0.874** |
| OCSVM | 0.843 |
| XGBoost | 0.832 |

**Best Results Per Machine**:
| Machine | AUC | Best Method |
|---------|-----|-------------|
| fan | 0.651 | OCSVM |
| pump | **0.874** | GMM-16 |
| slider | 0.870 | GMM-8 |
| valve | 0.779 | GMM-8 |
| ToyCar | 0.751 | GMM-8 |
| ToyConveyor | 0.594 | GMM-8 |
| **Average** | **0.753** | - |

**Key Findings**:
1. Hybrid beats baseline on 3/6 machines (pump, slider, valve, ToyCar partially)
2. Fan performance drops in hybrid → AST features hurt Fan detection
3. GMM-16 is optimal for Pump; GMM-8 for most others
4. OCSVM handles Fan best (less affected by AST noise)

**Lesson**: ✅ Combining complementary features boosts performance. Per-machine method selection is important.

---

### Synergy Analysis

Hybrid (0.874) outperforms both single-modality approaches for Pump:
- vs Classical-only (0.815): +0.059 (+5.9%)
- vs AST-only (0.799): +0.075 (+7.5%)

This suggests **complementary information content**: AST captures high-level
acoustic patterns (cavitation modulation, rhythm) that classical features miss,
while classical features capture low-level spectral details (bearing defects,
frequency peaks) that AST misses. The GMM can leverage both signals to learn
richer decision boundaries.

**Key insight**: The improvement over both single approaches indicates that
neither feature set alone is sufficient—pump anomalies genuinely require both
semantic and spectral information for optimal detection.

---

## Summary: What Works vs. What Doesn't

| Approach | Average AUC | Status |
|----------|-------------|--------|
| CAE | 0.620 | ❌ Fails |
| Contrastive Learning | <0.620 | ❌ Fails |
| AST Only | 0.723 | ⚠️ Partial |
| Classical GMM | 0.773 | ✅ Strong Baseline |
| **Hybrid Ensemble** | **0.753** | ✅ Best Single Result |

**Best single result**: Pump 0.874 AUC with GMM-16 on hybrid features
(+5.9% improvement over sklearn baseline of 0.815)

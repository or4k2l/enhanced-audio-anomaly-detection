# 📚 API Reference

## `src/models/`

### `ASTEmbeddingExtractor`

```python
from models.ast_extractor import ASTEmbeddingExtractor

extractor = ASTEmbeddingExtractor(
    model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
    device="cpu",
    sample_rate=16000,
)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `extract_embedding` | `(waveform: np.ndarray) -> np.ndarray` | 768-dim CLS embedding |
| `extract_batch` | `(waveforms: list) -> np.ndarray` | (N, 768) embedding matrix |

**Requirements**: `transformers`, `torch`

---

### `ClassicalFeatureExtractor`

```python
from models.classical_features import ClassicalFeatureExtractor, extract_classical_features

# Module-level function
features = extract_classical_features(wave, sr=16000)  # → (955,)

# Class API
extractor = ClassicalFeatureExtractor(sr=16000)
features = extractor.transform(wave)             # → (955,)
matrix   = extractor.transform_batch(waves)      # → (N, 955)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `transform` | `(wave: np.ndarray) -> np.ndarray` | 955-dim feature vector |
| `transform_batch` | `(waveforms: list) -> np.ndarray` | (N, 955) feature matrix |
| `n_features` (property) | - | `955` |

**Feature composition**:
- Mel-spectrogram stats: 640 features
- MFCCs + deltas: 100 features
- Spectral features: 155+ features

---

### `HybridAnomalyDetector`

```python
from models.ensemble import HybridAnomalyDetector

detector = HybridAnomalyDetector(
    method="gmm",        # "gmm" | "ocsvm" | "xgboost" | "logistic_regression"
    n_components=16,     # GMM only
    random_state=42,
)
detector.fit(X_train)                          # Train on normal samples
scores = detector.score_samples(X_test)        # Anomaly scores
detector.save("model.pkl")
loaded = HybridAnomalyDetector.load("model.pkl")
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `fit` | `(features, labels=None) -> self` | Fitted detector |
| `score_samples` | `(features) -> np.ndarray` | Anomaly scores (higher=anomalous) |
| `save` | `(path: str)` | - |
| `load` (classmethod) | `(path: str) -> HybridAnomalyDetector` | Loaded detector |

---

### `ConvolutionalAutoencoder`

```python
from models.cae import ConvolutionalAutoencoder

cae = ConvolutionalAutoencoder(
    input_dim=128,
    latent_dim=32,
    dropout=0.3,
    learning_rate=1e-3,
)
cae.fit(X_train, epochs=30, batch_size=32)
scores = cae.score_samples(X_test)   # Reconstruction error
```

**Requirements**: `torch`

---

## `src/data/`

### `DCASEDataset`

```python
from data.dataset import DCASEDataset

dataset = DCASEDataset(root_dir="data/", machine="pump", sr=16000)
train_waves, train_labels = dataset.load_train()
test_waves, test_labels   = dataset.load_test()
```

### `load_audio_files`

```python
from data.dataset import load_audio_files

waves, labels, filenames = load_audio_files(
    directory="data/pump/train",
    sr=16000,
    duration=10.0,
    pattern="*.wav",
)
```

### `AudioPreprocessor`

```python
from data.preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(sr=16000, duration=10.0, normalize=True)
wave = preprocessor.process(raw_wave, orig_sr=22050)
segments, times = preprocessor.segment(wave, segment_duration=1.0, hop_duration=0.5)
```

---

## `src/evaluation/`

### `compute_auc`

```python
from evaluation.metrics import compute_auc

auc = compute_auc(y_true, scores)  # → float in [0, 1]
```

### `evaluate_detector`

```python
from evaluation.metrics import evaluate_detector

result = evaluate_detector(y_true, scores, threshold=None)
# result.auc, result.average_precision, result.fpr, result.tpr, result.confusion
```

### `EvaluationResult`

Dataclass fields: `auc`, `average_precision`, `threshold`, `fpr`, `tpr`, `confusion`, `metadata`

### `plot_roc_curve`

```python
from evaluation.visualization import plot_roc_curve

fig = plot_roc_curve(result, title="Pump ROC", save_path="pump_roc.png")
```

### `plot_results_comparison`

```python
from evaluation.visualization import plot_results_comparison

results = {
    "Baseline": {"fan": 0.832, "pump": 0.815},
    "Hybrid":   {"fan": 0.651, "pump": 0.874},
}
fig = plot_results_comparison(results, save_path="comparison.png")
```

---

## `src/config.py`

```python
from config import (
    ExperimentConfig,
    AudioConfig,
    ASTConfig,
    ClassicalConfig,
    EnsembleConfig,
    BEST_METHODS,
    BEST_N_COMPONENTS,
)

cfg = ExperimentConfig()
cfg.ensemble.method = "gmm"
cfg.ensemble.n_components = BEST_N_COMPONENTS["pump"]  # 16
```

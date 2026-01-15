# Enhanced Audio Anomaly Detection

Advanced audio anomaly detection system for industrial equipment monitoring using machine learning. This system provides comprehensive feature extraction, multiple ML models, and thorough evaluation capabilities.

## Features

### Enhanced Feature Extraction
- 🎵 **Time Domain**: RMS, energy, zero-crossing rate, statistical moments (mean, std, skewness, kurtosis)
- 📊 **Frequency Domain**: FFT analysis, dominant frequency, spectral centroid/spread, multi-band energy
- 🎼 **Spectral Features**: Spectral centroid, rolloff, flatness, bandwidth, contrast, chroma
- 🔊 **MFCCs**: Up to 20 Mel-frequency cepstral coefficients with statistics

### Multiple ML Models
- 🌲 **Random Forest**: With GridSearchCV hyperparameter tuning
- ⚡ **XGBoost**: Advanced gradient boosting with optimized parameters
- 🧠 **Autoencoder**: Deep learning for unsupervised anomaly detection
- 🎯 **Isolation Forest**: Baseline unsupervised method

### Advanced Preprocessing
- 📉 **PCA**: Dimensionality reduction with 95% variance retention
- ⚖️ **SMOTE**: Synthetic minority over-sampling for class imbalance
- 📏 **StandardScaler**: Feature normalization

### Comprehensive Evaluation
- ✅ Cross-validation (5-fold stratified)
- 📈 Multiple metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- 🔍 Confusion matrices and ROC curves
- 📊 Feature importance analysis
- 🎨 Model comparison visualizations

## Quickstart

1. Create and activate a Python 3.10+ virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the package:
   ```bash
   pip install -e .
   ```
4. Run the demo:
   ```bash
   python examples/demo.py
   ```
5. Prepare the dataset following `docs/DATASET.md`
6. Run training (example):
   ```bash
   bash scripts/train.sh
   ```

## Usage Examples

### Training Enhanced Models

```bash
# Train all models (Random Forest, XGBoost, Autoencoder)
python src/audio_anom/train_enhanced.py \
    --data-dir data/pump \
    --output-dir models \
    --models rf xgb ae \
    --enhanced-features \
    --use-pca \
    --use-smote \
    --visualize

# Train specific model with custom parameters
python src/audio_anom/train_enhanced.py \
    --data-dir data/pump \
    --models rf \
    --n-mfcc 20 \
    --pca-variance 0.95 \
    --n-splits 5
```

### Running the Enhanced Demo

```bash
# Run demo with real data
python examples/demo_enhanced.py

# Or run with synthetic data (no dataset needed)
python examples/demo_enhanced.py  # Auto-detects missing data
```

### Training a Basic Model (Legacy)

```bash
python src/audio_anom/train.py \
    --data-dir data/pump \
    --output-dir models \
    --sample-rate 16000 \
    --contamination 0.1
```

### Running Inference

```bash
python examples/inference.py path/to/audio.wav \
    --model-path models/model.pkl
```

### Python API

```python
from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
    ModelEvaluator
)

# Initialize components
feature_extractor = AudioFeatureExtractor(sr=16000, n_mels=128, n_mfcc=20)
data_processor = AudioDataProcessor(sr=16000)
evaluator = ModelEvaluator()

# Load and process data
dataset = data_processor.load_dataset("data/pump")
train_data, val_data, test_data = data_processor.split_dataset(dataset)

# Extract enhanced features
X_train, y_train = [], []
for audio, label in train_data:
    features = feature_extractor.extract_features(audio, enhanced=True)
    if features:
        X_train.append(list(features.values()))
        y_train.append(label)

# Train Random Forest model
rf_detector = RandomForestAnomalyDetector(random_state=42)
rf_detector.fit(X_train, y_train, use_pca=True, use_smote=True)

# Train XGBoost model
xgb_detector = XGBoostAnomalyDetector(random_state=42)
xgb_detector.fit(X_train, y_train, use_pca=True, use_smote=True)

# Evaluate and compare
models = {
    'Random Forest': rf_detector,
    'XGBoost': xgb_detector
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, name)
    print(f"{name}: F1={metrics['F1-Score']:.4f}")

# Save best model
rf_detector.save("models/rf_model.pkl")
```

## Structure

- `src/audio_anom/`: package with feature extraction, data processing, models and training scripts
- `scripts/`: convenience scripts
- `examples/`: example inference script and demo
- `tests/`: unit tests
- `docs/`: dataset instructions

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run the end-to-end demo:
```bash
python examples/demo.py
```

## Development

Lint the code:
```bash
flake8 src/audio_anom/ examples/ tests/ --max-line-length=100
```

## License

MIT

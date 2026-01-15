# Enhanced Audio Anomaly Detection

This repository contains an enhanced pipeline for audio anomaly detection (pump sounds) including feature extraction, training, evaluation and model export. It is based on the MIMII Pump Sound dataset (download instructions included).

## Features

- 🎵 **Audio Feature Extraction**: Mel spectrograms, MFCCs, and statistical features
- 📊 **Data Processing**: Flexible audio loading, preprocessing, and dataset splitting
- 🤖 **Anomaly Detection**: Isolation Forest-based anomaly detection model
- 📈 **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and AUC-ROC
- 💾 **Model Persistence**: Save and load trained models
- 🧪 **Comprehensive Tests**: 20+ unit tests with full coverage

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

### Training a Model

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
from audio_anom import AudioFeatureExtractor, AudioDataProcessor, AnomalyDetector

# Initialize components
feature_extractor = AudioFeatureExtractor(sr=16000, n_mels=128)
data_processor = AudioDataProcessor(sr=16000)
detector = AnomalyDetector(contamination=0.1)

# Load and process data
dataset = data_processor.load_dataset("data/pump")
train_data, val_data, test_data = data_processor.split_dataset(dataset)

# Extract features and train
X_train, y_train = data_processor.prepare_features(train_data, feature_extractor)
detector.fit(X_train, y_train)

# Evaluate
X_test, y_test = data_processor.prepare_features(test_data, feature_extractor)
metrics = detector.evaluate(X_test, y_test)
print(metrics)

# Save model
detector.save("models/model.pkl")
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

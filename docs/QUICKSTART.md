# Quick Start Guide

This guide will help you get started with the Enhanced Audio Anomaly Detection system quickly.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/or4k2l/enhanced-audio-anomaly-detection.git
   cd enhanced-audio-anomaly-detection
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Basic Usage

### 1. Training a Model

Use the training script to train models on your data:

```bash
python scripts/train.py \
    --data-dir ./audio_data \
    --output-dir ./models \
    --model-type both \
    --use-grid-search
```

**Arguments:**
- `--data-dir`: Directory containing audio files (`.wav` format)
- `--output-dir`: Directory to save trained models
- `--model-type`: Choose `random_forest`, `xgboost`, or `both`
- `--use-grid-search`: Enable hyperparameter optimization for Random Forest

### 2. Making Predictions

Use the inference script to predict anomalies:

```bash
python examples/inference.py test_audio.wav \
    --model-path ./models/random_forest.pkl
```

### 3. Evaluating Models

Evaluate model performance on test data:

```bash
python scripts/evaluate.py \
    --model-path ./models/random_forest.pkl \
    --preprocessor-path ./models/preprocessor.pkl \
    --test-features ./data/test_features.npy \
    --test-labels ./data/test_labels.npy \
    --model-type random_forest \
    --save-plots
```

## Python API Usage

### Training Example

```python
from audio_anom import (
    DataPreprocessor,
    RandomForestAnomalyDetector,
    ModelConfig,
)
from sklearn.model_selection import train_test_split

# Load your data (X: features, y: labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize configuration
config = ModelConfig.default()

# Preprocess data
preprocessor = DataPreprocessor(config=config.preprocessing)
X_train_processed, y_train_processed = preprocessor.fit_transform_train(
    X_train, y_train
)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = RandomForestAnomalyDetector(config=config.random_forest)
model.train(X_train_processed, y_train_processed, use_grid_search=True)

# Make predictions
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)

# Save model
model.save("./models/my_model.pkl")
preprocessor.save("./models/my_preprocessor.pkl")
```

### Inference Example

```python
from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    DataPreprocessor,
    RandomForestAnomalyDetector,
    build_feature_vector,
)

# Load models
preprocessor = DataPreprocessor.load("./models/preprocessor.pkl")
model = RandomForestAnomalyDetector()
model.load("./models/random_forest.pkl")

# Initialize processors
feature_extractor = AudioFeatureExtractor()
data_processor = AudioDataProcessor()

# Process audio file
audio, sr = data_processor.load_audio("test_audio.wav")
features = feature_extractor.extract_features(audio)
feature_vector = build_feature_vector(features).reshape(1, -1)

# Preprocess and predict
feature_vector_processed = preprocessor.transform(feature_vector)
prediction = model.predict(feature_vector_processed)[0]
probability = model.predict_proba(feature_vector_processed)[0]

print(f"Prediction: {'Anomaly' if prediction == 1 else 'Normal'}")
print(f"Confidence: {probability[prediction]:.2%}")
```

## Data Format

### Audio Files
- **Format:** WAV files (16-bit PCM)
- **Sample Rate:** 16 kHz (configurable)
- **Naming Convention:**
  - Normal: `normal_*.wav` or place in `normal/` directory
  - Anomaly: `anomaly_*.wav` or place in `anomaly/` directory

### Directory Structure
```
audio_data/
├── normal/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── anomaly/
    ├── sample1.wav
    ├── sample2.wav
    └── ...
```

## Configuration

Create a custom configuration file:

```python
from audio_anom import ModelConfig

config = ModelConfig.default()

# Customize preprocessing
config.preprocessing.n_components = 15  # More PCA components
config.preprocessing.apply_smote = True

# Customize Random Forest
config.random_forest.param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
}

# Save configuration
config.save("./my_config.json")
```

Load and use custom configuration:

```python
config = ModelConfig.load("./my_config.json")
```

## Running the Complete Example

Run the complete training example:

```bash
python examples/train_example.py
```

This will:
1. Generate synthetic data for demonstration
2. Preprocess data with PCA and SMOTE
3. Train both Random Forest and XGBoost models
4. Evaluate and compare models
5. Generate comprehensive visualizations
6. Save models and results

## Next Steps

- Read the [Technical Whitepaper](TECHNICAL_WHITEPAPER.md) for detailed information
- Check out more examples in the `examples/` directory
- Explore advanced configuration options
- Customize models for your specific use case

## Troubleshooting

### Common Issues

1. **Import errors:**
   ```bash
   pip install -e .
   ```

2. **Audio file loading issues:**
   - Ensure files are in WAV format
   - Check sample rate matches configuration

3. **Memory issues with large datasets:**
   - Reduce batch size
   - Use fewer PCA components
   - Process data in chunks

## Support

For issues and questions:
- GitHub Issues: https://github.com/or4k2l/enhanced-audio-anomaly-detection/issues
- Documentation: Check `docs/` directory

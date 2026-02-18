# Enhanced Audio Anomaly Detection

A production-ready machine learning system for detecting anomalies in audio signals using Random Forest and XGBoost models with comprehensive preprocessing and evaluation pipelines.

![CI Tests](https://github.com/or4k2l/enhanced-audio-anomaly-detection/workflows/Tests/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🚀 Features

### Core Capabilities
- **Advanced Feature Extraction**: Mel spectrograms, MFCCs, statistical features
- **Multiple ML Models**: Random Forest with GridSearchCV, XGBoost with auto-balancing
- **Comprehensive Preprocessing**: StandardScaler, PCA (dimensionality reduction), SMOTE (class imbalance handling)
- **Production-Ready**: Centralized configuration, comprehensive logging, model persistence
- **Rich Visualizations**: Confusion matrices, ROC curves, feature importance, model comparison
- **Complete Pipeline**: Training scripts, evaluation tools, inference examples

### Machine Learning Components

#### 1. Data Preprocessing (`preprocessing.py`)
- **StandardScaler**: Feature normalization
- **PCA**: Dimensionality reduction (default: 10 components)
- **SMOTE**: Synthetic minority oversampling
- **Save/Load**: Persistent preprocessing pipelines

#### 2. Random Forest Model (`random_forest_model.py`)
- **GridSearchCV**: Automatic hyperparameter optimization
- **StratifiedKFold**: 3-fold cross-validation
- **Feature Importance**: Analyze most discriminative features
- **Configurable**: Extensive hyperparameter search space

#### 3. XGBoost Model (`xgboost_model.py`)
- **Auto scale_pos_weight**: Automatic class imbalance handling
- **Gradient Boosting**: Sequential weak learners
- **Model Persistence**: Save and load trained models

#### 4. Evaluation & Visualization
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**: Confusion matrix, ROC curves, feature importance
- **Model Comparison**: Side-by-side performance analysis

## 📦 Installation

### Requirements
- Python 3.8+
- System dependencies: `libsndfile1`, `ffmpeg`

### Quick Install

```bash
# Clone repository
git clone https://github.com/or4k2l/enhanced-audio-anomaly-detection.git
cd enhanced-audio-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 🎯 Quick Start

### 1. Training Models

Train both Random Forest and XGBoost models:

```bash
python scripts/train.py \
    --data-dir ./audio_data \
    --output-dir ./models \
    --model-type both \
    --use-grid-search
```

### 2. Making Predictions

Predict anomalies in audio files:

```bash
python examples/inference.py test_audio.wav \
    --model-path ./models/random_forest.pkl
```

### 3. Evaluating Models

Evaluate model performance:

```bash
python scripts/evaluate.py \
    --model-path ./models/random_forest.pkl \
    --preprocessor-path ./models/preprocessor.pkl \
    --test-features ./data/test_features.npy \
    --test-labels ./data/test_labels.npy \
    --model-type random_forest \
    --save-plots
```

### 4. Complete Training Example

Run the full pipeline example:

```bash
python examples/train_example.py
```

This demonstrates:
- Data preprocessing with PCA and SMOTE
- Training Random Forest and XGBoost
- Model evaluation and comparison
- Comprehensive visualizations
- Model persistence

## 💻 Python API

### Training Example

```python
from audio_anom import (
    DataPreprocessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    ModelEvaluator,
    ModelConfig,
)
from sklearn.model_selection import train_test_split

# Load configuration
config = ModelConfig.default()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Preprocess
preprocessor = DataPreprocessor(config=config.preprocessing)
X_train_proc, y_train_proc = preprocessor.fit_transform_train(X_train, y_train)
X_test_proc = preprocessor.transform(X_test)

# Train Random Forest with GridSearchCV
rf_model = RandomForestAnomalyDetector(config=config.random_forest)
rf_model.train(X_train_proc, y_train_proc, use_grid_search=True)

# Train XGBoost
xgb_model = XGBoostAnomalyDetector(config=config.xgboost)
xgb_model.train(X_train_proc, y_train_proc)

# Evaluate
evaluator = ModelEvaluator()
rf_metrics = evaluator.evaluate_model(
    y_test, rf_model.predict(X_test_proc), 
    rf_model.predict_proba(X_test_proc)[:, 1],
    "Random Forest"
)
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

# Process audio
feature_extractor = AudioFeatureExtractor()
data_processor = AudioDataProcessor()

audio, sr = data_processor.load_audio("test_audio.wav")
features = feature_extractor.extract_features(audio)
feature_vector = build_feature_vector(features).reshape(1, -1)

# Predict
feature_vector_proc = preprocessor.transform(feature_vector)
prediction = model.predict(feature_vector_proc)[0]
probability = model.predict_proba(feature_vector_proc)[0]

print(f"Prediction: {'Anomaly' if prediction == 1 else 'Normal'}")
print(f"Confidence: {probability[prediction]:.2%}")
```

## 📊 Project Structure

```
enhanced-audio-anomaly-detection/
├── src/audio_anom/           # Main package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── logger.py             # Logging utilities
│   ├── preprocessing.py      # Data preprocessing
│   ├── random_forest_model.py # Random Forest implementation
│   ├── xgboost_model.py      # XGBoost implementation
│   ├── evaluation.py         # Model evaluation
│   ├── visualization.py      # Visualization tools
│   ├── features.py           # Feature extraction
│   ├── data.py               # Data processing
│   ├── models.py             # Base model classes
│   └── export.py             # Model export utilities
├── scripts/                  # Utility scripts
│   ├── train.py             # Training pipeline
│   └── evaluate.py          # Evaluation pipeline
├── examples/                 # Usage examples
│   ├── train_example.py     # Complete training example
│   ├── inference.py         # Inference example
│   └── demo.py              # Demo script
├── tests/                    # Test suite
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_features.py
│   └── ...
├── docs/                     # Documentation
│   ├── QUICKSTART.md        # Quick start guide
│   ├── TECHNICAL_WHITEPAPER.md # Technical details
│   └── ...
├── .github/workflows/        # CI/CD pipelines
│   ├── tests.yml            # Automated testing
│   └── ci.yml               # Original CI
├── requirements.txt          # Python dependencies
├── setup.py                 # Package configuration
├── pytest.ini               # Pytest configuration
└── README.md                # This file
```

## 🧪 Testing

Run the complete test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_model.py -v
pytest tests/test_preprocessing.py -v
```

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)**: Get started quickly
- **[Technical Whitepaper](docs/TECHNICAL_WHITEPAPER.md)**: Detailed technical documentation
- **[API Reference](docs/)**: Complete API documentation in code docstrings

## 🔧 Configuration

Create custom configurations:

```python
from audio_anom import ModelConfig

config = ModelConfig.default()

# Customize preprocessing
config.preprocessing.n_components = 15
config.preprocessing.apply_smote = True

# Customize Random Forest
config.random_forest.param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
}

# Save configuration
config.save("./my_config.json")

# Load configuration
config = ModelConfig.load("./my_config.json")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn for machine learning algorithms
- XGBoost for gradient boosting
- librosa for audio processing
- imbalanced-learn for SMOTE implementation

## 📧 Support

For issues and questions:
- GitHub Issues: https://github.com/or4k2l/enhanced-audio-anomaly-detection/issues
- Documentation: Check `docs/` directory

---

**Note**: This is a production-ready implementation with comprehensive features for audio anomaly detection. All components are fully tested and documented.
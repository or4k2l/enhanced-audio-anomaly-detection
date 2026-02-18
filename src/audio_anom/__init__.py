"""Enhanced Audio Anomaly Detection Package."""

__version__ = "0.1.0"

from .features import AudioFeatureExtractor
from .data import AudioDataProcessor, build_feature_vector
from .models import (
    AnomalyDetector,
    AutoencoderAnomalyDetector,
)
from .random_forest_model import RandomForestAnomalyDetector
from .xgboost_model import XGBoostAnomalyDetector
from .preprocessing import DataPreprocessor
from .evaluation import ModelEvaluator
from .export import ModelExporter
from .config import (
    ModelConfig,
    FeatureConfig,
    PreprocessingConfig,
    RandomForestConfig,
    XGBoostConfig,
    TrainingConfig,
    EvaluationConfig,
    DEFAULT_CONFIG,
)
from .logger import setup_logger, get_logger

# Unsupervised Anomaly Detection
from .unsupervised_anomaly import (
    LocalOutlierFactorAnomalyDetector,
    IsolationForestAnomalyDetector,
    EllipticEnvelopeAnomalyDetector,
    create_detector,
)
from .preprocessing_unsupervised import UnsupervisedPreprocessor

__all__ = [
    # Core
    "AudioFeatureExtractor",
    "AudioDataProcessor",
    "build_feature_vector",
    # Supervised Models
    "AnomalyDetector",
    "RandomForestAnomalyDetector",
    "XGBoostAnomalyDetector",
    "AutoencoderAnomalyDetector",
    # Unsupervised Models
    "LocalOutlierFactorAnomalyDetector",
    "IsolationForestAnomalyDetector",
    "EllipticEnvelopeAnomalyDetector",
    "create_detector",
    # Preprocessing
    "DataPreprocessor",
    "UnsupervisedPreprocessor",
    # Evaluation & Export
    "ModelEvaluator",
    "ModelExporter",
    # Configuration
    "ModelConfig",
    "FeatureConfig",
    "PreprocessingConfig",
    "RandomForestConfig",
    "XGBoostConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DEFAULT_CONFIG",
    # Utilities
    "setup_logger",
    "get_logger",
]

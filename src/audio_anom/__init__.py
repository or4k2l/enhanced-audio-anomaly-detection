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

__all__ = [
    "AudioFeatureExtractor",
    "AudioDataProcessor",
    "AnomalyDetector",
    "RandomForestAnomalyDetector",
    "XGBoostAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "DataPreprocessor",
    "ModelEvaluator",
    "ModelExporter",
    "build_feature_vector",
    "ModelConfig",
    "FeatureConfig",
    "PreprocessingConfig",
    "RandomForestConfig",
    "XGBoostConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "DEFAULT_CONFIG",
    "setup_logger",
    "get_logger",
]

"""Enhanced Audio Anomaly Detection Package."""

__version__ = "0.1.0"

from .features import AudioFeatureExtractor
from .data import AudioDataProcessor, build_feature_vector
from .model import AnomalyDetector
from .models_advanced import (
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
)
from .evaluation import ModelEvaluator

__all__ = [
    "AudioFeatureExtractor",
    "AudioDataProcessor",
    "AnomalyDetector",
    "RandomForestAnomalyDetector",
    "XGBoostAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "ModelEvaluator",
    "build_feature_vector",
]

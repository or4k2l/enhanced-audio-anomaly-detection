"""Enhanced Audio Anomaly Detection Package."""

__version__ = "0.1.0"

from .features import AudioFeatureExtractor
from .data import AudioDataProcessor, build_feature_vector
from .models import (
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
)
from .evaluation import ModelEvaluator

__all__ = [
    "AudioFeatureExtractor",
    "AudioDataProcessor",
    "RandomForestAnomalyDetector",
    "XGBoostAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "ModelEvaluator",
    "build_feature_vector",
]

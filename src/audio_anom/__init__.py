"""Enhanced Audio Anomaly Detection Package."""

__version__ = "0.1.0"

from .features import AudioFeatureExtractor
from .data import AudioDataProcessor, build_feature_vector
from .models import (
    AnomalyDetector,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    AutoencoderAnomalyDetector,
)
from .evaluation import ModelEvaluator
from .export import ModelExporter

__all__ = [
    "AudioFeatureExtractor",
    "AudioDataProcessor",
    "AnomalyDetector",
    "RandomForestAnomalyDetector",
    "XGBoostAnomalyDetector",
    "AutoencoderAnomalyDetector",
    "ModelEvaluator",
    "ModelExporter",
    "build_feature_vector",
]

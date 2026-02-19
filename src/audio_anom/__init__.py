"""Enhanced Audio Anomaly Detection Package."""

__version__ = "3.0.0"

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

# Embedding-Based Anomaly Detection
from .feature_extractor import RobustFeatureExtractor
from .embedding_anomaly import (
    MahalanobisDetector,
    KNNDetector,
    IsolationForestDetector as EmbeddingIsolationForestDetector,
    EnsembleDetector,
    create_detector as create_embedding_detector,
)
from .augmentation import AudioAugmenter, create_augmenter
from .embedding_config import (
    EmbeddingAnomalyConfig,
    FeatureExtractionConfig,
    MahalanobisConfig,
    KNNConfig,
    IsolationForestConfig as EmbeddingIsolationForestConfig,
    EnsembleConfig,
    AugmentationConfig,
    create_default_config_file,
)

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
    # Embedding-Based Anomaly Detection
    "RobustFeatureExtractor",
    "MahalanobisDetector",
    "KNNDetector",
    "EmbeddingIsolationForestDetector",
    "EnsembleDetector",
    "create_embedding_detector",
    "AudioAugmenter",
    "create_augmenter",
    "EmbeddingAnomalyConfig",
    "FeatureExtractionConfig",
    "MahalanobisConfig",
    "KNNConfig",
    "EmbeddingIsolationForestConfig",
    "EnsembleConfig",
    "AugmentationConfig",
    "create_default_config_file",
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

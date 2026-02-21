"""Configuration system for embedding-based anomaly detection.

YAML-based configuration for:
- Feature extraction parameters
- Detector method selection
- Augmentation settings
- Ensemble weights
- Training hyperparameters
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""

    sr: int = 22050
    n_mels: int = 128
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    normalize: bool = True


@dataclass
class MahalanobisConfig:
    """Configuration for Mahalanobis detector."""

    use_ledoit_wolf: bool = True
    regularization: float = 1e-6
    threshold_percentile: float = 95.0


@dataclass
class KNNConfig:
    """Configuration for k-NN detector."""

    n_neighbors: int = 5
    threshold_percentile: float = 95.0
    metric: str = "euclidean"


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest detector."""

    contamination: float = 0.1
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: Optional[int] = 42


@dataclass
class EnsembleConfig:
    """Configuration for ensemble detector."""

    methods: List[str] = field(
        default_factory=lambda: ["mahalanobis", "knn", "isolation_forest"]
    )
    weights: Optional[List[float]] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    threshold_percentile: float = 95.0


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""

    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_specaug: bool = True
    use_timestretch: bool = True
    use_pitchshift: bool = True
    use_noise: bool = True
    random_state: Optional[int] = None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 32
    validation_split: float = 0.2
    random_state: int = 42
    save_best_model: bool = True


@dataclass
class EmbeddingAnomalyConfig:
    """Main configuration for embedding-based anomaly detection."""

    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
    detector_method: str = "ensemble"
    mahalanobis: MahalanobisConfig = field(default_factory=MahalanobisConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "EmbeddingAnomalyConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration object
        """
        logger.info(f"Loading configuration from {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configurations
        feature_config = FeatureExtractionConfig(
            **config_dict.get("feature_extraction", {})
        )
        mahal_config = MahalanobisConfig(**config_dict.get("mahalanobis", {}))
        knn_config = KNNConfig(**config_dict.get("knn", {}))
        isof_config = IsolationForestConfig(**config_dict.get("isolation_forest", {}))
        ensemble_config = EnsembleConfig(**config_dict.get("ensemble", {}))
        aug_config = AugmentationConfig(**config_dict.get("augmentation", {}))
        train_config = TrainingConfig(**config_dict.get("training", {}))

        detector_method = config_dict.get("detector_method", "ensemble")

        return cls(
            feature_extraction=feature_config,
            detector_method=detector_method,
            mahalanobis=mahal_config,
            knn=knn_config,
            isolation_forest=isof_config,
            ensemble=ensemble_config,
            augmentation=aug_config,
            training=train_config,
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        logger.info(f"Saving configuration to {path}")

        config_dict = {
            "detector_method": self.detector_method,
            "feature_extraction": asdict(self.feature_extraction),
            "mahalanobis": asdict(self.mahalanobis),
            "knn": asdict(self.knn),
            "isolation_forest": asdict(self.isolation_forest),
            "ensemble": asdict(self.ensemble),
            "augmentation": asdict(self.augmentation),
            "training": asdict(self.training),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "detector_method": self.detector_method,
            "feature_extraction": asdict(self.feature_extraction),
            "mahalanobis": asdict(self.mahalanobis),
            "knn": asdict(self.knn),
            "isolation_forest": asdict(self.isolation_forest),
            "ensemble": asdict(self.ensemble),
            "augmentation": asdict(self.augmentation),
            "training": asdict(self.training),
        }

    @classmethod
    def default(cls) -> "EmbeddingAnomalyConfig":
        """Create default configuration.

        Returns:
            Default configuration
        """
        return cls()


def create_default_config_file(path: str = "config/embedding_anomaly_config.yaml"):
    """Create default configuration file.

    Args:
        path: Path to save configuration file
    """
    config = EmbeddingAnomalyConfig.default()
    config.to_yaml(path)
    logger.info(f"Default configuration created at {path}")


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()

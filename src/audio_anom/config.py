"""Configuration management for audio anomaly detection."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class FeatureConfig:
    """Configuration for audio feature extraction."""

    sr: int = 16000  # Sample rate
    n_mels: int = 128  # Number of Mel bands
    n_fft: int = 1024  # FFT window size
    hop_length: int = 512  # Hop length for STFT
    n_mfcc: int = 20  # Number of MFCC coefficients


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    n_components: int = 10  # Number of PCA components
    apply_pca: bool = True  # Whether to apply PCA
    apply_smote: bool = True  # Whether to apply SMOTE
    apply_scaling: bool = True  # Whether to apply StandardScaler
    smote_random_state: int = 42  # Random state for SMOTE
    smote_k_neighbors: int = 5  # Number of neighbors for SMOTE


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""

    # GridSearchCV parameters
    param_grid: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced", None],
        }
    )

    # Cross-validation parameters
    cv_folds: int = 3  # StratifiedKFold folds
    cv_random_state: int = 42

    # Other parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    verbose: int = 1


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""

    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1
    use_label_encoder: bool = False
    eval_metric: str = "logloss"
    auto_scale_pos_weight: bool = True  # Auto-calculate scale_pos_weight


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    test_size: float = 0.2  # Test set size
    val_size: float = 0.1  # Validation set size (from training set)
    random_state: int = 42
    batch_size: int = 32
    shuffle: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    metrics: list = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    )
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


@dataclass
class ModelConfig:
    """Main configuration for audio anomaly detection models."""

    feature: FeatureConfig = field(default_factory=FeatureConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "feature": self.feature.__dict__,
            "preprocessing": self.preprocessing.__dict__,
            "random_forest": self.random_forest.__dict__,
            "xgboost": self.xgboost.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
        }

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save configuration file
        """
        config_dict = self.to_dict()
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """
        Load configuration from JSON file.

        Args:
            path: Path to configuration file

        Returns:
            ModelConfig instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)

        return cls(
            feature=FeatureConfig(**config_dict.get("feature", {})),
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            random_forest=RandomForestConfig(**config_dict.get("random_forest", {})),
            xgboost=XGBoostConfig(**config_dict.get("xgboost", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
        )

    @classmethod
    def default(cls) -> "ModelConfig":
        """
        Create default configuration.

        Returns:
            ModelConfig with default values
        """
        return cls()


# Default configuration instance
DEFAULT_CONFIG = ModelConfig.default()

"""XGBoost model for audio anomaly detection."""

import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from typing import Optional, Dict, Any

from .logger import get_logger
from .config import XGBoostConfig

logger = get_logger(__name__)


class XGBoostAnomalyDetector:
    """
    XGBoost classifier for audio anomaly detection.

    This class provides XGBoost-based anomaly detection with automatic
    scale_pos_weight calculation for handling class imbalance.

    Attributes:
        config: XGBoost configuration
        model: Trained XGBoost model
        best_estimator_: Best estimator (same as model)
        is_fitted: Whether the model has been fitted
        random_state: Random state for reproducibility
    """

    def __init__(
        self,
        config: Optional[XGBoostConfig] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize XGBoost anomaly detector.

        Args:
            config: XGBoost configuration. If None, uses defaults.
            random_state: Random state for reproducibility. Overrides config if provided.
        """
        self.config = config or XGBoostConfig()
        if random_state is not None:
            self.random_state = random_state
            self.config.random_state = random_state
        else:
            self.random_state = self.config.random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.best_estimator_: Optional[xgb.XGBClassifier] = None
        self.is_fitted = False

        logger.info(
            f"XGBoostAnomalyDetector initialized with "
            f"random_state={self.random_state}"
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> "XGBoostAnomalyDetector":
        """
        Train XGBoost model.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        logger.info(f"Training XGBoost on data with shape {X.shape}")

        # Calculate scale_pos_weight if auto-scaling enabled
        scale_pos_weight = None
        if self.config.auto_scale_pos_weight:
            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique, counts))

            if 0 in class_counts and 1 in class_counts:
                scale_pos_weight = class_counts[0] / class_counts[1]
                logger.info(
                    f"Auto-calculated scale_pos_weight: {scale_pos_weight:.4f} "
                    f"(negative={class_counts[0]}, positive={class_counts[1]})"
                )

        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            n_estimators=self.config.n_estimators,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.config.n_jobs,
            use_label_encoder=self.config.use_label_encoder,
            eval_metric=self.config.eval_metric,
            scale_pos_weight=scale_pos_weight,
        )

        # Train model
        self.model.fit(X, y)

        self.best_estimator_ = self.model
        self.is_fitted = True

        logger.info("XGBoost training completed")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostAnomalyDetector":
        """
        Fit the model (alias for train).

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        return self.train(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels

        Raises:
            Exception: If model is not fitted
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() before predicting.")

        logger.debug(f"Predicting on data with shape {X.shape}")
        predictions = self.best_estimator_.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Predicted class probabilities

        Raises:
            Exception: If model is not fitted
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() before predicting.")

        logger.debug(f"Predicting probabilities on data with shape {X.shape}")
        probabilities = self.best_estimator_.predict_proba(X)

        return probabilities

    def get_feature_importance(
        self, feature_names: Optional[list] = None, top_n: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: Names of features. If None, uses indices.
            top_n: Return only top N features. If None, returns all.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        if not hasattr(self.best_estimator_, "feature_importances_"):
            raise ValueError("Feature importances not available")

        feature_importances = self.best_estimator_.feature_importances_

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(feature_importances))]

        # Create importance dictionary
        importance_dict = dict(zip(feature_names, feature_importances))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Return top N if specified
        if top_n is not None:
            importance_dict = dict(list(importance_dict.items())[:top_n])

        return importance_dict

    def save(self, file_path: str) -> None:
        """
        Save trained model to disk.

        Args:
            file_path: Path to save the model

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "best_estimator_": self.best_estimator_,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "config": self.config,
        }

        joblib.dump(state, file_path)
        logger.info(f"Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load trained model from disk.

        Args:
            file_path: Path to load the model from
        """
        state = joblib.load(file_path)

        self.best_estimator_ = state["best_estimator_"]
        self.random_state = state["random_state"]
        self.is_fitted = state["is_fitted"]

        # Load config if available (backwards compatibility)
        if "config" in state:
            self.config = state["config"]

        # Set model reference
        self.model = self.best_estimator_

        logger.info(f"Model loaded from {file_path}")

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        if self.is_fitted and self.best_estimator_ is not None:
            return self.best_estimator_.get_params()
        return {}

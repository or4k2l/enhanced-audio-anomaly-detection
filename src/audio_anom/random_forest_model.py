"""Random Forest model with GridSearchCV for audio anomaly detection."""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .logger import get_logger
from .config import RandomForestConfig

logger = get_logger(__name__)


class RandomForestAnomalyDetector:
    """
    Random Forest classifier with GridSearchCV hyperparameter optimization.

    This class provides Random Forest-based anomaly detection with automatic
    hyperparameter tuning using GridSearchCV and StratifiedKFold cross-validation.

    Attributes:
        config: Random Forest configuration
        model: Trained model (GridSearchCV or RandomForestClassifier)
        best_estimator_: Best estimator from GridSearchCV
        is_fitted: Whether the model has been fitted
        feature_importances_: Feature importance scores
    """

    def __init__(
        self,
        config: Optional[RandomForestConfig] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Random Forest anomaly detector.

        Args:
            config: Random Forest configuration. If None, uses defaults.
            random_state: Random state for reproducibility. Overrides config if provided.
        """
        self.config = config or RandomForestConfig()
        if random_state is not None:
            self.random_state = random_state
            self.config.random_state = random_state
        else:
            self.random_state = self.config.random_state

        self.model: Optional[GridSearchCV] = None
        self.best_estimator_: Optional[RandomForestClassifier] = None
        self.is_fitted = False
        self.feature_importances_: Optional[np.ndarray] = None
        self.cv_results_: Optional[Dict[str, Any]] = None

        logger.info(
            f"RandomForestAnomalyDetector initialized with "
            f"random_state={self.random_state}"
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_grid_search: bool = True,
    ) -> "RandomForestAnomalyDetector":
        """
        Train Random Forest model with optional GridSearchCV.

        Args:
            X: Training features
            y: Training labels
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Training Random Forest on data with shape {X.shape}, "
            f"use_grid_search={use_grid_search}"
        )

        if use_grid_search:
            # Setup StratifiedKFold cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.cv_random_state,
            )

            # Create base estimator
            base_estimator = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.config.n_jobs,
            )

            # Setup GridSearchCV
            logger.info(
                f"Starting GridSearchCV with {self.config.cv_folds} folds, "
                f"param_grid={self.config.param_grid}"
            )

            self.model = GridSearchCV(
                estimator=base_estimator,
                param_grid=self.config.param_grid,
                cv=cv,
                scoring="f1",
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
                return_train_score=True,
            )

            # Fit model
            self.model.fit(X, y)

            # Get best estimator
            self.best_estimator_ = self.model.best_estimator_
            self.cv_results_ = self.model.cv_results_

            logger.info(f"Best parameters: {self.model.best_params_}")
            logger.info(f"Best cross-validation F1 score: {self.model.best_score_:.4f}")

        else:
            # Train simple Random Forest without GridSearchCV
            logger.info("Training simple Random Forest without GridSearchCV")

            self.best_estimator_ = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.config.n_jobs,
            )
            self.best_estimator_.fit(X, y)

        # Extract feature importances
        self.feature_importances_ = self.best_estimator_.feature_importances_

        self.is_fitted = True
        logger.info("Random Forest training completed")

        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestAnomalyDetector":
        """
        Fit the model (alias for train with simple training).

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        return self.train(X, y, use_grid_search=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels

        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predicting.")

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
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predicting.")

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

        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available")

        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importances_))]

        # Create importance dictionary
        importance_dict = dict(zip(feature_names, self.feature_importances_))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Return top N if specified
        if top_n is not None:
            importance_dict = dict(list(importance_dict.items())[:top_n])

        return importance_dict

    def get_cv_results(self) -> Optional[Dict[str, Any]]:
        """
        Get cross-validation results from GridSearchCV.

        Returns:
            Cross-validation results dictionary, or None if GridSearchCV not used
        """
        return self.cv_results_

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
            "feature_importances_": self.feature_importances_,
            "cv_results_": self.cv_results_,
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
        self.feature_importances_ = state.get("feature_importances_")
        self.cv_results_ = state.get("cv_results_")

        # Load config if available (backwards compatibility)
        if "config" in state:
            self.config = state["config"]

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

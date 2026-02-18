"""Data preprocessing module for audio anomaly detection."""

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from .logger import get_logger
from .config import PreprocessingConfig

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing including scaling, PCA, and SMOTE.

    This class provides a complete preprocessing pipeline for audio anomaly
    detection, including feature normalization, dimensionality reduction,
    and class imbalance handling.

    Attributes:
        config: Preprocessing configuration
        scaler: StandardScaler instance
        pca: PCA instance
        smote: SMOTE instance
        is_fitted: Whether the preprocessor has been fitted
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize data preprocessor.

        Args:
            config: Preprocessing configuration. If None, uses defaults.
        """
        self.config = config or PreprocessingConfig()
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.smote: Optional[SMOTE] = None
        self.is_fitted = False

        # Initialize components based on config
        if self.config.apply_scaling:
            self.scaler = StandardScaler()

        if self.config.apply_pca:
            self.pca = PCA(
                n_components=self.config.n_components,
                random_state=self.config.smote_random_state,
            )

        if self.config.apply_smote:
            self.smote = SMOTE(
                random_state=self.config.smote_random_state,
                k_neighbors=self.config.smote_k_neighbors,
            )

        logger.info(
            f"DataPreprocessor initialized with config: "
            f"scaling={self.config.apply_scaling}, "
            f"pca={self.config.apply_pca} (n_components={self.config.n_components}), "
            f"smote={self.config.apply_smote}"
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            X: Training features
            y: Training labels (optional, needed for SMOTE)

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting preprocessor on data with shape {X.shape}")

        X_processed = X.copy()

        # Fit scaler
        if self.scaler is not None:
            logger.debug("Fitting StandardScaler")
            self.scaler.fit(X_processed)
            X_processed = self.scaler.transform(X_processed)

        # Fit PCA
        if self.pca is not None:
            logger.debug(f"Fitting PCA with {self.config.n_components} components")
            self.pca.fit(X_processed)

            # Log explained variance
            if hasattr(self.pca, "explained_variance_ratio_"):
                total_var = np.sum(self.pca.explained_variance_ratio_)
                logger.info(
                    f"PCA explains {total_var:.2%} of variance with "
                    f"{self.config.n_components} components"
                )

        self.is_fitted = True
        logger.info("Preprocessor fitting completed")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            Transformed features

        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )

        logger.debug(f"Transforming data with shape {X.shape}")

        X_processed = X.copy()

        # Apply scaling
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)

        # Apply PCA
        if self.pca is not None:
            X_processed = self.pca.transform(X_processed)

        logger.debug(f"Transformed data shape: {X_processed.shape}")

        return X_processed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.

        Args:
            X: Features to fit and transform

        Returns:
            Transformed features
        """
        self.fit(X)
        return self.transform(X)

    def fit_transform_train(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor on training data, apply transformations including SMOTE.

        This method is specifically designed for training data where we want
        to apply SMOTE for class imbalance handling.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (transformed features, resampled labels)
        """
        logger.info(
            f"Fitting and transforming training data: X shape={X_train.shape}, "
            f"y shape={y_train.shape}"
        )

        # Log class distribution before SMOTE
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Class distribution before SMOTE: {dict(zip(unique, counts))}")

        # Fit and transform without SMOTE
        self.fit(X_train)
        X_processed = self.transform(X_train)

        # Apply SMOTE if configured
        if self.smote is not None:
            try:
                logger.debug("Applying SMOTE for class balancing")
                X_processed, y_train = self.smote.fit_resample(X_processed, y_train)

                # Log class distribution after SMOTE
                unique, counts = np.unique(y_train, return_counts=True)
                logger.info(
                    f"Class distribution after SMOTE: {dict(zip(unique, counts))}"
                )
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")

        logger.info(f"Training data processed: final shape {X_processed.shape}")

        return X_processed, y_train

    def save(self, path: str) -> None:
        """
        Save fitted preprocessor to disk.

        Args:
            path: Path to save the preprocessor

        Raises:
            ValueError: If preprocessor is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "scaler": self.scaler,
            "pca": self.pca,
            "smote": self.smote,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(state, path)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """
        Load fitted preprocessor from disk.

        Args:
            path: Path to load the preprocessor from

        Returns:
            Loaded DataPreprocessor instance
        """
        state = joblib.load(path)

        preprocessor = cls(config=state["config"])
        preprocessor.scaler = state["scaler"]
        preprocessor.pca = state["pca"]
        preprocessor.smote = state["smote"]
        preprocessor.is_fitted = state["is_fitted"]

        logger.info(f"Preprocessor loaded from {path}")

        return preprocessor

    def get_feature_names(self) -> list:
        """
        Get feature names after transformation.

        Returns:
            List of feature names (e.g., ['PC1', 'PC2', ...])
        """
        if self.pca is not None and self.is_fitted:
            return [f"PC{i+1}" for i in range(self.pca.n_components_)]
        elif self.scaler is not None and self.is_fitted:
            return [f"Feature_{i+1}" for i in range(self.scaler.n_features_in_)]
        else:
            return []

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio from PCA.

        Returns:
            Explained variance ratio array, or None if PCA not applied
        """
        if self.pca is not None and self.is_fitted:
            return self.pca.explained_variance_ratio_
        return None

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing pipeline.

        Returns:
            Dictionary with preprocessing information
        """
        info = {
            "is_fitted": self.is_fitted,
            "apply_scaling": self.config.apply_scaling,
            "apply_pca": self.config.apply_pca,
            "apply_smote": self.config.apply_smote,
        }

        if self.is_fitted:
            if self.scaler is not None:
                info["n_features_in"] = self.scaler.n_features_in_

            if self.pca is not None:
                info["n_components"] = self.pca.n_components_
                if hasattr(self.pca, "explained_variance_ratio_"):
                    info["total_variance_explained"] = float(
                        np.sum(self.pca.explained_variance_ratio_)
                    )

        return info

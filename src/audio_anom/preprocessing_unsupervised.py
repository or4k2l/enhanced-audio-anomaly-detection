"""Preprocessing pipeline for unsupervised anomaly detection.

This module provides a preprocessing pipeline specifically designed for
unsupervised anomaly detection on real-world audio data:
- StandardScaler for feature normalization
- PCA for dimensionality reduction (10 components, 88-91% variance)
- Fit on normal data only (production scenario)
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .logger import get_logger

logger = get_logger(__name__)


class UnsupervisedPreprocessor:
    """Preprocessing pipeline for unsupervised anomaly detection.
    
    This preprocessor is designed for production scenarios where only normal
    data is available for training. It applies:
    1. StandardScaler: Normalizes features to zero mean and unit variance
    2. PCA: Reduces dimensionality while preserving 88-91% variance
    
    Key difference from supervised preprocessing:
    - Fit only on normal data (no anomaly labels needed)
    - No SMOTE (no class balancing needed)
    - Optimized for anomaly detection (not classification)
    
    Attributes:
        n_components: Number of PCA components (default: 10)
        apply_pca: Whether to apply PCA (default: True)
        scaler: StandardScaler instance
        pca: PCA instance
        is_fitted: Whether the preprocessor has been fitted
        explained_variance_ratio: Variance explained by each component
    """
    
    def __init__(
        self,
        n_components: int = 10,
        apply_pca: bool = True,
    ):
        """Initialize preprocessor.
        
        Args:
            n_components: Number of PCA components to keep
            apply_pca: Whether to apply PCA dimensionality reduction
        """
        self.n_components = n_components
        self.apply_pca = apply_pca
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if apply_pca else None
        self.is_fitted = False
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray) -> "UnsupervisedPreprocessor":
        """Fit the preprocessor on normal data only.
        
        Important: X should contain only normal samples (no anomalies).
        This is the key difference from supervised preprocessing.
        
        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info(f"Fitting preprocessor on {X.shape[0]} normal samples "
                   f"with {X.shape[1]} features")
        
        # Step 1: Fit StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        logger.info("StandardScaler fitted")
        
        # Step 2: Fit PCA (optional)
        if self.apply_pca:
            X_pca = self.pca.fit_transform(X_scaled)
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            total_variance = np.sum(self.explained_variance_ratio_)
            logger.info(f"PCA fitted: {self.n_components} components explain "
                       f"{total_variance:.2%} of variance")
            logger.info(f"Variance per component: {self.explained_variance_ratio_}")
        
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data, shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Step 1: Scale
        X_scaled = self.scaler.transform(X)
        
        # Step 2: PCA (optional)
        if self.apply_pca:
            X_pca = self.pca.transform(X_scaled)
            return X_pca
        
        return X_scaled
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the preprocessor and transform data.
        
        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
        
    def save(self, path: str) -> None:
        """Save preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
            
        preprocessor_data = {
            'scaler': self.scaler,
            'pca': self.pca,
            'n_components': self.n_components,
            'apply_pca': self.apply_pca,
            'is_fitted': self.is_fitted,
            'explained_variance_ratio': self.explained_variance_ratio_,
        }
        joblib.dump(preprocessor_data, path)
        logger.info(f"Preprocessor saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> "UnsupervisedPreprocessor":
        """Load preprocessor from disk.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            preprocessor: Loaded preprocessor instance
        """
        preprocessor_data = joblib.load(path)
        
        preprocessor = cls(
            n_components=preprocessor_data['n_components'],
            apply_pca=preprocessor_data['apply_pca'],
        )
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.pca = preprocessor_data['pca']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        preprocessor.explained_variance_ratio_ = preprocessor_data.get('explained_variance_ratio')
        
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
        
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from PCA components.
        
        Returns the absolute values of the first principal component,
        which indicates which original features are most important.
        
        Returns:
            importance: Feature importance array, or None if PCA not applied
        """
        if not self.apply_pca or not self.is_fitted:
            return None
        
        # First component typically captures most variance
        return np.abs(self.pca.components_[0])
        
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Inverse transform back to original feature space.
        
        Useful for visualization and interpretation.
        
        Args:
            X_transformed: Transformed data, shape (n_samples, n_components)
            
        Returns:
            X_original: Data in original space, shape (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        if self.apply_pca:
            X_scaled = self.pca.inverse_transform(X_transformed)
        else:
            X_scaled = X_transformed
            
        return self.scaler.inverse_transform(X_scaled)
        
    def __repr__(self) -> str:
        """String representation of preprocessor."""
        status = "fitted" if self.is_fitted else "not fitted"
        if self.apply_pca:
            if self.explained_variance_ratio_ is not None:
                variance = np.sum(self.explained_variance_ratio_)
                return (f"UnsupervisedPreprocessor(n_components={self.n_components}, "
                       f"variance_explained={variance:.2%}, {status})")
            return f"UnsupervisedPreprocessor(n_components={self.n_components}, {status})"
        return f"UnsupervisedPreprocessor(PCA=False, {status})"

"""Embedding-based anomaly detection with multiple methods.

This module provides three detector implementations with a unified interface:
- MahalanobisDetector: Robust covariance estimation with Ledoit-Wolf
- KNNDetector: k-NN anomaly detection in embedding space
- IsolationForestDetector: Tree-based outlier detection
- EnsembleDetector: Weighted combination of multiple methods
"""

import warnings
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest as SKIsolationForest
from scipy.spatial.distance import mahalanobis

from .logger import get_logger

logger = get_logger(__name__)


class BaseEmbeddingDetector(ABC):
    """Base class for embedding-based anomaly detectors."""
    
    def __init__(self):
        """Initialize detector."""
        self.is_fitted = False
        self.threshold = None
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseEmbeddingDetector':
        """Fit the detector on normal data.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,), higher scores indicate anomalies
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (0=normal, 1=anomaly).
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Binary predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        scores = self.score(X)
        predictions = (scores > self.threshold).astype(int)
        return predictions
    
    def save(self, path: str):
        """Save detector to disk.
        
        Args:
            path: File path to save detector
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted detector")
        
        joblib.dump(self, path)
        logger.info(f"Detector saved to {path}")
    
    @staticmethod
    def load(path: str) -> 'BaseEmbeddingDetector':
        """Load detector from disk.
        
        Args:
            path: File path to load detector
            
        Returns:
            Loaded detector
        """
        detector = joblib.load(path)
        logger.info(f"Detector loaded from {path}")
        return detector


class MahalanobisDetector(BaseEmbeddingDetector):
    """Mahalanobis distance-based anomaly detector.
    
    Uses Ledoit-Wolf robust covariance estimation and percentile-based thresholding.
    """
    
    def __init__(
        self,
        use_ledoit_wolf: bool = True,
        regularization: float = 1e-6,
        threshold_percentile: float = 95.0,
    ):
        """Initialize Mahalanobis detector.
        
        Args:
            use_ledoit_wolf: Use Ledoit-Wolf shrinkage for covariance estimation
            regularization: Regularization parameter for numerical stability
            threshold_percentile: Percentile for threshold (e.g., 95 = 95th percentile)
        """
        super().__init__()
        self.use_ledoit_wolf = use_ledoit_wolf
        self.regularization = regularization
        self.threshold_percentile = threshold_percentile
        self.mean_ = None
        self.cov_ = None
        self.inv_cov_ = None
    
    def fit(self, X: np.ndarray) -> 'MahalanobisDetector':
        """Fit Mahalanobis detector on normal data.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            
        Returns:
            self
        """
        logger.info(f"Fitting MahalanobisDetector on {X.shape[0]} samples")
        
        # Compute mean
        self.mean_ = np.mean(X, axis=0)
        
        # Compute covariance
        if self.use_ledoit_wolf:
            logger.info("Using Ledoit-Wolf covariance estimation")
            lw = LedoitWolf()
            lw.fit(X)
            self.cov_ = lw.covariance_
        else:
            self.cov_ = np.cov(X, rowvar=False)
        
        # Add regularization for numerical stability
        self.cov_ += np.eye(self.cov_.shape[0]) * self.regularization
        
        # Compute inverse covariance
        try:
            self.inv_cov_ = np.linalg.inv(self.cov_)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance matrix, using pseudo-inverse")
            self.inv_cov_ = np.linalg.pinv(self.cov_)
        
        # Mark as fitted before computing threshold
        self.is_fitted = True
        
        # Compute threshold based on training scores
        train_scores = self.score(X)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)
        
        logger.info(f"MahalanobisDetector fitted, threshold: {self.threshold:.4f}")
        
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances as anomaly scores.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Mahalanobis distances (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Compute Mahalanobis distance for each sample
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            diff = x - self.mean_
            scores[i] = np.sqrt(diff @ self.inv_cov_ @ diff.T)
        
        return scores


class KNNDetector(BaseEmbeddingDetector):
    """k-NN based anomaly detector.
    
    Uses average distance to k nearest neighbors as anomaly score.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        threshold_percentile: float = 95.0,
        metric: str = 'euclidean',
    ):
        """Initialize k-NN detector.
        
        Args:
            n_neighbors: Number of neighbors to consider
            threshold_percentile: Percentile for threshold
            metric: Distance metric ('euclidean', 'manhattan', etc.)
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.threshold_percentile = threshold_percentile
        self.metric = metric
        self.nn_ = None
        self.X_train_ = None
    
    def fit(self, X: np.ndarray) -> 'KNNDetector':
        """Fit k-NN detector on normal data.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            
        Returns:
            self
        """
        logger.info(f"Fitting KNNDetector on {X.shape[0]} samples with k={self.n_neighbors}")
        
        # Store training data
        self.X_train_ = X.copy()
        
        # Fit nearest neighbors
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn_.fit(X)
        
        # Mark as fitted before computing threshold
        self.is_fitted = True
        
        # Compute threshold based on training scores
        train_scores = self.score(X)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)
        
        logger.info(f"KNNDetector fitted, threshold: {self.threshold:.4f}")
        
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute k-NN distances as anomaly scores.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Average k-NN distances (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Find k nearest neighbors and compute average distance
        distances, _ = self.nn_.kneighbors(X)
        scores = np.mean(distances, axis=1)
        
        return scores


class IsolationForestDetector(BaseEmbeddingDetector):
    """Isolation Forest-based anomaly detector.
    
    Uses tree-based outlier detection with contamination parameter tuning.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: Optional[int] = 42,
    ):
        """Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random state for reproducibility
        """
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_ = None
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit Isolation Forest detector on normal data.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            
        Returns:
            self
        """
        logger.info(
            f"Fitting IsolationForestDetector on {X.shape[0]} samples "
            f"with contamination={self.contamination}"
        )
        
        # Create and fit Isolation Forest
        self.model_ = SKIsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        
        # Mark as fitted before computing threshold
        self.is_fitted = True
        
        # Compute threshold (Isolation Forest decision function is negative for outliers)
        train_scores = self.score(X)
        self.threshold = np.percentile(train_scores, 100 - self.contamination * 100)
        
        logger.info(f"IsolationForestDetector fitted, threshold: {self.threshold:.4f}")
        
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute Isolation Forest anomaly scores.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Anomaly scores (n_samples,), higher is more anomalous
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Get anomaly scores (negative decision function)
        # Sklearn's decision_function returns negative scores for outliers
        # We negate it so higher scores = more anomalous
        scores = -self.model_.decision_function(X)
        
        return scores


class EnsembleDetector(BaseEmbeddingDetector):
    """Ensemble detector combining multiple methods.
    
    Uses weighted combination of multiple detectors with automatic score normalization.
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        weights: Optional[List[float]] = None,
        threshold_percentile: float = 95.0,
    ):
        """Initialize ensemble detector.
        
        Args:
            methods: List of method names ('mahalanobis', 'knn', 'isolation_forest')
            weights: Weights for each method (must sum to 1.0)
            threshold_percentile: Percentile for ensemble threshold
        """
        super().__init__()
        
        if methods is None:
            methods = ['mahalanobis', 'knn', 'isolation_forest']
        
        self.methods = methods
        self.threshold_percentile = threshold_percentile
        
        # Set default weights if not provided
        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)
        
        if len(weights) != len(methods):
            raise ValueError("Number of weights must match number of methods")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        self.weights = weights
        self.detectors_ = []
        self.normalizers_ = []  # Store mean and std for score normalization
    
    def fit(self, X: np.ndarray) -> 'EnsembleDetector':
        """Fit ensemble detector on normal data.
        
        Args:
            X: Training embeddings (n_samples, n_features)
            
        Returns:
            self
        """
        logger.info(f"Fitting EnsembleDetector with methods: {self.methods}")
        
        # Create and fit each detector
        self.detectors_ = []
        self.normalizers_ = []
        
        for method in self.methods:
            if method == 'mahalanobis':
                detector = MahalanobisDetector()
            elif method == 'knn':
                detector = KNNDetector()
            elif method == 'isolation_forest':
                detector = IsolationForestDetector()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Fit detector
            detector.fit(X)
            self.detectors_.append(detector)
            
            # Compute normalization parameters on training data
            train_scores = detector.score(X)
            mean = np.mean(train_scores)
            std = np.std(train_scores)
            if std < 1e-8:
                std = 1.0  # Avoid division by zero
            self.normalizers_.append((mean, std))
            
            logger.info(f"  {method}: threshold={detector.threshold:.4f}")
        
        # Mark as fitted before computing threshold
        self.is_fitted = True
        
        # Compute ensemble threshold
        train_scores = self.score(X)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)
        
        logger.info(f"EnsembleDetector fitted, ensemble threshold: {self.threshold:.4f}")
        
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble anomaly scores.
        
        Args:
            X: Input embeddings (n_samples, n_features)
            
        Returns:
            Ensemble scores (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before scoring")
        
        # Get scores from each detector and normalize
        ensemble_scores = np.zeros(X.shape[0])
        
        for i, detector in enumerate(self.detectors_):
            # Get raw scores
            scores = detector.score(X)
            
            # Normalize scores (z-score normalization)
            mean, std = self.normalizers_[i]
            normalized_scores = (scores - mean) / std
            
            # Add weighted contribution
            ensemble_scores += self.weights[i] * normalized_scores
        
        return ensemble_scores


def create_detector(
    method: str,
    **kwargs
) -> BaseEmbeddingDetector:
    """Factory function to create anomaly detector.
    
    Args:
        method: Detector method ('mahalanobis', 'knn', 'isolation_forest', 'ensemble')
        **kwargs: Additional arguments passed to detector constructor
        
    Returns:
        Detector instance
        
    Examples:
        >>> detector = create_detector('mahalanobis')
        >>> detector = create_detector('knn', n_neighbors=10)
        >>> detector = create_detector('ensemble', methods=['mahalanobis', 'knn'])
    """
    if method == 'mahalanobis':
        return MahalanobisDetector(**kwargs)
    elif method == 'knn':
        return KNNDetector(**kwargs)
    elif method == 'isolation_forest':
        return IsolationForestDetector(**kwargs)
    elif method == 'ensemble':
        return EnsembleDetector(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: mahalanobis, knn, isolation_forest, ensemble")

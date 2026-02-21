"""Unsupervised anomaly detection methods for audio anomaly detection.

This module implements three production-ready unsupervised anomaly detection methods:
- Local Outlier Factor (LOF): Density-based anomaly detection
- Isolation Forest: Tree-based outlier detection
- Elliptic Envelope: Robust covariance estimation
"""

import numpy as np
import joblib
from typing import Optional
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from .logger import get_logger

logger = get_logger(__name__)


class BaseUnsupervisedAnomalyDetector:
    """Base class for unsupervised anomaly detectors.

    Provides common interface for all unsupervised methods:
    - fit() on normal data only
    - predict() returns 1 for anomaly, 0 for normal
    - anomaly_score() returns anomaly scores
    - save/load functionality
    """

    def __init__(self, contamination: float = 0.1):
        """Initialize base detector.

        Args:
            contamination: Expected proportion of outliers in the dataset (0.0 to 0.5)
        """
        self.contamination = contamination
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "BaseUnsupervisedAnomalyDetector":
        """Fit the model on normal data only.

        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)

        Returns:
            self: Fitted estimator
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies.

        Args:
            X: Data to predict, shape (n_samples, n_features)

        Returns:
            predictions: Array of shape (n_samples,) with 1 for anomaly, 0 for normal
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # sklearn returns -1 for outliers, 1 for inliers
        # We convert to 1 for anomaly, 0 for normal
        raw_predictions = self.model.predict(X)
        return np.where(raw_predictions == -1, 1, 0)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Higher scores indicate higher anomaly likelihood.

        Args:
            X: Data to score, shape (n_samples, n_features)

        Returns:
            scores: Array of shape (n_samples,) with anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        # Get decision function (higher = more normal, lower = more anomalous)
        # We negate it so higher = more anomalous
        return -self.model.decision_function(X)

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> "BaseUnsupervisedAnomalyDetector":
        """Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            self: Loaded estimator
        """
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.contamination = model_data["contamination"]
        self.is_fitted = model_data["is_fitted"]
        logger.info(f"Model loaded from {path}")
        return self


class LocalOutlierFactorAnomalyDetector(BaseUnsupervisedAnomalyDetector):
    """Local Outlier Factor (LOF) anomaly detector.

    Density-based anomaly detection using local outlier factor.
    Best performing method with AUC 0.755+ on DCASE 2020 Task 2.

    LOF measures the local density deviation of a given data point with respect
    to its neighbors. Points with substantially lower density than neighbors
    are considered outliers.

    Args:
        n_neighbors: Number of neighbors to use (default: 20)
        contamination: Expected proportion of outliers (default: 0.1)
        metric: Distance metric to use (default: 'minkowski')
        p: Parameter for Minkowski metric (default: 2, Euclidean distance)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        metric: str = "minkowski",
        p: int = 2,
    ):
        super().__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X: np.ndarray) -> "LocalOutlierFactorAnomalyDetector":
        """Fit LOF model on normal data.

        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)

        Returns:
            self: Fitted estimator
        """
        logger.info(
            f"Fitting LOF with n_neighbors={self.n_neighbors}, "
            f"contamination={self.contamination}"
        )

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            p=self.p,
            novelty=True,  # Required for predict on new data
        )
        self.model.fit(X)
        self.is_fitted = True

        logger.info(f"LOF fitted on {X.shape[0]} samples with {X.shape[1]} features")
        return self


class IsolationForestAnomalyDetector(BaseUnsupervisedAnomalyDetector):
    """Isolation Forest anomaly detector.

    Tree-based outlier detection using isolation forest.
    Good performance with AUC 0.687+ on DCASE 2020 Task 2.

    Isolation Forest isolates anomalies by randomly selecting a feature and
    then randomly selecting a split value. Anomalies are easier to isolate
    (fewer splits required) than normal points.

    Args:
        n_estimators: Number of trees in the forest (default: 100)
        contamination: Expected proportion of outliers (default: 0.1)
        max_samples: Number of samples to draw for each tree (default: 'auto')
        random_state: Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        super().__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "IsolationForestAnomalyDetector":
        """Fit Isolation Forest model on normal data.

        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)

        Returns:
            self: Fitted estimator
        """
        logger.info(
            f"Fitting Isolation Forest with n_estimators={self.n_estimators}, "
            f"contamination={self.contamination}"
        )

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.is_fitted = True

        logger.info(
            f"Isolation Forest fitted on {X.shape[0]} samples with {X.shape[1]} features"
        )
        return self


class EllipticEnvelopeAnomalyDetector(BaseUnsupervisedAnomalyDetector):
    """Elliptic Envelope anomaly detector.

    Robust covariance estimation for outlier detection.
    Moderate performance with AUC 0.643+ on DCASE 2020 Task 2.

    Fits an ellipse (ellipsoid in higher dimensions) to the central data points,
    ignoring points outside the ellipse. Assumes data comes from a known distribution
    (typically Gaussian).

    Args:
        contamination: Expected proportion of outliers (default: 0.1)
        support_fraction: Proportion of points to include in support (default: None, auto)
        random_state: Random seed for reproducibility (default: 42)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        support_fraction: Optional[float] = None,
        random_state: int = 42,
    ):
        super().__init__(contamination=contamination)
        self.support_fraction = support_fraction
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "EllipticEnvelopeAnomalyDetector":
        """Fit Elliptic Envelope model on normal data.

        Args:
            X: Training data (normal samples only), shape (n_samples, n_features)

        Returns:
            self: Fitted estimator
        """
        logger.info(
            f"Fitting Elliptic Envelope with contamination={self.contamination}"
        )

        self.model = EllipticEnvelope(
            contamination=self.contamination,
            support_fraction=self.support_fraction,
            random_state=self.random_state,
        )
        self.model.fit(X)
        self.is_fitted = True

        logger.info(
            f"Elliptic Envelope fitted on {X.shape[0]} samples with {X.shape[1]} features"
        )
        return self


def create_detector(
    method: str, contamination: float = 0.1, **kwargs
) -> BaseUnsupervisedAnomalyDetector:
    """Factory function to create anomaly detectors.

    Args:
        method: Method name ('lof', 'isolation_forest', 'elliptic_envelope')
        contamination: Expected proportion of outliers
        **kwargs: Additional arguments for specific methods

    Returns:
        detector: Anomaly detector instance

    Example:
        >>> detector = create_detector('lof', contamination=0.1, n_neighbors=20)
        >>> detector.fit(X_train_normal)
        >>> predictions = detector.predict(X_test)
    """
    method = method.lower()

    if method == "lof":
        return LocalOutlierFactorAnomalyDetector(contamination=contamination, **kwargs)
    elif method == "isolation_forest" or method == "iforest":
        return IsolationForestAnomalyDetector(contamination=contamination, **kwargs)
    elif method == "elliptic_envelope" or method == "envelope":
        return EllipticEnvelopeAnomalyDetector(contamination=contamination, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from 'lof', 'isolation_forest', 'elliptic_envelope'"
        )

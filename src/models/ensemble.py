"""Hybrid ensemble anomaly detector combining AST and classical features.

Supports multiple detection methods (GMM, OCSVM, XGBoost, Logistic Regression)
trained on 1723-dimensional hybrid feature vectors (768 AST + 955 classical).
"""

import logging
from typing import Optional

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("gmm", "ocsvm", "xgboost", "logistic_regression")
HYBRID_DIM = 1723  # 768 AST + 955 classical


class HybridAnomalyDetector:
    """Ensemble anomaly detector on hybrid AST + Classical features.

    Trains on 1723-dimensional feature vectors concatenating 768-dim AST
    embeddings and 955-dim classical audio features.

    Supported detection methods:
    - ``"gmm"``: Gaussian Mixture Model (recommended for Pump/Slider)
    - ``"ocsvm"``: One-Class SVM (recommended for Fan)
    - ``"xgboost"``: Gradient boosting with synthetic anomalies
    - ``"logistic_regression"``: With synthetic negatives

    Args:
        method: Detection algorithm to use.
        n_components: Number of GMM components (used only when method='gmm').
        random_state: Random seed for reproducibility.

    Example:
        >>> detector = HybridAnomalyDetector(method="gmm", n_components=16)
        >>> features = np.random.randn(100, 1723)
        >>> detector.fit(features)
        >>> scores = detector.score_samples(features)
        >>> assert scores.shape == (100,)
    """

    def __init__(
        self,
        method: str = "gmm",
        n_components: int = 16,
        random_state: int = 42,
    ):
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {SUPPORTED_METHODS}"
            )
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self._model = None
        self._is_fitted = False

    def _build_model(self):
        """Instantiate the anomaly detection model."""
        if self.method == "gmm":
            return GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                reg_covar=1e-3,
                random_state=self.random_state,
            )
        elif self.method == "ocsvm":
            return OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
        elif self.method == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError:
                raise ImportError(
                    "XGBoost is required. Install with: pip install xgboost"
                )
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        elif self.method == "logistic_regression":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state
            )

    def fit(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> "HybridAnomalyDetector":
        """Train the detector on (normal) feature vectors.

        For GMM and OCSVM, only normal samples are used (unsupervised).
        For XGBoost and Logistic Regression, synthetic anomalies are generated
        if no labels are provided.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            labels: Optional ground-truth labels (0=normal, 1=anomaly).
                    If None and method is supervised, synthetic anomalies
                    are generated.

        Returns:
            self
        """
        logger.info(
            f"Training HybridAnomalyDetector (method={self.method}, "
            f"n_samples={features.shape[0]})"
        )

        X_scaled = self.scaler.fit_transform(features).astype(np.float64)
        self._model = self._build_model()

        if self.method in ("gmm", "ocsvm"):
            # Unsupervised: train on all provided features (assumed normal)
            self._model.fit(X_scaled)
        else:
            # Supervised: need labels
            if labels is None:
                # Generate synthetic anomalies by adding Gaussian noise
                rng = np.random.default_rng(self.random_state)
                noise_scale = np.std(X_scaled, axis=0) * 3.0
                n_synthetic = len(X_scaled)
                X_anomalies = X_scaled + rng.normal(
                    scale=noise_scale, size=(n_synthetic, X_scaled.shape[1])
                )
                X_combined = np.vstack([X_scaled, X_anomalies])
                y_combined = np.concatenate(
                    [np.zeros(len(X_scaled)), np.ones(n_synthetic)]
                )
            else:
                X_combined = X_scaled
                y_combined = labels

            self._model.fit(X_combined, y_combined)

        self._is_fitted = True
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous).

        Args:
            features: Feature matrix of shape (n_samples, n_features).

        Returns:
            Anomaly scores of shape (n_samples,).

        Raises:
            ValueError: If the detector has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "Detector must be fitted before scoring. Call fit() first."
            )

        X_scaled = self.scaler.transform(features).astype(np.float64)

        if self.method == "gmm":
            # Negate log-probability (higher = more anomalous)
            return -self._model.score_samples(X_scaled)
        elif self.method == "ocsvm":
            # Negate decision function (lower = more anomalous → negate)
            return -self._model.decision_function(X_scaled)
        elif self.method in ("xgboost", "logistic_regression"):
            # Probability of being anomalous (class 1)
            return self._model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str) -> None:
        """Persist the fitted detector to disk.

        Args:
            path: File path for the saved model (e.g. 'model.pkl').
        """
        if not self._is_fitted:
            raise ValueError("Cannot save an unfitted detector.")
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "HybridAnomalyDetector":
        """Load a previously saved detector.

        Args:
            path: File path of the saved model.

        Returns:
            Loaded HybridAnomalyDetector instance.
        """
        detector = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return detector

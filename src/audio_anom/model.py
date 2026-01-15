"""Anomaly detection model."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


class AnomalyDetector:
    """Anomaly detector for audio signals."""

    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in dataset
            random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fit the anomaly detector.

        Args:
            X: Feature matrix
            y: Labels (optional, used for semi-supervised learning)

        Returns:
            Self
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled)
        self.is_fitted = True

        return self

    def predict(self, X):
        """
        Predict anomalies.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions (1 for anomaly, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Convert from sklearn format (-1 for anomaly, 1 for normal)
        # to our format (1 for anomaly, 0 for normal)
        predictions = np.where(predictions == -1, 1, 0)

        return predictions

    def decision_function(self, X):
        """
        Compute anomaly scores.

        Args:
            X: Feature matrix

        Returns:
            Array of anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)

        return scores

    def save(self, model_path):
        """
        Save model to disk.

        Args:
            model_path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            "scaler": self.scaler,
            "model": self.model,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
        joblib.dump(model_data, model_path)

    def load(self, model_path):
        """
        Load model from disk.

        Args:
            model_path: Path to load model from

        Returns:
            Self
        """
        model_data = joblib.load(model_path)
        self.scaler = model_data["scaler"]
        self.model = model_data["model"]
        self.contamination = model_data["contamination"]
        self.random_state = model_data["random_state"]
        self.is_fitted = True

        return self

    def evaluate(self, X, y_true):
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y_true: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        y_pred = self.predict(X)
        scores = self.decision_function(X)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        try:
            metrics["auc"] = roc_auc_score(y_true, -scores)
        except ValueError:
            metrics["auc"] = 0.0

        return metrics

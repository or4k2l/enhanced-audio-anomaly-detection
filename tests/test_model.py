"""Unit tests for anomaly detection model."""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.model import AnomalyDetector  # noqa: E402


class TestAnomalyDetector:
    """Test suite for AnomalyDetector."""

    def test_initialization(self):
        """Test model initialization."""
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert not detector.is_fitted

    def test_fit(self):
        """Test model fitting."""
        detector = AnomalyDetector(contamination=0.1)

        # Create dummy data
        X = np.random.randn(100, 20)

        detector.fit(X)
        assert detector.is_fitted

    def test_predict(self):
        """Test prediction."""
        detector = AnomalyDetector(contamination=0.1)
        X_train = np.random.randn(100, 20)
        detector.fit(X_train)

        X_test = np.random.randn(10, 20)
        predictions = detector.predict(X_test)

        assert predictions.shape[0] == 10
        assert set(predictions).issubset({0, 1})

    def test_predict_unfitted(self):
        """Test prediction on unfitted model raises error."""
        detector = AnomalyDetector()
        X = np.random.randn(10, 20)

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(X)

    def test_decision_function(self):
        """Test anomaly score computation."""
        detector = AnomalyDetector(contamination=0.1)
        X_train = np.random.randn(100, 20)
        detector.fit(X_train)

        X_test = np.random.randn(10, 20)
        scores = detector.decision_function(X_test)

        assert scores.shape[0] == 10
        assert isinstance(scores, np.ndarray)

    def test_save_load(self):
        """Test model saving and loading."""
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        X_train = np.random.randn(100, 20)
        detector.fit(X_train)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            detector.save(model_path)

            # Load model
            detector2 = AnomalyDetector()
            detector2.load(model_path)

            assert detector2.is_fitted
            assert detector2.contamination == 0.1
            assert detector2.random_state == 42

            # Test predictions match
            X_test = np.random.randn(10, 20)
            pred1 = detector.predict(X_test)
            pred2 = detector2.predict(X_test)
            np.testing.assert_array_equal(pred1, pred2)
        finally:
            Path(model_path).unlink()

    def test_save_unfitted_raises(self):
        """Test saving unfitted model raises error."""
        detector = AnomalyDetector()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                detector.save(model_path)
        finally:
            Path(model_path).unlink(missing_ok=True)

    def test_evaluate(self):
        """Test model evaluation."""
        detector = AnomalyDetector(contamination=0.1)
        X_train = np.random.randn(100, 20)
        detector.fit(X_train)

        # Create test data with known labels
        X_test = np.random.randn(50, 20)
        y_test = np.random.randint(0, 2, 50)

        metrics = detector.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics

        # Check metric ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

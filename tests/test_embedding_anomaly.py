"""Unit tests for embedding-based anomaly detectors."""

import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.embedding_anomaly import (
    MahalanobisDetector,
    KNNDetector,
    IsolationForestDetector,
    EnsembleDetector,
    create_detector,
)


class TestMahalanobisDetector:
    """Test suite for MahalanobisDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = MahalanobisDetector(
            use_ledoit_wolf=True, regularization=1e-6, threshold_percentile=95.0
        )
        assert detector.use_ledoit_wolf is True
        assert detector.regularization == 1e-6
        assert detector.threshold_percentile == 95.0
        assert detector.is_fitted is False

    def test_fit(self):
        """Test fitting detector."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        detector = MahalanobisDetector()
        detector.fit(X)

        assert detector.is_fitted is True
        assert detector.mean_ is not None
        assert detector.cov_ is not None
        assert detector.inv_cov_ is not None
        assert detector.threshold is not None

    def test_score(self):
        """Test scoring."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = MahalanobisDetector()
        detector.fit(X_train)

        scores = detector.score(X_test)

        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()
        assert np.all(scores >= 0)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = MahalanobisDetector()
        detector.fit(X_train)

        predictions = detector.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_save_load(self):
        """Test saving and loading."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        detector = MahalanobisDetector()
        detector.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            detector.save(f.name)
            loaded = MahalanobisDetector.load(f.name)

        assert loaded.is_fitted is True
        assert np.allclose(loaded.mean_, detector.mean_)

        # Test predictions match
        X_test = np.random.randn(10, 256)
        assert np.allclose(loaded.score(X_test), detector.score(X_test))


class TestKNNDetector:
    """Test suite for KNNDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = KNNDetector(n_neighbors=5, threshold_percentile=95.0)
        assert detector.n_neighbors == 5
        assert detector.threshold_percentile == 95.0
        assert detector.is_fitted is False

    def test_fit(self):
        """Test fitting detector."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        detector = KNNDetector(n_neighbors=5)
        detector.fit(X)

        assert detector.is_fitted is True
        assert detector.nn_ is not None
        assert detector.X_train_ is not None
        assert detector.threshold is not None

    def test_score(self):
        """Test scoring."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = KNNDetector(n_neighbors=5)
        detector.fit(X_train)

        scores = detector.score(X_test)

        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()
        assert np.all(scores >= 0)

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = KNNDetector(n_neighbors=5)
        detector.fit(X_train)

        predictions = detector.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))


class TestIsolationForestDetector:
    """Test suite for IsolationForestDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(
            contamination=0.1, n_estimators=100, random_state=42
        )
        assert detector.contamination == 0.1
        assert detector.n_estimators == 100
        assert detector.random_state == 42
        assert detector.is_fitted is False

    def test_fit(self):
        """Test fitting detector."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X)

        assert detector.is_fitted is True
        assert detector.model_ is not None
        assert detector.threshold is not None

    def test_score(self):
        """Test scoring."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X_train)

        scores = detector.score(X_test)

        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = IsolationForestDetector(random_state=42)
        detector.fit(X_train)

        predictions = detector.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))


class TestEnsembleDetector:
    """Test suite for EnsembleDetector."""

    def test_initialization(self):
        """Test detector initialization."""
        methods = ["mahalanobis", "knn", "isolation_forest"]
        weights = [0.4, 0.35, 0.25]

        detector = EnsembleDetector(methods=methods, weights=weights)
        assert detector.methods == methods
        assert detector.weights == weights
        assert detector.is_fitted is False

    def test_initialization_default(self):
        """Test detector initialization with defaults."""
        detector = EnsembleDetector()
        assert len(detector.methods) == 3
        assert len(detector.weights) == 3
        assert abs(sum(detector.weights) - 1.0) < 1e-6

    def test_fit(self):
        """Test fitting detector."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        detector = EnsembleDetector()
        detector.fit(X)

        assert detector.is_fitted is True
        assert len(detector.detectors_) == 3
        assert len(detector.normalizers_) == 3
        assert detector.threshold is not None

    def test_score(self):
        """Test scoring."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = EnsembleDetector()
        detector.fit(X_train)

        scores = detector.score(X_test)

        assert scores.shape == (20,)
        assert not np.isnan(scores).any()
        assert not np.isinf(scores).any()

    def test_predict(self):
        """Test prediction."""
        np.random.seed(42)
        X_train = np.random.randn(100, 256)
        X_test = np.random.randn(20, 256)

        detector = EnsembleDetector()
        detector.fit(X_train)

        predictions = detector.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_invalid_weights(self):
        """Test initialization with invalid weights."""
        with pytest.raises(ValueError):
            EnsembleDetector(methods=["mahalanobis", "knn"], weights=[0.5])

        with pytest.raises(ValueError):
            EnsembleDetector(methods=["mahalanobis", "knn"], weights=[0.3, 0.3])


class TestCreateDetector:
    """Test suite for create_detector factory function."""

    def test_create_mahalanobis(self):
        """Test creating Mahalanobis detector."""
        detector = create_detector("mahalanobis")
        assert isinstance(detector, MahalanobisDetector)

    def test_create_knn(self):
        """Test creating k-NN detector."""
        detector = create_detector("knn", n_neighbors=10)
        assert isinstance(detector, KNNDetector)
        assert detector.n_neighbors == 10

    def test_create_isolation_forest(self):
        """Test creating Isolation Forest detector."""
        detector = create_detector("isolation_forest", contamination=0.05)
        assert isinstance(detector, IsolationForestDetector)
        assert detector.contamination == 0.05

    def test_create_ensemble(self):
        """Test creating ensemble detector."""
        detector = create_detector("ensemble")
        assert isinstance(detector, EnsembleDetector)

    def test_create_unknown(self):
        """Test creating unknown detector."""
        with pytest.raises(ValueError):
            create_detector("unknown_method")


class TestPredictBeforeFit:
    """Test error handling when predicting before fitting."""

    def test_mahalanobis_predict_before_fit(self):
        """Test Mahalanobis predict before fit."""
        detector = MahalanobisDetector()
        X = np.random.randn(10, 256)

        with pytest.raises(ValueError):
            detector.score(X)

        with pytest.raises(ValueError):
            detector.predict(X)

    def test_knn_predict_before_fit(self):
        """Test k-NN predict before fit."""
        detector = KNNDetector()
        X = np.random.randn(10, 256)

        with pytest.raises(ValueError):
            detector.score(X)

        with pytest.raises(ValueError):
            detector.predict(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

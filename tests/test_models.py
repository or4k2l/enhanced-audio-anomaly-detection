"""Tests for src/models/: CAE, AST extractor, classical features, and ensemble."""

import numpy as np
import pytest

from models.classical_features import (
    ClassicalFeatureExtractor,
    extract_classical_features,
)
from models.ensemble import HybridAnomalyDetector


class TestClassicalFeatures:
    """Tests for classical audio feature extraction."""

    def test_feature_shape(self):
        wave = np.random.randn(16000 * 10).astype(np.float32)
        features = extract_classical_features(wave)
        assert features.shape == (955,), f"Expected (955,), got {features.shape}"

    def test_no_nan_or_inf(self):
        wave = np.random.randn(16000 * 10).astype(np.float32)
        features = extract_classical_features(wave)
        assert not np.any(np.isnan(features)), "NaN found in features"
        assert not np.any(np.isinf(features)), "Inf found in features"

    def test_short_audio(self):
        wave = np.random.randn(8000).astype(np.float32)
        features = extract_classical_features(wave)
        assert features.shape == (955,)

    def test_silent_audio(self):
        wave = np.zeros(16000 * 5, dtype=np.float32)
        features = extract_classical_features(wave)
        assert features.shape == (955,)
        assert not np.any(np.isnan(features))

    def test_classical_feature_extractor_class(self):
        extractor = ClassicalFeatureExtractor(sr=16000)
        wave = np.random.randn(16000 * 5).astype(np.float32)
        features = extractor.transform(wave)
        assert features.shape == (955,)
        assert extractor.n_features == 955

    def test_batch_extraction(self):
        extractor = ClassicalFeatureExtractor(sr=16000)
        waves = [np.random.randn(16000 * 3).astype(np.float32) for _ in range(4)]
        result = extractor.transform_batch(waves)
        assert result.shape == (4, 955)


class TestHybridAnomalyDetector:
    """Tests for the HybridAnomalyDetector ensemble."""

    def test_gmm_fit_and_score(self):
        detector = HybridAnomalyDetector(method="gmm", n_components=4)
        features = np.random.randn(50, 1723)
        detector.fit(features)
        scores = detector.score_samples(features)
        assert scores.shape == (50,)
        assert not np.any(np.isnan(scores))

    def test_ocsvm_fit_and_score(self):
        detector = HybridAnomalyDetector(method="ocsvm")
        features = np.random.randn(50, 1723)
        detector.fit(features)
        scores = detector.score_samples(features)
        assert scores.shape == (50,)

    def test_xgboost_fit_and_score(self):
        detector = HybridAnomalyDetector(method="xgboost")
        features = np.random.randn(50, 1723)
        detector.fit(features)
        scores = detector.score_samples(features)
        assert scores.shape == (50,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_logistic_regression_fit_and_score(self):
        detector = HybridAnomalyDetector(method="logistic_regression")
        features = np.random.randn(50, 1723)
        detector.fit(features)
        scores = detector.score_samples(features)
        assert scores.shape == (50,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_score_before_fit_raises(self):
        detector = HybridAnomalyDetector(method="gmm")
        with pytest.raises(ValueError, match="fitted"):
            detector.score_samples(np.random.randn(10, 1723))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            HybridAnomalyDetector(method="unknown_method")

    def test_save_and_load(self, tmp_path):
        detector = HybridAnomalyDetector(method="gmm", n_components=4)
        features = np.random.randn(30, 1723)
        detector.fit(features)
        save_path = str(tmp_path / "test_model.pkl")
        detector.save(save_path)
        loaded = HybridAnomalyDetector.load(save_path)
        original_scores = detector.score_samples(features)
        loaded_scores = loaded.score_samples(features)
        np.testing.assert_allclose(original_scores, loaded_scores)

    def test_save_unfitted_raises(self, tmp_path):
        detector = HybridAnomalyDetector(method="gmm")
        with pytest.raises(ValueError, match="Cannot save"):
            detector.save(str(tmp_path / "model.pkl"))

    def test_smaller_feature_dim(self):
        """Detector should work with arbitrary input dimensions."""
        detector = HybridAnomalyDetector(method="gmm", n_components=2)
        features = np.random.randn(30, 100)
        detector.fit(features)
        scores = detector.score_samples(features)
        assert scores.shape == (30,)

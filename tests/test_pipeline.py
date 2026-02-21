"""Integration tests for the full hybrid anomaly detection pipeline."""

import numpy as np

from models.classical_features import extract_classical_features
from models.ensemble import HybridAnomalyDetector
from evaluation.metrics import compute_auc, evaluate_detector
from data.preprocessing import AudioPreprocessor


class TestEndToEndPipeline:
    """Test the full feature extraction + training + scoring pipeline."""

    def _make_features(self, n: int = 30, dim: int = 1723) -> np.ndarray:
        return np.random.randn(n, dim).astype(np.float32)

    def test_fit_and_score_gmm(self):
        X = self._make_features(50)
        detector = HybridAnomalyDetector(method="gmm", n_components=4)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert scores.shape == (50,)
        assert np.isfinite(scores).all()

    def test_evaluation_auc(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=100)
        scores = rng.random(100)
        auc = compute_auc(y_true, scores)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_detector(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=80)
        scores = rng.random(80)
        result = evaluate_detector(y_true, scores)
        assert 0.0 <= result.auc <= 1.0
        assert 0.0 <= result.average_precision <= 1.0
        assert result.fpr is not None
        assert result.tpr is not None
        assert result.confusion is not None
        assert result.confusion.shape == (2, 2)

    def test_classical_features_pipeline(self):
        wave = np.random.randn(16000 * 5).astype(np.float32)
        features = extract_classical_features(wave)
        assert features.shape == (955,)

    def test_full_pipeline_with_save_load(self, tmp_path):
        X = self._make_features(40)
        detector = HybridAnomalyDetector(method="gmm", n_components=2)
        detector.fit(X)

        save_path = str(tmp_path / "model.pkl")
        detector.save(save_path)

        loaded = HybridAnomalyDetector.load(save_path)
        scores_orig = detector.score_samples(X)
        scores_loaded = loaded.score_samples(X)
        np.testing.assert_allclose(scores_orig, scores_loaded)


class TestAudioPreprocessor:
    """Tests for the AudioPreprocessor."""

    def test_target_length(self):
        preprocessor = AudioPreprocessor(sr=16000, duration=5.0)
        assert preprocessor.target_length == 80000

    def test_pad_short_audio(self):
        preprocessor = AudioPreprocessor(sr=16000, duration=2.0)
        wave = np.random.randn(8000).astype(np.float32)
        processed = preprocessor.process(wave)
        assert len(processed) == 32000

    def test_trim_long_audio(self):
        preprocessor = AudioPreprocessor(sr=16000, duration=1.0)
        wave = np.random.randn(32000).astype(np.float32)
        processed = preprocessor.process(wave)
        assert len(processed) == 16000

    def test_normalize(self):
        preprocessor = AudioPreprocessor(sr=16000, normalize=True)
        wave = np.ones(16000, dtype=np.float32) * 5.0
        processed = preprocessor.process(wave)
        assert np.max(np.abs(processed)) <= 1.0 + 1e-6

    def test_no_normalize(self):
        preprocessor = AudioPreprocessor(sr=16000, normalize=False, duration=None)
        wave = np.ones(16000, dtype=np.float32) * 5.0
        processed = preprocessor.process(wave)
        assert np.allclose(processed, 5.0)

    def test_segment(self):
        preprocessor = AudioPreprocessor(sr=16000, duration=None)
        wave = np.random.randn(16000 * 3).astype(np.float32)
        segments, starts = preprocessor.segment(
            wave, segment_duration=1.0, hop_duration=0.5
        )
        assert len(segments) == len(starts)
        assert all(len(s) == 16000 for s in segments)

    def test_batch_process(self):
        preprocessor = AudioPreprocessor(sr=16000, duration=2.0)
        waves = [np.random.randn(10000).astype(np.float32) for _ in range(3)]
        results = preprocessor.process_batch(waves)
        assert len(results) == 3
        assert all(len(r) == 32000 for r in results)

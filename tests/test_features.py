"""Unit tests for audio feature extraction."""

import numpy as np
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.features import AudioFeatureExtractor  # noqa: E402


class TestAudioFeatureExtractor:
    """Test suite for AudioFeatureExtractor."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = AudioFeatureExtractor(sr=16000, n_mels=128)
        assert extractor.sr == 16000
        assert extractor.n_mels == 128
        assert extractor.n_fft == 1024
        assert extractor.hop_length == 512

    def test_extract_mel_spectrogram(self):
        """Test mel spectrogram extraction."""
        extractor = AudioFeatureExtractor(sr=16000, n_mels=64)
        # Generate dummy audio
        audio = np.random.randn(16000)  # 1 second

        mel_spec = extractor.extract_mel_spectrogram(audio)

        assert mel_spec.shape[0] == 64  # n_mels
        assert mel_spec.ndim == 2
        assert not np.isnan(mel_spec).any()

    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        extractor = AudioFeatureExtractor(sr=16000)
        audio = np.random.randn(16000)

        mfcc = extractor.extract_mfcc(audio, n_mfcc=13)

        assert mfcc.shape[0] == 13
        assert mfcc.ndim == 2
        assert not np.isnan(mfcc).any()

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        extractor = AudioFeatureExtractor()
        audio = np.random.randn(16000)

        stats = extractor.extract_statistical_features(audio)

        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert "min" in stats
        assert "rms" in stats
        assert "zcr" in stats

        # Check values are reasonable
        assert isinstance(stats["mean"], (int, float, np.number))
        assert stats["std"] >= 0
        assert stats["rms"] >= 0
        assert stats["zcr"] >= 0

    def test_extract_features(self):
        """Test comprehensive feature extraction."""
        extractor = AudioFeatureExtractor(sr=16000, n_mels=128)
        audio = np.random.randn(16000)

        features = extractor.extract_features(audio)

        # Check all feature types are present
        assert "mel_spec" in features
        assert "mfcc" in features
        assert "mel_spec_mean" in features
        assert "mel_spec_std" in features
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert "mean" in features
        assert "std" in features

        # Check shapes
        assert features["mel_spec_mean"].shape[0] == 128
        assert features["mfcc_mean"].shape[0] == 13

    def test_empty_audio(self):
        """Test handling of empty audio."""
        extractor = AudioFeatureExtractor()
        audio = np.zeros(16000)

        features = extractor.extract_features(audio)

        # Should still return valid features, even if zeros
        assert "mel_spec_mean" in features
        assert not np.isnan(features["mel_spec_mean"]).any()

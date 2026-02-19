"""Unit tests for RobustFeatureExtractor."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.feature_extractor import RobustFeatureExtractor


class TestRobustFeatureExtractor:
    """Test suite for RobustFeatureExtractor."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = RobustFeatureExtractor(sr=16000, n_mels=64, n_mfcc=13)
        assert extractor.sr == 16000
        assert extractor.n_mels == 64
        assert extractor.n_mfcc == 13
        assert extractor.n_fft == 2048
        assert extractor.hop_length == 512
        assert extractor.normalize is True
    
    def test_extract_features_empty_audio(self):
        """Test with empty audio."""
        extractor = RobustFeatureExtractor()
        audio = np.array([])
        features = extractor.extract_features(audio)
        assert features is None
    
    def test_extract_features_short_audio(self):
        """Test with very short audio."""
        extractor = RobustFeatureExtractor()
        audio = np.random.randn(100)
        features = extractor.extract_features(audio)
        assert features is None
    
    def test_extract_features_normal_audio(self):
        """Test with normal audio."""
        extractor = RobustFeatureExtractor(sr=22050)
        # Generate 1 second of audio
        audio = np.random.randn(22050)
        features = extractor.extract_features(audio)
        
        assert features is not None
        assert features.shape == (256,)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_extract_mel_spectrogram(self):
        """Test mel-spectrogram extraction."""
        extractor = RobustFeatureExtractor(sr=22050, n_mels=128)
        audio = np.random.randn(22050)
        
        mel_spec = extractor.extract_mel_spectrogram(audio)
        
        assert mel_spec is not None
        assert mel_spec.shape[0] == 128
        assert mel_spec.ndim == 2
        assert not np.isnan(mel_spec).any()
    
    def test_extract_mfcc(self):
        """Test MFCC extraction."""
        extractor = RobustFeatureExtractor(sr=22050, n_mfcc=13)
        audio = np.random.randn(22050)
        
        mfcc = extractor.extract_mfcc(audio)
        
        assert mfcc is not None
        assert mfcc.shape[0] == 13
        assert mfcc.ndim == 2
        assert not np.isnan(mfcc).any()
    
    def test_extract_spectral_features(self):
        """Test spectral features extraction."""
        extractor = RobustFeatureExtractor(sr=22050)
        audio = np.random.randn(22050)
        
        spectral = extractor.extract_spectral_features(audio)
        
        assert spectral is not None
        assert 'centroid' in spectral
        assert 'rolloff' in spectral
        assert 'zcr' in spectral
        assert not np.isnan(spectral['centroid']).any()
        assert not np.isnan(spectral['rolloff']).any()
        assert not np.isnan(spectral['zcr']).any()
    
    def test_extract_temporal_features(self):
        """Test temporal features extraction."""
        extractor = RobustFeatureExtractor(sr=22050)
        audio = np.random.randn(22050)
        
        temporal = extractor.extract_temporal_features(audio)
        
        assert temporal is not None
        assert 'rms_mean' in temporal
        assert 'rms_std' in temporal
        assert 'audio_mean' in temporal
        assert 'audio_std' in temporal
        assert 'audio_max' in temporal
        
        for value in temporal.values():
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_safe_normalize(self):
        """Test audio normalization."""
        extractor = RobustFeatureExtractor()
        
        # Normal audio
        audio = np.array([1.0, 2.0, -3.0, 0.5])
        normalized = extractor._safe_normalize(audio)
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6
        
        # Silent audio
        audio = np.array([0.0, 0.0, 0.0])
        normalized = extractor._safe_normalize(audio)
        assert np.all(normalized == 0.0)
        
        # Empty audio
        audio = np.array([])
        normalized = extractor._safe_normalize(audio)
        assert len(normalized) == 0
    
    def test_handle_nans(self):
        """Test NaN handling."""
        extractor = RobustFeatureExtractor()
        
        # Array with NaNs
        features = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
        cleaned = extractor._handle_nans(features, "test")
        
        assert not np.isnan(cleaned).any()
        assert not np.isinf(cleaned).any()
    
    def test_extract_features_batch(self):
        """Test batch feature extraction."""
        extractor = RobustFeatureExtractor(sr=22050)
        
        # Generate batch of audio
        audio_list = [np.random.randn(22050) for _ in range(5)]
        
        features = extractor.extract_features_batch(audio_list)
        
        assert features.shape == (5, 256)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()
    
    def test_extract_features_with_resampling(self):
        """Test feature extraction with resampling."""
        extractor = RobustFeatureExtractor(sr=22050)
        
        # Generate audio at different sample rate
        audio = np.random.randn(16000)
        features = extractor.extract_features(audio, target_sr=16000)
        
        assert features is not None
        assert features.shape == (256,)
        assert not np.isnan(features).any()
    
    def test_feature_dimension_consistency(self):
        """Test that features always have 256 dimensions."""
        extractor = RobustFeatureExtractor(sr=22050)
        
        # Test with different length audio
        for duration in [0.5, 1.0, 2.0]:
            audio = np.random.randn(int(22050 * duration))
            features = extractor.extract_features(audio)
            
            if features is not None:
                assert features.shape == (256,), f"Duration {duration}s gave wrong shape"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

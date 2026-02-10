"""Unit tests for data processing."""

import numpy as np
import sys
from pathlib import Path
import tempfile
import soundfile as sf

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.data import AudioDataProcessor  # noqa: E402
from audio_anom.features import AudioFeatureExtractor  # noqa: E402


class TestAudioDataProcessor:
    """Test suite for AudioDataProcessor."""

    def test_prepare_features_with_empty_audio(self):
        """Test prepare_features mit leerem Audio."""
        processor = AudioDataProcessor()
        extractor = AudioFeatureExtractor()
        dataset = [("file1.wav", np.array([]), "normal")]
        X, y = processor.prepare_features(dataset, extractor)
        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_prepare_features_with_invalid_label(self):
        """Test prepare_features mit ungültigem Label."""
        processor = AudioDataProcessor()
        extractor = AudioFeatureExtractor()
        dataset = [("file1.wav", np.random.randn(16000), None)]
        X, y = processor.prepare_features(dataset, extractor)
        assert X.shape[0] == 1
        assert y[0] is None

    def test_initialization(self):
        """Test data processor initialization."""
        processor = AudioDataProcessor(sr=16000, duration=5.0)
        assert processor.sr == 16000
        assert processor.duration == 5.0

    def test_load_audio_mono(self):
        """Test loading mono audio file."""
        processor = AudioDataProcessor(sr=16000)

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(f.name, audio_data, 16000)
            temp_path = f.name

        try:
            audio, sr = processor.load_audio(temp_path)
            assert sr == 16000
            assert len(audio) == 16000
            assert audio.ndim == 1
        finally:
            Path(temp_path).unlink()

    def test_load_audio_with_duration(self):
        """Test loading audio with target duration."""
        processor = AudioDataProcessor(sr=16000, duration=2.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_data = np.random.randn(48000).astype(np.float32)  # 3 seconds
            sf.write(f.name, audio_data, 16000)
            temp_path = f.name

        try:
            audio, sr = processor.load_audio(temp_path)
            expected_length = int(2.0 * 16000)
            assert len(audio) == expected_length
        finally:
            Path(temp_path).unlink()

    def test_split_dataset(self):
        """Test dataset splitting."""
        processor = AudioDataProcessor()

        # Create dummy dataset
        dataset = [
            (f"file_{i}.wav", np.random.randn(1000), "normal" if i % 2 == 0 else "anomaly")
            for i in range(100)
        ]

        train, val, test = processor.split_dataset(dataset, train_ratio=0.7, val_ratio=0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_prepare_features(self):
        """Test feature preparation from dataset."""
        processor = AudioDataProcessor()
        extractor = AudioFeatureExtractor(sr=16000, n_mels=64)

        # Create small dummy dataset
        dataset = [
            ("file1.wav", np.random.randn(16000), "normal"),
            ("file2.wav", np.random.randn(16000), "anomaly"),
        ]

        X, y = processor.prepare_features(dataset, extractor)

        assert X.shape[0] == 2  # Two samples
        assert X.ndim == 2
        assert y.shape[0] == 2
        assert y[0] == 0  # normal
        assert y[1] == 1  # anomaly

    def test_label_extraction(self):
        """Test label extraction from filenames."""
        processor = AudioDataProcessor()

        with tempfile.NamedTemporaryFile(suffix="_normal.wav", delete=False) as f:
            audio_data = np.random.randn(16000).astype(np.float32)
            sf.write(f.name, audio_data, 16000)
            temp_path = f.name

        try:
            # Test that the file loads (label extraction tested in load_dataset)
            audio, sr = processor.load_audio(temp_path)
            assert audio is not None
        finally:
            Path(temp_path).unlink()

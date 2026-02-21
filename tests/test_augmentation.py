"""Unit tests for audio augmentation."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom.augmentation import AudioAugmenter, create_augmenter


class TestAudioAugmenter:
    """Test suite for AudioAugmenter."""

    def test_initialization(self):
        """Test augmenter initialization."""
        augmenter = AudioAugmenter(
            sr=22050,
            use_mixup=True,
            use_specaug=True,
            use_timestretch=True,
            use_pitchshift=True,
            use_noise=True,
            mixup_alpha=0.2,
            random_state=42,
        )
        assert augmenter.sr == 22050
        assert augmenter.use_mixup is True
        assert augmenter.use_specaug is True
        assert augmenter.use_timestretch is True
        assert augmenter.use_pitchshift is True
        assert augmenter.use_noise is True
        assert augmenter.mixup_alpha == 0.2

    def test_mixup(self):
        """Test mixup augmentation."""
        np.random.seed(42)
        augmenter = AudioAugmenter(random_state=42)

        audio1 = np.random.randn(1000)
        audio2 = np.random.randn(1000)

        mixed = augmenter.mixup(audio1, audio2)

        assert len(mixed) == 1000
        assert not np.isnan(mixed).any()
        assert not np.isinf(mixed).any()
        assert not np.array_equal(mixed, audio1)
        assert not np.array_equal(mixed, audio2)

    def test_mixup_different_lengths(self):
        """Test mixup with different length audio."""
        augmenter = AudioAugmenter(random_state=42)

        audio1 = np.random.randn(1000)
        audio2 = np.random.randn(1500)

        mixed = augmenter.mixup(audio1, audio2)

        assert len(mixed) == 1500
        assert not np.isnan(mixed).any()

    def test_spec_augment(self):
        """Test SpecAugment."""
        np.random.seed(42)
        augmenter = AudioAugmenter(random_state=42)

        # Create dummy spectrogram
        spectrogram = np.random.randn(128, 100)

        augmented = augmenter.spec_augment(spectrogram)

        assert augmented.shape == spectrogram.shape
        assert not np.array_equal(augmented, spectrogram)
        # Check that some values are zeroed (masked)
        assert np.sum(augmented == 0) >= np.sum(spectrogram == 0)

    def test_time_stretch(self):
        """Test time stretching."""
        np.random.seed(42)
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio = np.random.randn(22050)  # 1 second

        stretched = augmenter.time_stretch(audio)

        assert not np.isnan(stretched).any()
        assert not np.isinf(stretched).any()
        # Length should change
        assert len(stretched) != len(audio)

    def test_pitch_shift(self):
        """Test pitch shifting."""
        np.random.seed(42)
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio = np.random.randn(22050)

        shifted = augmenter.pitch_shift(audio)

        assert not np.isnan(shifted).any()
        assert not np.isinf(shifted).any()
        assert len(shifted) == len(audio)

    def test_add_noise(self):
        """Test noise injection."""
        np.random.seed(42)
        augmenter = AudioAugmenter(random_state=42)

        audio = np.random.randn(1000)

        noisy = augmenter.add_noise(audio)

        assert len(noisy) == len(audio)
        assert not np.isnan(noisy).any()
        assert not np.isinf(noisy).any()
        assert not np.array_equal(noisy, audio)

    def test_augment(self):
        """Test combined augmentation."""
        np.random.seed(42)
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio = np.random.randn(22050)

        augmented = augmenter.augment(audio, apply_prob=1.0)

        assert not np.isnan(augmented).any()
        assert not np.isinf(augmented).any()

    def test_augment_spectrogram(self):
        """Test spectrogram augmentation."""
        np.random.seed(42)
        augmenter = AudioAugmenter(use_specaug=True, random_state=42)

        spectrogram = np.random.randn(128, 100)

        augmented = augmenter.augment_spectrogram(spectrogram, apply_prob=1.0)

        assert augmented.shape == spectrogram.shape
        assert not np.isnan(augmented).any()

    def test_augment_batch(self):
        """Test batch augmentation."""
        np.random.seed(42)
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio_list = [np.random.randn(22050) for _ in range(5)]

        augmented_list = augmenter.augment_batch(audio_list, apply_prob=1.0)

        assert len(augmented_list) == 5
        for augmented in augmented_list:
            assert not np.isnan(augmented).any()
            assert not np.isinf(augmented).any()

    def test_empty_audio_handling(self):
        """Test handling of empty audio."""
        augmenter = AudioAugmenter()

        audio = np.array([])

        # These should return empty audio without error
        assert len(augmenter.time_stretch(audio)) == 0
        assert len(augmenter.pitch_shift(audio)) == 0
        assert len(augmenter.add_noise(audio)) == 0

    def test_disable_augmentations(self):
        """Test with all augmentations disabled."""
        augmenter = AudioAugmenter(
            use_mixup=False,
            use_specaug=False,
            use_timestretch=False,
            use_pitchshift=False,
            use_noise=False,
        )

        audio = np.random.randn(1000)
        augmented = augmenter.augment(audio, apply_prob=1.0)

        # Should return original audio
        assert np.array_equal(augmented, audio)

    def test_create_augmenter(self):
        """Test factory function."""
        augmenter = create_augmenter(sr=16000, use_mixup=True)

        assert isinstance(augmenter, AudioAugmenter)
        assert augmenter.sr == 16000
        assert augmenter.use_mixup is True


class TestAugmentationRobustness:
    """Test augmentation robustness and error handling."""

    def test_mixup_with_custom_alpha(self):
        """Test mixup with custom alpha."""
        augmenter = AudioAugmenter(random_state=42)

        audio1 = np.random.randn(1000)
        audio2 = np.random.randn(1000)

        mixed = augmenter.mixup(audio1, audio2, alpha=0.5)

        assert not np.isnan(mixed).any()
        assert len(mixed) == 1000

    def test_spec_augment_custom_params(self):
        """Test SpecAugment with custom parameters."""
        augmenter = AudioAugmenter(random_state=42)

        spectrogram = np.random.randn(128, 100)

        augmented = augmenter.spec_augment(
            spectrogram,
            freq_mask_param=20,
            time_mask_param=30,
            n_freq_masks=2,
            n_time_masks=2,
        )

        assert augmented.shape == spectrogram.shape
        assert not np.isnan(augmented).any()

    def test_time_stretch_custom_range(self):
        """Test time stretching with custom rate range."""
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio = np.random.randn(22050)

        stretched = augmenter.time_stretch(audio, rate_range=(0.8, 1.2))

        assert not np.isnan(stretched).any()

    def test_pitch_shift_custom_range(self):
        """Test pitch shifting with custom step range."""
        augmenter = AudioAugmenter(sr=22050, random_state=42)

        audio = np.random.randn(22050)

        shifted = augmenter.pitch_shift(audio, n_steps_range=(-4, 4))

        assert not np.isnan(shifted).any()

    def test_add_noise_custom_range(self):
        """Test noise injection with custom factor range."""
        augmenter = AudioAugmenter(random_state=42)

        audio = np.random.randn(1000)

        noisy = augmenter.add_noise(audio, noise_factor_range=(0.01, 0.05))

        assert not np.isnan(noisy).any()
        assert len(noisy) == len(audio)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

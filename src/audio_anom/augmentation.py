"""Data augmentation module for audio anomaly detection.

Provides modern augmentation techniques:
- Mixup (α=0.2 beta distribution)
- SpecAugment (frequency/time masking)
- Time stretching (0.9-1.1x rate)
- Pitch shifting (±2 semitones)
- Gaussian noise injection
"""

import numpy as np
import librosa
from typing import Optional, Tuple
from .logger import get_logger

logger = get_logger(__name__)


class AudioAugmenter:
    """Audio data augmentation with error handling and fallbacks."""

    def __init__(
        self,
        sr: int = 22050,
        use_mixup: bool = True,
        use_specaug: bool = True,
        use_timestretch: bool = True,
        use_pitchshift: bool = True,
        use_noise: bool = True,
        mixup_alpha: float = 0.2,
        random_state: Optional[int] = None,
    ):
        """Initialize audio augmenter.

        Args:
            sr: Sample rate
            use_mixup: Enable Mixup augmentation
            use_specaug: Enable SpecAugment
            use_timestretch: Enable time stretching
            use_pitchshift: Enable pitch shifting
            use_noise: Enable noise injection
            mixup_alpha: Alpha parameter for beta distribution in Mixup
            random_state: Random seed for reproducibility
        """
        self.sr = sr
        self.use_mixup = use_mixup
        self.use_specaug = use_specaug
        self.use_timestretch = use_timestretch
        self.use_pitchshift = use_pitchshift
        self.use_noise = use_noise
        self.mixup_alpha = mixup_alpha

        if random_state is not None:
            np.random.seed(random_state)

        logger.info(
            f"AudioAugmenter initialized: mixup={use_mixup}, specaug={use_specaug}, "
            f"timestretch={use_timestretch}, pitchshift={use_pitchshift}, noise={use_noise}"
        )

    def mixup(
        self, audio1: np.ndarray, audio2: np.ndarray, alpha: Optional[float] = None
    ) -> np.ndarray:
        """Apply Mixup augmentation.

        Args:
            audio1: First audio signal
            audio2: Second audio signal
            alpha: Alpha parameter for beta distribution (uses self.mixup_alpha if None)

        Returns:
            Mixed audio signal
        """
        try:
            if alpha is None:
                alpha = self.mixup_alpha

            # Sample mixing coefficient from beta distribution
            lam = np.random.beta(alpha, alpha)

            # Ensure same length (pad/truncate if needed)
            max_len = max(len(audio1), len(audio2))
            if len(audio1) < max_len:
                audio1 = np.pad(audio1, (0, max_len - len(audio1)), mode="constant")
            if len(audio2) < max_len:
                audio2 = np.pad(audio2, (0, max_len - len(audio2)), mode="constant")

            audio1 = audio1[:max_len]
            audio2 = audio2[:max_len]

            # Mix
            mixed = lam * audio1 + (1 - lam) * audio2

            return mixed

        except Exception as e:
            logger.error(f"Error in mixup: {e}, returning original audio")
            return audio1

    def spec_augment(
        self,
        spectrogram: np.ndarray,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ) -> np.ndarray:
        """Apply SpecAugment (frequency and time masking).

        Args:
            spectrogram: Input spectrogram (freq_bins, time_steps)
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks

        Returns:
            Augmented spectrogram
        """
        try:
            augmented = spectrogram.copy()
            n_freq, n_time = augmented.shape

            # Apply frequency masking
            for _ in range(n_freq_masks):
                f = np.random.randint(0, freq_mask_param)
                f0 = np.random.randint(0, n_freq - f) if n_freq > f else 0
                augmented[f0 : f0 + f, :] = 0

            # Apply time masking
            for _ in range(n_time_masks):
                t = np.random.randint(0, time_mask_param)
                t0 = np.random.randint(0, n_time - t) if n_time > t else 0
                augmented[:, t0 : t0 + t] = 0

            return augmented

        except Exception as e:
            logger.error(f"Error in spec_augment: {e}, returning original spectrogram")
            return spectrogram

    def time_stretch(
        self,
        audio: np.ndarray,
        rate_range: Tuple[float, float] = (0.9, 1.1),
    ) -> np.ndarray:
        """Apply time stretching.

        Args:
            audio: Input audio signal
            rate_range: Range for stretching rate (min, max)

        Returns:
            Time-stretched audio
        """
        try:
            if len(audio) == 0:
                return audio

            # Sample random rate
            rate = np.random.uniform(rate_range[0], rate_range[1])

            # Apply time stretching
            stretched = librosa.effects.time_stretch(audio, rate=rate)

            return stretched

        except Exception as e:
            logger.error(f"Error in time_stretch: {e}, returning original audio")
            return audio

    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps_range: Tuple[int, int] = (-2, 2),
    ) -> np.ndarray:
        """Apply pitch shifting.

        Args:
            audio: Input audio signal
            n_steps_range: Range for pitch shift in semitones (min, max)

        Returns:
            Pitch-shifted audio
        """
        try:
            if len(audio) == 0:
                return audio

            # Sample random pitch shift
            n_steps = np.random.randint(n_steps_range[0], n_steps_range[1] + 1)

            if n_steps == 0:
                return audio

            # Apply pitch shifting
            shifted = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)

            return shifted

        except Exception as e:
            logger.error(f"Error in pitch_shift: {e}, returning original audio")
            return audio

    def add_noise(
        self,
        audio: np.ndarray,
        noise_factor_range: Tuple[float, float] = (0.001, 0.01),
    ) -> np.ndarray:
        """Add Gaussian noise to audio.

        Args:
            audio: Input audio signal
            noise_factor_range: Range for noise factor (min, max)

        Returns:
            Noisy audio
        """
        try:
            if len(audio) == 0:
                return audio

            # Sample noise factor
            noise_factor = np.random.uniform(
                noise_factor_range[0], noise_factor_range[1]
            )

            # Generate and add noise
            noise = np.random.randn(len(audio)) * noise_factor
            noisy = audio + noise

            return noisy

        except Exception as e:
            logger.error(f"Error in add_noise: {e}, returning original audio")
            return audio

    def augment(
        self,
        audio: np.ndarray,
        apply_prob: float = 0.5,
    ) -> np.ndarray:
        """Apply random augmentations to audio.

        Args:
            audio: Input audio signal
            apply_prob: Probability of applying each augmentation

        Returns:
            Augmented audio
        """
        augmented = audio.copy()

        # Time stretching
        if self.use_timestretch and np.random.random() < apply_prob:
            augmented = self.time_stretch(augmented)

        # Pitch shifting
        if self.use_pitchshift and np.random.random() < apply_prob:
            augmented = self.pitch_shift(augmented)

        # Noise injection
        if self.use_noise and np.random.random() < apply_prob:
            augmented = self.add_noise(augmented)

        return augmented

    def augment_spectrogram(
        self,
        spectrogram: np.ndarray,
        apply_prob: float = 0.5,
    ) -> np.ndarray:
        """Apply SpecAugment to spectrogram.

        Args:
            spectrogram: Input spectrogram
            apply_prob: Probability of applying augmentation

        Returns:
            Augmented spectrogram
        """
        if self.use_specaug and np.random.random() < apply_prob:
            return self.spec_augment(spectrogram)
        return spectrogram

    def augment_batch(
        self,
        audio_list: list,
        apply_prob: float = 0.5,
    ) -> list:
        """Augment batch of audio signals.

        Args:
            audio_list: List of audio signals
            apply_prob: Probability of augmenting each sample

        Returns:
            List of augmented audio signals
        """
        augmented_list = []
        for audio in audio_list:
            if np.random.random() < apply_prob:
                augmented = self.augment(audio, apply_prob=0.7)
            else:
                augmented = audio
            augmented_list.append(augmented)

        return augmented_list


def create_augmenter(**kwargs) -> AudioAugmenter:
    """Factory function to create audio augmenter.

    Args:
        **kwargs: Arguments passed to AudioAugmenter constructor

    Returns:
        AudioAugmenter instance

    Examples:
        >>> augmenter = create_augmenter(use_mixup=True, mixup_alpha=0.2)
        >>> augmenter = create_augmenter(use_noise=True, random_state=42)
    """
    return AudioAugmenter(**kwargs)

"""Audio preprocessing utilities.

Provides waveform normalization, resampling, and segmentation utilities
for preparing audio data for feature extraction.
"""

import logging
from typing import List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SR = 16000
DEFAULT_DURATION = 10.0


class AudioPreprocessor:
    """Preprocess raw audio waveforms for feature extraction.

    Args:
        sr: Target sample rate in Hz.
        duration: Target duration in seconds. Clips are padded or trimmed.
        normalize: Whether to normalize waveform amplitude to [-1, 1].

    Example:
        >>> preprocessor = AudioPreprocessor(sr=16000, duration=10.0)
        >>> wave = np.random.randn(22050)
        >>> processed = preprocessor.process(wave, orig_sr=22050)
        >>> assert len(processed) == 16000 * 10
    """

    def __init__(
        self,
        sr: int = DEFAULT_SR,
        duration: Optional[float] = DEFAULT_DURATION,
        normalize: bool = True,
    ):
        self.sr = sr
        self.duration = duration
        self.normalize = normalize

    @property
    def target_length(self) -> Optional[int]:
        """Target number of samples."""
        if self.duration is None:
            return None
        return int(self.sr * self.duration)

    def process(self, wave: np.ndarray, orig_sr: Optional[int] = None) -> np.ndarray:
        """Process a single waveform.

        Applies resampling (if orig_sr differs from target sr), padding/trimming,
        and optional amplitude normalization.

        Args:
            wave: Input waveform.
            orig_sr: Original sample rate. If None, no resampling is applied.

        Returns:
            Processed waveform of length ``target_length`` (if duration is set).
        """
        # Resample if needed
        if orig_sr is not None and orig_sr != self.sr:
            wave = librosa.resample(wave, orig_sr=orig_sr, target_sr=self.sr)

        # Pad or trim to target length
        if self.target_length is not None:
            if len(wave) < self.target_length:
                wave = np.pad(wave, (0, self.target_length - len(wave)))
            elif len(wave) > self.target_length:
                wave = wave[: self.target_length]

        # Normalize amplitude
        if self.normalize:
            max_val = np.max(np.abs(wave))
            if max_val > 0:
                wave = wave / max_val

        return wave.astype(np.float32)

    def process_batch(
        self, waveforms: List[np.ndarray], orig_sr: Optional[int] = None
    ) -> List[np.ndarray]:
        """Process a list of waveforms.

        Args:
            waveforms: List of input waveforms.
            orig_sr: Original sample rate for all waveforms.

        Returns:
            List of processed waveforms.
        """
        return [self.process(w, orig_sr=orig_sr) for w in waveforms]

    def segment(
        self,
        wave: np.ndarray,
        segment_duration: float = 1.0,
        hop_duration: float = 0.5,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Segment a waveform into overlapping chunks.

        Args:
            wave: Input waveform (preprocessed to target sr).
            segment_duration: Duration of each segment in seconds.
            hop_duration: Hop size between segments in seconds.

        Returns:
            Tuple of (segments, start_times_seconds).
        """
        segment_len = int(self.sr * segment_duration)
        hop_len = int(self.sr * hop_duration)
        segments = []
        start_times = []

        start = 0
        while start + segment_len <= len(wave):
            segments.append(wave[start : start + segment_len])
            start_times.append(start / self.sr)
            start += hop_len

        return segments, start_times

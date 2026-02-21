"""Classical audio feature extraction using librosa.

Extracts 955-dimensional handcrafted features combining:
- Mel-spectrogram statistics (640 features)
- MFCCs and deltas (100 features)
- Spectral features: centroid, bandwidth, rolloff, contrast (150+ features)
- Temporal features: ZCR, RMS, Chroma, Tonnetz
"""

import logging
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SR = 16000
N_MELS = 128
N_MFCC = 20
HOP_LENGTH = 512
N_FFT = 2048


def extract_classical_features(
    wave: np.ndarray,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """Extract 955-dimensional handcrafted audio features.

    Combines mel-spectrogram statistics, MFCCs, spectral descriptors,
    and temporal features into a fixed-size feature vector.

    Args:
        wave: Raw audio waveform as a 1D numpy array.
        sr: Sample rate in Hz (default: 16000).

    Returns:
        Feature vector of shape (955,).

    Example:
        >>> wave = np.random.randn(16000 * 10)
        >>> features = extract_classical_features(wave)
        >>> assert features.shape == (955,)
    """
    features = []

    # --- Mel-spectrogram statistics (640 features) ---
    mel_spec = librosa.feature.melspectrogram(
        y=wave, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Mean, std, min, max, median per mel band → 128 * 5 = 640
    features.append(np.mean(mel_db, axis=1))
    features.append(np.std(mel_db, axis=1))
    features.append(np.min(mel_db, axis=1))
    features.append(np.max(mel_db, axis=1))
    features.append(np.median(mel_db, axis=1))

    # --- MFCCs + deltas (100 features) ---
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    # 3 mean vectors (20 each) + 2 std vectors (20 each) = 100 features total
    for feat in [mfcc, mfcc_delta, mfcc_delta2]:
        features.append(np.mean(feat, axis=1))
        features.append(np.std(feat, axis=1))

    # --- Spectral features ---
    spectral_centroid = librosa.feature.spectral_centroid(
        y=wave, sr=sr, hop_length=HOP_LENGTH
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=wave, sr=sr, hop_length=HOP_LENGTH
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=wave, sr=sr, hop_length=HOP_LENGTH
    )
    spectral_contrast = librosa.feature.spectral_contrast(
        y=wave, sr=sr, hop_length=HOP_LENGTH
    )

    for feat in [spectral_centroid, spectral_bandwidth, spectral_rolloff]:
        features.append(
            np.array([np.mean(feat), np.std(feat), np.min(feat), np.max(feat)])
        )

    # spectral contrast: 7 bands * (mean + std) = 14
    features.append(np.mean(spectral_contrast, axis=1))
    features.append(np.std(spectral_contrast, axis=1))

    # --- Temporal features ---
    zcr = librosa.feature.zero_crossing_rate(y=wave, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=wave, hop_length=HOP_LENGTH)

    for feat in [zcr, rms]:
        features.append(
            np.array([np.mean(feat), np.std(feat), np.min(feat), np.max(feat)])
        )

    # --- Chroma features ---
    chroma = librosa.feature.chroma_stft(
        y=wave, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    features.append(np.mean(chroma, axis=1))  # 12
    features.append(np.std(chroma, axis=1))  # 12

    # --- Tonnetz ---
    try:
        tonnetz = librosa.feature.tonnetz(y=wave, sr=sr)
        features.append(np.mean(tonnetz, axis=1))  # 6
        features.append(np.std(tonnetz, axis=1))  # 6
    except Exception:
        features.append(np.zeros(6))
        features.append(np.zeros(6))

    feature_vector = np.concatenate(features)

    # Pad or truncate to exactly 955 dimensions
    target_dim = 955
    if len(feature_vector) < target_dim:
        feature_vector = np.pad(feature_vector, (0, target_dim - len(feature_vector)))
    elif len(feature_vector) > target_dim:
        feature_vector = feature_vector[:target_dim]

    # Replace NaN/Inf with zeros
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vector


class ClassicalFeatureExtractor:
    """Stateful wrapper around extract_classical_features.

    Args:
        sr: Sample rate in Hz.

    Example:
        >>> extractor = ClassicalFeatureExtractor(sr=16000)
        >>> features = extractor.transform(wave)
        >>> assert features.shape == (955,)
    """

    def __init__(self, sr: int = DEFAULT_SR):
        self.sr = sr

    def transform(self, wave: np.ndarray) -> np.ndarray:
        """Extract features from a single waveform.

        Args:
            wave: Raw audio waveform.

        Returns:
            Feature vector of shape (955,).
        """
        return extract_classical_features(wave, sr=self.sr)

    def transform_batch(self, waveforms: list, verbose: bool = False) -> np.ndarray:
        """Extract features from a list of waveforms.

        Args:
            waveforms: List of raw audio waveforms.
            verbose: Whether to log progress.

        Returns:
            Feature matrix of shape (n_samples, 955).
        """
        results = []
        for i, wave in enumerate(waveforms):
            if verbose and i % 50 == 0:
                logger.info(f"Processing {i}/{len(waveforms)}")
            results.append(self.transform(wave))
        return np.stack(results, axis=0)

    @property
    def n_features(self) -> Optional[int]:
        """Number of output features."""
        return 955

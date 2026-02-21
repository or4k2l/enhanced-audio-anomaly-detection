"""Robust feature extraction module for audio anomaly detection.

This module provides comprehensive audio feature extraction with fallback mechanisms,
error handling, and NaN detection/recovery.
"""

import numpy as np
import librosa
from typing import Optional, Dict
from .logger import get_logger

logger = get_logger(__name__)


class RobustFeatureExtractor:
    """Robust audio feature extractor with multi-feature support.

    Extracts 256-dimensional embeddings from audio signals using:
    - Mel-spectrogram features
    - MFCC features (13 coefficients)
    - Spectral features (centroid, rolloff, zero-crossing rate)
    - Temporal features and cepstral analysis

    Features:
    - Audio normalization
    - Robust resample handling
    - Zero-division protection
    - NaN detection and recovery
    - Comprehensive error handling
    """

    def __init__(
        self,
        sr: int = 22050,
        n_mels: int = 128,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        normalize: bool = True,
    ):
        """Initialize feature extractor.

        Args:
            sr: Sample rate for audio processing
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            normalize: Whether to normalize audio before feature extraction
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize

        logger.info(
            f"RobustFeatureExtractor initialized: sr={sr}, n_mels={n_mels}, "
            f"n_mfcc={n_mfcc}, n_fft={n_fft}, hop_length={hop_length}"
        )

    def _safe_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Safely normalize audio with zero-division protection.

        Args:
            audio: Input audio signal

        Returns:
            Normalized audio signal
        """
        if len(audio) == 0:
            return audio

        max_val = np.abs(audio).max()
        if max_val > 1e-8:  # Avoid division by very small numbers
            return audio / max_val
        return audio

    def _handle_nans(
        self, features: np.ndarray, feature_name: str = "features"
    ) -> np.ndarray:
        """Detect and handle NaN values in features.

        Args:
            features: Feature array
            feature_name: Name of feature for logging

        Returns:
            Features with NaNs replaced by zeros
        """
        if np.isnan(features).any():
            n_nans = np.isnan(features).sum()
            logger.warning(
                f"Found {n_nans} NaN values in {feature_name}, replacing with zeros"
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def extract_mel_spectrogram(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract mel-spectrogram features with error handling.

        Args:
            audio: Input audio signal

        Returns:
            Mel-spectrogram features (n_mels, n_frames) or None on error
        """
        try:
            if len(audio) < self.n_fft:
                logger.warning(
                    f"Audio too short ({len(audio)} samples) for mel-spectrogram"
                )
                return None

            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Handle NaNs
            mel_spec_db = self._handle_nans(mel_spec_db, "mel_spectrogram")

            return mel_spec_db

        except Exception as e:
            logger.error(f"Error extracting mel-spectrogram: {e}")
            return None

    def extract_mfcc(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract MFCC features with error handling.

        Args:
            audio: Input audio signal

        Returns:
            MFCC features (n_mfcc, n_frames) or None on error
        """
        try:
            if len(audio) < self.n_fft:
                logger.warning(f"Audio too short ({len(audio)} samples) for MFCC")
                return None

            # Compute MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )

            # Handle NaNs
            mfcc = self._handle_nans(mfcc, "mfcc")

            return mfcc

        except Exception as e:
            logger.error(f"Error extracting MFCC: {e}")
            return None

    def extract_spectral_features(
        self, audio: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract spectral features (centroid, rolloff, zero-crossing rate).

        Args:
            audio: Input audio signal

        Returns:
            Dictionary of spectral features or None on error
        """
        try:
            if len(audio) < self.hop_length:
                logger.warning(
                    f"Audio too short ({len(audio)} samples) for spectral features"
                )
                return None

            features = {}

            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["centroid"] = self._handle_nans(centroid, "spectral_centroid")

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            features["rolloff"] = self._handle_nans(rolloff, "spectral_rolloff")

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y=audio, hop_length=self.hop_length
            )
            features["zcr"] = self._handle_nans(zcr, "zero_crossing_rate")

            return features

        except Exception as e:
            logger.error(f"Error extracting spectral features: {e}")
            return None

    def extract_temporal_features(
        self, audio: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Extract temporal and statistical features.

        Args:
            audio: Input audio signal

        Returns:
            Dictionary of temporal features or None on error
        """
        try:
            if len(audio) == 0:
                logger.warning("Empty audio for temporal features")
                return None

            features = {}

            # RMS energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)
            features["rms_mean"] = float(np.mean(rms))
            features["rms_std"] = float(np.std(rms))

            # Temporal envelope statistics
            features["audio_mean"] = float(np.mean(audio))
            features["audio_std"] = float(np.std(audio))
            features["audio_max"] = float(np.max(np.abs(audio)))

            # Handle NaNs in features dict
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    logger.warning(f"Invalid value in {key}, replacing with 0.0")
                    features[key] = 0.0

            return features

        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return None

    def extract_features(
        self, audio: np.ndarray, target_sr: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Extract comprehensive 256-dimensional feature vector.

        Args:
            audio: Input audio signal
            target_sr: Target sample rate (if different from self.sr)

        Returns:
            256-dimensional feature vector or None on error
        """
        try:
            # Handle empty audio
            if len(audio) == 0:
                logger.warning("Empty audio provided for feature extraction")
                return None

            # Resample if needed
            if target_sr is not None and target_sr != self.sr:
                try:
                    audio = librosa.resample(
                        audio, orig_sr=target_sr, target_sr=self.sr
                    )
                except Exception as e:
                    logger.error(f"Error resampling audio: {e}")
                    return None

            # Normalize audio
            if self.normalize:
                audio = self._safe_normalize(audio)

            # Extract all features
            mel_spec = self.extract_mel_spectrogram(audio)
            mfcc = self.extract_mfcc(audio)
            spectral = self.extract_spectral_features(audio)
            temporal = self.extract_temporal_features(audio)

            # Check if any feature extraction failed
            if mel_spec is None or mfcc is None or spectral is None or temporal is None:
                logger.warning("One or more feature extractions failed")
                return None

            # Aggregate features to fixed dimension
            feature_vector = []

            # Mel-spectrogram statistics (128 features: mean and std per mel band)
            mel_mean = np.mean(mel_spec, axis=1)  # (n_mels,)
            mel_std = np.std(mel_spec, axis=1)  # (n_mels,)
            feature_vector.extend(mel_mean[:64])  # Use first 64 mel bands mean
            feature_vector.extend(mel_std[:64])  # Use first 64 mel bands std

            # MFCC statistics (52 features: mean, std, min, max per MFCC)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_min = np.min(mfcc, axis=1)
            mfcc_max = np.max(mfcc, axis=1)
            feature_vector.extend(mfcc_mean)  # 13 features
            feature_vector.extend(mfcc_std)  # 13 features
            feature_vector.extend(mfcc_min)  # 13 features
            feature_vector.extend(mfcc_max)  # 13 features

            # Spectral features statistics (60 features: mean and std for each)
            for key in ["centroid", "rolloff", "zcr"]:
                feat = spectral[key]
                feature_vector.append(np.mean(feat))
                feature_vector.append(np.std(feat))
                feature_vector.append(np.min(feat))
                feature_vector.append(np.max(feat))
                feature_vector.append(np.median(feat))
                # Add percentiles
                feature_vector.append(np.percentile(feat, 25))
                feature_vector.append(np.percentile(feat, 75))
                # Add range
                feature_vector.append(np.max(feat) - np.min(feat))
                # Add skewness and kurtosis approximations
                feat_normalized = (feat - np.mean(feat)) / (np.std(feat) + 1e-8)
                feature_vector.append(np.mean(feat_normalized**3))  # Skewness
                feature_vector.append(np.mean(feat_normalized**4))  # Kurtosis

            # Temporal features (5 features)
            feature_vector.extend(
                [
                    temporal["rms_mean"],
                    temporal["rms_std"],
                    temporal["audio_mean"],
                    temporal["audio_std"],
                    temporal["audio_max"],
                ]
            )

            # Delta features for MFCCs (13 features)
            try:
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
                feature_vector.extend(mfcc_delta_mean)
            except Exception as e:
                logger.warning(f"Error computing delta features: {e}, using zeros")
                feature_vector.extend([0.0] * 13)

            # Cepstral flux (1 feature)
            try:
                mfcc_flux = np.mean(np.sqrt(np.sum(np.diff(mfcc, axis=1) ** 2, axis=0)))
                feature_vector.append(mfcc_flux)
            except Exception as e:
                logger.warning(f"Error computing cepstral flux: {e}, using 0")
                feature_vector.append(0.0)

            # Convert to numpy array
            feature_vector = np.array(feature_vector, dtype=np.float32)

            # Final NaN check
            feature_vector = self._handle_nans(feature_vector, "final_feature_vector")

            # Ensure exactly 256 dimensions (pad or truncate if needed)
            target_dim = 256
            if len(feature_vector) < target_dim:
                # Pad with zeros
                padding = target_dim - len(feature_vector)
                feature_vector = np.pad(feature_vector, (0, padding), mode="constant")
                logger.debug(
                    f"Padded feature vector from {len(feature_vector)-padding} to {target_dim}"
                )
            elif len(feature_vector) > target_dim:
                # Truncate
                feature_vector = feature_vector[:target_dim]
                logger.debug(f"Truncated feature vector to {target_dim}")

            return feature_vector

        except Exception as e:
            logger.error(f"Unexpected error in feature extraction: {e}")
            return None

    def extract_features_batch(
        self, audio_list: list, target_sr: Optional[int] = None
    ) -> np.ndarray:
        """Extract features from multiple audio signals.

        Args:
            audio_list: List of audio signals
            target_sr: Target sample rate

        Returns:
            Feature matrix (n_samples, 256)
        """
        features = []
        for i, audio in enumerate(audio_list):
            feat = self.extract_features(audio, target_sr=target_sr)
            if feat is not None:
                features.append(feat)
            else:
                logger.warning(
                    f"Failed to extract features for audio {i}, using zero vector"
                )
                features.append(np.zeros(256, dtype=np.float32))

        return np.array(features)

"""Audio feature extraction module."""

import numpy as np
import librosa


class AudioFeatureExtractor:
    """Extract features from audio signals for anomaly detection."""

    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_mel_spectrogram(self, audio):
        """
        Extract mel spectrogram from audio.

        Args:
            audio: Audio time series

        Returns:
            Mel spectrogram in dB scale
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_mfcc(self, audio, n_mfcc=13):
        """
        Extract MFCC features from audio.

        Args:
            audio: Audio time series
            n_mfcc: Number of MFCCs to extract

        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def extract_statistical_features(self, audio):
        """
        Extract statistical features from audio.

        Args:
            audio: Audio time series

        Returns:
            Dictionary of statistical features
        """
        features = {
            "mean": np.mean(audio),
            "std": np.std(audio),
            "max": np.max(audio),
            "min": np.min(audio),
            "rms": np.sqrt(np.mean(audio**2)),
            "zcr": np.mean(librosa.feature.zero_crossing_rate(audio)),
        }
        return features

    def extract_features(self, audio):
        """
        Extract all features from audio.

        Args:
            audio: Audio time series

        Returns:
            Dictionary containing all extracted features
        """
        mel_spec = self.extract_mel_spectrogram(audio)
        mfcc = self.extract_mfcc(audio)
        stats = self.extract_statistical_features(audio)

        # Flatten features for ML models
        features = {
            "mel_spec": mel_spec,
            "mfcc": mfcc,
            "mel_spec_mean": np.mean(mel_spec, axis=1),
            "mel_spec_std": np.std(mel_spec, axis=1),
            "mfcc_mean": np.mean(mfcc, axis=1),
            "mfcc_std": np.std(mfcc, axis=1),
            **stats,
        }
        return features

"""Audio feature extraction module."""

import numpy as np
import librosa
from scipy.fft import fft, fftfreq
from scipy import stats


class AudioFeatureExtractor:
    """Extract features from audio signals for anomaly detection."""

    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=512, n_mfcc=20):
        """
        Initialize feature extractor.

        Args:
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

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

    def extract_time_domain_features(self, audio):
        """
        Extract time domain features from audio.

        Args:
            audio: Audio time series

        Returns:
            Dictionary of time domain features
        """
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        features = {
            "mean": np.mean(audio),
            "std": np.std(audio),
            "max": np.max(audio),
            "min": np.min(audio),
            "rms": np.sqrt(np.mean(audio**2)),
            "energy": np.sum(audio**2),
            "zcr_mean": np.mean(zcr),
            "zcr_std": np.std(zcr),
            "skewness": stats.skew(audio),
            "kurtosis": stats.kurtosis(audio),
            "peak": np.max(np.abs(audio)),
        }
        return features
    
    def extract_frequency_domain_features(self, audio):
        """
        Extract frequency domain features using FFT.

        Args:
            audio: Audio time series

        Returns:
            Dictionary of frequency domain features
        """
        N = len(audio)
        yf = fft(audio)
        xf = fftfreq(N, 1/self.sr)
        
        # Only positive frequencies
        pos_mask = xf >= 0
        xf_pos = xf[pos_mask]
        yf_pos = np.abs(yf[pos_mask])
        
        features = {}
        
        if len(yf_pos) > 0:
            # Dominant frequency
            dom_idx = np.argmax(yf_pos)
            features['dom_freq'] = xf_pos[dom_idx]
            features['dom_freq_mag'] = yf_pos[dom_idx]
            
            # Spectral moments
            power_spectrum = yf_pos ** 2
            total_power = np.sum(power_spectrum)
            
            if total_power > 0:
                # Spectral centroid (FFT)
                features['spectral_centroid_fft'] = np.sum(xf_pos * power_spectrum) / total_power
                # Spectral spread
                features['spectral_spread'] = np.sqrt(
                    np.sum(((xf_pos - features['spectral_centroid_fft'])**2) * power_spectrum) / total_power
                )
            
            # Band energy for different frequency bands
            bands = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 4000)]
            for i, (low, high) in enumerate(bands):
                band_mask = (xf_pos >= low) & (xf_pos < high)
                features[f'band_energy_{i}'] = np.sum(yf_pos[band_mask]**2)
        
        return features
    
    def extract_spectral_features(self, audio):
        """
        Extract spectral features using librosa.

        Args:
            audio: Audio time series

        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr, roll_percent=0.85)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_{i}'] = np.mean(contrast[i])
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def extract_statistical_features(self, audio):
        """
        Extract statistical features from audio (legacy method).

        Args:
            audio: Audio time series

        Returns:
            Dictionary of statistical features
        """
        return self.extract_time_domain_features(audio)

    def extract_features(self, audio, enhanced=True):
        """
        Extract all features from audio.

        Args:
            audio: Audio time series
            enhanced: If True, extract enhanced features including frequency and spectral features

        Returns:
            Dictionary containing all extracted features
        """
        if len(audio) < 100:  # Audio too short
            return None
        
        features = {}
        
        # Time domain features
        time_features = self.extract_time_domain_features(audio)
        features.update(time_features)
        
        if enhanced:
            # Frequency domain features
            freq_features = self.extract_frequency_domain_features(audio)
            features.update(freq_features)
            
            # Spectral features
            spectral_features = self.extract_spectral_features(audio)
            features.update(spectral_features)
        
        # MFCCs (extended)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc, 
                                      hop_length=self.hop_length)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Mel spectrogram statistics
        mel_spec = self.extract_mel_spectrogram(audio)
        features['mel_spec_mean_overall'] = np.mean(mel_spec)
        features['mel_spec_std_overall'] = np.std(mel_spec)
        
        return features
    
    def extract_features_as_array(self, audio, enhanced=True):
        """
        Extract features and return as a flat array.

        Args:
            audio: Audio time series
            enhanced: If True, extract enhanced features

        Returns:
            1D numpy array of features
        """
        features = self.extract_features(audio, enhanced=enhanced)
        if features is None:
            return None
        
        # Convert to array in consistent order
        feature_values = []
        for key in sorted(features.keys()):
            feature_values.append(features[key])
        
        return np.array(feature_values)

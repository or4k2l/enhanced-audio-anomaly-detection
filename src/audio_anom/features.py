"""Audio feature extraction module."""

import numpy as np
import librosa
from scipy.fft import fft, fftfreq
from scipy import stats


from typing import Optional, Dict, Any


class AudioFeatureExtractor:
    """
    Extrahiert Audio-Features für Anomalieerkennung.

    Attribute:
        sr (int): Sample Rate
        n_mels (int): Anzahl Mel-Bänder
        n_fft (int): FFT-Fenstergröße
        hop_length (int): Hop-Länge für STFT
        n_mfcc (int): Anzahl MFCC-Koeffizienten
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mfcc: int = 20,
    ) -> None:
        """
        Initialisiert den Feature-Extractor.

        Args:
            sr (int): Sample Rate
            n_mels (int): Anzahl Mel-Bänder
            n_fft (int): FFT-Fenstergröße
            hop_length (int): Hop-Länge für STFT
            n_mfcc (int): Anzahl MFCC-Koeffizienten
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extrahiert das Mel-Spektrogramm aus einem Audiosignal.

        Args:
            audio (np.ndarray): Audiosignal (1D)

        Returns:
            np.ndarray: Mel-Spektrogramm in dB-Skala (n_mels x Frames)
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

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extrahiert MFCC-Features aus einem Audiosignal.

        Args:
            audio (np.ndarray): Audiosignal
            n_mfcc (int): Anzahl MFCCs

        Returns:
            np.ndarray: MFCC-Features (n_mfcc x Frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mfcc

    def extract_time_domain_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extrahiert Zeitbereichs-Features aus einem Audiosignal.

        Args:
            audio (np.ndarray): Audiosignal

        Returns:
            Dict[str, float]: Zeitbereichs-Features
        """
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features = {
            "mean": float(np.mean(audio)),
            "std": float(np.std(audio)),
            "max": float(np.max(audio)),
            "min": float(np.min(audio)),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "energy": float(np.sum(audio**2)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "skewness": float(stats.skew(audio)),
            "kurtosis": float(stats.kurtosis(audio)),
            "peak": float(np.max(np.abs(audio))),
        }
        return features

    def extract_frequency_domain_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extrahiert Frequenzbereichs-Features mittels FFT.

        Args:
            audio (np.ndarray): Audiosignal

        Returns:
            Dict[str, float]: Frequenzbereichs-Features
        """
        N = len(audio)
        yf = fft(audio)
        xf = fftfreq(N, 1 / self.sr)

        # Nur positive Frequenzen
        pos_mask = xf >= 0
        xf_pos = xf[pos_mask]
        yf_pos = np.abs(yf[pos_mask])

        features: Dict[str, float] = {}

        if len(yf_pos) > 0:
            # Dominante Frequenz
            dom_idx = int(np.argmax(yf_pos))
            features["dom_freq"] = float(xf_pos[dom_idx])
            features["dom_freq_mag"] = float(yf_pos[dom_idx])

            # Spektrale Momente
            power_spectrum = yf_pos**2
            total_power = float(np.sum(power_spectrum))

            if total_power > 0:
                # Spektrales Zentrum (FFT)
                features["spectral_centroid_fft"] = float(
                    np.sum(xf_pos * power_spectrum) / total_power
                )
                # Spektrale Streuung
                features["spectral_spread"] = float(
                    np.sqrt(
                        np.sum(
                            ((xf_pos - features["spectral_centroid_fft"]) ** 2)
                            * power_spectrum
                        )
                        / total_power
                    )
                )

            # Bandenergie für verschiedene Frequenzbänder
            bands = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 4000)]
            for i, (low, high) in enumerate(bands):
                band_mask = (xf_pos >= low) & (xf_pos < high)
                features[f"band_energy_{i}"] = float(np.sum(yf_pos[band_mask] ** 2))

        return features

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extrahiert spektrale Features mittels librosa.

        Args:
            audio (np.ndarray): Audiosignal

        Returns:
            Dict[str, float]: Spektrale Features
        """
        features: Dict[str, float] = {}

        # Spektrales Zentrum
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))

        # Spektraler Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, roll_percent=0.85
        )[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))

        # Spektrale Flachheit
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))

        # Spektrale Bandbreite
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

        # Spektraler Kontrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_{i}"] = float(np.mean(contrast[i]))

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        return features

    def extract_statistical_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extrahiert statistische Features (Legacy-Methode).

        Args:
            audio (np.ndarray): Audiosignal

        Returns:
            Dict[str, float]: Statistische Features
        """
        stats = self.extract_time_domain_features(audio)
        # Kompatibilität: zcr als Mittelwert
        stats["zcr"] = stats.get("zcr_mean", 0.0)
        return stats

    def extract_features(
        self, audio: np.ndarray, enhanced: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Extrahiert alle Features aus einem Audiosignal.

        Args:
            audio (np.ndarray): Audiosignal
            enhanced (bool): Wenn True, werden zusätzliche Frequenz- und Spektralfeatures extrahiert

        Returns:
            Optional[Dict[str, Any]]: Feature-Dictionary oder None bei zu kurzem Audio
        """
        if len(audio) < 100:  # Audio zu kurz
            return None

        features: Dict[str, Any] = {}

        # Zeitbereichsfeatures
        time_features = self.extract_time_domain_features(audio)
        features.update(time_features)
        features["zcr"] = time_features.get("zcr_mean", 0.0)

        if enhanced:
            # Frequenzbereichsfeatures
            freq_features = self.extract_frequency_domain_features(audio)
            features.update(freq_features)
            # Spektralfeatures
            spectral_features = self.extract_spectral_features(audio)
            features.update(spectral_features)

        # MFCCs (klassisch: Mittelwert und Std über alle MFCCs)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=13, hop_length=self.hop_length
        )
        features["mfcc_mean"] = np.mean(mfccs, axis=1)
        features["mfcc_std"] = np.std(mfccs, axis=1)
        features["mfcc"] = mfccs

        # Mel-Spektrogramm-Statistiken
        mel_spec = self.extract_mel_spectrogram(audio)
        features["mel_spec_mean"] = np.mean(mel_spec, axis=1)
        features["mel_spec_std"] = np.std(mel_spec, axis=1)
        features["mel_spec"] = mel_spec

        return features

    def extract_features_as_array(
        self, audio: np.ndarray, enhanced: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extrahiert Features und gibt sie als flaches Array zurück.

        Args:
            audio (np.ndarray): Audiosignal
            enhanced (bool): Wenn True, werden zusätzliche Features extrahiert

        Returns:
            Optional[np.ndarray]: 1D-Array der Features oder None
        """
        features = self.extract_features(audio, enhanced=enhanced)
        if features is None:
            return None

        # In konsistenter Reihenfolge in Array umwandeln
        feature_values = [features[key] for key in sorted(features.keys())]
        return np.array(feature_values)

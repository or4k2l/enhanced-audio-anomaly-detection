"""Data processing and loading utilities."""

import numpy as np
import soundfile as sf
from pathlib import Path


class AudioDataProcessor:
    """Process and load audio data for anomaly detection."""

    def __init__(self, sr=16000, duration=None):
        """
        Initialize data processor.

        Args:
            sr: Target sample rate
            duration: Target duration in seconds (None for variable length)
        """
        self.sr = sr
        self.duration = duration

    def load_audio(self, file_path):
        """
        Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Audio time series and sample rate
        """
        audio, sr = sf.read(file_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Trim or pad to target duration
        if self.duration is not None:
            target_length = int(self.duration * self.sr)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))

        return audio, self.sr

    def load_dataset(self, data_dir, file_pattern="*.wav"):
        """
        Load multiple audio files from directory.

        Args:
            data_dir: Directory containing audio files
            file_pattern: Glob pattern for audio files

        Returns:
            List of (file_path, audio, label) tuples
        """
        data_dir = Path(data_dir)
        audio_files = sorted(data_dir.glob(file_pattern))

        dataset = []
        for file_path in audio_files:
            try:
                audio, sr = self.load_audio(file_path)
                # Extract label from filename (e.g., normal/anomaly)
                label = "normal" if "normal" in file_path.stem.lower() else "anomaly"
                dataset.append((str(file_path), audio, label))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return dataset

    def split_dataset(self, dataset, train_ratio=0.7, val_ratio=0.15):
        """
        Split dataset into train, validation, and test sets.

        Args:
            dataset: List of (file_path, audio, label) tuples
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data

        Returns:
            Tuple of (train, val, test) datasets
        """
        np.random.seed(42)
        n_samples = len(dataset)
        indices = np.random.permutation(n_samples)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train = [dataset[i] for i in train_idx]
        val = [dataset[i] for i in val_idx]
        test = [dataset[i] for i in test_idx]

        return train, val, test

    def prepare_features(self, dataset, feature_extractor):
        """
        Extract features from dataset.

        Args:
            dataset: List of (file_path, audio, label) tuples
            feature_extractor: AudioFeatureExtractor instance

        Returns:
            Features array and labels array
        """
        features_list = []
        labels = []

        for file_path, audio, label in dataset:
            features = feature_extractor.extract_features(audio)
            # Concatenate numeric features
            feature_vector = np.concatenate(
                [
                    features["mel_spec_mean"],
                    features["mel_spec_std"],
                    features["mfcc_mean"],
                    features["mfcc_std"],
                    [
                        features["mean"],
                        features["std"],
                        features["max"],
                        features["min"],
                        features["rms"],
                        features["zcr"],
                    ],
                ]
            )
            features_list.append(feature_vector)
            labels.append(1 if label == "anomaly" else 0)

        return np.array(features_list), np.array(labels)

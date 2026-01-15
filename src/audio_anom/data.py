"""Data processing and loading utilities."""

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path


def build_feature_vector(features):
    """
    Build feature vector from extracted features dictionary.

    Args:
        features: Dictionary of extracted features from AudioFeatureExtractor

    Returns:
        Flattened feature vector as numpy array
    """
    return np.concatenate(
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
            feature_vector = build_feature_vector(features)
            features_list.append(feature_vector)
            labels.append(1 if label == "anomaly" else 0)

        return np.array(features_list), np.array(labels)

    def process_dataset_with_metadata(
        self,
        base_path,
        segment_length_sec=1,
        max_audio_length_sec=10,
        feature_extractor=None
    ):
        """
        Process entire dataset with metadata tracking (pump_id, file_id, segment_id).
        
        This matches the enhanced_audio_anomaly_detection.py approach.
        
        Args:
            base_path: Base directory containing 'normal' and 'abnormal' subdirectories
            segment_length_sec: Length of each segment in seconds
            max_audio_length_sec: Maximum audio length to process per file
            feature_extractor: AudioFeatureExtractor instance (required)
            
        Returns:
            pandas DataFrame with features and metadata columns
        """
        if feature_extractor is None:
            raise ValueError("feature_extractor is required")
        
        all_data = []
        file_count = {'normal': 0, 'abnormal': 0}
        segment_length_samples = segment_length_sec * self.sr
        max_samples = max_audio_length_sec * self.sr
        
        for condition in ['normal', 'abnormal']:
            label = 0 if condition == 'normal' else 1
            condition_path = Path(base_path) / condition
            
            if not condition_path.exists():
                print(f"  ⚠ Path not found: {condition_path}")
                continue
            
            # Find all .wav files
            wav_files = list(condition_path.rglob('*.wav'))
            print(f"  - {condition}: {len(wav_files)} files found")
            
            for file_idx, file_path in enumerate(wav_files):
                try:
                    # Extract pump_id from path (e.g., id_00, id_02, etc.)
                    pump_id = 'unknown'
                    for part in file_path.parts:
                        if part.startswith('id_'):
                            pump_id = part
                            break
                    
                    # Load audio
                    audio, sr = sf.read(file_path)
                    if audio.ndim > 1:
                        audio = audio[:, 0]  # Convert to mono
                    if sr != self.sr:
                        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sr)
                    
                    # Limit to max length
                    audio = audio[:max_samples]
                    
                    # Segment audio
                    segment_id = 0
                    for i in range(0, len(audio), segment_length_samples):
                        segment = audio[i:i + segment_length_samples]
                        
                        if len(segment) == segment_length_samples:
                            features = feature_extractor.extract_features(segment, enhanced=True)
                            
                            if features is not None:
                                # Add metadata
                                features['label'] = label
                                features['pump_id'] = pump_id
                                features['file_id'] = f"{condition}_{file_idx}"
                                features['segment_id'] = segment_id
                                features['condition'] = condition
                                all_data.append(features)
                                segment_id += 1
                    
                    file_count[condition] += 1
                    
                    if (file_idx + 1) % 50 == 0:
                        print(f"    Progress: {file_idx + 1}/{len(wav_files)} files")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {file_path.name}: {e}")
                    continue
        
        df = pd.DataFrame(all_data)
        print(f"\n✓ Processing complete:")
        print(f"  - Normal files: {file_count['normal']}")
        print(f"  - Abnormal files: {file_count['abnormal']}")
        print(f"  - Total segments: {len(df)}")
        if len(df) > 0:
            metadata_cols = ['label', 'pump_id', 'file_id', 'segment_id', 'condition']
            print(f"  - Features per segment: {len(df.columns) - len(metadata_cols)}")
        
        return df


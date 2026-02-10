"""Data processing and loading utilities."""

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path


from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path

def build_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Baut einen Feature-Vektor aus einem Feature-Dictionary.

    Args:
        features (Dict[str, Any]): Extrahierte Features

    Returns:
        np.ndarray: 1D-Feature-Vektor
    """
    def safe_get(key: str, default: Any) -> Any:
        return features[key] if key in features else default

    return np.concatenate(
        [
            safe_get("mel_spec_mean", np.zeros(128)),
            safe_get("mel_spec_std", np.zeros(128)),
            safe_get("mfcc_mean", np.zeros(13)),
            safe_get("mfcc_std", np.zeros(13)),
            [
                safe_get("mean", 0.0),
                safe_get("std", 0.0),
                safe_get("max", 0.0),
                safe_get("min", 0.0),
                safe_get("rms", 0.0),
                safe_get("zcr", 0.0),
            ],
        ]
    )



class AudioDataProcessor:
    """
    Verarbeitet und lädt Audiodaten für die Anomalieerkennung.

    Attribute:
        sr (int): Ziel-Sample-Rate
        duration (Optional[float]): Ziel-Dauer in Sekunden
    """

    def __init__(self, sr: int = 16000, duration: Optional[float] = None) -> None:
        """
        Initialisiert den Datenprozessor.

        Args:
            sr (int): Ziel-Sample-Rate
            duration (Optional[float]): Ziel-Dauer in Sekunden
        """
        self.sr = sr
        self.duration = duration

    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Lädt eine Audiodatei.

        Args:
            file_path (str | Path): Pfad zur Audiodatei

        Returns:
            Tuple[np.ndarray, int]: Audiosignal und Sample-Rate
        """
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        if self.duration is not None:
            target_length = int(self.duration * self.sr)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
        return audio, self.sr

    def load_dataset(self, data_dir: Union[str, Path], file_pattern: str = "*.wav") -> List[Tuple[str, np.ndarray, str]]:
        """
        Lädt mehrere Audiodateien aus einem Verzeichnis.

        Args:
            data_dir (str | Path): Verzeichnis mit Audiodateien
            file_pattern (str): Glob-Pattern

        Returns:
            List[Tuple[str, np.ndarray, str]]: Liste aus (Pfad, Audio, Label)
        """
        data_dir = Path(data_dir)
        audio_files = sorted(data_dir.glob(file_pattern))
        dataset: List[Tuple[str, np.ndarray, str]] = []
        for file_path in audio_files:
            try:
                audio, sr = self.load_audio(file_path)
                label = "normal" if "normal" in file_path.stem.lower() else "anomaly"
                dataset.append((str(file_path), audio, label))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        return dataset

    def split_dataset(
        self,
        dataset: List[Tuple[str, np.ndarray, str]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]], List[Tuple[str, np.ndarray, str]]]:
        """
        Teilt das Dataset in Trainings-, Validierungs- und Testdaten.

        Args:
            dataset (List[Tuple[str, np.ndarray, str]]): Datensätze
            train_ratio (float): Trainingsanteil
            val_ratio (float): Validierungsanteil

        Returns:
            Tuple[List, List, List]: (train, val, test)
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

    def prepare_features(
        self,
        dataset: List[Tuple[str, np.ndarray, str]],
        feature_extractor: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrahiert Features aus einem Dataset.

        Args:
            dataset (List[Tuple[str, np.ndarray, str]]): Datensätze
            feature_extractor: Instanz von AudioFeatureExtractor

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features und Labels
        """
        features_list: List[np.ndarray] = []
        labels: List[Optional[int]] = []
        for file_path, audio, label in dataset:
            features = feature_extractor.extract_features(audio)
            if features is None:
                continue
            feature_vector = build_feature_vector(features)
            features_list.append(feature_vector)
            labels.append(1 if label == "anomaly" else 0 if label == "normal" else None)
        return np.array(features_list), np.array(labels)

    def process_dataset_with_metadata(
        self, base_path, segment_length_sec=1, max_audio_length_sec=10, feature_extractor=None
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
        file_count = {"normal": 0, "abnormal": 0}
        segment_length_samples = segment_length_sec * self.sr
        max_samples = max_audio_length_sec * self.sr

        for condition in ["normal", "abnormal"]:
            label = 0 if condition == "normal" else 1
            condition_path = Path(base_path) / condition

            if not condition_path.exists():
                print(f"  ⚠ Path not found: {condition_path}")
                continue

            # Find all .wav files
            wav_files = list(condition_path.rglob("*.wav"))
            print(f"  - {condition}: {len(wav_files)} files found")

            for file_idx, file_path in enumerate(wav_files):
                try:
                    # Extract pump_id from path (e.g., id_00, id_02, etc.)
                    pump_id = "unknown"
                    for part in file_path.parts:
                        if part.startswith("id_"):
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
                        segment = audio[i : i + segment_length_samples]

                        if len(segment) == segment_length_samples:
                            features = feature_extractor.extract_features(segment, enhanced=True)

                            if features is not None:
                                # Add metadata
                                features["label"] = label
                                features["pump_id"] = pump_id
                                features["file_id"] = f"{condition}_{file_idx}"
                                features["segment_id"] = segment_id
                                features["condition"] = condition
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
            metadata_cols = ["label", "pump_id", "file_id", "segment_id", "condition"]
            print(f"  - Features per segment: {len(df.columns) - len(metadata_cols)}")

        return df

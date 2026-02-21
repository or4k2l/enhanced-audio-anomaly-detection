"""DCASE 2020 Task 2 dataset loading utilities.

Supports the standard DCASE 2020 Task 2 directory structure:
    <machine_type>/
        train/
            normal_*.wav
        test/
            normal_*.wav
            anomaly_*.wav
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

MACHINE_TYPES = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
DEFAULT_SR = 16000
DEFAULT_DURATION = 10.0  # seconds


def load_audio_files(
    directory: str,
    sr: int = DEFAULT_SR,
    duration: Optional[float] = DEFAULT_DURATION,
    pattern: str = "*.wav",
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load all audio files from a directory.

    Files whose names contain 'anomaly' receive label 1, all others label 0.

    Args:
        directory: Path to the directory containing .wav files.
        sr: Target sample rate in Hz.
        duration: Duration to load in seconds. None loads the full file.
        pattern: Glob pattern for audio files.

    Returns:
        Tuple of (waveforms, labels, filenames) where:
        - waveforms: List of numpy arrays of shape (n_samples,).
        - labels: List of int labels (0=normal, 1=anomaly).
        - filenames: List of file paths.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if not files:
        logger.warning(f"No files found in {directory} matching '{pattern}'")
        return [], [], []

    waveforms, labels, names = [], [], []
    for fpath in files:
        try:
            wave, _ = librosa.load(str(fpath), sr=sr, duration=duration, mono=True)
            label = 1 if "anomaly" in fpath.name.lower() else 0
            waveforms.append(wave)
            labels.append(label)
            names.append(str(fpath))
        except Exception as exc:
            logger.warning(f"Failed to load {fpath}: {exc}")

    logger.info(
        f"Loaded {len(waveforms)} files from {directory} "
        f"(normal: {labels.count(0)}, anomaly: {labels.count(1)})"
    )
    return waveforms, labels, names


class DCASEDataset:
    """DCASE 2020 Task 2 dataset loader.

    Args:
        root_dir: Root directory containing machine-type sub-directories.
        machine: Machine type to load (e.g. 'pump').
        sr: Target sample rate in Hz.
        duration: Audio clip duration in seconds.

    Example:
        >>> dataset = DCASEDataset(root_dir="data/", machine="pump")
        >>> train_waves, train_labels = dataset.load_train()
        >>> test_waves, test_labels = dataset.load_test()
    """

    def __init__(
        self,
        root_dir: str,
        machine: str,
        sr: int = DEFAULT_SR,
        duration: Optional[float] = DEFAULT_DURATION,
    ):
        if machine not in MACHINE_TYPES:
            logger.warning(
                f"Machine '{machine}' not in known DCASE machines: {MACHINE_TYPES}"
            )
        self.root_dir = Path(root_dir)
        self.machine = machine
        self.sr = sr
        self.duration = duration

    @property
    def train_dir(self) -> Path:
        """Path to the training split directory."""
        return self.root_dir / self.machine / "train"

    @property
    def test_dir(self) -> Path:
        """Path to the test split directory."""
        return self.root_dir / self.machine / "test"

    def load_train(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load training audio files (all assumed normal).

        Returns:
            Tuple of (waveforms, labels).
        """
        waves, labels, _ = load_audio_files(
            self.train_dir, sr=self.sr, duration=self.duration
        )
        return waves, labels

    def load_test(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load test audio files (mixed normal and anomaly).

        Returns:
            Tuple of (waveforms, labels).
        """
        waves, labels, _ = load_audio_files(
            self.test_dir, sr=self.sr, duration=self.duration
        )
        return waves, labels

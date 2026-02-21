"""Centralized configuration for the hybrid audio anomaly detection system.

All tunable hyperparameters and path defaults are defined here.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16000
CLIP_DURATION: float = 10.0  # seconds

# ── AST ────────────────────────────────────────────────────────────────────
AST_MODEL_NAME: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_EMBEDDING_DIM: int = 768

# ── Classical features ──────────────────────────────────────────────────────
CLASSICAL_FEATURE_DIM: int = 955
N_MELS: int = 128
N_MFCC: int = 20
HOP_LENGTH: int = 512
N_FFT: int = 2048

# ── Hybrid ─────────────────────────────────────────────────────────────────
HYBRID_DIM: int = AST_EMBEDDING_DIM + CLASSICAL_FEATURE_DIM  # 1723

# ── Supported machines (DCASE 2020 Task 2) ─────────────────────────────────
MACHINE_TYPES: List[str] = [
    "fan",
    "pump",
    "slider",
    "valve",
    "ToyCar",
    "ToyConveyor",
]

# ── Best methods per machine (from experiments) ─────────────────────────────
BEST_METHODS: Dict[str, str] = {
    "fan": "ocsvm",
    "pump": "gmm",
    "slider": "gmm",
    "valve": "gmm",
    "ToyCar": "gmm",
    "ToyConveyor": "gmm",
}

BEST_N_COMPONENTS: Dict[str, int] = {
    "fan": 8,
    "pump": 16,
    "slider": 8,
    "valve": 8,
    "ToyCar": 8,
    "ToyConveyor": 8,
}


@dataclass
class AudioConfig:
    """Audio loading configuration.

    Attributes:
        sample_rate: Target sample rate in Hz.
        duration: Clip duration in seconds.
        normalize: Normalize waveform amplitude to [-1, 1].
    """

    sample_rate: int = SAMPLE_RATE
    duration: float = CLIP_DURATION
    normalize: bool = True


@dataclass
class ASTConfig:
    """AST embedding extractor configuration.

    Attributes:
        model_name: Hugging Face model identifier.
        device: Compute device ('cpu' or 'cuda').
        embedding_dim: Output embedding dimensionality.
    """

    model_name: str = AST_MODEL_NAME
    device: str = "cpu"
    embedding_dim: int = AST_EMBEDDING_DIM


@dataclass
class ClassicalConfig:
    """Classical feature extraction configuration.

    Attributes:
        n_mels: Number of mel filter banks.
        n_mfcc: Number of MFCC coefficients.
        hop_length: STFT hop length in samples.
        n_fft: FFT window size.
        feature_dim: Output feature dimensionality.
    """

    n_mels: int = N_MELS
    n_mfcc: int = N_MFCC
    hop_length: int = HOP_LENGTH
    n_fft: int = N_FFT
    feature_dim: int = CLASSICAL_FEATURE_DIM


@dataclass
class EnsembleConfig:
    """Hybrid ensemble detector configuration.

    Attributes:
        method: Detection algorithm ('gmm', 'ocsvm', 'xgboost', 'logistic_regression').
        n_components: Number of GMM components.
        random_state: Random seed.
    """

    method: str = "gmm"
    n_components: int = 16
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Full experiment configuration bundling all sub-configs.

    Attributes:
        audio: Audio loading settings.
        ast: AST extractor settings.
        classical: Classical feature settings.
        ensemble: Ensemble detector settings.
        machines: List of machine types to evaluate.
        output_dir: Directory for saving results and models.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    ast: ASTConfig = field(default_factory=ASTConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    machines: List[str] = field(default_factory=lambda: list(MACHINE_TYPES))
    output_dir: str = "experiments/results"

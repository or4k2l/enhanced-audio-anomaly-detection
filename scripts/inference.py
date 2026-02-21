"""Production inference: score a single audio file for anomaly.

Usage:
    python scripts/inference.py \\
        --model models/pump_hybrid_gmm16.pkl \\
        --audio test_sample.wav

Output:
    Anomaly score: 0.823 (likely anomaly)
"""

import argparse
import logging
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.ast_extractor import ASTEmbeddingExtractor  # noqa: E402
from models.classical_features import extract_classical_features  # noqa: E402
from models.ensemble import HybridAnomalyDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SR = 16000
DEFAULT_DURATION = 10.0
ANOMALY_THRESHOLD = 0.5


def run_inference(model_path: str, audio_path: str) -> dict:
    """Score a single audio file using a pretrained hybrid detector.

    Args:
        model_path: Path to a saved HybridAnomalyDetector (.pkl).
        audio_path: Path to the audio file (.wav).

    Returns:
        Dictionary with 'score', 'label', and 'threshold'.
    """
    logger.info(f"Loading model from {model_path}")
    detector = HybridAnomalyDetector.load(model_path)

    logger.info(f"Loading audio from {audio_path}")
    wave, _ = librosa.load(audio_path, sr=DEFAULT_SR, duration=DEFAULT_DURATION, mono=True)

    # Extract hybrid features
    ast_extractor = ASTEmbeddingExtractor()
    ast_feat = ast_extractor.extract_embedding(wave)
    classical_feat = extract_classical_features(wave)
    features = np.concatenate([ast_feat, classical_feat]).reshape(1, -1)

    score = float(detector.score_samples(features)[0])

    # Normalize score to [0, 1] using a simple sigmoid for display
    import math

    normalized = 1.0 / (1.0 + math.exp(-score))
    label = "likely anomaly" if normalized >= ANOMALY_THRESHOLD else "likely normal"

    result = {
        "audio_file": str(audio_path),
        "raw_score": score,
        "normalized_score": round(normalized, 4),
        "label": label,
        "threshold": ANOMALY_THRESHOLD,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run anomaly inference on an audio file")
    parser.add_argument("--model", required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav)")
    args = parser.parse_args()

    result = run_inference(args.model, args.audio)
    print(f"Anomaly score: {result['normalized_score']:.3f} ({result['label']})")


if __name__ == "__main__":
    main()

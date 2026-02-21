"""Train sklearn GMM baseline on classical audio features.

Usage:
    python scripts/train_baseline.py \\
        --train_dir data/pump/train \\
        --test_dir data/pump/test \\
        --machine pump \\
        --output models/pump_baseline.pkl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.classical_features import extract_classical_features  # noqa: E402
from models.ensemble import HybridAnomalyDetector  # noqa: E402
from data.dataset import load_audio_files  # noqa: E402
from evaluation.metrics import evaluate_detector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def train_baseline(
    train_dir: Path,
    test_dir: Path,
    machine: str,
    output_path: Path,
    n_components: int = 8,
) -> dict:
    """Train and evaluate a GMM baseline on classical features.

    Args:
        train_dir: Directory containing training (normal) audio files.
        test_dir: Directory containing test audio files.
        machine: Machine type label for logging.
        output_path: Path to save the trained model.
        n_components: Number of GMM components.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info(f"Training baseline for machine: {machine}")

    # Load audio
    train_waves, _, _ = load_audio_files(str(train_dir))
    test_waves, test_labels, _ = load_audio_files(str(test_dir))

    if not train_waves:
        raise RuntimeError(f"No training files found in {train_dir}")

    # Extract classical features
    logger.info("Extracting classical features...")
    X_train = np.stack([extract_classical_features(w) for w in train_waves])
    X_test = np.stack([extract_classical_features(w) for w in test_waves])
    y_test = np.array(test_labels)

    # Train GMM
    detector = HybridAnomalyDetector(method="gmm", n_components=n_components)
    detector.fit(X_train)

    # Evaluate
    scores = detector.score_samples(X_test)
    result = evaluate_detector(y_test, scores)
    logger.info(f"AUC: {result.auc:.4f}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(output_path))

    metrics = {"machine": machine, "auc": result.auc, "method": "gmm_baseline"}
    logger.info(f"Results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train GMM baseline")
    parser.add_argument("--train_dir", required=True, help="Path to training audio dir")
    parser.add_argument("--test_dir", required=True, help="Path to test audio dir")
    parser.add_argument("--machine", required=True, help="Machine type label")
    parser.add_argument(
        "--output", required=True, help="Output path for the model (.pkl)"
    )
    parser.add_argument("--n_components", type=int, default=8, help="GMM components")
    args = parser.parse_args()

    metrics = train_baseline(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        machine=args.machine,
        output_path=Path(args.output),
        n_components=args.n_components,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

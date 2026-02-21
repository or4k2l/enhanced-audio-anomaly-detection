"""Train hybrid ensemble on AST + classical features.

Usage:
    python scripts/train_hybrid.py \\
        --train_dir data/pump/train \\
        --test_dir data/pump/test \\
        --machine pump \\
        --method gmm \\
        --n_components 16 \\
        --output models/pump_hybrid_gmm16.pkl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.ast_extractor import ASTEmbeddingExtractor  # noqa: E402
from models.classical_features import extract_classical_features  # noqa: E402
from models.ensemble import HybridAnomalyDetector  # noqa: E402
from data.dataset import load_audio_files  # noqa: E402
from evaluation.metrics import evaluate_detector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_hybrid_features(
    waveforms: list,
    ast_extractor: ASTEmbeddingExtractor,
) -> np.ndarray:
    """Concatenate AST (768-dim) and classical (955-dim) features.

    Args:
        waveforms: List of raw audio waveforms.
        ast_extractor: Fitted ASTEmbeddingExtractor instance.

    Returns:
        Feature matrix of shape (n_samples, 1723).
    """
    features = []
    for i, wave in enumerate(waveforms):
        if i % 50 == 0:
            logger.info(f"  Extracting features {i}/{len(waveforms)}")
        ast_feat = ast_extractor.extract_embedding(wave)
        classical_feat = extract_classical_features(wave)
        features.append(np.concatenate([ast_feat, classical_feat]))
    return np.stack(features, axis=0)


def train_hybrid_ensemble(
    train_dir: Path,
    test_dir: Path,
    machine: str,
    method: str,
    n_components: int,
    output_path: Path,
) -> dict:
    """End-to-end training pipeline for the hybrid ensemble.

    Steps:
    1. Load audio files
    2. Extract AST embeddings (768-dim)
    3. Extract classical features (955-dim)
    4. Concatenate → 1723-dim hybrid features
    5. Train GMM/OCSVM/XGBoost
    6. Evaluate on test set
    7. Save model

    Args:
        train_dir: Directory with training audio.
        test_dir: Directory with test audio.
        machine: Machine type label.
        method: Detection method ('gmm', 'ocsvm', 'xgboost').
        n_components: GMM components (used only for method='gmm').
        output_path: Path to save trained model.

    Returns:
        Dictionary with AUC and experiment metadata.
    """
    logger.info(f"Training hybrid ensemble for {machine} using {method}")

    train_waves, _, _ = load_audio_files(str(train_dir))
    test_waves, test_labels, _ = load_audio_files(str(test_dir))

    if not train_waves:
        raise RuntimeError(f"No training files in {train_dir}")

    ast_extractor = ASTEmbeddingExtractor()

    logger.info("Extracting hybrid features for training set...")
    X_train = extract_hybrid_features(train_waves, ast_extractor)

    logger.info("Extracting hybrid features for test set...")
    X_test = extract_hybrid_features(test_waves, ast_extractor)
    y_test = np.array(test_labels)

    detector = HybridAnomalyDetector(method=method, n_components=n_components)
    detector.fit(X_train)

    scores = detector.score_samples(X_test)
    result = evaluate_detector(y_test, scores)
    logger.info(f"AUC: {result.auc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(output_path))

    metrics = {
        "machine": machine,
        "method": method,
        "n_components": n_components,
        "auc": result.auc,
        "feature_dim": X_train.shape[1],
    }
    logger.info(f"Results: {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train hybrid ensemble")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--method", default="gmm", choices=["gmm", "ocsvm", "xgboost"])
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = train_hybrid_ensemble(
        train_dir=Path(args.train_dir),
        test_dir=Path(args.test_dir),
        machine=args.machine,
        method=args.method,
        n_components=args.n_components,
        output_path=Path(args.output),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

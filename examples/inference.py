"""Example inference script for audio anomaly detection."""

import argparse
import sys
from pathlib import Path

# Allow standalone execution from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_anom import (
    AudioDataProcessor,
    AudioFeatureExtractor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
)


def _load_detector(model_path: str):
    """Load a persisted detector by trying supported model classes."""
    loaders = [RandomForestAnomalyDetector, XGBoostAnomalyDetector]
    last_error = None

    for detector_cls in loaders:
        detector = detector_cls()
        try:
            detector.load(model_path)
            return detector
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    raise ValueError(
        "Could not load model with supported detectors "
        "(RandomForestAnomalyDetector/XGBoostAnomalyDetector)."
    ) from last_error


def predict(model_path, audio_path, sample_rate=16000):
    """Predict if an audio file contains anomaly."""
    print("=" * 60)
    print("Audio Anomaly Detection - Inference")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    detector = _load_detector(model_path)

    # Initialize components
    feature_extractor = AudioFeatureExtractor(sr=sample_rate)
    data_processor = AudioDataProcessor(sr=sample_rate)

    # Load and process audio
    print(f"Loading audio from {audio_path}...")
    audio, sr = data_processor.load_audio(audio_path)
    print(f"Audio length: {len(audio) / sr:.2f} seconds")

    # Extract features and align feature vector shape with trained model
    print("Extracting features...")
    features = feature_extractor.extract_features(audio)
    feature_values = [features[k] for k in sorted(features.keys())]
    feature_vector = [feature_values]

    # Predict
    print("Running prediction...")
    prediction = detector.predict(feature_vector)[0]
    if hasattr(detector, "predict_proba"):
        score = float(detector.predict_proba(feature_vector)[0, 1])
    else:
        score = float(prediction)

    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"\nFile: {audio_path}")
    print(f"Prediction: {'ANOMALY' if prediction == 1 else 'NORMAL'}")
    print(f"Anomaly Probability/Score: {score:.4f}\n")

    return prediction, score


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference on audio file for anomaly detection"
    )
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.pkl",
        help="Path to trained model",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Sample rate for processing"
    )

    args = parser.parse_args()
    predict(args.model_path, args.audio_path, args.sample_rate)


if __name__ == "__main__":
    main()

"""Example inference script for audio anomaly detection."""

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom import (  # noqa: E402
    AudioFeatureExtractor,
    AudioDataProcessor,
    AnomalyDetector,
    build_feature_vector,
)


def predict(model_path, audio_path, sample_rate=16000):
    """
    Predict if an audio file contains anomaly.

    Args:
        model_path: Path to trained model
        audio_path: Path to audio file
        sample_rate: Sample rate for processing

    Returns:
        Prediction result
    """
    print("=" * 60)
    print("Audio Anomaly Detection - Inference")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    detector = AnomalyDetector()
    detector.load(model_path)

    # Initialize components
    feature_extractor = AudioFeatureExtractor(sr=sample_rate)
    data_processor = AudioDataProcessor(sr=sample_rate)

    # Load and process audio
    print(f"Loading audio from {audio_path}...")
    audio, sr = data_processor.load_audio(audio_path)
    print(f"Audio length: {len(audio) / sr:.2f} seconds")

    # Extract features
    print("Extracting features...")
    features = feature_extractor.extract_features(audio)

    # Prepare feature vector using shared utility
    feature_vector = build_feature_vector(features).reshape(1, -1)

    # Predict
    print("Running prediction...")
    prediction = detector.predict(feature_vector)[0]
    score = detector.decision_function(feature_vector)[0]

    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"\nFile: {audio_path}")
    print(f"Prediction: {'ANOMALY' if prediction == 1 else 'NORMAL'}")
    print(f"Anomaly Score: {score:.4f}")
    print(
        f"Confidence: {abs(score):.4f} (higher absolute value = more confident)\n"
    )

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

    try:
        predict(args.model_path, args.audio_path, args.sample_rate)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the model and audio file exist.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

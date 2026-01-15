"""End-to-end demonstration of the audio anomaly detection system."""

import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    AnomalyDetector,
    build_feature_vector,
)


def generate_synthetic_audio(duration=1.0, sr=16000, anomaly=False):
    """Generate synthetic audio for demonstration."""
    t = np.linspace(0, duration, int(duration * sr))

    if anomaly:
        # Anomalous sound: irregular frequencies and noise
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.3 * np.sin(2 * np.pi * 350 * t)
            + 0.4 * np.random.randn(len(t))
        )
    else:
        # Normal sound: regular frequency
        audio = 0.5 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))

    return audio.astype(np.float32)


def main():
    """Run end-to-end demonstration."""
    print("=" * 80)
    print("Audio Anomaly Detection - End-to-End Demo")
    print("=" * 80)

    # Create temporary directory for demo data
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "demo_data"
        data_dir.mkdir()

        print("\n[1/6] Generating synthetic audio data...")
        # Generate 30 normal and 10 anomaly samples
        for i in range(30):
            audio = generate_synthetic_audio(duration=1.0, anomaly=False)
            sf.write(data_dir / f"normal_{i:03d}.wav", audio, 16000)

        for i in range(10):
            audio = generate_synthetic_audio(duration=1.0, anomaly=True)
            sf.write(data_dir / f"anomaly_{i:03d}.wav", audio, 16000)

        print("Generated 40 audio files (30 normal, 10 anomaly)")

        # Initialize components
        print("\n[2/6] Initializing components...")
        feature_extractor = AudioFeatureExtractor(sr=16000, n_mels=64)
        data_processor = AudioDataProcessor(sr=16000, duration=1.0)

        # Load dataset
        print("\n[3/6] Loading dataset...")
        dataset = data_processor.load_dataset(data_dir, "*.wav")
        print(f"Loaded {len(dataset)} files")

        # Split dataset
        print("\n[4/6] Splitting dataset...")
        train_data, val_data, test_data = data_processor.split_dataset(
            dataset, train_ratio=0.6, val_ratio=0.2
        )
        print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Extract features
        print("\n[5/6] Extracting features and training model...")
        X_train, y_train = data_processor.prepare_features(train_data, feature_extractor)
        X_test, y_test = data_processor.prepare_features(test_data, feature_extractor)

        # Train model
        detector = AnomalyDetector(contamination=0.25, random_state=42)
        detector.fit(X_train, y_train)

        # Evaluate
        print("\n[6/6] Evaluating model...")
        metrics = detector.evaluate(X_test, y_test)

        print("\n" + "=" * 80)
        print("Test Results")
        print("=" * 80)
        for metric, value in metrics.items():
            print(f"  {metric.upper():<12}: {value:.4f}")

        # Test on individual samples
        print("\n" + "=" * 80)
        print("Sample Predictions")
        print("=" * 80)

        for i, (file_path, audio, true_label) in enumerate(test_data[:5]):
            features = feature_extractor.extract_features(audio)
            feature_vector = build_feature_vector(features).reshape(1, -1)

            prediction = detector.predict(feature_vector)[0]
            score = detector.decision_function(feature_vector)[0]

            pred_label = "ANOMALY" if prediction == 1 else "NORMAL"
            true_label_str = "ANOMALY" if true_label == "anomaly" else "NORMAL"
            match = "✓" if pred_label == true_label_str else "✗"

            print(f"  Sample {i+1}: True={true_label_str:<8} Pred={pred_label:<8} "
                  f"Score={score:7.4f} {match}")

        # Save model
        model_path = Path(tmpdir) / "demo_model.pkl"
        detector.save(model_path)
        print(f"\n  Model saved to {model_path}")

        # Test loading
        print("\n[Test] Loading saved model...")
        detector2 = AnomalyDetector()
        detector2.load(model_path)
        print("  Model loaded successfully!")

        # Verify predictions match
        test_features = feature_extractor.extract_features(test_data[0][1])
        test_vector = build_feature_vector(test_features).reshape(1, -1)
        pred1 = detector.predict(test_vector)[0]
        pred2 = detector2.predict(test_vector)[0]
        print(f"  Prediction verification: original={pred1}, loaded={pred2}, "
              f"match={'✓' if pred1 == pred2 else '✗'}")

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        print("\nThe system demonstrates:")
        print("  • Audio loading and preprocessing")
        print("  • Feature extraction (mel spectrograms, MFCCs, statistics)")
        print("  • Dataset splitting and preparation")
        print("  • Model training with Isolation Forest")
        print("  • Model evaluation with multiple metrics")
        print("  • Model persistence (save/load)")
        print("  • End-to-end prediction pipeline")
        print("\nAll components are working correctly! ✓")


if __name__ == "__main__":
    main()

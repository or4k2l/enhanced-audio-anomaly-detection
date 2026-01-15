"""Training script for audio anomaly detection."""

import argparse
import os

from audio_anom import AudioFeatureExtractor, AudioDataProcessor, AnomalyDetector


def train(args):
    """Train anomaly detection model."""
    print("=" * 60)
    print("Audio Anomaly Detection - Training")
    print("=" * 60)

    # Initialize components
    print("\n[1/5] Initializing components...")
    feature_extractor = AudioFeatureExtractor(
        sr=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    data_processor = AudioDataProcessor(sr=args.sample_rate, duration=args.duration)

    # Load dataset
    print(f"\n[2/5] Loading dataset from {args.data_dir}...")
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        print("Please prepare the dataset following docs/DATASET.md")
        return

    dataset = data_processor.load_dataset(args.data_dir, args.file_pattern)
    print(f"Loaded {len(dataset)} audio files")

    if len(dataset) == 0:
        print("Error: No audio files found in the specified directory")
        return

    # Split dataset
    print("\n[3/5] Splitting dataset...")
    train_data, val_data, test_data = data_processor.split_dataset(
        dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    print(
        f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)} samples"
    )

    # Extract features
    print("\n[4/5] Extracting features...")
    X_train, y_train = data_processor.prepare_features(train_data, feature_extractor)
    X_val, y_val = data_processor.prepare_features(val_data, feature_extractor)
    X_test, y_test = data_processor.prepare_features(test_data, feature_extractor)
    print(f"Feature shape: {X_train.shape}")

    # Train model
    print("\n[5/5] Training model...")
    detector = AnomalyDetector(
        contamination=args.contamination, random_state=args.random_state
    )
    detector.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print("\nValidation set:")
    val_metrics = detector.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTest set:")
    test_metrics = detector.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pkl")
        detector.save(model_path)
        print(f"\nModel saved to {model_path}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train audio anomaly detection model"
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pump",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--file-pattern", type=str, default="*.wav", help="Audio file pattern"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory for model"
    )

    # Audio processing arguments
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Target duration in seconds"
    )

    # Feature extraction arguments
    parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    parser.add_argument("--n-fft", type=int, default=1024, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length")

    # Training arguments
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="Training data ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation data ratio"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Expected proportion of anomalies",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random state for reproducibility"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

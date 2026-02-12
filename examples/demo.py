#!/usr/bin/env python3
"""Demo script for training/evaluating with real or synthetic audio data."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Allow standalone execution from repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_anom import (
    AudioDataProcessor,
    AudioFeatureExtractor,
    ModelEvaluator,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
)


def _make_synthetic_dataset(sr: int = 16000, samples: int = 200):
    """Create synthetic normal/anomaly waveforms for quick demo runs."""
    rng = np.random.default_rng(42)
    dataset = []
    for idx in range(samples):
        is_anomaly = idx % 2 == 1
        if is_anomaly:
            audio = rng.normal(0, 0.5, sr) + 0.4 * np.sin(np.linspace(0, 50, sr))
            label = "anomaly"
        else:
            audio = rng.normal(0, 0.2, sr)
            label = "normal"
        dataset.append((f"synthetic_{idx}.wav", audio.astype(np.float32), label))
    return dataset


def main():
    """Run demo end-to-end with optional real dataset path."""
    parser = argparse.ArgumentParser(description="Audio anomaly detection demo")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-files", type=int, default=500)
    args = parser.parse_args()

    print("=" * 80)
    print("ENHANCED AUDIO ANOMALY DETECTION - DEMO")
    print("=" * 80)

    data_processor = AudioDataProcessor(sr=args.sample_rate)
    feature_extractor = AudioFeatureExtractor(sr=args.sample_rate)
    evaluator = ModelEvaluator()

    if args.data_dir:
        dataset = data_processor.load_dataset(args.data_dir)
        if not dataset:
            raise ValueError("No audio files found in provided --data-dir")
        dataset = dataset[: args.max_files]
        print(f"Loaded real dataset: {len(dataset)} files")
    else:
        dataset = _make_synthetic_dataset(sr=args.sample_rate)
        print("Using synthetic dataset")

    train_data, _, test_data = data_processor.split_dataset(
        dataset, train_ratio=0.7, val_ratio=0.15
    )

    X_train, y_train = [], []
    for _, audio, label in train_data:
        features = feature_extractor.extract_features(audio, enhanced=True)
        X_train.append([features[k] for k in sorted(features.keys())])
        y_train.append(1 if label == "anomaly" else 0)

    X_test, y_test = [], []
    for _, audio, label in test_data:
        features = feature_extractor.extract_features(audio, enhanced=True)
        X_test.append([features[k] for k in sorted(features.keys())])
        y_test.append(1 if label == "anomaly" else 0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    models = {
        "Random Forest": RandomForestAnomalyDetector(random_state=42, n_splits=3),
        "XGBoost": XGBoostAnomalyDetector(random_state=42, n_splits=3),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train, use_pca=True, use_smote=True, verbose=0)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(evaluator.evaluate_model(y_test, y_pred, y_prob, name))

    evaluator.compare_models(results)
    print("\nDemo completed.")


if __name__ == "__main__":
    main()

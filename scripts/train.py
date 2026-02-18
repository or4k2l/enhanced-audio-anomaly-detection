#!/usr/bin/env python
"""Complete training pipeline for audio anomaly detection models."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    DataPreprocessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    ModelEvaluator,
    ModelConfig,
    setup_logger,
)

logger = setup_logger("train", level=20)  # INFO level


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train audio anomaly detection models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost", "both"],
        default="both",
        help="Model type to train",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0-1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--use-grid-search",
        action="store_true",
        help="Use GridSearchCV for Random Forest",
    )

    return parser.parse_args()


def load_or_prepare_features(data_dir: str, config: ModelConfig):
    """
    Load or prepare features from audio files.

    Args:
        data_dir: Directory containing audio files
        config: Model configuration

    Returns:
        Tuple of (X, y, metadata)
    """
    logger.info(f"Loading/preparing features from {data_dir}")

    data_path = Path(data_dir)

    # Check if preprocessed features exist
    features_file = data_path / "features.npy"
    labels_file = data_path / "labels.npy"

    if features_file.exists() and labels_file.exists():
        logger.info("Loading preprocessed features")
        X = np.load(features_file)
        y = np.load(labels_file)
        metadata = {}
    else:
        logger.info("Extracting features from audio files...")

        # Initialize processors
        feature_extractor = AudioFeatureExtractor(
            sr=config.feature.sr,
            n_mels=config.feature.n_mels,
            n_fft=config.feature.n_fft,
            hop_length=config.feature.hop_length,
            n_mfcc=config.feature.n_mfcc,
        )
        data_processor = AudioDataProcessor(sr=config.feature.sr)

        # Find audio files
        audio_files = list(data_path.glob("**/*.wav"))
        if not audio_files:
            raise ValueError(f"No audio files found in {data_dir}")

        logger.info(f"Found {len(audio_files)} audio files")

        # Extract features
        features_list = []
        labels_list = []

        for audio_file in audio_files:
            # Extract label from filename or directory structure
            # Assuming format: normal_*.wav or anomaly_*.wav
            if "normal" in audio_file.stem.lower():
                label = 0
            elif "anomaly" in audio_file.stem.lower():
                label = 1
            else:
                logger.warning(f"Cannot determine label for {audio_file}, skipping")
                continue

            # Load audio
            audio, sr = data_processor.load_audio(str(audio_file))

            # Extract features
            features = feature_extractor.extract_features(audio)

            # Build feature vector
            from audio_anom import build_feature_vector

            feature_vector = build_feature_vector(features)

            features_list.append(feature_vector)
            labels_list.append(label)

        X = np.array(features_list)
        y = np.array(labels_list)
        metadata = {}

        # Save features
        np.save(features_file, X)
        np.save(labels_file, y)
        logger.info(f"Features saved to {features_file}")

    logger.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    return X, y, metadata


def train_random_forest(
    X_train, y_train, X_test, y_test, config, use_grid_search=False
):
    """Train Random Forest model."""
    logger.info("=" * 80)
    logger.info("Training Random Forest Model")
    logger.info("=" * 80)

    # Initialize model
    model = RandomForestAnomalyDetector(config=config.random_forest)

    # Train
    model.train(X_train, y_train, use_grid_search=use_grid_search)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, "Random Forest")

    logger.info("\nRandom Forest Results:")
    for key, value in metrics.items():
        if key != "Model" and value is not None:
            logger.info(f"  {key}: {value:.4f}")

    return model, metrics, y_pred, y_prob


def train_xgboost(X_train, y_train, X_test, y_test, config):
    """Train XGBoost model."""
    logger.info("=" * 80)
    logger.info("Training XGBoost Model")
    logger.info("=" * 80)

    # Initialize model
    model = XGBoostAnomalyDetector(config=config.xgboost)

    # Train
    model.train(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(y_test, y_pred, y_prob, "XGBoost")

    logger.info("\nXGBoost Results:")
    for key, value in metrics.items():
        if key != "Model" and value is not None:
            logger.info(f"  {key}: {value:.4f}")

    return model, metrics, y_pred, y_prob


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = ModelConfig.load(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = ModelConfig.default()
        logger.info("Using default configuration")

    # Override config with command line arguments
    config.training.test_size = args.test_size
    config.training.random_state = args.random_state

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or prepare features
    X, y, metadata = load_or_prepare_features(args.data_dir, config)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.training.test_size, random_state=config.training.random_state, stratify=y
    )

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(config=config.preprocessing)
    X_train_processed, y_train_processed = preprocessor.fit_transform_train(
        X_train, y_train
    )
    X_test_processed = preprocessor.transform(X_test)

    logger.info(
        f"Processed train set: {X_train_processed.shape}, "
        f"Processed test set: {X_test_processed.shape}"
    )

    # Save preprocessor
    preprocessor_path = output_dir / "preprocessor.pkl"
    preprocessor.save(str(preprocessor_path))
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Train models
    results = {}

    if args.model_type in ["random_forest", "both"]:
        rf_model, rf_metrics, rf_pred, rf_prob = train_random_forest(
            X_train_processed,
            y_train_processed,
            X_test_processed,
            y_test,
            config,
            use_grid_search=args.use_grid_search,
        )

        # Save model
        rf_path = output_dir / "random_forest.pkl"
        rf_model.save(str(rf_path))
        logger.info(f"Random Forest model saved to {rf_path}")

        results["Random Forest"] = {
            "model": rf_model,
            "metrics": rf_metrics,
            "y_pred": rf_pred,
            "y_prob": rf_prob,
        }

    if args.model_type in ["xgboost", "both"]:
        xgb_model, xgb_metrics, xgb_pred, xgb_prob = train_xgboost(
            X_train_processed, y_train_processed, X_test_processed, y_test, config
        )

        # Save model
        xgb_path = output_dir / "xgboost.pkl"
        xgb_model.save(str(xgb_path))
        logger.info(f"XGBoost model saved to {xgb_path}")

        results["XGBoost"] = {
            "model": xgb_model,
            "metrics": xgb_metrics,
            "y_pred": xgb_pred,
            "y_prob": xgb_prob,
        }

    # Compare models if both trained
    if len(results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)

        evaluator = ModelEvaluator()
        metrics_list = [results[name]["metrics"] for name in results]
        evaluator.compare_models(metrics_list)

    # Save configuration
    config_path = output_dir / "config.json"
    config.save(str(config_path))
    logger.info(f"Configuration saved to {config_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

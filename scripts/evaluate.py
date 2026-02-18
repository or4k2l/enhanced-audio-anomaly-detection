#!/usr/bin/env python
"""Model evaluation script for audio anomaly detection."""

import argparse
import numpy as np
from pathlib import Path

from audio_anom import (
    DataPreprocessor,
    RandomForestAnomalyDetector,
    XGBoostAnomalyDetector,
    ModelEvaluator,
    setup_logger,
)
from audio_anom import visualization as viz

logger = setup_logger("evaluate", level=20)  # INFO level


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate audio anomaly detection models"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=str,
        required=True,
        help="Path to fitted preprocessor",
    )
    parser.add_argument(
        "--test-features",
        type=str,
        required=True,
        help="Path to test features (.npy file)",
    )
    parser.add_argument(
        "--test-labels",
        type=str,
        required=True,
        help="Path to test labels (.npy file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["random_forest", "xgboost"],
        required=True,
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots",
    )

    return parser.parse_args()


def load_model(model_path: str, model_type: str):
    """Load trained model."""
    logger.info(f"Loading {model_type} model from {model_path}")

    if model_type == "random_forest":
        model = RandomForestAnomalyDetector()
    elif model_type == "xgboost":
        model = XGBoostAnomalyDetector()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load(model_path)
    logger.info("Model loaded successfully")

    return model


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessor
    logger.info(f"Loading preprocessor from {args.preprocessor_path}")
    preprocessor = DataPreprocessor.load(args.preprocessor_path)

    # Load test data
    logger.info(f"Loading test features from {args.test_features}")
    X_test = np.load(args.test_features)

    logger.info(f"Loading test labels from {args.test_labels}")
    y_test = np.load(args.test_labels)

    logger.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Preprocess test data
    logger.info("Preprocessing test data...")
    X_test_processed = preprocessor.transform(X_test)
    logger.info(f"Processed test data shape: {X_test_processed.shape}")

    # Load model
    model = load_model(args.model_path, args.model_type)

    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_prob, model_name=args.model_type
    )

    # Print metrics
    for key, value in metrics.items():
        if key != "Model" and value is not None:
            if isinstance(value, float):
                logger.info(f"{key:15s}: {value:.4f}")
            else:
                logger.info(f"{key:15s}: {value}")

    # Print detailed report
    evaluator.print_evaluation_report(y_test, y_pred, model_name=args.model_type)

    # Generate plots if requested
    if args.save_plots:
        logger.info("\nGenerating evaluation plots...")

        # Confusion matrix
        viz.plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"Confusion Matrix - {args.model_type}",
            save_path=str(output_dir / "confusion_matrix.png"),
        )

        # ROC curve
        viz.plot_roc_curve(
            y_test,
            y_prob,
            label=args.model_type,
            save_path=str(output_dir / "roc_curve.png"),
        )

        # Label distribution
        viz.plot_label_distribution(
            y_test,
            title="Test Set Label Distribution",
            save_path=str(output_dir / "label_distribution.png"),
        )

        # Feature importance (if available)
        if hasattr(model.best_estimator_, "feature_importances_"):
            feature_names = preprocessor.get_feature_names()
            importance_dict = model.get_feature_importance(feature_names, top_n=15)

            viz.plot_feature_importance(
                importance_dict,
                title=f"Feature Importance - {args.model_type}",
                save_path=str(output_dir / "feature_importance.png"),
            )

        # PCA variance (if available)
        explained_variance = preprocessor.get_explained_variance_ratio()
        if explained_variance is not None:
            viz.plot_pca_variance(
                explained_variance,
                save_path=str(output_dir / "pca_variance.png"),
            )

        logger.info(f"Plots saved to {output_dir}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

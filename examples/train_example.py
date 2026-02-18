#!/usr/bin/env python
"""Complete training example for audio anomaly detection."""

import numpy as np
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
    build_feature_vector,
    setup_logger,
)
from audio_anom import visualization as viz

logger = setup_logger("train_example")


def generate_synthetic_data(n_samples: int = 500, random_state: int = 42):
    """
    Generate synthetic audio features for demonstration.

    Args:
        n_samples: Number of samples to generate
        random_state: Random state for reproducibility

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    np.random.seed(random_state)

    # Generate features (simulating audio features)
    # Normal samples (70% of data)
    n_normal = int(n_samples * 0.7)
    X_normal = np.random.randn(n_normal, 288) * 0.5 + np.random.rand(288)

    # Anomaly samples (30% of data)
    n_anomaly = n_samples - n_normal
    X_anomaly = np.random.randn(n_anomaly, 288) * 1.5 + np.random.rand(288) * 2

    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def main():
    """Main training example."""
    logger.info("=" * 80)
    logger.info("Audio Anomaly Detection - Complete Training Example")
    logger.info("=" * 80)

    # Configuration
    config = ModelConfig.default()

    # Generate synthetic data for demonstration
    logger.info("\nGenerating synthetic data for demonstration...")
    X, y = generate_synthetic_data(n_samples=500, random_state=42)
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")

    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: Normal={counts[0]}, Anomaly={counts[1]}")

    # Split data
    logger.info("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # ============================================================================
    # PREPROCESSING
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    preprocessor = DataPreprocessor(config=config.preprocessing)

    # Fit and transform training data (with SMOTE)
    X_train_processed, y_train_processed = preprocessor.fit_transform_train(
        X_train, y_train
    )
    logger.info(f"Processed train set: {X_train_processed.shape}")

    # Transform test data (without SMOTE)
    X_test_processed = preprocessor.transform(X_test)
    logger.info(f"Processed test set: {X_test_processed.shape}")

    # Log preprocessing info
    info = preprocessor.get_preprocessing_info()
    logger.info(f"Preprocessing info: {info}")

    # ============================================================================
    # TRAIN RANDOM FOREST
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 80)

    rf_model = RandomForestAnomalyDetector(config=config.random_forest)

    # Train with GridSearchCV
    rf_model.train(X_train_processed, y_train_processed, use_grid_search=True)

    # Evaluate
    rf_pred = rf_model.predict(X_test_processed)
    rf_prob = rf_model.predict_proba(X_test_processed)[:, 1]

    evaluator = ModelEvaluator()
    rf_metrics = evaluator.evaluate_model(y_test, rf_pred, rf_prob, "Random Forest")

    logger.info("\nRandom Forest Results:")
    for key, value in rf_metrics.items():
        if key != "Model" and value is not None:
            logger.info(f"  {key}: {value:.4f}")

    # Feature importance
    feature_names = preprocessor.get_feature_names()
    rf_importance = rf_model.get_feature_importance(feature_names, top_n=10)
    logger.info("\nTop 10 Feature Importances:")
    for feat, imp in list(rf_importance.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    # ============================================================================
    # TRAIN XGBOOST
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 80)

    xgb_model = XGBoostAnomalyDetector(config=config.xgboost)

    # Train
    xgb_model.train(X_train_processed, y_train_processed)

    # Evaluate
    xgb_pred = xgb_model.predict(X_test_processed)
    xgb_prob = xgb_model.predict_proba(X_test_processed)[:, 1]

    xgb_metrics = evaluator.evaluate_model(y_test, xgb_pred, xgb_prob, "XGBoost")

    logger.info("\nXGBoost Results:")
    for key, value in xgb_metrics.items():
        if key != "Model" and value is not None:
            logger.info(f"  {key}: {value:.4f}")

    # ============================================================================
    # MODEL COMPARISON
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    results_list = [rf_metrics, xgb_metrics]
    df_comparison = evaluator.compare_models(results_list)

    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    output_dir = Path("./example_outputs")
    output_dir.mkdir(exist_ok=True)

    # Confusion matrices
    viz.plot_confusion_matrix(
        y_test,
        rf_pred,
        title="Confusion Matrix - Random Forest",
        save_path=str(output_dir / "rf_confusion_matrix.png"),
    )

    viz.plot_confusion_matrix(
        y_test,
        xgb_pred,
        title="Confusion Matrix - XGBoost",
        save_path=str(output_dir / "xgb_confusion_matrix.png"),
    )

    # ROC curves
    viz.plot_multiple_roc_curves(
        [
            ("Random Forest", y_test, rf_prob),
            ("XGBoost", y_test, xgb_prob),
        ],
        save_path=str(output_dir / "roc_curves.png"),
    )

    # Feature importance
    viz.plot_feature_importance(
        rf_importance,
        title="Feature Importance - Random Forest",
        top_n=10,
        save_path=str(output_dir / "feature_importance.png"),
    )

    # PCA variance
    explained_variance = preprocessor.get_explained_variance_ratio()
    if explained_variance is not None:
        viz.plot_pca_variance(
            explained_variance,
            save_path=str(output_dir / "pca_variance.png"),
        )

    # Model comparison
    viz.plot_model_comparison(
        df_comparison,
        save_path=str(output_dir / "model_comparison.png"),
    )

    # Comprehensive dashboard
    models_results = {
        "Random Forest": {
            "y_pred": rf_pred,
            "y_prob": rf_prob,
            "model": rf_model.best_estimator_,
        },
        "XGBoost": {
            "y_pred": xgb_pred,
            "y_prob": xgb_prob,
            "model": xgb_model.best_estimator_,
        },
    }

    viz.create_evaluation_dashboard(
        y_test,
        models_results,
        save_dir=str(output_dir),
    )

    logger.info(f"\nVisualizations saved to: {output_dir}")

    # ============================================================================
    # SAVE MODELS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODELS AND PREPROCESSOR")
    logger.info("=" * 80)

    models_dir = Path("./example_models")
    models_dir.mkdir(exist_ok=True)

    # Save preprocessor
    preprocessor.save(str(models_dir / "preprocessor.pkl"))
    logger.info(f"Preprocessor saved to: {models_dir / 'preprocessor.pkl'}")

    # Save Random Forest
    rf_model.save(str(models_dir / "random_forest.pkl"))
    logger.info(f"Random Forest saved to: {models_dir / 'random_forest.pkl'}")

    # Save XGBoost
    xgb_model.save(str(models_dir / "xgboost.pkl"))
    logger.info(f"XGBoost saved to: {models_dir / 'xgboost.pkl'}")

    # Save config
    config.save(str(models_dir / "config.json"))
    logger.info(f"Configuration saved to: {models_dir / 'config.json'}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"\nModels saved to: {models_dir}")
    logger.info(f"Visualizations saved to: {output_dir}")
    logger.info("\nBest Model: " + df_comparison.loc[df_comparison["F1-Score"].idxmax(), "Model"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

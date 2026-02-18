#!/usr/bin/env python
"""Complete working example of unsupervised anomaly detection.

This example demonstrates the complete workflow:
1. Generate synthetic audio data (normal and anomaly)
2. Extract audio features
3. Preprocess with StandardScaler + PCA
4. Train unsupervised models (LOF, Isolation Forest, Elliptic Envelope)
5. Evaluate and compare models
6. Generate visualizations

Usage:
    python examples/unsupervised_example.py
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_anom import setup_logger
from audio_anom.unsupervised_anomaly import (
    LocalOutlierFactorAnomalyDetector,
    IsolationForestAnomalyDetector,
    EllipticEnvelopeAnomalyDetector,
)
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor
from audio_anom.evaluation_unsupervised import ModelComparator
from audio_anom.visualization_unsupervised import (
    plot_confusion_matrix,
    plot_multiple_roc_curves,
    create_results_summary_figure,
)

logger = setup_logger("unsupervised_example")


def generate_synthetic_data(n_train=1000, n_test_normal=300, n_test_anomaly=200, n_features=284):
    """Generate synthetic audio feature data.
    
    Simulates real audio features with:
    - Normal data: Gaussian distribution
    - Anomaly data: Shifted/scaled Gaussian distribution
    
    Args:
        n_train: Number of training samples (normal only)
        n_test_normal: Number of normal test samples
        n_test_anomaly: Number of anomaly test samples
        n_features: Number of features (default: 284 = mel_spec + mfcc + stats)
        
    Returns:
        X_train_normal: Training data (normal only)
        X_test: Test data (mixed)
        y_test: Test labels (0=normal, 1=anomaly)
    """
    logger.info("Generating synthetic audio feature data...")
    
    np.random.seed(42)
    
    # Normal distribution parameters
    mean_normal = np.random.randn(n_features) * 0.5
    cov_normal = np.eye(n_features) * 0.5
    
    # Training data: normal only
    X_train_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_train)
    
    # Test data: normal samples
    X_test_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_test_normal)
    
    # Test data: anomaly samples (shifted distribution)
    mean_anomaly = mean_normal + np.random.randn(n_features) * 2.0
    cov_anomaly = np.eye(n_features) * 1.5
    X_test_anomaly = np.random.multivariate_normal(mean_anomaly, cov_anomaly, n_test_anomaly)
    
    # Combine test data
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * n_test_normal + [1] * n_test_anomaly)
    
    # Shuffle test set
    indices = np.random.permutation(len(y_test))
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    logger.info(f"Generated data:")
    logger.info(f"  Train (normal only): {X_train_normal.shape}")
    logger.info(f"  Test (mixed): {X_test.shape}")
    logger.info(f"  Test labels: {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomaly")
    
    return X_train_normal, X_test, y_test


def main():
    """Main example workflow."""
    logger.info("\n" + "="*80)
    logger.info("UNSUPERVISED ANOMALY DETECTION - COMPLETE EXAMPLE")
    logger.info("="*80)
    
    # Step 1: Generate synthetic data
    logger.info("\nStep 1: Generate Synthetic Data")
    logger.info("-"*80)
    X_train_normal, X_test, y_test = generate_synthetic_data()
    
    # Step 2: Preprocess data
    logger.info("\nStep 2: Preprocess Data")
    logger.info("-"*80)
    logger.info("Applying StandardScaler + PCA (10 components)...")
    
    preprocessor = UnsupervisedPreprocessor(n_components=10, apply_pca=True)
    X_train_proc = preprocessor.fit_transform(X_train_normal)
    X_test_proc = preprocessor.transform(X_test)
    
    logger.info(f"Original features: {X_train_normal.shape[1]}")
    logger.info(f"Reduced features: {X_train_proc.shape[1]}")
    if preprocessor.explained_variance_ratio_ is not None:
        total_variance = np.sum(preprocessor.explained_variance_ratio_)
        logger.info(f"Variance explained: {total_variance:.2%}")
    
    # Step 3: Train models
    logger.info("\nStep 3: Train Unsupervised Models")
    logger.info("-"*80)
    
    models = {}
    
    # Local Outlier Factor (LOF)
    logger.info("\nTraining LOF (Local Outlier Factor)...")
    lof_model = LocalOutlierFactorAnomalyDetector(n_neighbors=20, contamination=0.1)
    lof_model.fit(X_train_proc)
    models['LOF'] = lof_model
    logger.info("✓ LOF training complete")
    
    # Isolation Forest
    logger.info("\nTraining Isolation Forest...")
    iforest_model = IsolationForestAnomalyDetector(n_estimators=100, contamination=0.1)
    iforest_model.fit(X_train_proc)
    models['Isolation Forest'] = iforest_model
    logger.info("✓ Isolation Forest training complete")
    
    # Elliptic Envelope
    logger.info("\nTraining Elliptic Envelope...")
    envelope_model = EllipticEnvelopeAnomalyDetector(contamination=0.1)
    envelope_model.fit(X_train_proc)
    models['Elliptic Envelope'] = envelope_model
    logger.info("✓ Elliptic Envelope training complete")
    
    # Step 4: Evaluate and compare models
    logger.info("\nStep 4: Evaluate and Compare Models")
    logger.info("-"*80)
    
    comparator = ModelComparator()
    
    for name, model in models.items():
        comparator.add_model(name, model, X_test_proc, y_test)
    
    # Print comparison
    comparator.print_summary()
    comparison_df = comparator.get_comparison()
    
    # Step 5: Generate visualizations
    logger.info("\nStep 5: Generate Visualizations")
    logger.info("-"*80)
    
    output_dir = Path("./output_unsupervised_example")
    output_dir.mkdir(exist_ok=True)
    
    # Individual confusion matrices
    for name, model in models.items():
        y_pred = model.predict(X_test_proc)
        plot_confusion_matrix(
            y_test, y_pred,
            title=f"{name} - Confusion Matrix",
            save_path=output_dir / f"{name.replace(' ', '_').lower()}_confusion_matrix.png"
        )
    
    # Combined ROC curves
    roc_data = {}
    for name, model in models.items():
        y_score = model.anomaly_score(X_test_proc)
        roc_data[name] = (y_test, y_score)
    
    plot_multiple_roc_curves(
        roc_data,
        title="ROC Curves - Model Comparison",
        save_path=output_dir / "roc_curves_comparison.png"
    )
    
    # Summary figure
    create_results_summary_figure(
        comparison_df,
        roc_data=roc_data,
        title="Unsupervised Anomaly Detection - Results Summary",
        save_path=output_dir / "results_summary.png"
    )
    
    logger.info(f"Visualizations saved to {output_dir}/")
    
    # Step 6: Save models
    logger.info("\nStep 6: Save Models")
    logger.info("-"*80)
    
    # Save best model (LOF typically performs best)
    best_model_name = comparator.get_best_model('roc_auc')
    best_model = models[best_model_name]
    
    model_path = output_dir / "best_model.pkl"
    preprocessor_path = output_dir / "preprocessor.pkl"
    
    best_model.save(str(model_path))
    preprocessor.save(str(preprocessor_path))
    
    logger.info(f"Best model ({best_model_name}) saved to {model_path}")
    logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    # Step 7: Demonstrate inference
    logger.info("\nStep 7: Demonstrate Inference on New Data")
    logger.info("-"*80)
    
    # Generate a few new samples
    new_normal = np.random.randn(5, X_test.shape[1])
    new_anomaly = np.random.randn(5, X_test.shape[1]) * 3  # More extreme values
    
    # Preprocess
    new_normal_proc = preprocessor.transform(new_normal)
    new_anomaly_proc = preprocessor.transform(new_anomaly)
    
    # Predict
    logger.info("\nPredictions on new normal samples:")
    for i, sample in enumerate(new_normal_proc, 1):
        pred = best_model.predict(sample.reshape(1, -1))[0]
        score = best_model.anomaly_score(sample.reshape(1, -1))[0]
        logger.info(f"  Sample {i}: {'ANOMALY' if pred == 1 else 'NORMAL'} (score: {score:.4f})")
    
    logger.info("\nPredictions on new anomaly samples:")
    for i, sample in enumerate(new_anomaly_proc, 1):
        pred = best_model.predict(sample.reshape(1, -1))[0]
        score = best_model.anomaly_score(sample.reshape(1, -1))[0]
        logger.info(f"  Sample {i}: {'ANOMALY' if pred == 1 else 'NORMAL'} (score: {score:.4f})")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE COMPLETE")
    logger.info("="*80)
    logger.info("\nKey Takeaways:")
    logger.info("  ✓ Trained on normal data only (unsupervised learning)")
    logger.info("  ✓ Successfully detected anomalies in test set")
    logger.info(f"  ✓ Best method: {best_model_name}")
    logger.info(f"  ✓ Best ROC-AUC: {comparison_df.loc[best_model_name, 'roc_auc']:.4f}")
    logger.info(f"  ✓ Models and visualizations saved to {output_dir}/")
    logger.info("\nNext Steps:")
    logger.info("  1. Replace synthetic data with real audio features")
    logger.info("  2. Tune contamination parameter based on your dataset")
    logger.info("  3. Deploy best model to production using deploy_production.py")
    logger.info("="*80)


if __name__ == "__main__":
    main()

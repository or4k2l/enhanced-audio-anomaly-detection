#!/usr/bin/env python
"""Training script for unsupervised anomaly detection on DCASE 2020 Task 2 dataset.

This script trains unsupervised anomaly detection models on real-world industrial
machine sounds. It supports three methods:
- Local Outlier Factor (LOF)
- Isolation Forest
- Elliptic Envelope

Usage:
    python scripts/train_unsupervised.py --machine fan --contamination 0.1
    python scripts/train_unsupervised.py --machine pump --method lof --output models/
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_anom import (
    AudioFeatureExtractor,
    AudioDataProcessor,
    build_feature_vector,
    setup_logger,
)
from audio_anom.unsupervised_anomaly import create_detector
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor
from audio_anom.evaluation_unsupervised import evaluate_anomaly_detector, ModelComparator
from audio_anom.visualization_unsupervised import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_anomaly_scores_distribution,
    create_results_summary_figure,
)

logger = setup_logger("train_unsupervised")


def load_dc2020_data(
    machine_type: str,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load DCASE 2020 Task 2 data for a specific machine type.
    
    This is a placeholder function. In production, you would:
    1. Download data using kagglehub or from https://zenodo.org/record/3678171
    2. Load audio files from train/ (normal only) and test/ (mixed) directories
    3. Extract features using AudioFeatureExtractor
    4. Return train (normal only) and test (mixed) data
    
    Args:
        machine_type: Machine type (e.g., 'fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor')
        data_dir: Root directory containing the data
        
    Returns:
        X_train_normal: Training features (normal samples only)
        X_test: Test features (mixed normal and anomaly)
        y_test: Test labels (0 for normal, 1 for anomaly)
        test_file_names: List of test file names
    """
    logger.warning("Using synthetic data for demonstration. Replace with actual DC2020 data loader.")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    
    # Training: 1000 normal samples, 284 features (mel_spec + mfcc + stats)
    X_train_normal = np.random.randn(1000, 284) + np.random.randn(284)
    
    # Test: 600 normal + 400 anomaly samples
    X_test_normal = np.random.randn(600, 284) + np.random.randn(284)
    X_test_anomaly = np.random.randn(400, 284) + 2 * np.random.randn(284)  # Different distribution
    
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * 600 + [1] * 400)
    
    # Shuffle test set
    indices = np.random.permutation(len(y_test))
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    test_file_names = [f"{machine_type}_test_{i:04d}.wav" for i in range(len(y_test))]
    
    logger.info(f"Loaded {machine_type} data:")
    logger.info(f"  Train (normal only): {X_train_normal.shape}")
    logger.info(f"  Test (mixed): {X_test.shape}")
    logger.info(f"  Test labels: {np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomaly")
    
    return X_train_normal, X_test, y_test, test_file_names


def train_model(
    X_train: np.ndarray,
    method: str = 'lof',
    contamination: float = 0.1,
) -> tuple:
    """Train unsupervised anomaly detection model.
    
    Args:
        X_train: Training data (normal samples only)
        method: Detection method ('lof', 'isolation_forest', 'elliptic_envelope')
        contamination: Expected proportion of outliers
        
    Returns:
        model: Trained model
        preprocessor: Fitted preprocessor
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {method.upper()} with contamination={contamination}")
    logger.info(f"{'='*80}")
    
    # Preprocess
    logger.info("Preprocessing training data...")
    preprocessor = UnsupervisedPreprocessor(n_components=10, apply_pca=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    logger.info(f"Preprocessed shape: {X_train_proc.shape}")
    
    # Train model
    logger.info(f"Training {method} model...")
    model = create_detector(method, contamination=contamination)
    model.fit(X_train_proc)
    
    logger.info("Training complete!")
    return model, preprocessor


def evaluate_model(
    model,
    preprocessor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method_name: str,
    output_dir: Path,
) -> Dict[str, float]:
    """Evaluate trained model on test data.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X_test: Test data
        y_test: True labels
        method_name: Name of the method
        output_dir: Directory to save results
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {method_name}")
    logger.info(f"{'='*80}")
    
    # Preprocess test data
    X_test_proc = preprocessor.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_proc)
    y_score = model.anomaly_score(X_test_proc)
    
    # Evaluate
    metrics = evaluate_anomaly_detector(y_test, y_pred, y_score, method_name)
    
    # Create visualizations
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"{method_name} - Confusion Matrix",
        save_path=viz_dir / f"{method_name}_confusion_matrix.png",
    )
    
    # ROC curve
    plot_roc_curve(
        y_test, y_score,
        title=f"{method_name} - ROC Curve",
        model_name=method_name,
        save_path=viz_dir / f"{method_name}_roc_curve.png",
    )
    
    # Anomaly score distribution
    plot_anomaly_scores_distribution(
        y_test, y_score,
        title=f"{method_name} - Anomaly Score Distribution",
        save_path=viz_dir / f"{method_name}_score_distribution.png",
    )
    
    logger.info(f"Visualizations saved to {viz_dir}")
    
    return metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train unsupervised anomaly detection models on DCASE 2020 Task 2 data"
    )
    parser.add_argument(
        '--machine',
        type=str,
        default='fan',
        choices=['fan', 'pump', 'slider', 'valve', 'ToyCar', 'ToyConveyor'],
        help='Machine type to train on'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='all',
        choices=['lof', 'isolation_forest', 'elliptic_envelope', 'all'],
        help='Detection method (default: train all methods)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Expected proportion of outliers in training data (default: 0.1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./models_unsupervised',
        help='Output directory for trained models (default: ./models_unsupervised)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Root directory containing DCASE 2020 data'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output) / args.machine
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("UNSUPERVISED ANOMALY DETECTION TRAINING")
    logger.info(f"{'='*80}")
    logger.info(f"Machine Type: {args.machine}")
    logger.info(f"Method(s): {args.method}")
    logger.info(f"Contamination: {args.contamination}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"{'='*80}\n")
    
    # Load data
    logger.info("Loading data...")
    X_train_normal, X_test, y_test, test_file_names = load_dc2020_data(
        args.machine,
        args.data_dir,
    )
    
    # Determine methods to train
    if args.method == 'all':
        methods = ['lof', 'isolation_forest', 'elliptic_envelope']
    else:
        methods = [args.method]
    
    # Train and evaluate all methods
    all_results = {}
    comparator = ModelComparator()
    
    for method in methods:
        # Train
        model, preprocessor = train_model(
            X_train_normal,
            method=method,
            contamination=args.contamination,
        )
        
        # Save model and preprocessor
        model_path = output_dir / f"{method}_model.pkl"
        preprocessor_path = output_dir / f"{method}_preprocessor.pkl"
        model.save(str(model_path))
        preprocessor.save(str(preprocessor_path))
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Evaluate
        metrics = evaluate_model(
            model, preprocessor, X_test, y_test,
            method, output_dir,
        )
        all_results[method] = metrics
        
        # Add to comparator
        X_test_proc = preprocessor.transform(X_test)
        comparator.add_model(method, model, X_test_proc, y_test)
    
    # Compare models if multiple methods
    if len(methods) > 1:
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        comparator.print_summary()
        
        # Save comparison
        comparison_df = comparator.get_comparison()
        comparison_path = output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path)
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Create summary figure
        roc_data = {}
        for method in methods:
            model_path = output_dir / f"{method}_model.pkl"
            preprocessor_path = output_dir / f"{method}_preprocessor.pkl"
            model = create_detector(method)
            model.load(str(model_path))
            preprocessor = UnsupervisedPreprocessor.load(str(preprocessor_path))
            X_test_proc = preprocessor.transform(X_test)
            y_score = model.anomaly_score(X_test_proc)
            roc_data[method] = (y_test, y_score)
        
        create_results_summary_figure(
            comparison_df,
            roc_data=roc_data,
            title=f"{args.machine.upper()} - Unsupervised Anomaly Detection Results",
            save_path=output_dir / 'visualizations' / 'results_summary.png',
        )
    
    # Save detailed results
    results_df = pd.DataFrame(all_results).T
    results_path = output_dir / 'results.csv'
    results_df.to_csv(results_path)
    logger.info(f"\nResults saved to {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Visualizations saved to: {output_dir / 'visualizations'}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

"""Evaluation metrics and utilities for unsupervised anomaly detection.

This module provides comprehensive evaluation tools for unsupervised
anomaly detection models:
- ROC-AUC, F1-Score, Accuracy, Precision, Recall
- Confusion matrices
- Classification reports
- Model comparison utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from .logger import get_logger

logger = get_logger(__name__)


def evaluate_anomaly_detector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Evaluate anomaly detector performance.
    
    Computes comprehensive metrics for binary anomaly detection:
    - ROC-AUC: Area under ROC curve (requires y_score)
    - F1-Score: Harmonic mean of precision and recall
    - Accuracy: Overall correctness
    - Precision: True anomalies / predicted anomalies
    - Recall: True anomalies detected / total anomalies
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_pred: Predicted labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores (optional, for ROC-AUC)
        model_name: Name of the model for logging
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # ROC-AUC (requires scores)
    if y_score is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_score)
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = None
    
    # Log results
    logger.info(f"\n{model_name} Evaluation Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix.
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_pred: Predicted labels (0 for normal, 1 for anomaly)
        
    Returns:
        cm: Confusion matrix [[TN, FP], [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Generate classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for each class (default: ['Normal', 'Anomaly'])
        
    Returns:
        report: Classification report string
    """
    if target_names is None:
        target_names = ['Normal', 'Anomaly']
    
    return classification_report(y_true, y_pred, target_names=target_names)


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve.
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
    """
    return roc_curve(y_true, y_score)


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'roc_auc',
) -> pd.DataFrame:
    """Compare multiple models based on evaluation metrics.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metric: Primary metric to sort by (default: 'roc_auc')
        
    Returns:
        comparison_df: DataFrame with model comparison
        
    Example:
        >>> results = {
        ...     'LOF': {'roc_auc': 0.755, 'f1_score': 0.704},
        ...     'Isolation Forest': {'roc_auc': 0.687, 'f1_score': 0.637},
        ...     'Elliptic Envelope': {'roc_auc': 0.643, 'f1_score': 0.528},
        ... }
        >>> df = compare_models(results)
    """
    df = pd.DataFrame(results).T
    
    # Sort by specified metric (descending)
    if metric in df.columns:
        df = df.sort_values(by=metric, ascending=False)
    
    return df


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = 'f1',
) -> Tuple[float, float]:
    """Find optimal decision threshold for anomaly scores.
    
    Searches for the threshold that maximizes the specified metric.
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
        
    Returns:
        optimal_threshold: Best threshold value
        best_score: Best metric value
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    best_score = 0.0
    optimal_threshold = 0.0
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            optimal_threshold = threshold
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} "
               f"({metric}={best_score:.4f})")
    
    return optimal_threshold, best_score


class ModelComparator:
    """Compare multiple anomaly detection models.
    
    This class facilitates easy comparison of multiple models on the same dataset.
    
    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('LOF', lof_model, X_test, y_test)
        >>> comparator.add_model('IForest', iforest_model, X_test, y_test)
        >>> results_df = comparator.get_comparison()
        >>> comparator.print_summary()
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.results = {}
        
    def add_model(
        self,
        name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Add a model to compare.
        
        Args:
            name: Model name
            model: Fitted anomaly detector (must have predict and anomaly_score methods)
            X_test: Test data
            y_test: True labels
        """
        logger.info(f"\nEvaluating {name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_score = model.anomaly_score(X_test)
        
        # Evaluate
        metrics = evaluate_anomaly_detector(y_test, y_pred, y_score, name)
        
        # Store results
        self.results[name] = metrics
        
    def get_comparison(self, metric: str = 'roc_auc') -> pd.DataFrame:
        """Get comparison DataFrame sorted by metric.
        
        Args:
            metric: Metric to sort by
            
        Returns:
            comparison_df: Model comparison DataFrame
        """
        return compare_models(self.results, metric=metric)
        
    def print_summary(self) -> None:
        """Print formatted summary of all models."""
        if not self.results:
            logger.info("No models to compare")
            return
        
        df = self.get_comparison()
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("="*80)
        logger.info(f"\n{df.to_string()}")
        logger.info("\n" + "="*80)
        
        # Highlight best model
        best_model = df.index[0]
        best_auc = df.loc[best_model, 'roc_auc'] if 'roc_auc' in df.columns else None
        best_f1 = df.loc[best_model, 'f1_score'] if 'f1_score' in df.columns else None
        
        logger.info(f"\nBest Model: {best_model}")
        if best_auc is not None:
            logger.info(f"  ROC-AUC: {best_auc:.4f}")
        if best_f1 is not None:
            logger.info(f"  F1-Score: {best_f1:.4f}")
        logger.info("="*80)
        
    def get_best_model(self, metric: str = 'roc_auc') -> str:
        """Get the name of the best model.
        
        Args:
            metric: Metric to compare by
            
        Returns:
            best_model_name: Name of best performing model
        """
        df = self.get_comparison(metric=metric)
        return df.index[0]


def evaluate_machine_type(
    machine_type: str,
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate multiple models on a specific machine type.
    
    Args:
        machine_type: Name of machine type (e.g., 'fan', 'pump')
        models: Dictionary mapping method names to fitted models
        X_test: Test data
        y_test: True labels
        
    Returns:
        results_df: DataFrame with results for all models
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating Machine Type: {machine_type.upper()}")
    logger.info(f"{'='*80}")
    
    comparator = ModelComparator()
    
    for method_name, model in models.items():
        comparator.add_model(method_name, model, X_test, y_test)
    
    results_df = comparator.get_comparison()
    comparator.print_summary()
    
    # Add machine type column
    results_df['machine_type'] = machine_type
    
    return results_df

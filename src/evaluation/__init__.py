"""Evaluation package for anomaly detection metrics and visualization."""

from .metrics import compute_auc, evaluate_detector, EvaluationResult
from .visualization import (
    plot_roc_curve,
    plot_score_distribution,
    plot_results_comparison,
)

__all__ = [
    "compute_auc",
    "evaluate_detector",
    "EvaluationResult",
    "plot_roc_curve",
    "plot_score_distribution",
    "plot_results_comparison",
]

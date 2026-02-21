"""Evaluation metrics for anomaly detection.

Provides AUC, ROC, and summary statistics for comparing detectors.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for anomaly detection evaluation results.

    Attributes:
        auc: Area Under the ROC Curve.
        average_precision: Area Under the Precision-Recall Curve.
        threshold: Decision threshold.
        fpr: False positive rates for ROC curve.
        tpr: True positive rates for ROC curve.
        confusion: Confusion matrix as a 2x2 numpy array.
        metadata: Arbitrary extra metadata.
    """

    auc: float
    average_precision: float
    threshold: float = 0.5
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    confusion: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"AUC={self.auc:.4f}, "
            f"AP={self.average_precision:.4f}, "
            f"threshold={self.threshold:.4f}"
        )


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute the Area Under the ROC Curve.

    Args:
        y_true: Ground-truth binary labels (0=normal, 1=anomaly).
        scores: Anomaly scores (higher = more anomalous).

    Returns:
        AUC score in [0, 1].
    """
    return float(roc_auc_score(y_true, scores))


def evaluate_detector(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> EvaluationResult:
    """Compute comprehensive evaluation metrics for an anomaly detector.

    Args:
        y_true: Ground-truth binary labels (0=normal, 1=anomaly).
        scores: Anomaly scores (higher = more anomalous).
        threshold: Decision threshold. If None, uses the median score.

    Returns:
        EvaluationResult with AUC, AP, ROC curve, and confusion matrix.
    """
    auc = float(roc_auc_score(y_true, scores))
    ap = float(average_precision_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)

    if threshold is None:
        threshold = float(np.median(scores))

    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"Evaluation: AUC={auc:.4f}, AP={ap:.4f}")

    return EvaluationResult(
        auc=auc,
        average_precision=ap,
        threshold=threshold,
        fpr=fpr,
        tpr=tpr,
        confusion=cm,
    )


def evaluate_all_machines(
    results_dict: Dict[str, EvaluationResult],
) -> Dict[str, float]:
    """Summarize AUC results across multiple machine types.

    Args:
        results_dict: Mapping of machine_name → EvaluationResult.

    Returns:
        Dictionary with per-machine AUC values and the macro average.
    """
    summary = {name: result.auc for name, result in results_dict.items()}
    summary["average"] = float(np.mean(list(summary.values())))
    return summary

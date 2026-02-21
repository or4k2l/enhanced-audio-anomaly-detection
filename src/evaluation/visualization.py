"""Visualization utilities for anomaly detection evaluation.

Provides ROC curves, score distribution histograms, and result comparison plots.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .metrics import EvaluationResult

logger = logging.getLogger(__name__)


def plot_roc_curve(
    result: EvaluationResult,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a single ROC curve.

    Args:
        result: EvaluationResult containing fpr, tpr, and auc.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        result.fpr,
        result.tpr,
        color="steelblue",
        lw=2,
        label=f"ROC (AUC = {result.auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot histograms of anomaly scores for normal vs. anomalous samples.

    Args:
        scores: Anomaly scores of shape (n_samples,).
        labels: Ground-truth labels (0=normal, 1=anomaly).
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        scores[labels == 0],
        bins=40,
        alpha=0.6,
        color="steelblue",
        label="Normal",
        density=True,
    )
    ax.hist(
        scores[labels == 1],
        bins=40,
        alpha=0.6,
        color="crimson",
        label="Anomaly",
        density=True,
    )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Score distribution saved to {save_path}")

    return fig


def plot_results_comparison(
    results: Dict[str, Dict[str, float]],
    methods: Optional[List[str]] = None,
    title: str = "AUC Comparison Across Methods",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing AUC across machines and detection methods.

    Args:
        results: Nested dict {method_name: {machine_name: auc_value}}.
        methods: Subset of methods to include. If None, all are included.
        title: Plot title.
        save_path: If given, save the figure to this path.

    Returns:
        Matplotlib figure.
    """
    if methods is None:
        methods = list(results.keys())

    # Collect machines (exclude 'average' key)
    machines = sorted(
        {m for method in methods for m in results[method] if m != "average"}
    )

    x = np.arange(len(machines))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(machines) * 1.5), 5))

    for i, method in enumerate(methods):
        aucs = [results[method].get(machine, 0.0) for machine in machines]
        ax.bar(x + i * width, aucs, width, label=method, alpha=0.85)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(machines, rotation=30, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim([0.0, 1.0])
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, label="Random")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info(f"Comparison chart saved to {save_path}")

    return fig

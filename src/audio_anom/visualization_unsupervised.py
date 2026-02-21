"""Visualization utilities for unsupervised anomaly detection.

This module provides professional visualizations for unsupervised
anomaly detection results:
- Confusion matrices (heatmaps)
- ROC curves with AUC scores
- Performance metric comparisons
- Anomaly score distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from .evaluation_unsupervised import compute_roc_curve, get_confusion_matrix
from .logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot confusion matrix as heatmap.

    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_pred: Predicted labels (0 for normal, 1 for anomaly)
        title: Plot title
        class_names: Names for each class (default: ['Normal', 'Anomaly'])
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    if class_names is None:
        class_names = ["Normal", "Anomaly"]

    cm = get_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot ROC curve with AUC score.

    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores
        title: Plot title
        model_name: Model name for legend
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.metrics import roc_auc_score

    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc:.3f})",
        color="#2E86AB",
    )

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_multiple_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
) -> plt.Figure:
    """Plot multiple ROC curves for model comparison.

    Args:
        results: Dictionary mapping model names to (y_true, y_score) tuples
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.metrics import roc_auc_score

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    for idx, (model_name, (y_true, y_score)) in enumerate(results.items()):
        fpr, tpr, _ = compute_roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        color = colors[idx % len(colors)]
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {auc:.3f})",
            color=color,
        )

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Multi-ROC curve saved to {save_path}")

    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot bar chart comparing models by metric.

    Args:
        results_df: DataFrame with model results (models as index)
        metric: Metric to plot
        title: Plot title (auto-generated if None)
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    if title is None:
        metric_name = metric.replace("_", " ").title()
        title = f"Model Comparison - {metric_name}"

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by metric
    sorted_df = results_df.sort_values(by=metric, ascending=True)

    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df)))
    bars = ax.barh(sorted_df.index, sorted_df[metric], color=colors)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metric comparison saved to {save_path}")

    return fig


def plot_anomaly_scores_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot distribution of anomaly scores for normal and anomaly classes.

    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_score: Anomaly scores
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate scores by true class
    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]

    # Plot histograms
    ax.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label=f"Normal (n={len(normal_scores)})",
        color="#2E86AB",
        edgecolor="black",
    )
    ax.hist(
        anomaly_scores,
        bins=50,
        alpha=0.6,
        label=f"Anomaly (n={len(anomaly_scores)})",
        color="#C73E1D",
        edgecolor="black",
    )

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Score distribution saved to {save_path}")

    return fig


def plot_model_comparison_grid(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Plot grid of bar charts comparing models across multiple metrics.

    Args:
        results_df: DataFrame with model results
        metrics: List of metrics to plot (default: all numeric columns)
        title: Overall plot title
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure
    """
    if metrics is None:
        # Use all numeric columns
        metrics = results_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove None values if present
        metrics = [m for m in metrics if results_df[m].notna().any()]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Sort by metric
        sorted_df = results_df.sort_values(by=metric, ascending=True)

        # Create horizontal bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df)))
        bars = ax.barh(sorted_df.index, sorted_df[metric], color=colors)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if not np.isnan(width):
                ax.text(
                    width,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    ha="left",
                    va="center",
                    fontsize=9,
                )

        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim([0, 1.05])

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Comparison grid saved to {save_path}")

    return fig


def create_results_summary_figure(
    results_df: pd.DataFrame,
    roc_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    title: str = "Unsupervised Anomaly Detection Results",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive results summary figure.

    Combines multiple plots into one summary figure:
    - ROC curves (if roc_data provided)
    - Metric comparison bars
    - Results table

    Args:
        results_df: DataFrame with model results
        roc_data: Optional dictionary mapping model names to (y_true, y_score)
        title: Overall figure title
        save_path: Path to save the figure (optional)

    Returns:
        fig: Matplotlib figure
    """
    from sklearn.metrics import roc_auc_score

    if roc_data:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # ROC curves (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ["#2E86AB", "#A23B72", "#F18F01"]
        for idx, (model_name, (y_true, y_score)) in enumerate(roc_data.items()):
            fpr, tpr, _ = compute_roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax1.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{model_name} (AUC={auc:.3f})",
                color=colors[idx % len(colors)],
            )
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curves", fontweight="bold")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ROC-AUC comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_df = results_df.sort_values(by="roc_auc", ascending=True)
        colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df)))
        ax2.barh(sorted_df.index, sorted_df["roc_auc"], color=colors_bar)
        for idx, (name, value) in enumerate(zip(sorted_df.index, sorted_df["roc_auc"])):
            ax2.text(value, idx, f"{value:.3f}", ha="left", va="center", fontsize=9)
        ax2.set_xlabel("ROC-AUC Score")
        ax2.set_title("Model Performance (ROC-AUC)", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")

        # F1-Score comparison (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        sorted_df_f1 = results_df.sort_values(by="f1_score", ascending=True)
        ax3.barh(sorted_df_f1.index, sorted_df_f1["f1_score"], color=colors_bar)
        for idx, (name, value) in enumerate(
            zip(sorted_df_f1.index, sorted_df_f1["f1_score"])
        ):
            ax3.text(value, idx, f"{value:.3f}", ha="left", va="center", fontsize=9)
        ax3.set_xlabel("F1-Score")
        ax3.set_title("Model Performance (F1-Score)", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="x")

        # Results table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        table_data = results_df[["roc_auc", "f1_score", "precision", "recall"]].round(4)
        table = ax4.table(
            cellText=table_data.values,
            colLabels=["ROC-AUC", "F1", "Prec.", "Rec."],
            rowLabels=table_data.index,
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.2, 0.2, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax4.set_title("Detailed Metrics", fontweight="bold")
    else:
        # Simpler layout without ROC curves
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        metrics = ["roc_auc", "f1_score", "precision", "recall"]
        for idx, metric in enumerate(metrics):
            if metric not in results_df.columns:
                continue
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            sorted_df = results_df.sort_values(by=metric, ascending=True)
            colors_bar = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_df)))
            ax.barh(sorted_df.index, sorted_df[metric], color=colors_bar)
            for i, (name, value) in enumerate(zip(sorted_df.index, sorted_df[metric])):
                if not np.isnan(value):
                    ax.text(
                        value, i, f"{value:.3f}", ha="left", va="center", fontsize=9
                    )
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Results summary saved to {save_path}")

    return fig

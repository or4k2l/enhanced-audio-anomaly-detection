"""Visualization utilities for audio anomaly detection."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

# Set seaborn style
sns.set_style("whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels for display
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting confusion matrix: {title}")

    if labels is None:
        labels = ["Normal", "Anomaly"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_feature_importance(
    importance_dict: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot feature importance as horizontal bar chart.

    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        title: Plot title
        top_n: Number of top features to display
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting feature importance: {title}")

    # Sort and get top N
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    feature_names = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances, color="skyblue", edgecolor="navy")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Highest at top
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    label: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot ROC curve for a single model.

    Args:
        y_true: True labels
        y_scores: Predicted probabilities for positive class
        label: Model label for legend
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting ROC curve: {label}")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    plt.show()


def plot_multiple_roc_curves(
    models_data: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot ROC curves for multiple models on the same plot.

    Args:
        models_data: List of tuples (model_name, y_true, y_scores)
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug("Plotting multiple ROC curves")

    fig, ax = plt.subplots(figsize=figsize)

    for model_name, y_true, y_scores in models_data:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = roc_auc_score(y_true, y_scores)
            ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_score:.3f})")
        except Exception as e:
            logger.warning(f"Could not plot ROC for {model_name}: {e}")

    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Multiple ROC curves saved to {save_path}")

    plt.show()


def plot_label_distribution(
    y: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Label Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot label distribution as bar chart.

    Args:
        y: Labels array
        labels: Class labels for display
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting label distribution: {title}")

    if labels is None:
        labels = ["Normal", "Anomaly"]

    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["green", "red"]
    ax.bar(range(len(unique)), counts, color=colors[: len(unique)], edgecolor="black")

    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([labels[i] for i in unique])
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Label distribution plot saved to {save_path}")

    plt.show()


def plot_classification_report_heatmap(
    classification_report_dict: Dict[str, Any],
    title: str = "Classification Report",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot classification report as heatmap.

    Args:
        classification_report_dict: Classification report as dictionary
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting classification report: {title}")

    # Convert to DataFrame
    df = pd.DataFrame(classification_report_dict).T

    # Remove non-numeric rows
    df = df[df.index.isin(["0", "1", "macro avg", "weighted avg"])]

    # Select numeric columns
    numeric_cols = ["precision", "recall", "f1-score"]
    df_numeric = df[numeric_cols].astype(float)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        df_numeric,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "Score"},
        vmin=0,
        vmax=1,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Class / Average", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Classification report heatmap saved to {save_path}")

    plt.show()


def plot_pca_variance(
    explained_variance_ratio: np.ndarray,
    title: str = "PCA Explained Variance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot PCA explained variance.

    Args:
        explained_variance_ratio: Explained variance ratio from PCA
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting PCA variance: {title}")

    cumsum_var = np.cumsum(explained_variance_ratio)
    n_components = len(explained_variance_ratio)

    fig, ax = plt.subplots(figsize=figsize)

    x = range(1, n_components + 1)
    ax.bar(x, explained_variance_ratio, alpha=0.6, label="Individual", color="skyblue")
    ax.plot(x, cumsum_var, "r-o", label="Cumulative", linewidth=2, markersize=5)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Explained Variance Ratio", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add total variance text
    total_var = cumsum_var[-1]
    ax.text(
        n_components * 0.7,
        total_var * 0.5,
        f"Total: {total_var:.2%}",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"PCA variance plot saved to {save_path}")

    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot model comparison metrics.

    Args:
        results_df: DataFrame with model comparison results
        metrics: List of metrics to plot. If None, uses all numeric columns except 'Model'.
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
    """
    logger.debug(f"Plotting model comparison: {title}")

    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Filter available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]

    if not available_metrics:
        logger.warning("No valid metrics found for plotting")
        return

    fig, ax = plt.subplots(figsize=figsize)

    df_plot = results_df.set_index("Model")[available_metrics]
    df_plot.plot(kind="bar", ax=ax, width=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Model comparison plot saved to {save_path}")

    plt.show()


def create_evaluation_dashboard(
    y_true: np.ndarray,
    models_results: Dict[str, Dict[str, Any]],
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> None:
    """
    Create comprehensive evaluation dashboard with multiple visualizations.

    Args:
        y_true: True labels
        models_results: Dictionary of {model_name: {'y_pred': ..., 'y_prob': ..., 'model': ...}}
        save_dir: Directory to save the dashboard
        figsize: Figure size
    """
    logger.info("Creating evaluation dashboard")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle("Model Evaluation Dashboard", fontsize=16, fontweight="bold")

    # 1. Model comparison metrics
    results_list = []
    for name, data in models_results.items():
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        acc = accuracy_score(y_true, data["y_pred"])
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, data["y_pred"], average="binary", zero_division=0
        )
        results_list.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
            }
        )

    df_results = pd.DataFrame(results_list).set_index("Model")
    df_results.plot(kind="bar", ax=axes[0, 0], legend=True)
    axes[0, 0].set_title("Model Comparison")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Best model confusion matrix
    best_model_name = df_results["F1-Score"].idxmax()
    best_data = models_results[best_model_name]
    cm = confusion_matrix(y_true, best_data["y_pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0, 1],
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    axes[0, 1].set_title(f"Confusion Matrix - {best_model_name}")
    axes[0, 1].set_ylabel("True Label")
    axes[0, 1].set_xlabel("Predicted Label")

    # 3. ROC Curves
    for name, data in models_results.items():
        if data.get("y_prob") is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, data["y_prob"])
                auc = roc_auc_score(y_true, data["y_prob"])
                axes[0, 2].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
            except Exception as e:
                logger.warning(f"Could not plot ROC for {name}: {e}")

    axes[0, 2].plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    axes[0, 2].set_xlabel("False Positive Rate")
    axes[0, 2].set_ylabel("True Positive Rate")
    axes[0, 2].set_title("ROC Curves")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Label distribution
    unique, counts = np.unique(y_true, return_counts=True)
    axes[1, 0].bar(
        range(len(unique)), counts, color=["green", "red"], edgecolor="black"
    )
    axes[1, 0].set_xticks(range(len(unique)))
    axes[1, 0].set_xticklabels(["Normal", "Anomaly"])
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Label Distribution")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for i, count in enumerate(counts):
        axes[1, 0].text(i, count, str(count), ha="center", va="bottom")

    # 5. Feature importance (if available)
    if hasattr(best_data.get("model"), "feature_importances_"):
        importances = best_data["model"].feature_importances_
        top_n = min(10, len(importances))
        indices = np.argsort(importances)[-top_n:]
        axes[1, 1].barh(range(top_n), importances[indices], color="skyblue")
        axes[1, 1].set_yticks(range(top_n))
        axes[1, 1].set_yticklabels([f"PC{i+1}" for i in indices])
        axes[1, 1].set_xlabel("Importance")
        axes[1, 1].set_title("Top 10 Feature Importance")
        axes[1, 1].invert_yaxis()
    else:
        axes[1, 1].text(
            0.5, 0.5, "Feature importance\nnot available", ha="center", va="center"
        )
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    # 6. Metrics table
    axes[1, 2].axis("tight")
    axes[1, 2].axis("off")
    table_data = [
        [
            row["Model"],
            f"{row['Accuracy']:.3f}",
            f"{row['Precision']:.3f}",
            f"{row['Recall']:.3f}",
            f"{row['F1-Score']:.3f}",
        ]
        for _, row in df_results.reset_index().iterrows()
    ]
    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=["Model", "Acc", "Prec", "Rec", "F1"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    axes[1, 2].set_title("Metrics Summary")

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "evaluation_dashboard.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Evaluation dashboard saved to {save_path}")

    plt.show()

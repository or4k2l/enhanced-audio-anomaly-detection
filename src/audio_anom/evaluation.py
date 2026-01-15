"""Model evaluation and visualization utilities."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


class ModelEvaluator:
    """Evaluate and visualize model performance."""

    def __init__(self):
        """Initialize evaluator."""
        sns.set_style("whitegrid")

    def evaluate_model(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        Evaluate model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            model_name: Name of the model

        Returns:
            Dictionary of metrics
        """
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        metrics = {
            'Model': model_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            try:
                auc = roc_auc_score(y_true, y_prob)
                metrics['AUC-ROC'] = auc
            except ValueError as e:
                print(f"Warning: Could not calculate AUC for {model_name} - {e}")
                metrics['AUC-ROC'] = None
        
        return metrics

    def print_evaluation_report(self, y_true, y_pred, model_name="Model"):
        """
        Print detailed evaluation report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
        """
        print(f"\n{'='*60}")
        print(f"{model_name}")
        print(f"{'='*60}")
        
        # Metrics
        metrics = self.evaluate_model(y_true, y_pred, model_name=model_name)
        for key, value in metrics.items():
            if key != 'Model' and value is not None:
                if isinstance(value, float):
                    print(f"{key:12s}: {value:.4f}")
                else:
                    print(f"{key:12s}: {value}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
        print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    def compare_models(self, results_list):
        """
        Compare multiple models.

        Args:
            results_list: List of dictionaries containing evaluation metrics

        Returns:
            DataFrame with comparison results
        """
        df_results = pd.DataFrame(results_list)
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        print(df_results.to_string(index=False))
        
        return df_results

    def plot_model_comparison(self, df_results, save_path=None):
        """
        Plot model comparison metrics.

        Args:
            df_results: DataFrame with model comparison results
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_plot = df_results.set_index('Model')
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Filter out None values
        df_plot_filtered = df_plot[metrics_to_plot].dropna(how='all')
        
        df_plot_filtered.plot(kind='bar', ax=ax)
        ax.set_title('Model Comparison: Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", save_path=None):
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_roc_curves(self, models_data, save_path=None):
        """
        Plot ROC curves for multiple models.

        Args:
            models_data: List of tuples (model_name, y_true, y_prob)
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, y_true, y_prob in models_data:
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
            except Exception as e:
                print(f"Warning: Could not plot ROC for {model_name} - {e}")
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_feature_importance(self, model, feature_names=None, top_n=15, save_path=None):
        """
        Plot feature importance for tree-based models.

        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names (optional)
            top_n: Number of top features to show
            save_path: Path to save the plot (optional)
        """
        if not hasattr(model, 'feature_importances_'):
            print("Warning: Model does not have feature_importances_ attribute")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create DataFrame and sort
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        ax.barh(range(len(df_importance)), df_importance['Importance'])
        ax.set_yticks(range(len(df_importance)))
        ax.set_yticklabels(df_importance['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_pca_variance(self, pca, save_path=None):
        """
        Plot PCA explained variance.

        Args:
            pca: Fitted PCA object
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        var_ratio = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(var_ratio)
        
        x = range(1, len(var_ratio) + 1)
        ax.bar(x, var_ratio, alpha=0.6, label='Individual')
        ax.plot(x, cumsum_var, 'r-o', label='Cumulative', linewidth=2)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def create_comprehensive_report(self, models_results, y_true_test, save_dir=None):
        """
        Create comprehensive evaluation report with multiple plots.

        Args:
            models_results: Dictionary of {model_name: {'y_pred': ..., 'y_prob': ..., 'model': ...}}
            y_true_test: True test labels
            save_dir: Directory to save plots (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model comparison
        results_list = []
        for name, data in models_results.items():
            metrics = self.evaluate_model(
                y_true_test, 
                data['y_pred'], 
                data.get('y_prob'), 
                name
            )
            results_list.append(metrics)
        
        df_results = pd.DataFrame(results_list).set_index('Model')
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        df_results[metrics_to_plot].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Comparison: Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True, alpha=0.3)
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Confusion matrix (best model)
        best_model_name = df_results['F1-Score'].idxmax()
        best_data = models_results[best_model_name]
        cm = confusion_matrix(y_true_test, best_data['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
        axes[0, 1].set_title(f'Confusion Matrix ({best_model_name})')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. ROC Curves
        for name, data in models_results.items():
            if data.get('y_prob') is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true_test, data['y_prob'])
                    auc = roc_auc_score(y_true_test, data['y_prob'])
                    axes[0, 2].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
                except Exception as e:
                    print(f"Warning: Could not plot ROC for {name} - {e}")
        
        axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 2].set_xlabel('False Positive Rate')
        axes[0, 2].set_ylabel('True Positive Rate')
        axes[0, 2].set_title('ROC Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature importance (if available)
        if hasattr(best_data.get('model'), 'feature_importances_'):
            model = best_data['model']
            importances = model.feature_importances_
            
            # Get top features
            n_features = min(15, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            axes[1, 0].barh(range(n_features), importances[indices])
            axes[1, 0].set_yticks(range(n_features))
            axes[1, 0].set_yticklabels([f'PC{i+1}' for i in indices])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title(f'Top {n_features} Feature Importance')
            axes[1, 0].invert_yaxis()
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
        
        # 5. PCA variance (if available)
        pca = best_data.get('pca')
        if pca is not None and hasattr(pca, 'explained_variance_ratio_'):
            var_ratio = pca.explained_variance_ratio_
            cumsum_var = np.cumsum(var_ratio)
            x = range(1, len(var_ratio) + 1)
            axes[1, 1].bar(x, var_ratio, alpha=0.6, label='Individual')
            axes[1, 1].plot(x, cumsum_var, 'r-o', label='Cumulative')
            axes[1, 1].set_xlabel('Principal Component')
            axes[1, 1].set_ylabel('Explained Variance Ratio')
            axes[1, 1].set_title('PCA Explained Variance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'PCA not used', 
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
        
        # 6. Metrics summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table_data = []
        for name, data in models_results.items():
            metrics = self.evaluate_model(y_true_test, data['y_pred'], data.get('y_prob'), name)
            table_data.append([
                metrics['Model'],
                f"{metrics['Accuracy']:.3f}",
                f"{metrics['Precision']:.3f}",
                f"{metrics['Recall']:.3f}",
                f"{metrics['F1-Score']:.3f}"
            ])
        
        table = axes[1, 2].table(cellText=table_data,
                                colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        axes[1, 2].set_title('Metrics Summary')
        
        plt.tight_layout()
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'comprehensive_report.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return df_results

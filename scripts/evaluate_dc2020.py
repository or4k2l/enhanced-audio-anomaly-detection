#!/usr/bin/env python
"""Run a deterministic DCASE-shaped synthetic-data evaluation.

This runner does not load the DCASE 2020 dataset. It creates synthetic feature
matrices with DCASE-like dimensions to exercise the unsupervised pipeline.
Use it for demonstrations and regression checks, not for benchmark claims.

Usage:
    python scripts/evaluate_dc2020.py --output results_dc2020.csv
    python scripts/evaluate_dc2020.py --seed 42 --methods lof isolation_forest
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_anom import setup_logger
from audio_anom.unsupervised_anomaly import create_detector
from audio_anom.preprocessing_unsupervised import UnsupervisedPreprocessor
from audio_anom.evaluation_unsupervised import evaluate_machine_type

logger = setup_logger("evaluate_dc2020")


def load_machine_data(machine_type: str, seed: int):
    """Generate deterministic DCASE-shaped synthetic data for one machine."""
    logger.info(f"Loading {machine_type} data...")

    machine_seed = seed + sum(ord(character) for character in machine_type)
    rng = np.random.default_rng(machine_seed)
    mean_normal = rng.standard_normal(284)
    X_train_normal = rng.standard_normal((1000, 284)) + mean_normal
    X_test_normal = rng.standard_normal((600, 284)) + mean_normal
    X_test_anomaly = rng.standard_normal((400, 284)) + 2 * rng.standard_normal(284)

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0] * 600 + [1] * 400)

    indices = rng.permutation(len(y_test))
    X_test = X_test[indices]
    y_test = y_test[indices]

    return X_train_normal, X_test, y_test


def evaluate_all_machines(
    machines: List[str],
    methods: List[str],
    contamination: float,
    seed: int,
) -> pd.DataFrame:
    """Evaluate all methods on all machine types.

    Args:
        machines: List of machine types
        methods: List of detection methods
        contamination: Contamination parameter
        seed: Base random seed for synthetic data generation

    Returns:
        results_df: DataFrame with all results
    """
    all_results = []

    for machine_type in machines:
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATING MACHINE TYPE: {machine_type.upper()}")
        logger.info(f"{'='*80}")

        # Load data
        X_train, X_test, y_test = load_machine_data(machine_type, seed)

        # Train and evaluate each method
        models = {}
        for method in methods:
            logger.info(f"\nTraining {method}...")

            # Preprocess
            preprocessor = UnsupervisedPreprocessor(n_components=10)
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            # Train
            model = create_detector(method, contamination=contamination)
            model.fit(X_train_proc)

            models[method] = model

        # Evaluate all models for this machine
        machine_results = evaluate_machine_type(
            machine_type,
            models,
            X_test_proc,
            y_test,
        )

        all_results.append(machine_results)

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=False)
    return results_df


def generate_summary_report(results_df: pd.DataFrame) -> str:
    """Generate a formatted summary report.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        report: Formatted text report
    """
    report = []
    report.append("=" * 80)
    report.append("DCASE-SHAPED SYNTHETIC UNSUPERVISED EVALUATION RESULTS")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append("OVERALL RESULTS")
    report.append("-" * 80)

    # Average performance by method
    if "machine_type" in results_df.columns:
        avg_by_method = results_df.groupby(level=0)[
            ["roc_auc", "f1_score", "accuracy"]
        ].mean()
        avg_by_method = avg_by_method.sort_values("roc_auc", ascending=False)

        report.append("\nAverage Performance Across All Machines:")
        report.append(avg_by_method.to_string())

        best_method = avg_by_method.index[0]
        best_auc = avg_by_method.loc[best_method, "roc_auc"]
        best_f1 = avg_by_method.loc[best_method, "f1_score"]

        report.append(f"\nBest Method: {best_method}")
        report.append(f"  Average ROC-AUC: {best_auc:.4f}")
        report.append(f"  Average F1-Score: {best_f1:.4f}")

    report.append("\n" + "=" * 80)

    # Per-machine results
    report.append("\nPER-MACHINE RESULTS")
    report.append("-" * 80)

    if "machine_type" in results_df.columns:
        for machine in results_df["machine_type"].unique():
            machine_df = results_df[results_df["machine_type"] == machine]
            report.append(f"\n{machine.upper()}:")
            report.append(
                machine_df[
                    ["roc_auc", "f1_score", "accuracy", "precision", "recall"]
                ].to_string()
            )

    report.append("\n" + "=" * 80)

    # Performance ratings
    report.append("\nPERFORMANCE RATINGS")
    report.append("-" * 80)
    report.append("EXCELLENT (AUC >= 0.80)")
    report.append("GOOD      (AUC >= 0.70)")
    report.append("OK        (AUC >= 0.60)")
    report.append("POOR      (AUC < 0.60)")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Deterministic synthetic evaluation with DCASE-shaped features"
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        default=["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"],
        help="Machine types to evaluate (default: all 6 machines)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["lof", "isolation_forest", "elliptic_envelope"],
        help="Detection methods to evaluate (default: all 3 methods)",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.1,
        help="Contamination parameter (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible synthetic data (default: 42)",
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info("DCASE-SHAPED SYNTHETIC EVALUATION")
    logger.info(f"{'='*80}")
    logger.info(f"Machines: {args.machines}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Contamination: {args.contamination}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output}")
    logger.info(f"{'='*80}\n")

    # Evaluate all machines
    results_df = evaluate_all_machines(
        args.machines,
        args.methods,
        args.contamination,
        args.seed,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path)
    logger.info(f"\nResults saved to {output_path}")

    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "data_source": "synthetic DCASE-shaped features",
                "seed": args.seed,
                "machines": args.machines,
                "methods": args.methods,
                "contamination": args.contamination,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"Experiment metadata saved to {metadata_path}")

    # Generate and save summary report
    report = generate_summary_report(results_df)
    report_path = output_path.parent / (output_path.stem + "_report.txt")
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Summary report saved to {report_path}")

    # Print summary
    print("\n" + report)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

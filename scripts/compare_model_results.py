"""
Compare model results from MLflow runs and generate comprehensive report.

This script extracts metrics from MLflow runs and creates a comparison table
for baseline vs hybrid models.
"""
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# Set MLflow tracking URI to use SQLite database
mlflow.set_tracking_uri("sqlite:///fraud-detection-mlflow.db")


def get_run_metrics(run_id: str) -> Dict[str, float]:
    """Extract all metrics from an MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics


def get_run_params(run_id: str) -> Dict[str, str]:
    """Extract all parameters from an MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def get_run_tags(run_id: str) -> Dict[str, str]:
    """Extract all tags from an MLflow run."""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.tags


def format_metric_value(value: float, metric_name: str) -> str:
    """Format metric value based on metric type."""
    if "auc" in metric_name.lower() or "p@" in metric_name.lower():
        return f"{value:.4f}"
    elif "count" in metric_name.lower() or "size" in metric_name.lower():
        return f"{int(value):,}"
    else:
        return f"{value:.2f}"


def compare_models(
    baseline_run_id: str,
    sage_run_id: str,
    hgt_run_id: str,
    output_dir: Path
) -> pd.DataFrame:
    """Compare metrics across all three models."""
    client = MlflowClient()
    
    # Get metrics for each model
    baseline_metrics = get_run_metrics(baseline_run_id)
    sage_metrics = get_run_metrics(sage_run_id)
    hgt_metrics = get_run_metrics(hgt_run_id)
    
    # Get tags for model info
    baseline_tags = get_run_tags(baseline_run_id)
    sage_tags = get_run_tags(sage_run_id)
    hgt_tags = get_run_tags(hgt_run_id)
    
    # Key metrics to compare
    key_metrics = [
        "mean_auc_pr",
        "best_auc_pr",
        "mean_p_at_100",
        "mean_auc_roc",
        "num_windows",
    ]
    
    # Build comparison table
    comparison_data = []
    
    for metric in key_metrics:
        row = {
            "Metric": metric.replace("_", " ").title(),
            "Baseline": baseline_metrics.get(metric, 0.0),
            "SAGE Hybrid": sage_metrics.get(metric, 0.0),
            "HGT Hybrid": hgt_metrics.get(metric, 0.0),
        }
        
        # Calculate differences
        baseline_val = row["Baseline"]
        row["SAGE vs Baseline"] = row["SAGE Hybrid"] - baseline_val
        row["HGT vs Baseline"] = row["HGT Hybrid"] - baseline_val
        
        # Calculate percentage differences
        if baseline_val > 0:
            row["SAGE % Diff"] = ((row["SAGE Hybrid"] - baseline_val) / baseline_val) * 100
            row["HGT % Diff"] = ((row["HGT Hybrid"] - baseline_val) / baseline_val) * 100
        else:
            row["SAGE % Diff"] = 0.0
            row["HGT % Diff"] = 0.0
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Add model metadata
    metadata = {
        "Model": ["Baseline", "SAGE Hybrid", "HGT Hybrid"],
        "Run ID": [baseline_run_id, sage_run_id, hgt_run_id],
        "Training Time": [
            baseline_tags.get("training_time_minutes", "N/A"),
            sage_tags.get("training_time_minutes", "N/A"),
            hgt_tags.get("training_time_minutes", "N/A"),
        ],
        "Feature Count": [
            baseline_tags.get("feature_count", "N/A"),
            sage_tags.get("feature_count", "N/A"),
            hgt_tags.get("feature_count", "N/A"),
        ],
    }
    
    metadata_df = pd.DataFrame(metadata)
    
    # Save to files
    df.to_csv(output_dir / "metrics_comparison.csv", index=False)
    metadata_df.to_csv(output_dir / "model_metadata.csv", index=False)
    
    # Generate text report
    report_lines = [
        "=" * 80,
        "MODEL COMPARISON REPORT",
        "=" * 80,
        "",
        "Experiment Configuration:",
        f"  Baseline Run ID: {baseline_run_id}",
        f"  SAGE Run ID: {sage_run_id}",
        f"  HGT Run ID: {hgt_run_id}",
        "",
        "=" * 80,
        "PERFORMANCE METRICS",
        "=" * 80,
        "",
    ]
    
    # Format metrics table
    for _, row in df.iterrows():
        metric_name = row["Metric"]
        baseline_val = row["Baseline"]
        sage_val = row["SAGE Hybrid"]
        hgt_val = row["HGT Hybrid"]
        sage_diff = row["SAGE vs Baseline"]
        hgt_diff = row["HGT vs Baseline"]
        sage_pct = row["SAGE % Diff"]
        hgt_pct = row["HGT % Diff"]
        
        report_lines.append(f"{metric_name}:")
        report_lines.append(f"  Baseline:  {format_metric_value(baseline_val, metric_name)}")
        report_lines.append(f"  SAGE:      {format_metric_value(sage_val, metric_name)} ({sage_pct:+.2f}%)")
        report_lines.append(f"  HGT:       {format_metric_value(hgt_val, metric_name)} ({hgt_pct:+.2f}%)")
        report_lines.append("")
    
    # Add metadata
    report_lines.extend([
        "=" * 80,
        "MODEL METADATA",
        "=" * 80,
        "",
    ])
    
    for _, row in metadata_df.iterrows():
        report_lines.append(f"{row['Model']}:")
        report_lines.append(f"  Run ID: {row['Run ID']}")
        report_lines.append(f"  Training Time: {row['Training Time']}")
        report_lines.append(f"  Feature Count: {row['Feature Count']}")
        report_lines.append("")
    
    # Add summary
    baseline_auc_pr = df[df["Metric"] == "Mean Auc Pr"]["Baseline"].values[0]
    sage_auc_pr = df[df["Metric"] == "Mean Auc Pr"]["SAGE Hybrid"].values[0]
    hgt_auc_pr = df[df["Metric"] == "Mean Auc Pr"]["HGT Hybrid"].values[0]
    
    report_lines.extend([
        "=" * 80,
        "SUMMARY",
        "=" * 80,
        "",
        f"Baseline AUC-PR: {baseline_auc_pr:.4f}",
        f"SAGE Hybrid AUC-PR: {sage_auc_pr:.4f} ({(sage_auc_pr - baseline_auc_pr) / baseline_auc_pr * 100:+.2f}%)",
        f"HGT Hybrid AUC-PR: {hgt_auc_pr:.4f} ({(hgt_auc_pr - baseline_auc_pr) / baseline_auc_pr * 100:+.2f}%)",
        "",
        "Recommendation:",
    ])
    
    if baseline_auc_pr > max(sage_auc_pr, hgt_auc_pr):
        report_lines.append("  ✅ Baseline outperforms hybrid models")
        report_lines.append("  → Continue optimizing baseline (Experiment 13A)")
    elif max(sage_auc_pr, hgt_auc_pr) > baseline_auc_pr:
        best_hybrid = "SAGE" if sage_auc_pr > hgt_auc_pr else "HGT"
        report_lines.append(f"  ⚠️ {best_hybrid} Hybrid outperforms baseline")
        report_lines.append("  → Investigate hybrid model improvements (Experiment 13B)")
    else:
        report_lines.append("  ⚠️ Models are competitive")
        report_lines.append("  → Consider cost-benefit analysis")
    
    report_lines.append("")
    
    # Write report
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print("\n".join(report_lines))
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare model results from MLflow")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="exp13-baseline-vs-hybrid-365d",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        required=True,
        help="MLflow run ID for baseline model"
    )
    parser.add_argument(
        "--sage-run-id",
        type=str,
        required=True,
        help="MLflow run ID for SAGE hybrid model"
    )
    parser.add_argument(
        "--hgt-run-id",
        type=str,
        required=True,
        help="MLflow run ID for HGT hybrid model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compare_models(
        baseline_run_id=args.baseline_run_id,
        sage_run_id=args.sage_run_id,
        hgt_run_id=args.hgt_run_id,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()


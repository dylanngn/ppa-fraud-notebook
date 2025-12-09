"""
Experiment 6: Business Value Evaluation

Answers practical business questions before production deployment:
1. How does our model compare to Seon (current production)?
2. At what operating point should we deploy?
3. What is the business value?

Run with: python -m src.experiments.exp6_business_value experiment_name=business-value-eval
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from omegaconf import DictConfig

from src.data.loader import load_data
from src.features.definitions.base import compute_base_features
from src.features.processor import FeatureProcessor
from src.models.utils.common import setup_mlflow
from src.models.xgboost.utils import get_optimal_tree_method, validate_features, get_categorical_features
from src.utils.metrics import calculate_metrics
import xgboost as xgb

logger = logging.getLogger(__name__)


def load_seon_baseline() -> Dict[str, Any]:
    """
    Load Seon baseline performance from results registry.
    
    Returns:
        Dictionary with Seon metrics
    """
    from src.experiments.results import load_result, ExperimentID
    
    result = load_result(ExperimentID.SEON_BASELINE)
    if result and result.get("metrics"):
        metrics = result["metrics"]
        return {
            "mean_auc_pr": metrics.get("auc_pr"),
            "mean_precision": metrics.get("precision"),
            "mean_recall": metrics.get("recall"),
        }
    
    # Return placeholder if not available
    return {
        "mean_auc_pr": None,
        "mean_precision": None,
        "mean_recall": None,
        "note": "Seon baseline not available - run evaluate_seon first"
    }


def calculate_operating_points(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    review_budgets: List[int] = [50, 100, 200, 300, 500]
) -> pd.DataFrame:
    """
    Calculate precision, recall at different review budgets (top-K).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        review_budgets: List of K values for top-K analysis
        
    Returns:
        DataFrame with operating point metrics
    """
    total_fraud = y_true.sum()
    n_samples = len(y_true)
    
    # Sort by probability descending
    sorted_indices = np.argsort(y_proba)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    results = []
    for k in review_budgets:
        if k > n_samples:
            continue
            
        top_k_labels = sorted_labels[:k]
        fraud_in_top_k = top_k_labels.sum()
        
        precision_at_k = fraud_in_top_k / k
        recall_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0
        fraud_missed = total_fraud - fraud_in_top_k
        
        results.append({
            "reviews_per_window": k,
            "precision": precision_at_k,
            "recall": recall_at_k,
            "fraud_caught": fraud_in_top_k,
            "fraud_missed": fraud_missed,
            "false_positives": k - fraud_in_top_k,
        })
    
    return pd.DataFrame(results)


def train_and_evaluate(
    df: pl.DataFrame,
    config: DictConfig,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Train model and get predictions for evaluation.
    
    Returns:
        Tuple of (y_true, y_proba, metrics_dict)
    """
    xgb_params = dict(config.model.params)
    target = "is_fraud"
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Time-based split
    split_idx = int(len(df) * train_ratio)
    train_data = df[:split_idx]
    test_data = df[split_idx:]
    
    cutoff_date = train_data["submission_at"].max()
    
    # Compute features
    processor = FeatureProcessor.from_config(config)
    
    logger.info(f"Computing features for {len(train_data)} train, {len(test_data)} test samples...")
    train_features = processor.process(train_data, cutoff_date)
    test_features = processor.process(test_data, cutoff_date)
    
    # Convert to pandas
    train_pdf = train_features.to_pandas()
    test_pdf = test_features.to_pandas()
    
    # Validate and prepare features
    valid_features, _ = validate_features(train_pdf, strict=False)
    
    categorical_features = get_categorical_features(valid_features)
    for cat_feat in categorical_features:
        unique_vals = train_pdf[cat_feat].dropna().unique()
        train_pdf[cat_feat] = pd.Categorical(train_pdf[cat_feat], categories=unique_vals)
        test_pdf[cat_feat] = pd.Categorical(test_pdf[cat_feat], categories=unique_vals)
    
    X_train = train_pdf[valid_features]
    X_test = test_pdf[valid_features]
    y_train = train_features[target].to_pandas().values
    y_test = test_features[target].to_pandas().values
    
    # Train model
    tree_method = get_optimal_tree_method()
    xgb_params["tree_method"] = tree_method
    if categorical_features:
        xgb_params["enable_categorical"] = True
    
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Predict
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_proba)
    
    return y_test, y_proba, metrics


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path
):
    """Generate and save precision-recall curve with operating points."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Main PR curve
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'Our Model (AUC-PR = {ap:.3f})')
    
    # Random baseline
    fraud_rate = y_true.mean()
    ax.axhline(y=fraud_rate, color='gray', linestyle='--', label=f'Random ({fraud_rate:.3f})')
    
    # Mark operating points
    operating_points = calculate_operating_points(y_true, y_proba, [50, 100, 200, 500])
    for _, row in operating_points.iterrows():
        ax.scatter(row['recall'], row['precision'], s=100, zorder=5)
        ax.annotate(f"Top {int(row['reviews_per_window'])}\nP={row['precision']:.2f}\nR={row['recall']:.2f}",
                   xy=(row['recall'], row['precision']),
                   xytext=(row['recall'] + 0.05, row['precision'] + 0.05),
                   fontsize=9)
    
    ax.set_xlabel('Recall (Fraud Caught / Total Fraud)', fontsize=12)
    ax.set_ylabel('Precision (Fraud / Flagged)', fontsize=12)
    ax.set_title('Precision-Recall Curve with Operating Points', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")


def generate_business_value_report(
    metrics: Dict[str, Any],
    operating_points: pd.DataFrame,
    seon_baseline: Dict[str, Any],
    output_path: Path
):
    """Generate business value report."""
    
    report = []
    report.append("=" * 70)
    report.append("EXPERIMENT 6: BUSINESS VALUE EVALUATION")
    report.append("=" * 70)
    report.append("")
    
    # Model Performance
    report.append("## Model Performance")
    report.append(f"AUC-PR: {metrics['auc_pr']:.4f}")
    report.append(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    report.append(f"P@100: {metrics['p@100']:.4f}")
    report.append("")
    
    # Seon Comparison
    report.append("## Seon Comparison")
    if seon_baseline.get("mean_auc_pr"):
        seon_auc = seon_baseline["mean_auc_pr"]
        improvement = (metrics['auc_pr'] - seon_auc) / seon_auc * 100
        report.append(f"Seon AUC-PR: {seon_auc:.4f}")
        report.append(f"Our AUC-PR: {metrics['auc_pr']:.4f}")
        report.append(f"Improvement: {improvement:+.1f}%")
    else:
        report.append("Seon baseline not available - manual comparison needed")
    report.append("")
    
    # Operating Points
    report.append("## Operating Points")
    report.append(operating_points.to_string(index=False))
    report.append("")
    
    # Recommendation
    report.append("## Recommendation")
    if metrics['auc_pr'] > 0.5:
        report.append("✅ Model provides significant lift over random baseline")
        report.append(f"   Lift: {metrics['auc_pr'] / 0.082:.1f}x over random")
    else:
        report.append("⚠️ Model performance needs investigation")
    report.append("")
    
    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Saved: {output_path}")
    
    return '\n'.join(report)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 6: Business Value Evaluation
    
    Run with: python -m src.experiments.exp6_business_value experiment_name=business-value-eval
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 6: BUSINESS VALUE EVALUATION")
    logger.info("="*60)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info("")
    logger.info("Answering business questions:")
    logger.info("  1. How do we compare to Seon?")
    logger.info("  2. What operating point should we use?")
    logger.info("  3. What is the business value?")
    logger.info("="*60)
    
    # Setup
    setup_mlflow(cfg.experiment_name)
    output_dir = Path("artifacts/business_value")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(), config=None)
    
    # Run evaluation with MLflow tracking
    with mlflow.start_run(
        run_name=f"business_value_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "business_value_evaluation"}
    ):
        # Train and evaluate
        y_true, y_proba, metrics = train_and_evaluate(df, cfg)
        
        # Calculate operating points
        operating_points = calculate_operating_points(
            y_true, y_proba,
            review_budgets=[50, 100, 150, 200, 300, 500]
        )
        
        # Load Seon baseline
        seon_baseline = load_seon_baseline()
        
        # Generate plots
        plot_precision_recall_curve(y_true, y_proba, output_dir / "pr_curve.png")
        
        # Generate report
        report = generate_business_value_report(
            metrics, operating_points, seon_baseline,
            output_dir / "business_value_report.txt"
        )
        
        # Save operating points
        operating_points.to_csv(output_dir / "operating_points.csv", index=False)
        
        # Log to MLflow
        mlflow.log_metrics({
            "auc_pr": metrics["auc_pr"],
            "auc_roc": metrics["auc_roc"],
            "p_at_100": metrics["p@100"],
        })
        mlflow.log_artifacts(str(output_dir), artifact_path="business_value")
        
        # Print summary
        print("\n" + report)
        print("\n" + "="*60)
        print("OPERATING POINTS SUMMARY")
        print("="*60)
        print(operating_points.to_string(index=False))
        print("\n")
        
        # Save to results registry
        from src.experiments.results import save_result, ExperimentID
        save_result(ExperimentID.EXP6_BUSINESS_VALUE, {
            "auc_pr": metrics["auc_pr"],
            "auc_roc": metrics["auc_roc"],
            "p_at_100": metrics["p@100"],
            "p_at_500": metrics.get("p@500"),
            "seon_auc_pr": seon_baseline.get("mean_auc_pr"),
            "operating_points": operating_points.to_dict(orient="records"),
        })
        
        logger.info("Business value evaluation complete!")
        logger.info(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


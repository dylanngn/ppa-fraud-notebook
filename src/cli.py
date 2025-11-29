"""
Fraud Detection Pipeline CLI

Follows Typer best practices: https://typer.tiangolo.com/
"""
import typer
import sys
import os
from typing import List, Optional

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import etl, graph_builder
from src.models import train_baseline
from src.models.experiment_config import ExperimentConfig, FeatureCategory
from src.models.training_window import train_accumulating_window
from src.models.hyperopt_xgboost import optimize_xgboost_hyperparameters
from src.models.hyperopt_pytorch import optimize_pytorch_hyperparameters
from src.models.feature_engineering import load_data, add_base_tabular_features
from src.utils import evaluate_seon

app = typer.Typer(
    help="Fraud Detection Pipeline CLI",
    add_completion=False,
    no_args_is_help=True
)


@app.command()
def extract_data():
    """Run the ETL pipeline to extract data from Aurora and save to Parquet."""
    etl.main()


@app.command()
def build_graph():
    """Construct the PyTorch Geometric Graph from processed Parquet files."""
    graph_builder.build_graph()


@app.command()
def train_baseline(
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    initial_window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Step size between evaluation windows in days"),
    feature_categories: Optional[str] = typer.Option(
        None,
        help="Comma-separated feature categories: base,graph,advanced_graph,time_weighted,interaction,text"
    ),
):
    """
    Train Baseline XGBoost with accumulating window.
    
    Always includes graph features with temporal filtering to prevent data leakage.
    Automatically tracks with MLflow and registers best model to Model Registry.
    """
    # Parse feature categories
    categories = None
    if feature_categories:
        categories = [FeatureCategory(cat.strip()) for cat in feature_categories.split(",")]
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days,
        feature_categories=categories
    )
    
    df = load_data()
    df = add_base_tabular_features(df)
    
    result = train_accumulating_window(
        df=df,
        model_name="baseline_graph",
        config=config
    )
    
    return result


@app.command()
def train_hybrid_sage(
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    initial_window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Step size between evaluation windows in days"),
    epochs: int = typer.Option(25, help="Number of GNN training epochs"),
):
    """
    Train SAGE Hybrid Model (XGBoost + SAGE Embeddings).
    
    Automatically trains SAGE embeddings, uses accumulating window,
    tracks with MLflow, and registers model to Model Registry.
    """
    from src.models.train_hybrid_sage import main as train_sage_main
    
    train_sage_main(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days,
        epochs=epochs
    )


@app.command()
def train_hybrid_hgt(
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    initial_window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Step size between evaluation windows in days"),
    epochs: int = typer.Option(30, help="Number of GNN training epochs"),
):
    """
    Train HGT Hybrid Model (XGBoost + HGT Embeddings with RTE).
    
    Automatically trains HGT embeddings with RTE, uses accumulating window,
    tracks with MLflow, and registers model to Model Registry.
    """
    from src.models.train_hybrid_hgt import main as train_hgt_main
    
    train_hgt_main(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days,
        epochs=epochs
    )


@app.command()
def optimize_xgboost(
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    initial_window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Step size between evaluation windows in days"),
    n_trials: int = typer.Option(100, help="Number of Optuna trials"),
    n_windows: int = typer.Option(5, help="Number of evaluation windows per trial"),
    timeout_minutes: Optional[int] = typer.Option(None, help="Optional timeout in minutes"),
    feature_categories: Optional[str] = typer.Option(
        None,
        help="Comma-separated feature categories: base,graph,advanced_graph,time_weighted,interaction,text"
    ),
):
    """
    Hyperparameter optimization for XGBoost models using Optuna.
    
    Tunes XGBoost hyperparameters to maximize: 0.7 * AUC-PR + 0.3 * P@100
    
    Search space includes:
    - n_estimators: [100, 500]
    - max_depth: [4, 10]
    - learning_rate: [0.01, 0.2] (log scale)
    - min_child_weight: [1, 20]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - gamma: [0, 1]
    - reg_alpha: [0, 10]
    - reg_lambda: [1, 10]
    """
    # Parse feature categories
    categories = None
    if feature_categories:
        categories = [FeatureCategory(cat.strip()) for cat in feature_categories.split(",")]
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days,
        feature_categories=categories
    )
    
    result = optimize_xgboost_hyperparameters(
        config=config,
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes
    )
    
    typer.echo(f"Best parameters: {result['best_params']}")
    typer.echo(f"Best score: {result['best_score']:.4f}")
    
    return result


@app.command()
def optimize_pytorch(
    model_type: str = typer.Option(..., help="Model type: 'hgt' or 'sage'"),
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    n_trials: int = typer.Option(50, help="Number of Optuna trials"),
    epochs: int = typer.Option(25, help="Number of training epochs per trial"),
    timeout_minutes: Optional[int] = typer.Option(None, help="Optional timeout in minutes"),
):
    """
    Hyperparameter optimization for PyTorch GNN models using Optuna.
    
    Tunes GNN hyperparameters to maximize: 0.7 * AUC-PR + 0.3 * P@100
    
    Search space includes:
    - hidden_channels: [32, 128]
    - out_channels: [32, 128]
    - num_layers: [1, 3]
    - learning_rate: [1e-4, 1e-2] (log scale)
    - num_heads (HGT only): [2, 8]
    """
    from src.models.train_hybrid_hgt import HGTWrapper
    from src.models.train_hybrid_sage import SAGEWrapper
    
    model_class = HGTWrapper if model_type.lower() == "hgt" else SAGEWrapper
    
    config = ExperimentConfig(experiment_name=experiment_name)
    
    result = optimize_pytorch_hyperparameters(
        model_class=model_class,
        config=config,
        n_trials=n_trials,
        epochs=epochs,
        timeout_minutes=timeout_minutes
    )
    
    typer.echo(f"Best parameters: {result['best_params']}")
    typer.echo(f"Best score: {result['best_score']:.4f}")
    
    return result


@app.command()
def evaluate_seon(
    evaluation_start_days: int = typer.Option(90, help="Days to skip before starting evaluation"),
    step_days: int = typer.Option(14, help="Step size between evaluation windows"),
    include_fallback: bool = typer.Option(True, help="Include fallback predictions when Seon data unavailable"),
    log_to_mlflow: bool = typer.Option(True, help="Log metrics to MLflow for comparison"),
):
    """
    Evaluate Seon (production baseline) performance.
    
    This evaluates the currently-used Seon fraud detection system as a baseline
    for comparison with our trained models.
    """
    evaluate_seon.run_seon_evaluation(
        evaluation_start_days=evaluation_start_days,
        step_days=step_days,
        include_fallback=include_fallback,
        log_to_mlflow=log_to_mlflow
    )
    

# Legacy commands for backward compatibility
@app.command()
def check_density():
    """Check fraud density in different time windows."""
    from src.utils import check_window_density
    check_window_density.check_density()


@app.command()
def data_quality_report(
    data_path: str = typer.Option("artifacts/raw_insertions.parquet", help="Path to data file"),
    output: str = typer.Option("artifacts/data_quality_report.txt", help="Path to save report"),
    null_threshold_unusable: float = typer.Option(95.0, help="Fields with >= this % null are unusable"),
    null_threshold_high_coverage: float = typer.Option(50.0, help="Fields with < this % null are high coverage"),
):
    """Generate a comprehensive data quality report."""
    from src.utils.data_quality_report import generate_report
    
    generate_report(
        data_path=data_path,
        output_path=output,
        null_threshold_unusable=null_threshold_unusable,
        null_threshold_high_coverage=null_threshold_high_coverage
    )
    
    typer.echo(f"Report saved to: {output}")


@app.command()
def mlflow_compare_models(
    production_run_id: str = typer.Option(..., help="MLflow run ID of production model"),
    candidate_run_id: str = typer.Option(..., help="MLflow run ID of candidate model"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to compare"),
    improvement_threshold: float = typer.Option(0.01, help="Minimum improvement threshold"),
):
    """Compare two MLflow runs to evaluate model improvements."""
    from src.utils.mlflow_model_comparison import compare_models
    
    result = compare_models(
        production_run_id=production_run_id,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
    )
    
    if "error" in result:
        typer.echo(f"Error: {result['error']}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Primary Metric: {primary_metric}")
    typer.echo(f"Production: {result['production_value']:.4f}")
    typer.echo(f"Candidate: {result['candidate_value']:.4f}")
    typer.echo(f"Improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
    typer.echo(f"Recommendation: {'✓ Deploy' if result['should_deploy'] else '✗ Reject'}")
    typer.echo(f"Reason: {result['reason']}")


@app.command()
def mlflow_deployment_recommendation(
    model_name: str = typer.Option("fraud-detection-baseline_graph", help="Registered model name"),
    candidate_run_id: str = typer.Option(..., help="MLflow run ID of candidate model"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to evaluate"),
    improvement_threshold: float = typer.Option(0.01, help="Absolute improvement threshold"),
    min_improvement_pct: float = typer.Option(1.0, help="Minimum percentage improvement"),
):
    """Get deployment recommendation using MLflow Model Registry."""
    from src.utils.mlflow_model_comparison import recommend_deployment
    
    recommendation = recommend_deployment(
        model_name=model_name,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
        min_improvement_pct=min_improvement_pct,
    )
    
    typer.echo(f"Status: {recommendation['recommendation']}")
    typer.echo(f"Stage: {recommendation['stage'] or 'N/A'}")
    typer.echo(f"Reason: {recommendation['reason']}")


@app.command()
def mlflow_drift_summary(
    model_name: str = typer.Option("fraud-detection-baseline_graph", help="Registered model name"),
    n_versions: int = typer.Option(5, help="Number of recent versions to analyze"),
):
    """Analyze model drift by comparing recent model versions."""
    from src.utils.mlflow_model_comparison import get_model_drift_summary
    
    summary = get_model_drift_summary(
        model_name=model_name,
        n_recent_versions=n_versions,
    )
    
    if "error" in summary:
        typer.echo(f"Error: {summary['error']}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Drift Detected: {'⚠️ Yes' if summary['drift_detected'] else '✅ No'}")
    typer.echo(f"Production Version: {summary.get('production_version', 'N/A')}")
    typer.echo(f"Versions Analyzed: {summary['versions_analyzed']}")
    
    if summary['drift_detected']:
        typer.echo("Drift Reasons:")
        for reason in summary['drift_reasons']:
            typer.echo(f"  • {reason}")


if __name__ == "__main__":
    app()

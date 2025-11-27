import typer
from rich.console import Console
from datetime import datetime
import sys
import os

# Add project root to sys.path to allow importing 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import etl, graph_builder
from src.features import graph_features as graph_features_module
from src.models import train_baseline as baseline_module
from src.utils import evaluate_seon as seon_module
from src.features import advanced_graph_features as advanced_features_module
from src.features import time_weighted_features as time_weighted_module
from src.features import interaction_features as interaction_module

app = typer.Typer(help="Fraud Detection Pipeline CLI")
console = Console()

@app.command()
def extract_data():
    """
    Run the ETL pipeline to extract data from Aurora and save to Parquet.
    """
    console.print("[bold green]Starting ETL Pipeline...[/bold green]")
    etl.main()

@app.command()
def build_graph():
    """
    Construct the PyTorch Geometric Graph from processed Parquet files.
    """
    console.print("[bold green]Building Graph...[/bold green]")
    graph_builder.build_graph()

@app.command()
def train_baseline(
    include_graph: bool = typer.Option(True, help="Include graph features")
):
    """
    Train Baseline XGBoost with accumulating window.
    
    Automatically:
    - Uses accumulating window (all historical data)
    - Tracks with MLflow
    - Registers best model to Model Registry
    """
    console.print(f"[bold green]Training Baseline XGBoost...[/bold green]")
    baseline_module.run_baseline(include_graph_features=include_graph)



# train-graph-baseline removed - use: train-baseline --include-graph

@app.command()
def train_hybrid_sage():
    """
    Train SAGE Hybrid Model (XGBoost + SAGE Embeddings).
    
    Automatically:
    - Trains SAGE embeddings with optimized parameters
    - Uses accumulating window (all historical data)
    - Tracks with MLflow
    - Registers model to Model Registry
    """
    console.print("[bold green]Training SAGE Hybrid Model...[/bold green]")
    from src.models import train_hybrid_sage
    train_hybrid_sage.main()

@app.command()
def train_hybrid_hgt():
    """
    Train HGT Hybrid Model (XGBoost + HGT Embeddings with RTE).
    
    Automatically:
    - Trains HGT embeddings with RTE and optimized parameters
    - Uses accumulating window (all historical data)
    - Tracks with MLflow
    - Registers model to Model Registry
    """
    console.print("[bold green]Training HGT Hybrid Model (with RTE)...[/bold green]")
    from src.models import train_hybrid_hgt
    train_hybrid_hgt.main()


@app.command()
def graph_features():
    """
    Generate manual graph statistics for listings.
    """
    console.print("[bold green]Generating Graph Features...[/bold green]")
    graph_features_module.generate_graph_features()

@app.command()
def advanced_graph_features():
    """
    Generate advanced graph statistics (isolation, clustering, etc.).
    """
    console.print("[bold green]Generating Advanced Graph Features...[/bold green]")
    advanced_features_module.generate_advanced_features()


@app.command()
def time_weighted_features():
    """
    Generate time-weighted graph features (Experiment 9).
    
    Features include:
    - Recency-weighted connection counts (recent connections weighted higher)
    - Velocity metrics (connections per day)
    - Acceleration (velocity change)
    - Burst detection (sudden activity spikes)
    - Dormant reactivation patterns
    """
    console.print("[bold green]Generating Time-Weighted Graph Features...[/bold green]")
    time_weighted_module.generate_time_weighted_features()


@app.command()
def interaction_features():
    """
    Generate interaction features (Experiment 10).
    
    Features include:
    - new_account_high_reuse: New account + high email/phone reuse (13.4x lift)
    - new_account_invoice_payment: New account without direct payment (28% FR)
    - new_account_small_listing: New account with small property
    - suspicious_combo_score: Weighted combination of risk indicators
    """
    console.print("[bold green]Generating Interaction Features...[/bold green]")
    interaction_module.generate_interaction_features()



# Removed deprecated commands:
# - retrain-production → use train-baseline (same logic)

@app.command()
def optimize_graph_window():
    """
    Test different time windows for graph feature computation.
    Experiment 8: Find optimal window size (30, 60, 90, 120, 180, 365 days).
    """
    console.print("[bold green]Optimizing Graph Feature Window Size...[/bold green]")
    from src.experiments import optimize_graph_window as opt
    opt.run_window_optimization()


@app.command()
def optimize_hyperparams(
    n_trials: int = typer.Option(100, help="Number of Optuna trials"),
    n_windows: int = typer.Option(5, help="CV windows per trial for faster optimization"),
    timeout_minutes: int = typer.Option(None, help="Optional timeout in minutes"),
    skip_validation: bool = typer.Option(False, help="Skip full validation after optimization"),
):
    """
    Experiment 11: Hyperparameter optimization using Optuna.
    
    Tunes XGBoost hyperparameters to maximize:
        0.7 * AUC-PR + 0.3 * P@100
    
    Search space includes:
    - n_estimators: [100, 500]
    - max_depth: [4, 10]
    - learning_rate: [0.01, 0.2]
    - min_child_weight: [1, 20]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - gamma: [0, 1]
    - reg_alpha: [0, 10]
    - reg_lambda: [1, 10]
    
    Target: Push AUC-PR from 0.6713 to 0.70+
    """
    console.print("[bold green]Experiment 11: Hyperparameter Optimization...[/bold green]")
    from src.experiments import optimize_hyperparams as hyperopt_module
    hyperopt_module.main(
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes,
        validate=not skip_validation,
    )


@app.command()
def staged_hyperopt(
    n_windows: int = typer.Option(5, help="CV windows per configuration"),
):
    """
    Experiment 11b: Staged hyperparameter optimization.
    
    A structured 3-stage approach:
    
    Stage 1: Parameter sensitivity analysis
        - Tests individual parameter variations
        - Identifies which hyperparameters matter most
    
    Stage 2: Promising combinations
        - Tests 6 pre-defined configurations based on:
          * XGBoost best practices
          * Fraud detection literature
          * Imbalanced learning principles
    
    Stage 3: Compose final combination
        - Combines winning elements from Stage 1 & 2
        - Tests hybrid configurations
    
    More interpretable than blind Optuna search!
    """
    console.print("[bold green]Experiment 11b: Staged Hyperparameter Optimization...[/bold green]")
    from src.experiments import staged_hyperopt as staged_hyperopt_module
    staged_hyperopt_module.main(n_windows=n_windows)

@app.command()
def check_graph_timestamps():
    """
    Check timestamps in the built graph artifact.
    """
    console.print("[bold yellow]Checking Graph Timestamps...[/bold yellow]")
    from src.utils import check_graph_timestamps
    check_graph_timestamps.check_graph_timestamps()

@app.command()
def check_timestamps():
    """
    Check timestamps in the raw parquet files.
    """
    console.print("[bold yellow]Checking Parquet Timestamps...[/bold yellow]")
    from src.utils import check_timestamps
    check_timestamps.check_timestamps()

@app.command()
def check_density():
    """
    Check fraud density in different time windows.
    """
    console.print("[bold yellow]Checking Window Density...[/bold yellow]")
    from src.utils import check_window_density
    check_window_density.check_density()

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
    for comparison with our trained models. Seon operates BEFORE listing publication,
    making binary approve/reject decisions.
    
    Automatically:
    - Evaluates Seon on evaluation windows (matching model evaluation approach)
    - Logs metrics to MLflow for easy comparison
    - Saves results CSV
    
    Note: This is evaluation-only (no training). The windows are used to evaluate
    Seon on different time periods, matching how we evaluate our trained models.
    """
    console.print("[bold green]Evaluating Seon Baseline...[/bold green]")
    seon_module.run_seon_evaluation(
        evaluation_start_days=evaluation_start_days,
        step_days=step_days,
        include_fallback=include_fallback,
        log_to_mlflow=log_to_mlflow
    )
    
# Explainability commands removed - use MLflow UI for SHAP and model comparison
# MLflow automatically logs SHAP plots via mlflow.evaluate()
# Compare models in MLflow UI by selecting multiple runs

# analyze-failures command removed - use MLflow UI for SHAP analysis
# MLflow.evaluate() automatically generates SHAP plots for all models
# View SHAP explanations in MLflow UI under run artifacts


# Model comparison commands removed - use MLflow UI
# Compare models by selecting multiple runs in MLflow UI
# View side-by-side metrics, parameters, and SHAP plots

@app.command()
def mlflow_compare_models(
    production_run_id: str = typer.Option(..., help="MLflow run ID of production model"),
    candidate_run_id: str = typer.Option(..., help="MLflow run ID of candidate model"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to compare"),
    improvement_threshold: float = typer.Option(0.01, help="Minimum improvement threshold"),
):
    """
    Compare two MLflow runs to evaluate model improvements.
    
    Uses MLflow-native model comparison following best practices.
    
    Example:
        python src/cli.py mlflow-compare-models --production-run-id abc123 --candidate-run-id def456
    """
    console.print(f"[bold cyan]Comparing models...[/bold cyan]")
    
    from src.utils.mlflow_model_comparison import compare_models
    
    result = compare_models(
        production_run_id=production_run_id,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
    )
    
    if "error" in result:
        console.print(f"[bold red]Error: {result['error']}[/bold red]")
        return
    
    console.print(f"\n[bold]Comparison Results:[/bold]")
    console.print(f"  Primary Metric: {primary_metric}")
    console.print(f"  Production: {result['production_value']:.4f}")
    console.print(f"  Candidate: {result['candidate_value']:.4f}")
    console.print(f"  Improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.2f}%)")
    console.print(f"  Recommendation: {'✓ Deploy' if result['should_deploy'] else '✗ Reject'}")
    console.print(f"  Reason: {result['reason']}")


@app.command()
def mlflow_deployment_recommendation(
    model_name: str = typer.Option("fraud-detection-baseline_graph", help="Registered model name"),
    candidate_run_id: str = typer.Option(..., help="MLflow run ID of candidate model"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to evaluate"),
    improvement_threshold: float = typer.Option(0.01, help="Absolute improvement threshold"),
    min_improvement_pct: float = typer.Option(1.0, help="Minimum percentage improvement"),
):
    """
    Get deployment recommendation using MLflow Model Registry.
    
    Compares candidate model against production and recommends:
    - PRODUCTION: Significant improvement
    - STAGING: Small improvement (needs validation)
    - REJECT: No improvement or degradation
    
    Example:
        python src/cli.py mlflow-deployment-recommendation --candidate-run-id abc123
    """
    console.print(f"[bold cyan]Evaluating deployment recommendation...[/bold cyan]")
    
    from src.utils.mlflow_model_comparison import recommend_deployment
    
    recommendation = recommend_deployment(
        model_name=model_name,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
        min_improvement_pct=min_improvement_pct,
    )
    
    console.print(f"\n[bold]Deployment Recommendation:[/bold]")
    console.print(f"  Status: {recommendation['recommendation']}")
    console.print(f"  Stage: {recommendation['stage'] or 'N/A'}")
    console.print(f"  Reason: {recommendation['reason']}")
    
    if "comparison" in recommendation:
        comp = recommendation["comparison"]
        if "production_value" in comp:
            console.print(f"\n[bold]Metrics Comparison:[/bold]")
            console.print(f"  Production {primary_metric}: {comp['production_value']:.4f}")
            console.print(f"  Candidate {primary_metric}: {comp['candidate_value']:.4f}")
            console.print(f"  Improvement: {comp.get('improvement', 0):+.4f} ({comp.get('improvement_pct', 0):+.2f}%)")


@app.command()
def mlflow_drift_summary(
    model_name: str = typer.Option("fraud-detection-baseline_graph", help="Registered model name"),
    n_versions: int = typer.Option(5, help="Number of recent versions to analyze"),
):
    """
    Analyze model drift by comparing recent model versions.
    
    Uses MLflow Model Registry to detect performance degradation.
    
    Example:
        python src/cli.py mlflow-drift-summary --model-name fraud-detection-baseline_graph
    """
    console.print(f"[bold cyan]Analyzing model drift...[/bold cyan]")
    
    from src.utils.mlflow_model_comparison import get_model_drift_summary
    
    summary = get_model_drift_summary(
        model_name=model_name,
        n_recent_versions=n_versions,
    )
    
    if "error" in summary:
        console.print(f"[bold red]Error: {summary['error']}[/bold red]")
        return
    
    console.print(f"\n[bold]Drift Analysis:[/bold]")
    console.print(f"  Drift Detected: {'⚠️ Yes' if summary['drift_detected'] else '✅ No'}")
    console.print(f"  Production Version: {summary.get('production_version', 'N/A')}")
    console.print(f"  Versions Analyzed: {summary['versions_analyzed']}")
    
    if summary['drift_detected']:
        console.print(f"\n[bold yellow]Drift Reasons:[/bold yellow]")
        for reason in summary['drift_reasons']:
            console.print(f"  • {reason}")
    else:
        console.print(f"\n[bold green]No significant drift detected[/bold green]")




# --- MLflow Commands ---
# Note: train-mlflow is deprecated. Use train-baseline or train-hybrid instead.
# All training commands now use MLflow by default.



@app.command()
def mlflow_ui():
    """
    Start MLflow UI to view experiments.
    
    Opens browser at http://localhost:5000
    """
    console.print("[bold cyan]Starting MLflow UI...[/bold cyan]")
    console.print("Open http://localhost:5000 in your browser")
    console.print("Press Ctrl+C to stop")
    
    import subprocess
    subprocess.run(["mlflow", "ui", "--port", "5000"])


@app.command()
def mlflow_compare(
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    metric: str = typer.Option("mean_auc_pr", help="Metric to compare"),
    top_n: int = typer.Option(10, help="Number of top runs to show"),
):
    """
    Compare MLflow runs by metric.
    
    Shows top N runs sorted by the specified metric.
    """
    console.print(f"[bold cyan]Comparing runs in {experiment_name}...[/bold cyan]")
    
    import mlflow
    
    try:
        mlflow.set_experiment(experiment_name)
        
        # Search runs
        runs = mlflow.search_runs(
            order_by=[f"metrics.{metric} DESC"],
            max_results=top_n,
        )
        
        if runs.empty:
            console.print("[yellow]No runs found in this experiment.[/yellow]")
            return
        
        # Display results
        console.print(f"\n[bold]Top {len(runs)} runs by {metric}:[/bold]")
        console.print("-" * 80)
        
        for i, row in runs.iterrows():
            run_name = row.get('tags.mlflow.runName', row['run_id'][:8])
            metric_value = row.get(f'metrics.{metric}', 'N/A')
            model_type = row.get('params.model_type', 'unknown')
            
            if isinstance(metric_value, float):
                console.print(f"  {i+1}. {run_name}: {metric_value:.4f} ({model_type})")
            else:
                console.print(f"  {i+1}. {run_name}: {metric_value} ({model_type})")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def mlflow_promote(
    model_name: str = typer.Option("fraud-detection", help="Registered model name"),
    version: int = typer.Option(..., help="Model version to promote"),
    stage: str = typer.Option("Production", help="Target stage: Staging or Production"),
):
    """
    Promote a model version to Staging or Production.
    
    Example:
        python src/cli.py mlflow-promote --version 2 --stage Production
    """
    console.print(f"[bold cyan]Promoting model {model_name} v{version} to {stage}...[/bold cyan]")
    
    import mlflow
    from mlflow.tracking import MlflowClient
    
    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True
        )
        
        console.print(f"[bold green]✓ Model {model_name} v{version} promoted to {stage}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    app()

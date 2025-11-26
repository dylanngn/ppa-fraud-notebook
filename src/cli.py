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
from src.models import train_graph_baseline as graph_baseline_module
from src.models import train_hybrid as hybrid_module
from src.models import train_hybrid as hybrid_module
from src.models import evaluate_seon as seon_module
from src.features import advanced_graph_features as advanced_features_module
from src.features import time_weighted_features as time_weighted_module
from src.features import interaction_features as interaction_module
# Lazy imports for optional dependencies (optuna)
# from src.experiments import optimize_hyperparams as hyperopt_module
# from src.experiments import staged_hyperopt as staged_hyperopt_module

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
    window_days: int = typer.Option(90, help="Training window size in days"),
    step_days: int = typer.Option(7, help="Sliding window step size in days"),
    save_models: bool = typer.Option(True, help="Save models for SHAP analysis")
):
    """
    Train the Baseline XGBoost model using Sliding Window Backtesting.
    """
    console.print(f"[bold green]Training Baseline XGBoost (Window: {window_days} days)...[/bold green]")
    baseline_module.run_baseline(
        window_days=window_days, 
        step_days=step_days, 
        save_models=save_models,
        models_dir="artifacts/models/baseline"
    )


@app.command()
def train_graph_baseline(
    window_days: int = typer.Option(90, help="Training window size in days"),
    step_days: int = typer.Option(7, help="Sliding window step size in days")
):
    """
    Train the Graph-Feature XGBoost baseline.
    """
    console.print(f"[bold green]Training Graph-Feature Baseline (Window: {window_days} days)...[/bold green]")
    graph_baseline_module.main(window_days=window_days, step_days=step_days)

@app.command()
def train_embeddings(
    model: str = typer.Option("hgt", help="Model type: gat, gcn, hgt, hgt_rte"),
    epochs: int = typer.Option(20, help="Number of training epochs")
):
    """
    Train GNN model and generate embeddings.
    """
    console.print(f"[bold green]Training Embeddings ({model.upper()})...[/bold green]")
    from src.models import train_embeddings as train_embeddings_module
    train_embeddings_module.train_embeddings(model_name=model, epochs=epochs)

@app.command()
def train_hybrid(
    model: str = typer.Option("hgt", help="Model type for embeddings: gat, gcn, hgt, hgt_rte"),
    save_models: bool = typer.Option(True, help="Save models for SHAP analysis")
):
    """
    Train the Hybrid Model (XGBoost + GNN Embeddings).
    """
    console.print(f"[bold green]Training Hybrid Model ({model.upper()})...[/bold green]")
    hybrid_module.main(model_name=model, save_models=save_models)


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


@app.command()
def train_baseline_expanding(
    window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Sliding window step size in days")
):
    """
    Train baseline XGBoost with expanding window (accumulating data).
    Simulates production continuous learning.
    """
    console.print("[bold green]Training Baseline (Expanding Window)...[/bold green]")
    from src.models import train_baseline_expanding
    train_baseline_expanding.main(window_days=window_days, step_days=step_days)


@app.command()
def train_hybrid_expanding(
    model: str = typer.Option("hgt", help="Model type: hgt, gat, sage"),
    window_days: int = typer.Option(180, help="Initial training window size in days"),
    step_days: int = typer.Option(7, help="Sliding window step size in days")
):
    """
    Train hybrid model with expanding window (accumulating data).
    Gives HGT/GNN access to complete graph structure.
    """
    console.print(f"[bold green]Training Hybrid {model.upper()} (Expanding Window)...[/bold green]")
    from src.models import train_hybrid_expanding
    train_hybrid_expanding.main(model_name=model, window_days=window_days, step_days=step_days)


@app.command()
def retrain_production(
    include_production: bool = typer.Option(True, help="Include production data")
):
    """
    Retrain production model with all historical + production data.
    Used for weekly continuous learning.
    """
    console.print("[bold green]Retraining Production Model...[/bold green]")
    from src.training import continuous_learner
    continuous_learner.train_production_model(include_production_data=include_production)


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
def debug_polars():
    """
    Run Polars casting debug script.
    """
    console.print("[bold yellow]Debugging Polars Casting...[/bold yellow]")
    from src.utils import debug_polars_cast
    debug_polars_cast.debug_polars_cast()

@app.command()
def explain_model(
    model_type: str = typer.Option(..., help="Model type: baseline, graph_baseline, or hybrid_MODEL"),
    window_idx: int = typer.Option(-1, help="Window index to explain (-1 for latest)"),
    output_dir: str = typer.Option(None, help="Output directory for plots")
):
    """
    Generate SHAP explanations for a trained model window.
    """
    console.print(f"[bold cyan]Generating SHAP Explanations for {model_type}...[/bold cyan]")
    
    from src.utils.explainability import ModelExplainer, load_saved_model, save_explainer
    import glob
    
    # Determine models directory
    models_dir = f"artifacts/models/{model_type}"
    
    if not os.path.exists(models_dir):
        console.print(f"[bold red]Error: Models directory not found: {models_dir}[/bold red]")
        console.print("Run training with --save-models flag first.")
        return
    
    # Find model files
    model_files = sorted(glob.glob(os.path.join(models_dir, "model_window_*.pkl")))
    
    if not model_files:
        console.print(f"[bold red]Error: No model files found in {models_dir}[/bold red]")
        return
    
    # Select model
    if window_idx == -1:
        model_path = model_files[-1]  # Latest window
        console.print(f"Using latest model window: {os.path.basename(model_path)}")
    else:
        model_path = os.path.join(models_dir, f"model_window_{window_idx}.pkl")
        if not os.path.exists(model_path):
            console.print(f"[bold red]Error: Model window {window_idx} not found[/bold red]")
            return
    
    # Load model
    console.print(f"Loading model from {model_path}...")
    model_bundle = load_saved_model(model_path)
    
    # Create explainer
    explainer = ModelExplainer(
        model=model_bundle['model'],
        feature_names=model_bundle['features'],
        X_test=model_bundle['X_test'],
        y_test=model_bundle['y_test'],
        y_pred=model_bundle['y_pred'],
        model_type=model_type,
        window_info=model_bundle.get('window_info', {})
    )
    
    # Determine output directory
    if output_dir is None:
        window_idx_actual = model_bundle.get('window_info', {}).get('window_idx', 0)
        output_dir = f"artifacts/shap/{model_type}/window_{window_idx_actual}"
    
    # Generate all explanations
    save_explainer(explainer, output_dir, prefix="")
    
    console.print(f"[bold green]✓ SHAP explanations saved to {output_dir}[/bold green]")

@app.command()
def compare_models_shap(
    models: str = typer.Option(..., help="Comma-separated model types (e.g., baseline,hybrid_hgt)"),
    window_idx: int = typer.Option(-1, help="Window index to compare (-1 for latest)"),
    output_path: str = typer.Option("artifacts/shap/model_comparison.png", help="Output path for comparison plot")
):
    """
    Compare SHAP feature importance across multiple models.
    """
    console.print("[bold cyan]Comparing Models with SHAP...[/bold cyan]")
    
    from src.utils.explainability import ModelExplainer, load_saved_model, compare_models
    import glob
    
    model_types = [m.strip() for m in models.split(',')]
    explainers = {}
    
    for model_type in model_types:
        models_dir = f"artifacts/models/{model_type}"
        
        if not os.path.exists(models_dir):
            console.print(f"[yellow]Warning: Skipping {model_type} - directory not found[/yellow]")
            continue
        
        # Find model files
        model_files = sorted(glob.glob(os.path.join(models_dir, "model_window_*.pkl")))
        
        if not model_files:
            console.print(f"[yellow]Warning: Skipping {model_type} - no models found[/yellow]")
            continue
        
        # Select model
        if window_idx == -1:
            model_path = model_files[-1]
        else:
            model_path = os.path.join(models_dir, f"model_window_{window_idx}.pkl")
            if not os.path.exists(model_path):
                console.print(f"[yellow]Warning: Skipping {model_type} - window {window_idx} not found[/yellow]")
                continue
        
        # Load and create explainer
        console.print(f"Loading {model_type} from {os.path.basename(model_path)}...")
        model_bundle = load_saved_model(model_path)
        
        explainer = ModelExplainer(
            model=model_bundle['model'],
            feature_names=model_bundle['features'],
            X_test=model_bundle['X_test'],
            y_test=model_bundle['y_test'],
            y_pred=model_bundle['y_pred'],
            model_type=model_type,
            window_info=model_bundle.get('window_info', {})
        )
        
        explainers[model_type] = explainer
    
    if len(explainers) < 2:
        console.print("[bold red]Error: Need at least 2 models to compare[/bold red]")
        return
    
    # Generate comparison
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    compare_models(explainers, output_path=output_path)
    
    console.print(f"[bold green]✓ Model comparison saved to {output_path}[/bold green]")

@app.command()
def analyze_failures(
    model_type: str = typer.Option(..., help="Model type: baseline, graph_baseline, or hybrid_MODEL"),
    window_idx: int = typer.Option(-1, help="Window index to analyze (-1 for latest)"),
    prediction_type: str = typer.Option("FP", help="Prediction type to analyze: TP, FP, TN, FN"),
    top_k: int = typer.Option(20, help="Number of instances to explain")
):
    """
    Analyze false positives, false negatives, or other prediction types with SHAP.
    """
    console.print(f"[bold cyan]Analyzing {prediction_type} predictions for {model_type}...[/bold cyan]")
    
    from src.utils.explainability import ModelExplainer, load_saved_model
    import glob
    
    models_dir = f"artifacts/models/{model_type}"
    
    if not os.path.exists(models_dir):
        console.print(f"[bold red]Error: Models directory not found: {models_dir}[/bold red]")
        return
    
    # Find and load model
    model_files = sorted(glob.glob(os.path.join(models_dir, "model_window_*.pkl")))
    
    if not model_files:
        console.print(f"[bold red]Error: No model files found[/bold red]")
        return
    
    model_path = model_files[-1] if window_idx == -1 else os.path.join(models_dir, f"model_window_{window_idx}.pkl")
    
    if not os.path.exists(model_path):
        console.print(f"[bold red]Error: Model not found at {model_path}[/bold red]")
        return
    
    console.print(f"Loading model from {model_path}...")
    model_bundle = load_saved_model(model_path)
    
    # Create explainer
    explainer = ModelExplainer(
        model=model_bundle['model'],
        feature_names=model_bundle['features'],
        X_test=model_bundle['X_test'],
        y_test=model_bundle['y_test'],
        y_pred=model_bundle['y_pred'],
        model_type=model_type,
        window_info=model_bundle.get('window_info', {})
    )
    
    # Generate explanations
    window_idx_actual = model_bundle.get('window_info', {}).get('window_idx', 0)
    output_dir = f"artifacts/shap/{model_type}/window_{window_idx_actual}/{prediction_type}_analysis"
    
    explainer.explain_top_predictions(
        top_k=top_k,
        prediction_type=prediction_type,
        output_dir=output_dir
    )
    
    # Also generate cohort comparison if analyzing FP or FN
    if prediction_type in ['FP', 'FN']:
        comparison_type = 'TP' if prediction_type == 'FP' else 'TP'
        comparison_path = os.path.join(
            f"artifacts/shap/{model_type}/window_{window_idx_actual}",
            f"{prediction_type}_vs_{comparison_type}_comparison.png"
        )
        explainer.analyze_cohort_differences(
            cohort1_type=comparison_type,
            cohort2_type=prediction_type,
            output_path=comparison_path
        )
    
    console.print(f"[bold green]✓ Analysis saved to {output_dir}[/bold green]")

@app.command()
def compare_baseline_hybrid(
    baseline: str = typer.Option("baseline", help="Baseline model type"),
    hybrid: str = typer.Option("hybrid_hgt", help="Hybrid model type"),
    output_dir: str = typer.Option("artifacts/shap/comparison", help="Output directory")
):
    """
    Generate comprehensive comparison report between baseline and hybrid models.
    
    This answers: Why doesn't the hybrid model outperform the baseline?
    """
    console.print("[bold cyan]Comparing Baseline vs Hybrid Models...[/bold cyan]")
    
    from src.utils.compare_baseline_hybrid import generate_comparison_report
    
    try:
        generate_comparison_report(baseline, hybrid, output_dir)
        console.print(f"\n[bold green]✓ Comparison report complete![/bold green]")
        console.print(f"[bold green]Results saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

@app.command()
def evaluate_seon(
    window_days: int = typer.Option(90, help="Training window size in days"),
    step_days: int = typer.Option(7, help="Sliding window step size in days"),
    include_fallback: bool = typer.Option(True, help="Include fallback predictions")
):
    """
    Evaluate Seon (production baseline) performance using sliding window.
    
    Seon is the current production fraud detection system that we want to beat.
    This command calculates its performance metrics for comparison with our models.
    """
    console.print(f"[bold green]Evaluating Seon Baseline (Window: {window_days} days)...[/bold green]")
    seon_module.main(window_days=window_days, step_days=step_days, include_fallback=include_fallback)

@app.command()
def compare_all_models(
    window_days: int = typer.Option(90, help="Window size for comparison"),
    step_days: int = typer.Option(7, help="Step size for windows"),
    output_dir: str = typer.Option("artifacts/results/comparison", help="Output directory")
):
    """
    Compare ALL models: Baseline XGBoost, Hybrid (GNN+XGBoost), and Seon.
    
    This generates a comprehensive comparison showing:
    - Performance metrics across all models
    - Which model performs best
    - Whether research models beat production Seon baseline
    """
    console.print("[bold cyan]Comparing All Models: XGBoost vs Hybrid vs Seon...[/bold cyan]")
    
    from src.utils.compare_all_models import generate_comprehensive_comparison
    
    try:
        generate_comprehensive_comparison(
            window_days=window_days,
            step_days=step_days,
            output_dir=output_dir
        )
        console.print(f"\n[bold green]✓ Comprehensive comparison complete![/bold green]")
        console.print(f"[bold green]Results saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def analyze_adaptation(
    model_type: str = typer.Option("baseline", help="Model type: baseline, baseline_graph, hybrid_*"),
    window_idx: int = typer.Option(-1, help="Window index to analyze (-1 for latest)"),
    output_dir: str = typer.Option("artifacts/reports", help="Output directory for reports"),
    compute_interactions: bool = typer.Option(False, help="Compute SHAP interactions (slow)"),
    all_windows: bool = typer.Option(False, help="Analyze all available windows"),
):
    """
    Run SHAP-based adaptation analysis on trained models.
    
    Generates actionable adaptation suggestions:
    - Rule suggestions (high-importance features with thresholds)
    - Pruning candidates (zero-importance features)
    - Drift alerts (feature importance changes)
    - Retrain recommendations
    
    Example:
        python src/cli.py analyze-adaptation --model-type baseline
        python src/cli.py analyze-adaptation --all-windows
    """
    console.print(f"[bold cyan]Running Adaptation Analysis for {model_type}...[/bold cyan]")
    
    import glob
    from src.explainability.adaptation_engine import AdaptationEngine, run_adaptation_analysis
    from src.utils.explainability import load_saved_model
    
    models_dir = f"artifacts/models/{model_type}"
    
    if not os.path.exists(models_dir):
        console.print(f"[bold red]Error: Models directory not found: {models_dir}[/bold red]")
        console.print("Run training with --save-models flag first.")
        return
    
    model_files = sorted(glob.glob(os.path.join(models_dir, "model_window_*.pkl")))
    
    if not model_files:
        console.print(f"[bold red]Error: No model files found in {models_dir}[/bold red]")
        return
    
    if all_windows:
        # Analyze all windows
        console.print(f"[bold yellow]Analyzing {len(model_files)} windows...[/bold yellow]")
        
        # Load or create engine with history
        state_path = os.path.join(output_dir, f"adaptation_engine_{model_type}_state.pkl")
        
        if os.path.exists(state_path):
            engine = AdaptationEngine.load_state(state_path)
            console.print(f"Loaded engine state with {len(engine.importance_history)} history windows")
        else:
            # Get feature names from first model
            bundle = load_saved_model(model_files[0])
            engine = AdaptationEngine(feature_names=bundle["features"])
        
        for i, model_path in enumerate(model_files):
            console.print(f"\n[dim]Processing window {i+1}/{len(model_files)}[/dim]")
            
            try:
                bundle = load_saved_model(model_path)
                report = engine.analyze_window(
                    model=bundle["model"],
                    X_test=bundle["X_test"],
                    y_test=bundle["y_test"],
                    y_pred=bundle["y_pred"],
                    window_info=bundle.get("window_info", {"window_idx": i}),
                    compute_interactions=compute_interactions,
                )
                
                # Save report
                report.save_markdown(f"{output_dir}/adaptation_report_window_{i}.md")
                report.save_json(f"{output_dir}/adaptation_report_window_{i}.json")
                
                # Print summary
                console.print(
                    f"  AUC-PR: {report.model_performance.get('auc_pr', 0):.4f}, "
                    f"Drift: {len(report.drift_alerts)}, "
                    f"Rules: {len(report.rule_suggestions)}"
                )
                
            except Exception as e:
                console.print(f"[red]Error on window {i}: {e}[/red]")
        
        # Save engine state
        engine.save_state(state_path)
        console.print(f"\n[bold green]✓ Analyzed all windows. Reports saved to {output_dir}[/bold green]")
        
    else:
        # Single window analysis
        model_path = model_files[window_idx]
        console.print(f"Analyzing: {os.path.basename(model_path)}")
        
        try:
            report = run_adaptation_analysis(
                model_path=model_path,
                output_dir=output_dir,
                compute_interactions=compute_interactions,
            )
            
            console.print(f"\n[bold green]✓ Adaptation analysis complete![/bold green]")
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(f"  Window: {report.window_idx}")
            console.print(f"  AUC-PR: {report.model_performance.get('auc_pr', 0):.4f}")
            console.print(f"  Drift Alerts: {len(report.drift_alerts)}")
            console.print(f"  Rule Suggestions: {len(report.rule_suggestions)}")
            console.print(f"  Pruning Candidates: {len(report.pruning_candidates)}")
            console.print(f"  Retrain Recommended: {'Yes' if report.retrain_recommendation else 'No'}")
            
            if report.rule_suggestions:
                console.print(f"\n[bold]Top Rule Suggestions:[/bold]")
                for rule in report.rule_suggestions[:3]:
                    console.print(f"  • {rule.title} ({rule.priority})")
            
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            import traceback
            traceback.print_exc()


@app.command()
def generate_adaptation_report(
    model_type: str = typer.Option("baseline", help="Model type to analyze"),
    n_windows: int = typer.Option(10, help="Number of recent windows to analyze"),
    output_path: str = typer.Option("artifacts/reports/adaptation_summary.md", help="Output path"),
):
    """
    Generate a comprehensive adaptation report across multiple windows.
    
    Summarizes:
    - Performance trends
    - Feature importance stability
    - Emerging patterns
    - Recommendations
    """
    console.print(f"[bold cyan]Generating Adaptation Summary Report...[/bold cyan]")
    
    import glob
    import json
    from pathlib import Path
    
    reports_dir = Path("artifacts/reports")
    report_files = sorted(reports_dir.glob("adaptation_report_window_*.json"))
    
    if not report_files:
        console.print("[bold red]No adaptation reports found. Run analyze-adaptation first.[/bold red]")
        return
    
    # Load recent reports
    reports = []
    for f in report_files[-n_windows:]:
        with open(f) as fp:
            reports.append(json.load(fp))
    
    console.print(f"Loaded {len(reports)} reports")
    
    # Generate summary markdown
    lines = [
        "# Adaptation Summary Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Windows Analyzed**: {len(reports)}",
        "",
        "## Performance Trend",
        "",
    ]
    
    # Performance table
    lines.append("| Window | AUC-PR | P@100 | Drift Alerts | Rules |")
    lines.append("|--------|--------|-------|--------------|-------|")
    
    for r in reports:
        perf = r.get("model_performance", {})
        lines.append(
            f"| {r.get('window_idx', '?')} | "
            f"{perf.get('auc_pr', 0):.4f} | "
            f"{perf.get('p@100', 0):.4f} | "
            f"{len(r.get('drift_alerts', []))} | "
            f"{len(r.get('rule_suggestions', []))} |"
        )
    
    lines.append("")
    
    # Aggregate feature importance (average across windows)
    from collections import defaultdict
    feature_importance_sum = defaultdict(float)
    feature_importance_count = defaultdict(int)
    
    for r in reports:
        for feat, imp in r.get("current_importance", {}).items():
            feature_importance_sum[feat] += imp
            feature_importance_count[feat] += 1
    
    avg_importance = {
        k: feature_importance_sum[k] / feature_importance_count[k]
        for k in feature_importance_sum
    }
    
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    lines.append("## Most Important Features (Average)")
    lines.append("")
    for feat, imp in sorted_features[:15]:
        lines.append(f"- **{feat}**: {imp:.4f}")
    
    lines.append("")
    
    # Aggregate rule suggestions
    rule_counts = defaultdict(int)
    for r in reports:
        for rule in r.get("rule_suggestions", []):
            rule_counts[rule.get("title", "Unknown")] += 1
    
    if rule_counts:
        lines.append("## Most Frequent Rule Suggestions")
        lines.append("")
        for rule, count in sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"- **{rule}**: {count} windows")
        lines.append("")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    console.print(f"[bold green]✓ Summary saved to {output_path}[/bold green]")


# --- MLflow Commands ---

@app.command()
def train_mlflow(
    model_type: str = typer.Option("baseline", help="Model type: baseline, baseline_graph"),
    experiment_name: str = typer.Option("ppa-fraud-detection", help="MLflow experiment name"),
    window_days: int = typer.Option(90, help="Training window size in days"),
    step_days: int = typer.Option(14, help="Sliding window step size in days"),
    register_model: bool = typer.Option(False, help="Register model in MLflow Model Registry"),
):
    """
    Train fraud detection model with MLflow tracking.
    
    Features:
    - Automatic experiment tracking (hyperparams, metrics, artifacts)
    - Model versioning and registry
    - SHAP explanations logged as artifacts
    - Comparison with production model
    
    Example:
        python src/cli.py train-mlflow --model-type baseline_graph
        python src/cli.py train-mlflow --register-model
    """
    console.print(f"[bold cyan]Training {model_type} with MLflow tracking...[/bold cyan]")
    
    from src.training.mlflow_trainer import train_with_mlflow
    
    try:
        result = train_with_mlflow(
            experiment_name=experiment_name,
            model_type=model_type,
            window_days=window_days,
            step_days=step_days,
            register_model=register_model,
        )
        
        console.print(f"\n[bold green]✓ Training complete![/bold green]")
        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Run ID: {result['run_id']}")
        console.print(f"  Mean AUC-PR: {result['mean_auc_pr']:.4f}")
        console.print(f"  Best AUC-PR: {result['best_auc_pr']:.4f}")
        console.print(f"  Model URI: {result['model_uri']}")
        console.print(f"\n[dim]View in MLflow UI: mlflow ui --port 5000[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()


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
    
    from src.training.mlflow_trainer import MLflowTrainer
    
    try:
        trainer = MLflowTrainer()
        trainer.transition_model_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing=True,
        )
        
        console.print(f"[bold green]✓ Model {model_name} v{version} promoted to {stage}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

if __name__ == "__main__":
    app()

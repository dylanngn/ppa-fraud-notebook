import typer
from rich.console import Console
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

if __name__ == "__main__":
    app()

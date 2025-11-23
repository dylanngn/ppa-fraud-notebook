import typer
from rich.console import Console
import sys
import os

# Add project root to sys.path to allow importing 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import etl, graph_builder
from src.models import train_baseline as baseline_module
from src.models import train_gnn as gnn_module
from src.models import train_hybrid as hybrid_module

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
    step_days: int = typer.Option(7, help="Sliding window step size in days")
):
    """
    Train the Baseline XGBoost model using Sliding Window Backtesting.
    """
    console.print(f"[bold green]Training Baseline XGBoost (Window: {window_days} days)...[/bold green]")
    baseline_module.main(window_days=window_days, step_days=step_days)

@app.command()
def train_gnn(
    epochs: int = typer.Option(20, help="Number of training epochs"),
    split_percent: float = typer.Option(0.8, help="Train/Test split percentage (time-based)")
):
    """
    Train the GNN (HGT) model and extract node embeddings.
    """
    console.print(f"[bold green]Training GNN (Epochs: {epochs}, Split: {split_percent})...[/bold green]")
    gnn_module.train(epochs=epochs, split_percent=split_percent)

@app.command()
def train_gnn_rte(
    epochs: int = typer.Option(20, help="Number of training epochs"),
    split_percent: float = typer.Option(0.8, help="Train/Test split percentage (time-based)")
):
    """
    Train the GNN (HGT) model WITH Relative Temporal Encoding (RTE).
    """
    console.print(f"[bold cyan]Training GNN with RTE (Epochs: {epochs})...[/bold cyan]")
    from src.models import train_gnn_rte as rte_module
    rte_module.train_with_rte(epochs=epochs, split_percent=split_percent)

@app.command()
def train_hybrid():
    """
    Train the Hybrid Model (XGBoost + GNN Embeddings).
    """
    console.print("[bold green]Training Hybrid Model...[/bold green]")
    hybrid_module.main()

if __name__ == "__main__":
    app()

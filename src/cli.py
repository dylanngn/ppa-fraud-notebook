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
def train_embeddings(
    model: str = typer.Option("hgt", help="Model type: gat, gcn, hgt, hgt_rte"),
    epochs: int = typer.Option(20, help="Number of training epochs")
):
    """
    Train GNN model and generate embeddings.
    """
    console.print(f"[bold green]Training Embeddings ({model.upper()})...[/bold green]")
    from src.models import train_embeddings
    train_embeddings.train_embeddings(model_name=model, epochs=epochs)

@app.command()
def train_hybrid(
    model: str = typer.Option("hgt", help="Model type for embeddings: gat, gcn, hgt, hgt_rte")
):
    """
    Train the Hybrid Model (XGBoost + GNN Embeddings).
    """
    console.print(f"[bold green]Training Hybrid Model ({model.upper()})...[/bold green]")
    hybrid_module.main(model_name=model)

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

if __name__ == "__main__":
    app()

import polars as pl
from datetime import datetime, timedelta
import os
import typer

def main(
    start_date_str: str = typer.Option("2023-11-01", help="Start date for analysis (YYYY-MM-DD)"),
    nodes_path: str = typer.Option("artifacts/nodes_listing.parquet", help="Path to listing nodes parquet")
):
    """Check fraud density in time windows."""
    print("Loading Listing Nodes...")
    if not os.path.exists(nodes_path):
        typer.echo(f"Error: {nodes_path} not found.", err=True)
        raise typer.Exit(1)
        
    df = pl.read_parquet(nodes_path)
    
    # Ensure timestamps are correct (ns)
    df = df.with_columns(
        pl.col("submission_at").cast(pl.Datetime("ns")).alias("ts")
    )
    
    # Filter valid range
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        typer.echo("Error: Invalid date format. Use YYYY-MM-DD.", err=True)
        raise typer.Exit(1)
        
    df = df.filter(pl.col("ts") >= start_date)
    
    print(f"Total Listings (>= {start_date.date()}): {len(df)}")
    print(f"Total Frauds: {df['is_fraud'].sum()}")
    
    # Analyze 14-day windows
    print("\n--- 14-Day Windows ---")
    current = start_date
    window = timedelta(days=14)
    
    frauds_14d = []
    for _ in range(5): # Check first 5 windows
        end = current + window
        subset = df.filter((pl.col("ts") >= current) & (pl.col("ts") < end))
        count = len(subset)
        frauds = subset["is_fraud"].sum()
        frauds_14d.append(frauds)
        print(f"Window {current.date()} to {end.date()}: {count} listings, {frauds} frauds")
        current = end
        
    avg_14 = sum(frauds_14d) / len(frauds_14d) if frauds_14d else 0
    print(f"Avg Frauds per 14d: {avg_14:.1f}")

    # Analyze 90-day windows
    print("\n--- 90-Day Windows ---")
    current = start_date
    window = timedelta(days=90)
    
    frauds_90d = []
    for _ in range(3): # Check first 3 windows
        end = current + window
        subset = df.filter((pl.col("ts") >= current) & (pl.col("ts") < end))
        count = len(subset)
        frauds = subset["is_fraud"].sum()
        frauds_90d.append(frauds)
        print(f"Window {current.date()} to {end.date()}: {count} listings, {frauds} frauds")
        current += timedelta(days=14) # Slide by 14 days
        
    avg_90 = sum(frauds_90d) / len(frauds_90d) if frauds_90d else 0
    print(f"Avg Frauds per 90d: {avg_90:.1f}")

if __name__ == "__main__":
    import typer
    typer.run(main)

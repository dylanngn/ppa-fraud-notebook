"""Check fraud density in time windows."""
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

NODES_LISTING = resolve_path("artifacts/nodes_listing.parquet")


def check_density(start_date_str: str = "2023-11-01", nodes_path: str = None):
    """Check fraud density in time windows."""
    if nodes_path is None:
        nodes_path = NODES_LISTING
    
    logger.info("Loading listing nodes...")
    if not Path(nodes_path).exists():
        logger.error(f"{nodes_path} not found.")
        raise SystemExit(1)
        
    df = pl.read_parquet(nodes_path)
    
    # Ensure timestamps are correct (ns)
    df = df.with_columns(
        pl.col("submission_at").cast(pl.Datetime("ns")).alias("ts")
    )
    
    # Filter valid range
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Use YYYY-MM-DD.")
        raise SystemExit(1)
        
    df = df.filter(pl.col("ts") >= start_date)
    
    logger.info(f"Total Listings (>= {start_date.date()}): {len(df)}")
    logger.info(f"Total Frauds: {df['is_fraud'].sum()}")
    
    # Analyze 14-day windows
    logger.info("--- 14-Day Windows ---")
    current = start_date
    window = timedelta(days=14)
    
    frauds_14d = []
    for _ in range(5):  # Check first 5 windows
        end = current + window
        subset = df.filter((pl.col("ts") >= current) & (pl.col("ts") < end))
        count = len(subset)
        frauds = subset["is_fraud"].sum()
        frauds_14d.append(frauds)
        logger.info(f"Window {current.date()} to {end.date()}: {count} listings, {frauds} frauds")
        current = end
        
    avg_14 = sum(frauds_14d) / len(frauds_14d) if frauds_14d else 0
    logger.info(f"Avg Frauds per 14d: {avg_14:.1f}")

    # Analyze 90-day windows
    logger.info("--- 90-Day Windows ---")
    current = start_date
    window = timedelta(days=90)
    
    frauds_90d = []
    for _ in range(3):  # Check first 3 windows
        end = current + window
        subset = df.filter((pl.col("ts") >= current) & (pl.col("ts") < end))
        count = len(subset)
        frauds = subset["is_fraud"].sum()
        frauds_90d.append(frauds)
        logger.info(f"Window {current.date()} to {end.date()}: {count} listings, {frauds} frauds")
        current += timedelta(days=14)  # Slide by 14 days
        
    avg_90 = sum(frauds_90d) / len(frauds_90d) if frauds_90d else 0
    logger.info(f"Avg Frauds per 90d: {avg_90:.1f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check fraud density in time windows")
    parser.add_argument("--start-date", default="2023-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--nodes-path", default=None, help=f"Path to listing nodes (default: {NODES_LISTING})")
    
    args = parser.parse_args()
    check_density(start_date_str=args.start_date, nodes_path=args.nodes_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

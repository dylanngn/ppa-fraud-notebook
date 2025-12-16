"""
Data loading module for SEON + Snowflake events pipeline.
Handles saving processed data to parquet format.
"""
import logging
import os
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def save_parquet(
    df: pl.DataFrame,
    path: str,
    compression: str = "zstd",
) -> None:
    """
    Save DataFrame to parquet file.
    
    Args:
        df: DataFrame to save
        path: Output path
        compression: Compression algorithm (default: zstd)
    """
    path = Path(path)
    
    # Create parent directories
    os.makedirs(path.parent, exist_ok=True)
    
    logger.info(f"Saving to {path}...")
    df.write_parquet(path, compression=compression)
    
    # Log file size
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved {len(df):,} rows, {len(df.columns)} columns ({size_mb:.2f} MB)")


def save_summary(
    df: pl.DataFrame,
    path: str,
) -> None:
    """
    Save summary statistics to JSON file.
    
    Args:
        df: DataFrame to summarize
        path: Output path for summary JSON
    """
    import json
    
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    
    # Compute summary stats
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_insertions": df.select("INSERTION_ID").n_unique() if "INSERTION_ID" in df.columns else None,
        "fraud_events": df.filter(pl.col("is_fraud") == 1).height if "is_fraud" in df.columns else None,
        "non_fraud_events": df.filter(pl.col("is_fraud") == 0).height if "is_fraud" in df.columns else None,
    }
    
    # Status distribution
    if "STATUS" in df.columns:
        status_counts = (
            df.group_by("STATUS")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        summary["status_distribution"] = {
            row["STATUS"]: row["count"]
            for row in status_counts.iter_rows(named=True)
        }
    
    # SEON state distribution (benchmark)
    if "seon_state" in df.columns:
        seon_counts = (
            df.group_by("seon_state")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        summary["seon_state_distribution"] = {
            str(row["seon_state"]): row["count"]
            for row in seon_counts.iter_rows(named=True)
        }
    
    # Save
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved summary to {path}")


def load_data(df: pl.DataFrame, output_path: str) -> None:
    """
    Main load function - saves processed data.
    
    Args:
        df: Processed DataFrame
        output_path: Path for output parquet file
    """
    # Save main data
    save_parquet(df, output_path)
    
    # Save summary
    summary_path = str(output_path).replace(".parquet", "_summary.json")
    save_summary(df, summary_path)

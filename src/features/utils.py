"""
Utility functions for feature engineering and temporal filtering.
"""
from pathlib import Path
from datetime import datetime
from typing import Optional
import polars as pl

ARTIFACTS_DIR = Path("artifacts")
LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"


def _groupby(df: pl.DataFrame, *args, **kwargs):
    """
    Polars version compatibility shim for groupby/group_by.
    
    Args:
        df: Polars DataFrame
        *args, **kwargs: Arguments passed to groupby/group_by
        
    Returns:
        Grouped DataFrame
    """
    method = getattr(df, "groupby", None)
    if method is None:
        method = getattr(df, "group_by", None)
    if method is None:
        raise AttributeError("DataFrame has no groupby/group_by method. Please update Polars.")
    return method(*args, **kwargs)


def ensure_artifact(path: Path) -> bool:
    """
    Check if an artifact file exists.
    
    Args:
        path: Path to artifact file
        
    Returns:
        True if file exists, False otherwise
    """
    if not path.exists():
        return False
    return True


def load_listing_ids() -> pl.DataFrame:
    """
    Load all listing IDs from nodes_listing.parquet.
    
    Returns:
        DataFrame with 'insertion_id' column
        
    Raises:
        FileNotFoundError: If listing nodes file doesn't exist
    """
    if not ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes file missing: {LISTING_NODES}")
    return pl.read_parquet(LISTING_NODES).select("insertion_id")


def load_listings_with_timestamps() -> pl.DataFrame:
    """
    Load listings with their submission timestamps.
    
    Returns:
        DataFrame with 'insertion_id' and 'submission_at' columns
        
    Raises:
        FileNotFoundError: If listing nodes file doesn't exist
    """
    if not ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes not found: {LISTING_NODES}")
    return pl.read_parquet(LISTING_NODES).select(["insertion_id", "submission_at"])


def safe_load_edges(path: Path) -> Optional[pl.DataFrame]:
    """
    Safely load edge file with validation.
    
    Args:
        path: Path to edge parquet file
        
    Returns:
        DataFrame with 'listing_id' and 'target' columns, or None if file doesn't exist or is invalid
    """
    if not ensure_artifact(path):
        return None
    
    df = pl.read_parquet(path)
    if "source" not in df.columns or "target" not in df.columns:
        return None
    
    df = df.drop_nulls(["source", "target"])
    if df.is_empty():
        return None
    
    return df.select([pl.col("source").alias("listing_id"), pl.col("target")])


def filter_edges_by_time(
    edges: pl.DataFrame,
    listings_df: pl.DataFrame,
    cutoff_date: datetime,
    edge_type: str = "listing_to_target"
) -> pl.DataFrame:
    """
    Filter edges to only include listings before cutoff_date.
    
    This prevents temporal leakage by ensuring graph features only use
    historical data available at the time of prediction.
    
    Args:
        edges: DataFrame with 'source'/'listing_id' and 'target' columns
        listings_df: DataFrame with 'insertion_id' and 'submission_at' columns
        cutoff_date: Only include listings with submission_at < cutoff_date
        edge_type: Type of edge:
            - "listing_to_target": source is listing_id (e.g., listing -> email)
            - "target_to_listing": target is listing_id (e.g., user -> listing)
            - "listing_to_listing": both source and target are listing_ids
    
    Returns:
        Filtered edges DataFrame with only 'source' and 'target' columns
    """
    if edges.is_empty():
        return edges
    
    # Normalize column names: handle both 'source' and 'listing_id' as the listing column
    if "listing_id" in edges.columns and "source" not in edges.columns:
        edges = edges.rename({"listing_id": "source"})
    
    # Get listing timestamps
    listing_times = listings_df.select(["insertion_id", "submission_at"])
    
    if edge_type == "listing_to_target":
        # source is listing_id
        edges_with_time = edges.join(
            listing_times,
            left_on="source",
            right_on="insertion_id",
            how="left"
        )
        filtered = edges_with_time.filter(
            pl.col("submission_at") < cutoff_date
        ).select(["source", "target"])
        
    elif edge_type == "target_to_listing":
        # target is listing_id
        edges_with_time = edges.join(
            listing_times,
            left_on="target",
            right_on="insertion_id",
            how="left"
        )
        filtered = edges_with_time.filter(
            pl.col("submission_at") < cutoff_date
        ).select(["source", "target"])
        
    elif edge_type == "listing_to_listing":
        # Both source and target are listing_ids
        edges_with_src_time = edges.join(
            listing_times,
            left_on="source",
            right_on="insertion_id",
            how="left"
        ).rename({"submission_at": "source_time"})
        
        edges_with_both_times = edges_with_src_time.join(
            listing_times,
            left_on="target",
            right_on="insertion_id",
            how="left"
        ).rename({"submission_at": "target_time"})
        
        filtered = edges_with_both_times.filter(
            (pl.col("source_time") < cutoff_date) &
            (pl.col("target_time") < cutoff_date)
        ).select(["source", "target"])
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")
    
    return filtered


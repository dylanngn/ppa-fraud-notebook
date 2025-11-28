"""
Utility functions for temporal filtering of graph edges.
"""

from datetime import datetime
import polars as pl


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
        edges: DataFrame with 'source' and 'target' columns
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
        # Filter by cutoff_date
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
        # Filter by cutoff_date
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
        
        # Filter: both source and target must be before cutoff_date
        filtered = edges_with_both_times.filter(
            (pl.col("source_time") < cutoff_date) &
            (pl.col("target_time") < cutoff_date)
        ).select(["source", "target"])
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")
    
    return filtered


def load_listings_with_timestamps() -> pl.DataFrame:
    """
    Load listings with their submission timestamps.
    
    Returns:
        DataFrame with 'insertion_id' and 'submission_at' columns
    """
    from pathlib import Path
    LISTING_NODES = Path("artifacts/nodes_listing.parquet")
    
    if not LISTING_NODES.exists():
        raise FileNotFoundError(f"Listing nodes not found: {LISTING_NODES}")
    
    df = pl.read_parquet(LISTING_NODES).select(["insertion_id", "submission_at"])
    return df


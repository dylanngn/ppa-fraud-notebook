"""
Graph feature definitions.

Computes all graph-derived features (basic + advanced) in a single category.
"""
import polars as pl
from datetime import datetime
from typing import Any
from src.features.xgboost.registry import FeatureRegistry
from src.features.xgboost import generators


@FeatureRegistry.register("graph")
def compute_graph_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute ALL graph features (basic + advanced).
    
    Generates ~17 features:
    - Contact-based: shared emails, phones, counts
    - User-level: listing counts, IP sharing
    - Network structure: PageRank, connected components
    - Isolation: degree, isolated flag
    - Clustering: neighbor overlap, avg neighbor degree
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future configuration (currently unused)
    """
    # Generate all graph features (merged basic + advanced)
    graph_df = generators.graph_features.generate_graph_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if graph_df is None:
        raise ValueError("Failed to generate graph features")
    
    # Cast UInt32 to Int64 for XGBoost compatibility
    graph_df = graph_df.select([
        pl.col(c).cast(pl.Int64) if graph_df[c].dtype == pl.UInt32 else pl.col(c)
        for c in graph_df.columns
    ])
    
    df = df.join(graph_df, on="insertion_id", how="left")
    
    return df

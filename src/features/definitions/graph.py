"""
Graph feature definitions.

Computes all graph-derived features (basic + advanced) in a single category.
"""
import polars as pl
from datetime import datetime
from typing import Any
from src.features.registry import FeatureRegistry
from src.features import generators


@FeatureRegistry.register("graph")
def compute_graph_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute ALL graph features (basic + advanced).
    
    Generates 17 features:
    - Basic (12): contact reuse, user behavior, component analysis
    - Advanced (5): degree, isolation, neighbor overlap, pagerank
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future configuration (currently unused)
    """
    # Basic graph features
    graph_df = generators.graph_features.generate_graph_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if graph_df is None:
        raise ValueError("Failed to generate graph features")
    
    # Cast UInt32 to Int64
    graph_df = graph_df.select([
        pl.col(c).cast(pl.Int64) if graph_df[c].dtype == pl.UInt32 else pl.col(c)
        for c in graph_df.columns
    ])
    
    df = df.join(graph_df, on="insertion_id", how="left")
    
    # Advanced graph features
    advanced_df = generators.advanced_graph_features.generate_advanced_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if advanced_df is None:
        raise ValueError("Failed to generate advanced graph features")
    
    df = df.join(advanced_df, on="insertion_id", how="left")
    
    return df

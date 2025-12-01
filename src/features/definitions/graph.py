"""
Graph feature definitions.
"""
import polars as pl
from datetime import datetime
from typing import Any
from src.features.registry import FeatureRegistry
from src.features import generators

@FeatureRegistry.register("graph")
def compute_graph_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute basic graph features.
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future per-category configuration (currently unused)
    """
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
    
    return df.join(graph_df, on="insertion_id", how="left")

@FeatureRegistry.register("advanced_graph")
def compute_advanced_graph_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute advanced graph features.
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future per-category configuration (currently unused)
    """
    advanced_df = generators.advanced_graph_features.generate_advanced_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if advanced_df is None:
        raise ValueError("Failed to generate advanced graph features")
        
    return df.join(advanced_df, on="insertion_id", how="left")

@FeatureRegistry.register("time_weighted")
def compute_time_weighted_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute time-weighted features.
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future per-category configuration (currently unused)
    """
    time_df = generators.time_weighted_features.generate_time_weighted_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if time_df is None:
        raise ValueError("Failed to generate time-weighted features")
        
    return df.join(time_df, on="insertion_id", how="left")


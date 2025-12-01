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

@FeatureRegistry.register("interaction")
def compute_interaction_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute interaction features.
    
    NOTE: Interaction features are DEPRECATED for model training (Exp 10 showed
    XGBoost learns these automatically). They are kept for explainability/rules.
    
    Requires: graph and advanced_graph categories must be computed first.
    
    Args:
        df: Input DataFrame with raw listing data
        cutoff_date: Temporal cutoff for feature computation
        config: Reserved for future per-category configuration (currently unused)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Check if required graph feature columns exist in the dataframe
    required_cols = ["shared_contact_email_count", "shared_contact_phone_count", "listing_component_size"]
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        logger.warning(f"Interaction features skipped - missing required columns: {missing}. "
                      f"Ensure 'graph' and 'advanced_graph' categories are computed first.")
        return df
    
    # Extract graph feature columns for the generator
    graph_cols = ["insertion_id", "shared_contact_email_count", "shared_contact_phone_count",
                  "listing_component_size", "listing_pagerank"]
    graph_cols = [c for c in graph_cols if c in df.columns]
    graph_features_df = df.select(graph_cols)
    
    advanced_cols = ["insertion_id", "is_isolated", "degree_total"]
    advanced_cols = [c for c in advanced_cols if c in df.columns]
    advanced_features_df = df.select(advanced_cols)
    
    from src.features import generators as gen_module
    
    interaction_df = gen_module.interaction_features.generate_interaction_features(
        output_path=None,
        graph_features_df=graph_features_df,
        advanced_features_df=advanced_features_df
    )
    
    if interaction_df is None:
        raise ValueError("Failed to generate interaction features")
        
    return df.join(interaction_df, on="insertion_id", how="left")

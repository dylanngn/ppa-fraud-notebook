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
    """Compute basic graph features."""
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
    """Compute advanced graph features."""
    advanced_df = generators.advanced_graph_features.generate_advanced_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if advanced_df is None:
        raise ValueError("Failed to generate advanced graph features")
        
    return df.join(advanced_df, on="insertion_id", how="left")

@FeatureRegistry.register("time_weighted")
def compute_time_weighted_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """Compute time-weighted features."""
    time_df = generators.time_weighted_features.generate_time_weighted_features(
        output_path=None,
        cutoff_date=cutoff_date
    )
    if time_df is None:
        raise ValueError("Failed to generate time-weighted features")
        
    return df.join(time_df, on="insertion_id", how="left")

@FeatureRegistry.register("interaction")
def compute_interaction_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """Compute interaction features."""
    # For interaction features, we need graph features to already be in the dataframe
    # The FeatureProcessor will ensure they're computed in order
    
    # Check if required graph feature columns exist
    required_cols = ["shared_contact_email_count", "shared_contact_phone_count"]
    has_required = all(col in df.columns for col in required_cols)
    
    if not has_required:
        # If graph features aren't in the dataframe yet, skip interaction features
        # This can happen if interaction is requested without graph features
        import logging
        logging.warning("Interaction features skipped - graph features not available in dataframe")
        return df
    
    # Call the original function but pass None for the separate dataframes
    # since the features are already joined into the main df
    from src.features import generators as gen_module
    
    # Extract just the insertion_id and graph feature columns for the call
    # This is a workaround - ideally we'd refactor interaction_features to work directly with the df
    interaction_df = gen_module.interaction_features.generate_interaction_features(
        output_path=None,
        cutoff_date=cutoff_date,
        graph_features_df=None,  # Pass None - it will recompute if needed
        advanced_features_df=None
    )
    
    if interaction_df is None:
        raise ValueError("Failed to generate interaction features")
        
    return df.join(interaction_df, on="insertion_id", how="left")

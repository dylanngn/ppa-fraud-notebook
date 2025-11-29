"""
Feature engineering functions for model training.

This module provides functions to compute and combine features for fraud detection.
All graph features support temporal filtering to prevent data leakage.
"""
import numpy as np
import polars as pl
from datetime import datetime
from typing import Optional, List

from src.models.constants import (
    TEXT_FEATURE_COLUMNS,
    ALL_GRAPH_FEATURE_COLUMNS,
    BASE_FEATURES,
    GRAPH_FEATURE_COLUMNS,
    ADVANCED_GRAPH_FEATURE_COLUMNS,
    TIME_WEIGHTED_FEATURE_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
)
from src.models.experiment_config import ExperimentConfig, FeatureCategory


def compute_text_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute text features from listing descriptions.
    
    These are per-listing properties that don't need temporal filtering.
    
    Args:
        df: DataFrame with listings, must have 'insertion_id' and 'description_text' columns
        
    Returns:
        DataFrame with text features
        
    Raises:
        ValueError: If required columns are missing
    """
    if "insertion_id" not in df.columns:
        raise ValueError("DataFrame must contain 'insertion_id' column")
    if "description_text" not in df.columns:
        raise ValueError("DataFrame must contain 'description_text' column for text feature computation")
    
    text_features = df.select([
        "insertion_id",
        pl.col("description_text").fill_null("")
    ]).with_columns([
        # Character count
        pl.col("description_text").str.len_chars().alias("description_length"),
        
        # Word count (split by whitespace)
        pl.col("description_text").str.split(" ").list.len().alias("description_word_count"),
        
        # URL detection (simple pattern: http:// or https://)
        pl.col("description_text").str.contains(r"(?i)https?://").cast(pl.Int8).alias("description_has_url"),
        
        # Email detection (simple pattern: @)
        pl.col("description_text").str.contains(r"@").cast(pl.Int8).alias("description_has_email"),
        
        # Phone detection (simple pattern: digits with separators)
        pl.col("description_text").str.contains(r"\+?\d[\d\s\-\(\)]{7,}").cast(pl.Int8).alias("description_has_phone"),
        
        # Uppercase ratio
        (
            pl.col("description_text").str.count_matches(r"[A-Z]") / 
            pl.max_horizontal([
                pl.col("description_text").str.len_chars(),
                pl.lit(1)  # Avoid division by zero
            ])
        ).alias("description_caps_ratio"),
        
        # Exclamation count
        pl.col("description_text").str.count_matches(r"!").alias("description_exclamation_count"),
        
        # Question mark count
        pl.col("description_text").str.count_matches(r"\?").alias("description_question_count"),
        
        # All caps words count (words that are entirely uppercase)
        pl.col("description_text").str.count_matches(r"\b[A-Z]{2,}\b").alias("description_all_caps_words"),
        
        # Average word length
        (
            pl.col("description_text").str.len_chars() / 
            pl.max_horizontal([
                pl.col("description_text").str.split(" ").list.len(),
                pl.lit(1)
            ])
        ).alias("description_avg_word_length"),
    ]).select([
        "insertion_id",
        *TEXT_FEATURE_COLUMNS
    ])
    
    # Fill any nulls with 0
    numerical_cols = [col for col in text_features.columns if col != "insertion_id"]
    text_features = text_features.with_columns([
        pl.col(col).fill_null(0) for col in numerical_cols
    ])
    
    return text_features


def compute_graph_features_for_window(
    cutoff_date: datetime,
    feature_categories: List[FeatureCategory]
) -> pl.DataFrame:
    """
    Compute graph features for a specific time window.
    
    This function computes graph features using only data before cutoff_date,
    preventing temporal leakage. Supports selective feature category inclusion.
    
    Args:
        cutoff_date: Only use edges from listings before this date
        feature_categories: List of feature categories to include
        
    Returns:
        DataFrame with graph features for all listings (features computed with cutoff_date)
        
    Raises:
        ValueError: If feature generation fails
    """
    from src.features import (
        graph_features,
        advanced_graph_features,
        time_weighted_features,
        interaction_features
    )
    from src.features.utils import load_listing_ids
    
    # Start with all listing IDs as the base DataFrame
    df = load_listing_ids()
    
    graph_df = None
    advanced_df = None
    
    # Graph features - compute in memory (no disk I/O)
    graph_df = None
    if FeatureCategory.GRAPH in feature_categories:
        graph_df = graph_features.generate_graph_features(
            output_path=None,
            cutoff_date=cutoff_date
        )
        if graph_df is None:
            raise ValueError("Failed to generate graph features")
        df = df.join(graph_df, on="insertion_id", how="left")
    
    # Advanced graph features - compute in memory
    advanced_df = None
    if FeatureCategory.ADVANCED_GRAPH in feature_categories:
        advanced_df = advanced_graph_features.generate_advanced_features(
            output_path=None,
            cutoff_date=cutoff_date
        )
        if advanced_df is None:
            raise ValueError("Failed to generate advanced graph features")
        df = df.join(advanced_df, on="insertion_id", how="left")
    
    # Time-weighted features - compute in memory
    if FeatureCategory.TIME_WEIGHTED in feature_categories:
        time_df = time_weighted_features.generate_time_weighted_features(
            output_path=None,
            cutoff_date=cutoff_date
        )
        if time_df is None:
            raise ValueError("Failed to generate time-weighted features")
        df = df.join(time_df, on="insertion_id", how="left")
    
    # Interaction features (depends on graph features)
    if FeatureCategory.INTERACTION in feature_categories:
        interaction_df = interaction_features.generate_interaction_features(
            output_path=None,
            cutoff_date=cutoff_date,
            graph_features_df=graph_df,
            advanced_features_df=advanced_df
        )
        if interaction_df is None:
            raise ValueError("Failed to generate interaction features")
        df = df.join(interaction_df, on="insertion_id", how="left")
    
    return df


def add_base_tabular_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add base tabular features that don't depend on graph structure.
    
    These are computed once and reused across all windows because they are
    intrinsic properties of each listing (fixed at submission time):
    - account_age_days: Age of account when listing was submitted (fixed)
    - log_price: Price of the listing (fixed)
    - living_space, rooms, etc.: Properties of the property (fixed)
    
    Unlike graph features (which are relational and change as graph grows),
    tabular features are point-in-time snapshots that don't depend on other
    listings or future data.
    
    Graph features are computed separately per window with temporal filtering.
    """
    # 1. Account Age (The critical feature)
    df = df.with_columns(
        (pl.col("submission_at") - pl.col("account_created_at")).dt.total_days().alias("account_age_days")
    )
    
    # 2. Price Normalization (Simple log)
    df = df.with_columns(
        pl.col("price_rent_gross").log1p().alias("log_price")
    )
    
    # 3. Boolean features (Cast to Int)
    bool_cols = ["is_new", "has_balcony", "has_elevator", "has_parking"]
    bool_exprs = [
        pl.col(col).fill_null(False).cast(pl.Int8) if col in df.columns else pl.lit(0).alias(col)
        for col in bool_cols
    ]
    df = df.with_columns(bool_exprs)
            
    # 4. Bundle Info
    # Keep bundle_tier as categorical string for XGBoost native categorical support
    if "bundle_tier" in df.columns:
        df = df.with_columns([
            pl.col("bundle_period").fill_null(7) if "bundle_period" in df.columns else pl.lit(7).alias("bundle_period"),
            pl.col("bundle_tier").fill_null("basic").str.to_lowercase().alias("bundle_tier"),
        ])
    else:
        df = df.with_columns([
            pl.col("bundle_period").fill_null(7) if "bundle_period" in df.columns else pl.lit(7).alias("bundle_period"),
            pl.lit("basic").alias("bundle_tier"),
        ])

    # 5. Payment Type - Keep as categorical string for XGBoost native categorical support
    if "payment_type" in df.columns:
        df = df.with_columns([
            pl.col("payment_type").fill_null("INVOICE").str.to_uppercase().alias("payment_type"),
        ])
    else:
        df = df.with_columns([
            pl.lit("INVOICE").alias("payment_type"),
        ])
        
    # 6. Offer Type - Keep as categorical string for XGBoost native categorical support
    if "offer_type" in df.columns:
        df = df.with_columns([
            pl.col("offer_type").fill_null("RENT").str.to_uppercase().alias("offer_type"),
        ])
    else:
        df = df.with_columns([
            pl.lit("RENT").alias("offer_type"),
        ])
        
    # 7. Location features
    location_exprs = [
        pl.col(col).fill_null(0.0) if col in df.columns else pl.lit(0.0).alias(col)
        for col in ["latitude", "longitude"]
    ]
    df = df.with_columns(location_exprs)

    return df


def feature_engineering(
    df: pl.DataFrame,
    cutoff_date: datetime,
    config: ExperimentConfig
) -> pl.DataFrame:
    """
    Creates tabular features for model training.
    
    Supports configurable feature categories via ExperimentConfig.
    All graph features use temporal filtering to prevent data leakage.
    Text features are computed on-the-fly from listing descriptions.
    
    Args:
        df: DataFrame with listings
        cutoff_date: Compute graph features using only data before this date
                    to prevent temporal leakage
        config: ExperimentConfig to control which feature categories to include
        
    Returns:
        DataFrame with all requested features added
        
    Raises:
        ValueError: If required columns are missing or feature generation fails
    """
    if "insertion_id" not in df.columns:
        raise ValueError("DataFrame must contain 'insertion_id' column")
    
    # Compute graph features per window with temporal filtering
    graph_features_df = compute_graph_features_for_window(
        cutoff_date,
        feature_categories=config.feature_categories
    )
    
    # Cast UInt32 columns to Int64 to avoid MLflow warnings
    graph_features_df = graph_features_df.select([
        pl.col(c).cast(pl.Int64) if graph_features_df[c].dtype == pl.UInt32 else pl.col(c)
        for c in graph_features_df.columns
    ])
    
    # Join graph features
    df = df.join(graph_features_df, on="insertion_id", how="left")
    
    # Compute text features on-the-fly if requested (no temporal filtering needed)
    if FeatureCategory.TEXT in config.feature_categories:
        text_features = compute_text_features(df)
        df = df.join(text_features, on="insertion_id", how="left")
    
    # Determine which feature columns to process based on config
    expected_features = config.get_feature_columns()
    feature_cols_to_process = [
        col for col in expected_features
        if col in df.columns and col not in BASE_FEATURES
    ]
    
    # Fill nulls with np.nan for graph/text feature columns to let XGBoost handle missing values natively
    # XGBoost's sparsity-aware algorithm can learn optimal split directions for missing values
    for col in feature_cols_to_process:
        if col in df.columns:
            # Convert to float to support NaN, then fill nulls with NaN
            if df[col].dtype not in [pl.Float32, pl.Float64]:
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            df = df.with_columns(pl.col(col).fill_null(np.nan))
    
    return df


def build_feature_sources(feature_names: List[str]) -> dict:
    """
    Build a mapping of feature names to their source categories.
    
    Uses constants for consistent categorization across the codebase.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to their source categories
    """
    sources = {}
    
    # Create sets for efficient lookup
    base_set = set(BASE_FEATURES)
    graph_set = set(GRAPH_FEATURE_COLUMNS)
    advanced_set = set(ADVANCED_GRAPH_FEATURE_COLUMNS)
    time_weighted_set = set(TIME_WEIGHTED_FEATURE_COLUMNS)
    interaction_set = set(INTERACTION_FEATURE_COLUMNS)
    text_set = set(TEXT_FEATURE_COLUMNS)
    
    # Categorize features
    for feat in feature_names:
        if feat in base_set:
            sources[feat] = "base"
        elif feat in graph_set:
            sources[feat] = "graph"
        elif feat in advanced_set:
            sources[feat] = "graph_advanced"
        elif feat in time_weighted_set:
            sources[feat] = "time_weighted"
        elif feat in interaction_set:
            sources[feat] = "interaction"
        elif feat in text_set:
            sources[feat] = "text"
        else:
            sources[feat] = "unknown"
    
    return sources


def load_data() -> pl.DataFrame:
    """
    Loads and joins listing and user data.
    
    Returns:
        DataFrame with listings and user data joined
    """
    df_listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    
    return df_listings.join(df_users, on="user_id", how="left")


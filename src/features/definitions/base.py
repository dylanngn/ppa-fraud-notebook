"""
Base feature definitions.
"""
import polars as pl
from datetime import datetime
from typing import Any
from src.features.registry import FeatureRegistry

@FeatureRegistry.register("base")
def compute_base_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Add base tabular features that don't depend on graph structure.
    """
    # 1. Account Age
    if "submission_at" in df.columns and "account_created_at" in df.columns:
        df = df.with_columns(
            (pl.col("submission_at") - pl.col("account_created_at")).dt.total_days().cast(pl.Float64).alias("account_age_days")
        )
    
    # 2. Price Normalization
    if "price_rent_gross" in df.columns:
        df = df.with_columns(
            pl.col("price_rent_gross").log1p().cast(pl.Float64).alias("log_price")
        )
    
    # 3. Boolean features
    bool_cols = ["is_new", "has_balcony", "has_elevator", "has_parking"]
    bool_exprs = [
        pl.col(col).fill_null(False).cast(pl.Int8) if col in df.columns else pl.lit(0).alias(col)
        for col in bool_cols
    ]
    df = df.with_columns(bool_exprs)
            
    # 4. Bundle Info
    if "bundle_tier" in df.columns:
        df = df.with_columns([
            pl.col("bundle_period").cast(pl.Float64) if "bundle_period" in df.columns else pl.lit(None).cast(pl.Float64).alias("bundle_period"),
            pl.col("bundle_tier").fill_null("basic").str.to_lowercase().alias("bundle_tier"),
        ])
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("bundle_period"),
            pl.lit("basic").alias("bundle_tier"),
        ])

    # 5. Payment Type
    if "payment_type" in df.columns:
        df = df.with_columns([
            pl.col("payment_type").fill_null("INVOICE").str.to_uppercase().alias("payment_type"),
        ])
    else:
        df = df.with_columns([
            pl.lit("INVOICE").alias("payment_type"),
        ])
        
    # 6. Offer Type
    if "offer_type" in df.columns:
        df = df.with_columns([
            pl.col("offer_type").fill_null("RENT").str.to_uppercase().alias("offer_type"),
        ])
    else:
        df = df.with_columns([
            pl.lit("RENT").alias("offer_type"),
        ])
        
    # 7. Location features
    location_exprs = []
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            location_exprs.append(pl.col(col).cast(pl.Float64))
        else:
            location_exprs.append(pl.lit(None).cast(pl.Float64).alias(col))
    df = df.with_columns(location_exprs)
    
    # 8. Ensure other numerical base features support NaN
    numerical_base_cols = ["living_space", "rooms"]
    for col in numerical_base_cols:
        if col in df.columns:
            if df[col].dtype not in [pl.Float32, pl.Float64]:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

    return df

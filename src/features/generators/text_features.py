"""
Text Features for XGBoost

Simple, tabular-friendly text features extracted from listing descriptions.
These are lightweight alternatives to embeddings for tree-based models.

Features:
- description_length: Character count
- description_word_count: Word count
- description_has_url: Binary indicator for URLs
- description_has_email: Binary indicator for email addresses
- description_has_phone: Binary indicator for phone numbers
- description_has_caps_ratio: Ratio of uppercase characters
- description_exclamation_count: Count of exclamation marks
- description_question_count: Count of question marks
"""
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

from src.features.utils import ensure_artifact, ARTIFACTS_DIR

LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_text_features.parquet"


def generate_text_features(output_path: Path = OUTPUT_PATH) -> None:
    """
    Generate simple text features from listing descriptions.
    
    These features are optimized for XGBoost and don't require
    expensive embedding generation.
    """
    logger.info("Loading listings...")
    
    if not ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes missing: {LISTING_NODES}")
    
    # Load listings - description_text must be in nodes_listing.parquet
    df_listings = pl.read_parquet(LISTING_NODES)
    
    if "description_text" not in df_listings.columns:
        raise ValueError("description_text column not found in nodes_listing.parquet. Ensure ETL includes description_text extraction.")
    
    logger.info(f"Processing {len(df_listings):,} listings...")
    
    # Compute text features
    df_features = df_listings.select([
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
        "description_length",
        "description_word_count",
        "description_has_url",
        "description_has_email",
        "description_has_phone",
        "description_caps_ratio",
        "description_exclamation_count",
        "description_question_count",
        "description_all_caps_words",
        "description_avg_word_length",
    ])
    
    # Fill any nulls with 0
    numerical_cols = [col for col in df_features.columns if col != "insertion_id"]
    df_features = df_features.with_columns([
        pl.col(col).fill_null(0) for col in numerical_cols
    ])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.write_parquet(output_path)
    
    logger.info(f"Generated {len(numerical_cols)} text features")
    logger.info(f"Saved to {output_path}")
    
    # Log feature summary
    logger.info("Feature Summary:")
    for col in numerical_cols:
        stats = df_features.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).max().alias("max"),
            (pl.col(col) > 0).sum().alias("non_zero")
        ]).row(0)
        logger.info(f"  {col}: mean={stats[0]:.4f}, std={stats[1]:.4f}, max={stats[2]:.2f}, non_zero={stats[3]:,}")


if __name__ == "__main__":
    generate_text_features()


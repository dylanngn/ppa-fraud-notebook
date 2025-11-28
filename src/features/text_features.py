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

from pathlib import Path

import polars as pl

ARTIFACTS_DIR = Path("artifacts")
LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_text_features.parquet"


def _ensure_artifact(path: Path) -> bool:
    if not path.exists():
        print(f"[text-features] Skipping missing artifact: {path}")
        return False
    return True


def generate_text_features(output_path: Path = OUTPUT_PATH) -> None:
    """
    Generate simple text features from listing descriptions.
    
    These features are optimized for XGBoost and don't require
    expensive embedding generation.
    """
    print("[text-features] Loading listings...")
    
    if not _ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes missing: {LISTING_NODES}")
    
    # Load listings - we need description_text which should be in the processed data
    # But it might not be in nodes_listing.parquet. Let's check what we have.
    # Actually, description_text is extracted in process_listings() but might not
    # be saved to nodes_listing.parquet. We need to check the raw data or
    # add description_text to nodes_listing.
    
    # For now, let's try to load from nodes_listing and see if description_text exists
    # If not, we'll need to load from raw insertions or add it to ETL
    
    # Check if we have description_text in nodes_listing
    df_listings = pl.read_parquet(LISTING_NODES)
    
    # If description_text is not in nodes_listing, we need to get it from raw data
    # For now, let's assume we'll add description_text to nodes_listing in ETL
    # But as a fallback, we can compute features on empty strings
    
    if "description_text" not in df_listings.columns:
        print("[text-features] Warning: description_text not found in nodes_listing.parquet")
        print("[text-features] Using empty descriptions. Consider adding description_text to ETL.")
        df_listings = df_listings.with_columns(
            pl.lit("").alias("description_text")
        )
    
    print(f"[text-features] Processing {len(df_listings):,} listings...")
    
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
    
    print(f"[text-features] Generated {len(numerical_cols)} text features")
    print(f"[text-features] Saved to {output_path}")
    
    # Print feature summary
    print("\n[text-features] Feature Summary:")
    for col in numerical_cols:
        stats = df_features.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).max().alias("max"),
            (pl.col(col) > 0).sum().alias("non_zero")
        ]).row(0)
        print(f"  {col}: mean={stats[0]:.4f}, std={stats[1]:.4f}, max={stats[2]:.2f}, non_zero={stats[3]:,}")


if __name__ == "__main__":
    generate_text_features()


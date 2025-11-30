"""
Text feature definitions.
"""
import polars as pl
from datetime import datetime
from typing import Any
from src.features.registry import FeatureRegistry
from src.models.config.constants import TEXT_FEATURE_COLUMNS

@FeatureRegistry.register("text")
def compute_text_features(df: pl.DataFrame, cutoff_date: datetime, config: Any = None) -> pl.DataFrame:
    """
    Compute text features from listing descriptions.
    """
    if "description_text" not in df.columns:
        # Return empty features if column missing
        return df.with_columns([
            pl.lit(0).alias(col) for col in TEXT_FEATURE_COLUMNS
        ])
    
    text_features = df.select([
        "insertion_id",
        pl.col("description_text").fill_null("")
    ]).with_columns([
        # Character count
        pl.col("description_text").str.len_chars().alias("description_length"),
        
        # Word count
        pl.col("description_text").str.split(" ").list.len().alias("description_word_count"),
        
        # URL detection
        pl.col("description_text").str.contains(r"(?i)https?://").cast(pl.Int8).alias("description_has_url"),
        
        # Email detection
        pl.col("description_text").str.contains(r"@").cast(pl.Int8).alias("description_has_email"),
        
        # Phone detection
        pl.col("description_text").str.contains(r"\+?\d[\d\s\-\(\)]{7,}").cast(pl.Int8).alias("description_has_phone"),
        
        # Uppercase ratio
        (
            pl.col("description_text").str.count_matches(r"[A-Z]") / 
            pl.max_horizontal([
                pl.col("description_text").str.len_chars(),
                pl.lit(1)
            ])
        ).alias("description_caps_ratio"),
        
        # Exclamation count
        pl.col("description_text").str.count_matches(r"!").alias("description_exclamation_count"),
        
        # Question mark count
        pl.col("description_text").str.count_matches(r"\?").alias("description_question_count"),
        
        # All caps words count
        pl.col("description_text").str.count_matches(r"\b[A-Z]{2,}\b").alias("description_all_caps_words"),
        
        # Average word length
        (
            pl.col("description_text").str.len_chars() / 
            pl.max_horizontal([
                pl.col("description_text").str.split(" ").list.len(),
                pl.lit(1)
            ])
        ).alias("description_avg_word_length"),
    ])
    
    # Select only the feature columns
    text_cols = [c for c in text_features.columns if c in TEXT_FEATURE_COLUMNS]
    text_features = text_features.select(["insertion_id"] + text_cols)
    
    # Join back to original dataframe
    # Note: The registry contract expects returning the full dataframe with new features
    # But here we computed a subset. We should join it.
    
    # Fill nulls with 0
    numerical_cols = [col for col in text_features.columns if col != "insertion_id"]
    text_features = text_features.with_columns([
        pl.col(col).fill_null(0) for col in numerical_cols
    ])
    
    return df.join(text_features, on="insertion_id", how="left")

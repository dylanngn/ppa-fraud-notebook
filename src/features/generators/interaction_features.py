"""
Interaction Features for Fraud Detection

Experiment 10: Feature interaction discovery and engineering.
Based on SHAP/correlation analysis, certain feature combinations have synergistic effects.

Key findings from analysis:
- NEW ACCOUNT + HIGH EMAIL REUSE: 13.4x lift (21.6% fraud rate)
- NEW ACCOUNT + NO DIRECT PAYMENT: 28% fraud rate
- Fraudsters list smaller properties (rooms < 2.5, living_space < 60)
- Fraudsters are in larger graph components

Features engineered:
1. new_account_high_reuse: Account <30 days AND shared_email >3
2. new_account_no_direct: Account <14 days AND no direct payment
3. new_account_small_listing: New account with small property
4. large_component_new_account: Large component with new account
5. suspicious_combo_score: Weighted sum of fraud indicators
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

from src.features.utils import ensure_artifact, ARTIFACTS_DIR

LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"
USER_NODES = ARTIFACTS_DIR / "nodes_user.parquet"


def generate_interaction_features(
    output_path: Optional[Path] = None,
    graph_features_df: Optional[pl.DataFrame] = None,
    advanced_features_df: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """
    Generate interaction features based on discovered fraud patterns.
    
    These features capture synergistic combinations that are more predictive
    than individual features alone.
    
    Args:
        output_path: Optional path to save features. If None, returns DataFrame without saving.
        graph_features_df: Pre-computed graph features DataFrame.
                          Must have 'insertion_id' column.
        advanced_features_df: Pre-computed advanced features DataFrame.
                             Must have 'insertion_id' column.
    
    Returns:
        DataFrame with interaction features
        
    Raises:
        FileNotFoundError: If required artifact files are missing
        ValueError: If graph_features_df or advanced_features_df are not provided
    """
    logger.info("Loading data...")
    
    # Load listing nodes
    if not ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes missing: {LISTING_NODES}")
    
    listings = pl.read_parquet(LISTING_NODES)
    
    # Load user nodes for account age
    if not ensure_artifact(USER_NODES):
        raise FileNotFoundError(f"User nodes missing: {USER_NODES}")
    
    users = pl.read_parquet(USER_NODES)
    
    # Join to get account_created_at
    df = listings.join(users, on="user_id", how="left")
    
    logger.info(f"Loaded {df.shape[0]:,} listings")
    
    # Compute account_age_days
    df = df.with_columns(
        (pl.col("submission_at") - pl.col("account_created_at")).dt.total_days().alias("account_age_days")
    )
    
    # Require graph features to be provided
    if graph_features_df is None:
        raise ValueError("graph_features_df is required. Interaction features depend on graph features.")
    if "insertion_id" not in graph_features_df.columns:
        raise ValueError("graph_features_df must contain 'insertion_id' column")
    
    df = df.join(graph_features_df, on="insertion_id", how="left")
    logger.info("Joined graph features")
    
    # Require advanced features to be provided
    if advanced_features_df is None:
        raise ValueError("advanced_features_df is required. Interaction features depend on advanced features.")
    if "insertion_id" not in advanced_features_df.columns:
        raise ValueError("advanced_features_df must contain 'insertion_id' column")
    
    df = df.join(advanced_features_df, on="insertion_id", how="left")
    logger.info("Joined advanced features")
    
    # Fill nulls for graph features
    graph_cols = [
        "shared_contact_email_count", "shared_contact_phone_count",
        "listing_component_size", "is_isolated", "degree_total"
    ]
    for col in graph_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0))
    
    logger.info("Engineering interaction features...")
    
    # === INTERACTION FEATURES ===
    
    interaction_exprs = []
    
    # 1. NEW ACCOUNT + HIGH EMAIL REUSE (13.4x lift!)
    # Account <30 days AND shared_email >3
    if "shared_contact_email_count" in df.columns:
        interaction_exprs.append(
            ((pl.col("account_age_days") < 30) & (pl.col("shared_contact_email_count") > 3))
            .cast(pl.Int8).alias("new_account_high_email_reuse")
        )
        
        # Also with phone
        if "shared_contact_phone_count" in df.columns:
            interaction_exprs.append(
                ((pl.col("account_age_days") < 30) & (pl.col("shared_contact_phone_count") > 3))
                .cast(pl.Int8).alias("new_account_high_phone_reuse")
            )
            
            # Either email or phone reuse
            interaction_exprs.append(
                ((pl.col("account_age_days") < 30) & 
                 ((pl.col("shared_contact_email_count") > 3) | (pl.col("shared_contact_phone_count") > 3)))
                .cast(pl.Int8).alias("new_account_high_reuse_any")
            )
    
    # 2. NEW ACCOUNT + NO DIRECT PAYMENT (28% fraud rate)
    # Fraudsters avoid direct payment - invoice payment is suspicious for new accounts
    if "payment_type" in df.columns:
        interaction_exprs.append(
            ((pl.col("account_age_days") < 14) & (pl.col("payment_type") != "DIRECT"))
            .cast(pl.Int8).alias("new_account_invoice_payment")
        )
        
        # Very new account (< 7 days) + invoice
        interaction_exprs.append(
            ((pl.col("account_age_days") < 7) & (pl.col("payment_type") != "DIRECT"))
            .cast(pl.Int8).alias("very_new_account_invoice")
        )
    
    # 3. NEW ACCOUNT + SMALL LISTING
    # Fraudsters list smaller properties
    if "living_space" in df.columns and "rooms" in df.columns:
        # Small listing: living_space < 60 OR rooms < 2.5
        interaction_exprs.append(
            ((pl.col("living_space").fill_null(0) < 60) | (pl.col("rooms").fill_null(0) < 2.5))
            .cast(pl.Int8).alias("is_small_listing")
        )
        
        interaction_exprs.append(
            ((pl.col("account_age_days") < 30) & 
             ((pl.col("living_space").fill_null(0) < 60) | (pl.col("rooms").fill_null(0) < 2.5)))
            .cast(pl.Int8).alias("new_account_small_listing")
        )
    
    # 4. LARGE COMPONENT + NEW ACCOUNT
    # Fraudsters tend to be in larger graph components
    if "listing_component_size" in df.columns:
        # Component size > 20000 (large)
        interaction_exprs.append(
            (pl.col("listing_component_size") > 20000)
            .cast(pl.Int8).alias("in_large_component")
        )
        
        interaction_exprs.append(
            ((pl.col("account_age_days") < 30) & (pl.col("listing_component_size") > 20000))
            .cast(pl.Int8).alias("new_account_large_component")
        )
    
    # 5. ISOLATED + NEW ACCOUNT
    # New accounts that are isolated (no graph connections)
    if "is_isolated" in df.columns:
        interaction_exprs.append(
            ((pl.col("account_age_days") < 14) & (pl.col("is_isolated") == 1))
            .cast(pl.Int8).alias("new_account_isolated")
        )
    
    # 6. CONTINUOUS INTERACTION SCORES
    
    # Account age risk score (higher for newer accounts)
    # Uses sigmoid-like decay: 1 / (1 + age/30)
    interaction_exprs.append(
        (1.0 / (1.0 + pl.col("account_age_days").clip(0, 365) / 30.0))
        .alias("account_age_risk_score")
    )
    
    # Reuse intensity (normalized)
    if "shared_contact_email_count" in df.columns:
        interaction_exprs.append(
            (pl.col("shared_contact_email_count") / 
             (pl.col("shared_contact_email_count") + 10.0))  # Soft normalization
            .alias("email_reuse_intensity")
        )
    
    # Component risk score
    if "listing_component_size" in df.columns:
        interaction_exprs.append(
            (pl.col("listing_component_size") / 
             (pl.col("listing_component_size") + 10000.0))
            .alias("component_risk_score")
        )
    
    # 7. SUSPICIOUS COMBO SCORE
    # Weighted combination of all risk indicators
    # Weight based on observed fraud rate lifts
    score_parts = []
    
    # Account age contribution (high weight - most predictive)
    score_parts.append(
        pl.when(pl.col("account_age_days") < 7)
        .then(pl.lit(3.0))  # Very new
        .when(pl.col("account_age_days") < 14)
        .then(pl.lit(2.0))  # New
        .when(pl.col("account_age_days") < 30)
        .then(pl.lit(1.0))  # Somewhat new
        .otherwise(pl.lit(0.0))
    )
    
    # No direct payment contribution
    if "payment_type" in df.columns:
        score_parts.append(
            pl.when(pl.col("payment_type") != "DIRECT")
            .then(pl.lit(2.0))
            .otherwise(pl.lit(0.0))
        )
    
    # High reuse contribution
    if "shared_contact_email_count" in df.columns:
        score_parts.append(
            pl.when(pl.col("shared_contact_email_count") > 5)
            .then(pl.lit(2.0))
            .when(pl.col("shared_contact_email_count") > 3)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
        )
    
    # Small listing contribution
    if "living_space" in df.columns:
        score_parts.append(
            pl.when(pl.col("living_space").fill_null(100) < 50)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
        )
    
    # Large component contribution
    if "listing_component_size" in df.columns:
        score_parts.append(
            pl.when(pl.col("listing_component_size") > 30000)
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
        )
    
    # Sum all parts for suspicious combo score
    if score_parts:
        combo_expr = score_parts[0]
        for part in score_parts[1:]:
            combo_expr = combo_expr + part
        interaction_exprs.append(combo_expr.alias("suspicious_combo_score"))
    
    # Apply all interaction expressions
    if interaction_exprs:
        df = df.with_columns(interaction_exprs)
    
    # Select only the new features + insertion_id
    feature_cols = [
        "insertion_id",
        # Binary interactions
        "new_account_high_email_reuse",
        "new_account_high_phone_reuse", 
        "new_account_high_reuse_any",
        "new_account_invoice_payment",
        "very_new_account_invoice",
        "is_small_listing",
        "new_account_small_listing",
        "in_large_component",
        "new_account_large_component",
        "new_account_isolated",
        # Continuous scores
        "account_age_risk_score",
        "email_reuse_intensity",
        "component_risk_score",
        "suspicious_combo_score",
    ]
    
    # Only select columns that exist
    existing_cols = [c for c in feature_cols if c in df.columns]
    result = df.select(existing_cols)
    
    # Fill any remaining nulls with 0
    for col in result.columns:
        if col != "insertion_id":
            result = result.with_columns(pl.col(col).fill_null(0))
    
    # Save to disk if output_path is provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info(f"Saved to {output_path}")
    
    feature_count = len(existing_cols) - 1  # Exclude insertion_id
    logger.info(f"Generated {feature_count} interaction features")
    
    return result


if __name__ == "__main__":
    generate_interaction_features()


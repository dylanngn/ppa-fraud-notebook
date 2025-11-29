"""
Time-Weighted Graph Features for Fraud Detection

Experiment 9: Recency weighting for graph connections.
Hypothesis: Recent connections should have higher weight than old ones.

Features computed:
- recent_email_count: Weighted count (last 30 days * 2.0 + 30-90 days * 1.0)
- email_velocity_7d: Connections per day over last 7 days
- email_acceleration: Velocity change (last 7 vs prev 7 days)
- is_email_burst: Sudden spike (>3 in 7 days, 0 in prev 90 days)
- is_dormant_reactivation: Reactivation after dormancy
- Same metrics for phone connections

OPTIMIZATION: Uses chunked processing to avoid memory issues with large graphs.
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import timedelta, datetime
import math

import polars as pl
import numpy as np

from src.features.utils import (
    _groupby,
    ensure_artifact,
    safe_load_edges,
    filter_edges_by_time,
    load_listings_with_timestamps,
    ARTIFACTS_DIR,
)

EDGE_LISTING_CONTACT_EMAIL = ARTIFACTS_DIR / "edges_listing_contact_email.parquet"
EDGE_LISTING_CONTACT_PHONE = ARTIFACTS_DIR / "edges_listing_contact_phone.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_time_weighted_features.parquet"

# Chunk size for processing
CHUNK_SIZE = 10000


def _compute_time_weighted_features_chunked(
    listings_df: pl.DataFrame,
    edges: Optional[pl.DataFrame],
    prefix: str
) -> Optional[pl.DataFrame]:
    """
    Compute time-weighted features efficiently using chunked processing.
    
    Strategy:
    1. For each target (email/phone), get all listings that share it with timestamps
    2. Process targets in chunks to avoid memory explosion
    3. For each listing, aggregate time-based metrics from its connected listings
    """
    if edges is None:
        return None
    
    # Normalize column name: handle both 'source' and 'listing_id' as the listing column
    if "source" in edges.columns and "listing_id" not in edges.columns:
        edges = edges.rename({"source": "listing_id"})
    
    print(f"[time-weighted] Processing {prefix} features...")
    
    # Join edges with listing timestamps
    edges_with_time = edges.join(
        listings_df.rename({"insertion_id": "listing_id"}),
        on="listing_id",
        how="left"
    )
    
    # Get all unique targets
    unique_targets = edges_with_time.select("target").unique()["target"].to_list()
    n_targets = len(unique_targets)
    print(f"[time-weighted] Processing {n_targets:,} unique {prefix}s in chunks...")
    
    # Initialize result accumulators
    all_results = []
    
    # Process in chunks
    n_chunks = math.ceil(n_targets / CHUNK_SIZE)
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min((chunk_idx + 1) * CHUNK_SIZE, n_targets)
        chunk_targets = unique_targets[start_idx:end_idx]
        
        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            print(f"[time-weighted]   Chunk {chunk_idx + 1}/{n_chunks} ({end_idx:,}/{n_targets:,} {prefix}s)")
        
        # Filter edges to this chunk of targets
        chunk_edges = edges_with_time.filter(pl.col("target").is_in(chunk_targets))
        
        # For each target, compute features for connected listings
        chunk_result = _process_target_chunk(chunk_edges, prefix)
        
        if chunk_result is not None:
            all_results.append(chunk_result)
    
    if not all_results:
        return None
    
    # Combine all chunks and aggregate per listing
    combined = pl.concat(all_results)
    
    # Aggregate across all targets for each listing
    final_features = _aggregate_listing_features(combined, prefix)
    
    return final_features


def _process_target_chunk(
    chunk_edges: pl.DataFrame,
    prefix: str
) -> Optional[pl.DataFrame]:
    """
    Process a chunk of targets and compute time-weighted features.
    
    For each listing, we look at OTHER listings that share the same contact.
    We compute time differences and aggregate.
    """
    if chunk_edges.is_empty():
        return None
    
    # Group by target to get all listings per target with their times
    target_groups = _groupby(chunk_edges, "target").agg([
        pl.col("listing_id"),
        pl.col("submission_at")
    ])
    
    results = []
    
    for row in target_groups.iter_rows(named=True):
        listing_ids = row["listing_id"]
        submission_times = row["submission_at"]
        
        n_listings = len(listing_ids)
        if n_listings < 2:
            # No connections possible with only one listing
            continue
        
        # For each listing, compute features based on other listings with same target
        for i, (lid, ltime) in enumerate(zip(listing_ids, submission_times)):
            if ltime is None:
                continue
            
            # Compute time differences to all OTHER listings
            counts = {
                "in_7d": 0,
                "in_prev_7d": 0,
                "in_30d": 0,
                "in_30_90d": 0,
                "in_prev_90d": 0,
                "in_prev_180d": 0,
                "total_historical": 0,
                "recency_sum": 0.0,
                "min_days": float('inf'),
                "max_days": 0.0,
            }
            
            for j, (other_lid, other_time) in enumerate(zip(listing_ids, submission_times)):
                if i == j or other_time is None:
                    continue
                
                # Days difference: positive = other is older
                days_diff = (ltime - other_time).total_seconds() / 86400.0
                
                if days_diff < 0:
                    # Other listing is in the future, skip
                    continue
                
                counts["total_historical"] += 1
                counts["min_days"] = min(counts["min_days"], days_diff)
                counts["max_days"] = max(counts["max_days"], days_diff)
                counts["recency_sum"] += math.exp(-days_diff / 30.0)
                
                if days_diff <= 7:
                    counts["in_7d"] += 1
                elif days_diff <= 14:
                    counts["in_prev_7d"] += 1
                
                if days_diff <= 30:
                    counts["in_30d"] += 1
                elif days_diff <= 90:
                    counts["in_30_90d"] += 1
                
                if 7 < days_diff <= 97:
                    counts["in_prev_90d"] += 1
                
                if 30 < days_diff <= 210:
                    counts["in_prev_180d"] += 1
            
            if counts["total_historical"] > 0:
                results.append({
                    "listing_id": lid,
                    f"{prefix}_target_count_7d": counts["in_7d"],
                    f"{prefix}_target_count_prev_7d": counts["in_prev_7d"],
                    f"{prefix}_target_count_30d": counts["in_30d"],
                    f"{prefix}_target_count_30_90d": counts["in_30_90d"],
                    f"{prefix}_target_count_prev_90d": counts["in_prev_90d"],
                    f"{prefix}_target_count_prev_180d": counts["in_prev_180d"],
                    f"{prefix}_target_total_historical": counts["total_historical"],
                    f"{prefix}_target_recency_sum": counts["recency_sum"],
                    f"{prefix}_target_min_days": counts["min_days"] if counts["min_days"] != float('inf') else 0,
                    f"{prefix}_target_max_days": counts["max_days"],
                })
    
    if not results:
        return None
    
    return pl.DataFrame(results)


def _aggregate_listing_features(
    df: pl.DataFrame,
    prefix: str
) -> pl.DataFrame:
    """
    Aggregate per-target features into per-listing features.
    
    A listing may connect to multiple emails/phones, so we sum/max across targets.
    """
    agg_exprs = [
        # Sum across all targets
        pl.col(f"{prefix}_target_count_7d").sum().alias(f"{prefix}_count_7d"),
        pl.col(f"{prefix}_target_count_prev_7d").sum().alias(f"{prefix}_count_prev_7d"),
        pl.col(f"{prefix}_target_count_30d").sum().alias(f"{prefix}_count_30d"),
        pl.col(f"{prefix}_target_count_30_90d").sum().alias(f"{prefix}_count_30_90d"),
        pl.col(f"{prefix}_target_count_prev_90d").sum().alias(f"{prefix}_count_prev_90d"),
        pl.col(f"{prefix}_target_count_prev_180d").sum().alias(f"{prefix}_count_prev_180d"),
        pl.col(f"{prefix}_target_total_historical").sum().alias(f"{prefix}_total_historical"),
        pl.col(f"{prefix}_target_recency_sum").sum().alias(f"{prefix}_recency_weighted"),
        
        # Min/Max across all targets
        pl.col(f"{prefix}_target_min_days").min().alias(f"{prefix}_min_days_ago"),
        pl.col(f"{prefix}_target_max_days").max().alias(f"{prefix}_max_days_ago"),
    ]
    
    agg_df = _groupby(df, "listing_id").agg(agg_exprs)
    
    # Compute derived features
    agg_df = agg_df.with_columns([
        # Recent weighted: last 30 days * 2 + 30-90 days * 1
        (pl.col(f"{prefix}_count_30d") * 2.0 + pl.col(f"{prefix}_count_30_90d")).alias(
            f"{prefix}_recent_weighted"
        ),
        
        # Velocity: connections per day in last 7 days
        (pl.col(f"{prefix}_count_7d") / 7.0).alias(f"{prefix}_velocity_7d"),
        
        # Acceleration: velocity change
        ((pl.col(f"{prefix}_count_7d") - pl.col(f"{prefix}_count_prev_7d")) / 7.0).alias(
            f"{prefix}_acceleration"
        ),
        
        # Burst detection: high recent activity with no prior history
        (
            (pl.col(f"{prefix}_count_7d") > 3) & 
            (pl.col(f"{prefix}_count_prev_90d") == 0)
        ).cast(pl.Int8).alias(f"{prefix}_is_burst"),
        
        # Dormant reactivation: activity after long silence
        (
            (pl.col(f"{prefix}_count_prev_180d") == 0) & 
            (pl.col(f"{prefix}_count_30d") > 2)
        ).cast(pl.Int8).alias(f"{prefix}_is_dormant_reactivation"),
        
        # Time spread: how spread out are connections
        (pl.col(f"{prefix}_max_days_ago") - pl.col(f"{prefix}_min_days_ago")).alias(
            f"{prefix}_time_spread"
        ),
    ])
    
    return agg_df.rename({"listing_id": "insertion_id"})


def _compute_combined_features(
    email_features: Optional[pl.DataFrame],
    phone_features: Optional[pl.DataFrame]
) -> Optional[pl.DataFrame]:
    """Compute combined email+phone time-weighted features."""
    if email_features is None and phone_features is None:
        return None
    
    # Start with whichever is available
    if email_features is not None and phone_features is not None:
        combined = email_features.join(phone_features, on="insertion_id", how="outer", suffix="_phone")
        # Remove any duplicate columns from join
        combined = combined.drop([c for c in combined.columns if c.endswith("_phone") and c.replace("_phone", "") in combined.columns])
    elif email_features is not None:
        combined = email_features
    else:
        combined = phone_features
    
    # Fill nulls with 0 for numerical columns
    numerical_cols = [c for c in combined.columns if c != "insertion_id"]
    combined = combined.with_columns([
        pl.col(col).fill_null(0) for col in numerical_cols
    ])
    
    # Add combined features
    combined_features = []
    
    # Combined velocity
    if "email_velocity_7d" in combined.columns and "phone_velocity_7d" in combined.columns:
        combined_features.append(
            (pl.col("email_velocity_7d") + pl.col("phone_velocity_7d")).alias("combined_velocity_7d")
        )
    
    # Combined acceleration
    if "email_acceleration" in combined.columns and "phone_acceleration" in combined.columns:
        combined_features.append(
            (pl.col("email_acceleration") + pl.col("phone_acceleration")).alias("combined_acceleration")
        )
    
    # Any burst (email OR phone)
    if "email_is_burst" in combined.columns and "phone_is_burst" in combined.columns:
        combined_features.append(
            ((pl.col("email_is_burst") == 1) | (pl.col("phone_is_burst") == 1))
            .cast(pl.Int8).alias("any_burst")
        )
    
    # Any dormant reactivation
    if "email_is_dormant_reactivation" in combined.columns and "phone_is_dormant_reactivation" in combined.columns:
        combined_features.append(
            ((pl.col("email_is_dormant_reactivation") == 1) | (pl.col("phone_is_dormant_reactivation") == 1))
            .cast(pl.Int8).alias("any_dormant_reactivation")
        )
    
    # Combined recency weighted
    if "email_recency_weighted" in combined.columns and "phone_recency_weighted" in combined.columns:
        combined_features.append(
            (pl.col("email_recency_weighted") + pl.col("phone_recency_weighted")).alias("combined_recency_weighted")
        )
    
    if combined_features:
        combined = combined.with_columns(combined_features)
    
    return combined


def generate_time_weighted_features(
    output_path: Optional[Path] = None,
    cutoff_date: Optional[datetime] = None
) -> pl.DataFrame:
    """
    Generate time-weighted graph features for all listings.
    
    Args:
        output_path: Optional path to save features. If None, returns DataFrame without saving.
        cutoff_date: If provided, only use edges from listings before this date.
                     This prevents temporal leakage. If None, uses all edges.
    
    Returns:
        DataFrame with time-weighted features
        
    Raises:
        FileNotFoundError: If required artifact files are missing
        ValueError: If no features can be generated
    """
    print("[time-weighted] Loading listings with timestamps...")
    listings_df = load_listings_with_timestamps()
    print(f"[time-weighted] Loaded {listings_df.shape[0]:,} listings")
    
    # Load edges
    email_edges = safe_load_edges(EDGE_LISTING_CONTACT_EMAIL)
    phone_edges = safe_load_edges(EDGE_LISTING_CONTACT_PHONE)
    
    # Apply temporal filtering if cutoff_date is provided
    if cutoff_date is not None:
        print(f"[time-weighted] Filtering edges by cutoff_date: {cutoff_date}")
        listings_with_time = load_listings_with_timestamps()
        
        if email_edges is not None:
            email_edges = filter_edges_by_time(
                email_edges,
                listings_with_time,
                cutoff_date,
                edge_type="listing_to_target"
            )
        
        if phone_edges is not None:
            phone_edges = filter_edges_by_time(
                phone_edges,
                listings_with_time,
                cutoff_date,
                edge_type="listing_to_target"
            )
    
    if email_edges is not None:
        print(f"[time-weighted] Email edges: {email_edges.shape[0]:,}")
    if phone_edges is not None:
        print(f"[time-weighted] Phone edges: {phone_edges.shape[0]:,}")
    
    # Compute time-weighted features for each edge type
    email_features = _compute_time_weighted_features_chunked(listings_df, email_edges, "email")
    phone_features = _compute_time_weighted_features_chunked(listings_df, phone_edges, "phone")
    
    # Combine features
    print("[time-weighted] Combining features...")
    combined = _compute_combined_features(email_features, phone_features)
    
    if combined is None:
        raise ValueError("No time-weighted features were generated. Check that edge files exist and contain data.")
    
    # Ensure all listings are present (left join with listing IDs)
    all_listings = listings_df.select("insertion_id")
    combined = all_listings.join(combined, on="insertion_id", how="left")
    
    # Fill nulls for listings with no connections
    numerical_cols = [c for c in combined.columns if c != "insertion_id"]
    combined = combined.with_columns([
        pl.col(col).fill_null(0) for col in numerical_cols
    ])
    
    # Save to disk if output_path is provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)
        print(f"[time-weighted] Saved to {output_path}")
    
    print(f"[time-weighted] Generated {len(numerical_cols)} features for {combined.shape[0]:,} listings")
    
    # Print feature summary (only if saving to disk)
    if output_path is not None:
        print("\n[time-weighted] Feature Summary:")
        for col in numerical_cols[:12]:
            stats = combined.select([
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).max().alias("max"),
                (pl.col(col) > 0).sum().alias("non_zero")
            ]).row(0)
            print(f"  {col}: mean={stats[0]:.4f}, std={stats[1]:.4f}, max={stats[2]:.2f}, non_zero={stats[3]}")
    
    return combined


if __name__ == "__main__":
    generate_time_weighted_features()

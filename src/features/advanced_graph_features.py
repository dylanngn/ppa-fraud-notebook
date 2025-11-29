from pathlib import Path
from typing import List, Optional
from datetime import datetime

import polars as pl

from src.features.utils import (
    _groupby,
    ensure_artifact,
    load_listing_ids,
    safe_load_edges,
    filter_edges_by_time,
    load_listings_with_timestamps,
    ARTIFACTS_DIR,
)

EDGE_LISTING_CONTACT_EMAIL = ARTIFACTS_DIR / "edges_listing_contact_email.parquet"
EDGE_LISTING_CONTACT_PHONE = ARTIFACTS_DIR / "edges_listing_contact_phone.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_advanced_features.parquet"


def _calculate_isolation_scores(listing_ids: List[int], 
                                email_edges: Optional[pl.DataFrame], 
                                phone_edges: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """
    Calculates isolation metrics:
    - degree_total: Total connections (email + phone)
    - is_isolated: Boolean (degree == 0)
    - unique_identifier_count: Number of unique emails + phones
    """
    if email_edges is None and phone_edges is None:
        return None

    # Normalize column names: handle both 'source' and 'listing_id' as the listing column
    if email_edges is not None:
        if "source" in email_edges.columns and "listing_id" not in email_edges.columns:
            email_edges = email_edges.rename({"source": "listing_id"})
    
    if phone_edges is not None:
        if "source" in phone_edges.columns and "listing_id" not in phone_edges.columns:
            phone_edges = phone_edges.rename({"source": "listing_id"})

    # Count degrees
    email_counts = pl.DataFrame({"listing_id": listing_ids})
    if email_edges is not None:
        counts = _groupby(email_edges, "listing_id").agg(pl.len().alias("email_count"))
        email_counts = email_counts.join(counts, on="listing_id", how="left").fill_null(0)
    else:
        email_counts = email_counts.with_columns(pl.lit(0).alias("email_count"))

    phone_counts = pl.DataFrame({"listing_id": listing_ids})
    if phone_edges is not None:
        counts = _groupby(phone_edges, "listing_id").agg(pl.len().alias("phone_count"))
        phone_counts = phone_counts.join(counts, on="listing_id", how="left").fill_null(0)
    else:
        phone_counts = phone_counts.with_columns(pl.lit(0).alias("phone_count"))
        
    # Join
    df = email_counts.join(phone_counts, on="listing_id", how="left")
    
    # Calculate metrics
    df = df.with_columns([
        (pl.col("email_count") + pl.col("phone_count")).alias("degree_total"),
        ((pl.col("email_count") + pl.col("phone_count")) == 0).cast(pl.Int8).alias("is_isolated"),
        (pl.col("email_count") + pl.col("phone_count")).alias("unique_identifier_count") # Proxy for now
    ])
    
    return df.rename({"listing_id": "insertion_id"}).select([
        "insertion_id", "degree_total", "is_isolated", "unique_identifier_count"
    ])


def _calculate_clustering_coefficient(listing_ids: List[int],
                                      email_edges: Optional[pl.DataFrame],
                                      phone_edges: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """
    Calculates bipartite clustering coefficient for listings.
    cc_bipartite(u) = number of pairs of neighbors of u that share a neighbor other than u
                      -------------------------------------------------------------------
                      d(u) * (d(u) - 1)
                      
    This is computationally expensive, so we'll use a simplified "neighbor overlap" metric.
    neighbor_overlap = sum(degree(v) - 1) for v in neighbors(u)
    """
    if email_edges is None and phone_edges is None:
        return None
    
    # Normalize column names: handle both 'source' and 'listing_id' as the listing column
    if email_edges is not None:
        if "source" in email_edges.columns and "listing_id" not in email_edges.columns:
            email_edges = email_edges.rename({"source": "listing_id"})
    
    if phone_edges is not None:
        if "source" in phone_edges.columns and "listing_id" not in phone_edges.columns:
            phone_edges = phone_edges.rename({"source": "listing_id"})
        
    # Combine edges with type
    edges_list = []
    if email_edges is not None:
        edges_list.append(email_edges.with_columns(pl.lit("email").alias("type")))
    if phone_edges is not None:
        edges_list.append(phone_edges.with_columns(pl.lit("phone").alias("type")))
        
    if not edges_list:
        return None
        
    all_edges = pl.concat(edges_list)
    
    # Calculate target degrees (how many listings share this email/phone)
    target_degrees = _groupby(all_edges, ["target", "type"]).agg(pl.len().alias("target_degree"))
    
    # Join back to edges
    enriched_edges = all_edges.join(target_degrees, on=["target", "type"], how="left")
    
    # Aggregate per listing
    # neighbor_overlap_score: Sum of (degree - 1) of all neighbors. 
    # High score = neighbors are hubs (connected to many other listings).
    # Low score = neighbors are unique to this listing.
    agg = _groupby(enriched_edges, "listing_id").agg([
        (pl.col("target_degree") - 1).sum().alias("neighbor_overlap_score"),
        (pl.col("target_degree") - 1).mean().alias("avg_neighbor_degree")
    ])
    
    return agg.rename({"listing_id": "insertion_id"})


def generate_advanced_features(
    output_path: Optional[Path] = None,
    cutoff_date: Optional[datetime] = None
) -> pl.DataFrame:
    """
    Generate advanced graph features.
    
    Args:
        output_path: Optional path to save features. If None, returns DataFrame without saving.
        cutoff_date: If provided, only use edges from listings before this date.
                     This prevents temporal leakage. If None, uses all edges.
    
    Returns:
        DataFrame with advanced features
        
    Raises:
        FileNotFoundError: If required artifact files are missing
        ValueError: If no features can be generated
    """
    print("[advanced-features] Loading data...")
    listings_df = load_listing_ids()
    listing_ids = listings_df["insertion_id"].to_list()

    contact_email_edges = safe_load_edges(EDGE_LISTING_CONTACT_EMAIL)
    contact_phone_edges = safe_load_edges(EDGE_LISTING_CONTACT_PHONE)
    
    # Apply temporal filtering if cutoff_date is provided
    if cutoff_date is not None:
        print(f"[advanced-features] Filtering edges by cutoff_date: {cutoff_date}")
        listings_with_time = load_listings_with_timestamps()
        
        if contact_email_edges is not None:
            contact_email_edges = filter_edges_by_time(
                contact_email_edges,
                listings_with_time,
                cutoff_date,
                edge_type="listing_to_target"
            )
        
        if contact_phone_edges is not None:
            contact_phone_edges = filter_edges_by_time(
                contact_phone_edges,
                listings_with_time,
                cutoff_date,
                edge_type="listing_to_target"
            )

    feature_frames = [listings_df]

    # 1. Isolation Scores
    print("[advanced-features] Calculating isolation scores...")
    isolation = _calculate_isolation_scores(listing_ids, contact_email_edges, contact_phone_edges)
    if isolation is not None:
        feature_frames.append(isolation)

    # 2. Clustering / Overlap
    print("[advanced-features] Calculating clustering/overlap metrics...")
    clustering = _calculate_clustering_coefficient(listing_ids, contact_email_edges, contact_phone_edges)
    if clustering is not None:
        feature_frames.append(clustering)

    if len(feature_frames) == 1:
        raise ValueError("No advanced features were generated. Check that edge files exist and contain data.")

    print("[advanced-features] Joining features...")
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.join(frame, on="insertion_id", how="left")

    numerical_cols = [col for col in features.columns if col != "insertion_id"]
    if numerical_cols:
        features = features.with_columns([
            pl.col(col).fill_null(0) for col in numerical_cols
        ])

    # Save to disk if output_path is provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.write_parquet(output_path)
        print(f"[advanced-features] Saved advanced graph features to {output_path}")
    
    return features


if __name__ == "__main__":
    generate_advanced_features()

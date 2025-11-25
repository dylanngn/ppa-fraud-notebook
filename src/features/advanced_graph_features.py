from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"
EDGE_LISTING_CONTACT_EMAIL = ARTIFACTS_DIR / "edges_listing_contact_email.parquet"
EDGE_LISTING_CONTACT_PHONE = ARTIFACTS_DIR / "edges_listing_contact_phone.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_advanced_features.parquet"


def _groupby(df: pl.DataFrame, *args, **kwargs):
    method = getattr(df, "groupby", None)
    if method is None:
        method = getattr(df, "group_by", None)
    if method is None:
        raise AttributeError("DataFrame has no groupby/group_by method. Please update Polars.")
    return method(*args, **kwargs)


def _ensure_artifact(path: Path) -> bool:
    if not path.exists():
        print(f"[advanced-features] Skipping missing artifact: {path}")
        return False
    return True


def _load_listing_ids() -> pl.DataFrame:
    if not _ensure_artifact(LISTING_NODES):
        raise FileNotFoundError(f"Listing nodes file missing: {LISTING_NODES}")
    df = pl.read_parquet(LISTING_NODES).select("insertion_id")
    return df


def _safe_edges(path: Path) -> Optional[pl.DataFrame]:
    if not _ensure_artifact(path):
        return None
    df = pl.read_parquet(path)
    if "source" not in df.columns or "target" not in df.columns:
        return None
    df = df.drop_nulls(["source", "target"])
    if df.is_empty():
        return None
    return df.select([pl.col("source").alias("listing_id"), pl.col("target")])


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


def generate_advanced_features(output_path: Path = OUTPUT_PATH) -> None:
    print("[advanced-features] Loading data...")
    listings_df = _load_listing_ids()
    listing_ids = listings_df["insertion_id"].to_list()

    contact_email_edges = _safe_edges(EDGE_LISTING_CONTACT_EMAIL)
    contact_phone_edges = _safe_edges(EDGE_LISTING_CONTACT_PHONE)

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
        print("[advanced-features] No advanced features were generated.")
        return

    print("[advanced-features] Joining features...")
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.join(frame, on="insertion_id", how="left")

    numerical_cols = [col for col in features.columns if col != "insertion_id"]
    if numerical_cols:
        features = features.with_columns([
            pl.col(col).fill_null(0) for col in numerical_cols
        ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    print(f"[advanced-features] Saved advanced graph features to {output_path}")


if __name__ == "__main__":
    generate_advanced_features()

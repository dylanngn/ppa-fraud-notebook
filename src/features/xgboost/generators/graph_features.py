"""
Graph feature generators for fraud detection.

This module computes all graph-derived features from edge files:
- Contact-based features (shared emails, phones)
- User-level features (listing counts, IP sharing)
- Network structure (PageRank, connected components)
- Isolation and clustering metrics
"""
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)

from src.features.xgboost.utils import (
    _groupby,
    ensure_artifact,
    load_listing_ids,
    safe_load_edges,
    filter_edges_by_time,
    load_listings_with_timestamps,
    ARTIFACTS_DIR,
)

EDGE_LISTING_CONTACT_EMAIL = ARTIFACTS_DIR / "edges_listing_contact_email.parquet"
EDGE_LISTING_CONTACT_PHONE = ARTIFACTS_DIR / "edges_listing_phone.parquet"
EDGE_USER_POSTS_LISTING = ARTIFACTS_DIR / "edges_user_posts_listing.parquet"
EDGE_USER_IP = ARTIFACTS_DIR / "edges_user_uses_ip.parquet"


def _clip_zero(expr: pl.Expr) -> pl.Expr:
    """Clamp expression to zero without relying on version-specific APIs."""
    return pl.when(expr > 0).then(expr).otherwise(pl.lit(0))


# =============================================================================
# BASIC GRAPH FEATURES
# =============================================================================

def _contact_edge_features(edges: Optional[pl.DataFrame], prefix: str) -> Optional[pl.DataFrame]:
    """
    Compute contact-based edge features (email/phone).
    
    Features:
    - {prefix}_count: Number of contacts for this listing
    - shared_{prefix}_count: Sum of shared contacts
    - max_shared_{prefix}: Maximum sharing for any contact
    """
    if edges is None:
        return None

    # Normalize column name: handle both 'source' and 'listing_id' as the listing column
    if "source" in edges.columns and "listing_id" not in edges.columns:
        edges = edges.rename({"source": "listing_id"})

    degree_col = f"{prefix}_count"
    shared_sum_col = f"shared_{prefix}_count"
    shared_max_col = f"max_shared_{prefix}"

    edge_degrees = (
        _groupby(edges, "target")
        .agg(pl.len().alias("target_degree"))
    )

    enriched = edges.join(edge_degrees, on="target", how="left")

    agg = (
        _groupby(enriched, "listing_id")
        .agg([
            pl.len().alias(degree_col),
            _clip_zero(pl.col("target_degree") - 1).sum().alias(shared_sum_col),
            _clip_zero(pl.col("target_degree") - 1).max().alias(shared_max_col),
        ])
    )

    return agg.rename({"listing_id": "insertion_id"})


def _user_features(cutoff_date: Optional[datetime] = None) -> Optional[pl.DataFrame]:
    """
    Compute user-level features.
    
    Features:
    - user_listing_count: Number of listings by this user
    - user_unique_ip_count: Number of unique IPs used by this user
    - shared_ip_user_count: Sum of users sharing IPs with this user
    - max_shared_ip_users: Maximum sharing for any IP
    """
    if not (ensure_artifact(EDGE_USER_POSTS_LISTING) and ensure_artifact(EDGE_USER_IP)):
        return None

    posts = pl.read_parquet(EDGE_USER_POSTS_LISTING).drop_nulls(["source", "target"])
    if posts.is_empty():
        return None

    # Apply temporal filtering if cutoff_date is provided
    if cutoff_date is not None:
        listings_with_time = load_listings_with_timestamps()
        posts = filter_edges_by_time(
            posts,
            listings_with_time,
            cutoff_date,
            edge_type="target_to_listing"  # target is listing_id
        )
        if posts.is_empty():
            return None

    posts = posts.rename({"source": "user_id", "target": "insertion_id"})
    listing_to_user = posts.select(["insertion_id", "user_id"])

    user_listing_counts = (
        _groupby(posts, "user_id")
        .agg(pl.len().alias("user_listing_count"))
    )

    user_ip = pl.read_parquet(EDGE_USER_IP).drop_nulls(["source", "target"])
    
    # Apply temporal filtering to user_ip edges if cutoff_date is provided
    if cutoff_date is not None:
        filtered_user_ids = posts.select("user_id").unique()
        filtered_user_ids = filtered_user_ids.rename({"user_id": "source"})
        user_ip = user_ip.join(filtered_user_ids, on="source", how="inner")
    
    user_ip = user_ip.rename({"source": "user_id", "target": "ip"})

    ip_user_counts = (
        _groupby(user_ip, "ip")
        .agg(pl.len().alias("ip_user_count"))
    )

    user_ip = user_ip.join(ip_user_counts, on="ip", how="left")

    user_ip_stats = (
        _groupby(user_ip, "user_id")
        .agg([
            pl.len().alias("user_unique_ip_count"),
            _clip_zero(pl.col("ip_user_count") - 1).sum().alias("shared_ip_user_count"),
            _clip_zero(pl.col("ip_user_count") - 1).max().alias("max_shared_ip_users"),
        ])
    )

    user_stats = user_listing_counts.join(user_ip_stats, on="user_id", how="left")

    listing_user_features = listing_to_user.join(user_stats, on="user_id", how="left").select([
        "insertion_id",
        "user_listing_count",
        "user_unique_ip_count",
        "shared_ip_user_count",
        "max_shared_ip_users",
    ])

    return listing_user_features


class DisjointSet:
    """Union-Find data structure for connected components."""
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.sz = [1] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.sz[ra] += self.sz[rb]

    def component_size(self, x: int) -> int:
        root = self.find(x)
        return self.sz[root]


def _component_sizes(listing_ids: List[int],
                     email_edges: Optional[pl.DataFrame],
                     phone_edges: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """
    Compute connected component sizes for listings.
    
    Features:
    - listing_component_size: Size of connected component this listing belongs to
    """
    if not listing_ids:
        return None

    index_map = {lid: idx for idx, lid in enumerate(listing_ids)}
    dsu = DisjointSet(len(listing_ids))

    def union_from_edges(edges: Optional[pl.DataFrame]) -> None:
        if edges is None:
            return
        # Normalize column name: handle both 'source' and 'listing_id' as the listing column
        if "source" in edges.columns and "listing_id" not in edges.columns:
            edges = edges.rename({"source": "listing_id"})
        grouped = _groupby(edges, "target").agg(pl.col("listing_id"))
        for target_listings in grouped["listing_id"]:
            valid = [index_map[listing] for listing in target_listings if listing in index_map]
            if len(valid) < 2:
                continue
            anchor = valid[0]
            for other in valid[1:]:
                dsu.union(anchor, other)

    union_from_edges(email_edges)
    union_from_edges(phone_edges)

    sizes = [dsu.component_size(i) for i in range(len(listing_ids))]
    return pl.DataFrame({
        "insertion_id": listing_ids,
        "listing_component_size": sizes,
    })


def _pagerank_feature(listing_ids: List[int],
                      email_edges: Optional[pl.DataFrame],
                      phone_edges: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """
    Compute PageRank scores for listings in the bipartite graph.
    
    Features:
    - listing_pagerank: PageRank score (0-1, higher = more central)
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for PageRank feature. Install with: pip install networkx")

    if email_edges is None and phone_edges is None:
        return None

    G = nx.Graph()

    def add_edges(edges: Optional[pl.DataFrame], prefix: str) -> None:
        if edges is None:
            return
        # Normalize column name: handle both 'source' and 'listing_id' as the listing column
        if "source" in edges.columns and "listing_id" not in edges.columns:
            edges = edges.rename({"source": "listing_id"})
        # Select columns explicitly to ensure correct order
        for listing, target in edges.select(["listing_id", "target"]).iter_rows():
            if listing is None or target is None:
                continue
            listing_node = f"L_{listing}"
            target_node = f"{prefix}_{target}"
            G.add_edge(listing_node, target_node)

    add_edges(email_edges, "E")
    add_edges(phone_edges, "P")

    if G.number_of_nodes() == 0:
        return None

    pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)
    data = []
    for listing in listing_ids:
        node = f"L_{listing}"
        data.append((listing, pr.get(node, 0.0)))

    return pl.DataFrame(data, schema=["insertion_id", "listing_pagerank"], orient="row")


# =============================================================================
# ADVANCED GRAPH FEATURES
# =============================================================================

def _calculate_isolation_scores(listing_ids: List[int], 
                                email_edges: Optional[pl.DataFrame], 
                                phone_edges: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
    """
    Calculate isolation metrics for listings.
    
    Features:
    - degree_total: Total connections (email + phone)
    - is_isolated: Boolean (1 if degree == 0, 0 otherwise)
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
    Calculate bipartite clustering coefficient for listings.
    
    Uses simplified "neighbor overlap" metric:
    - neighbor_overlap_score: Sum of (degree - 1) for all neighbors
    - avg_neighbor_degree: Average degree of neighbors
    
    High score = neighbors are hubs (connected to many other listings).
    Low score = neighbors are unique to this listing.
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
    agg = _groupby(enriched_edges, "listing_id").agg([
        (pl.col("target_degree") - 1).sum().alias("neighbor_overlap_score"),
        (pl.col("target_degree") - 1).mean().alias("avg_neighbor_degree")
    ])
    
    return agg.rename({"listing_id": "insertion_id"})


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_graph_features(
    output_path: Optional[Path] = None,
    cutoff_date: Optional[datetime] = None
) -> pl.DataFrame:
    """
    Generate ALL graph features for listings (basic + advanced).
    
    Features Generated:
    - Contact-based: shared emails, phones, counts
    - User-level: listing counts, IP sharing
    - Network structure: PageRank, connected components
    - Isolation: degree, isolated flag
    - Clustering: neighbor overlap, avg neighbor degree
    
    Args:
        output_path: Optional path to save features. If None, returns DataFrame without saving.
        cutoff_date: If provided, only use edges from listings before this date.
                     This prevents temporal leakage. If None, uses all edges.
    
    Returns:
        DataFrame with all graph features
        
    Raises:
        FileNotFoundError: If required artifact files are missing
        ValueError: If no features can be generated
    """
    logger.info("Loading data...")
    listings_df = load_listing_ids()
    listing_ids = listings_df["insertion_id"].to_list()

    # Load edges
    contact_email_edges = safe_load_edges(EDGE_LISTING_CONTACT_EMAIL)
    contact_phone_edges = safe_load_edges(EDGE_LISTING_CONTACT_PHONE)
    
    # Apply temporal filtering if cutoff_date is provided
    if cutoff_date is not None:
        logger.info(f"Filtering edges by cutoff_date: {cutoff_date}")
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

    # Basic features
    logger.info("Calculating contact email features...")
    email_features = _contact_edge_features(contact_email_edges, "contact_email")
    if email_features is not None:
        feature_frames.append(email_features)

    logger.info("Calculating contact phone features...")
    phone_features = _contact_edge_features(contact_phone_edges, "contact_phone")
    if phone_features is not None:
        feature_frames.append(phone_features)

    logger.info("Calculating user features...")
    user_features = _user_features(cutoff_date=cutoff_date)
    if user_features is not None:
        feature_frames.append(user_features)

    logger.info("Calculating connected components...")
    component_sizes = _component_sizes(listing_ids, contact_email_edges, contact_phone_edges)
    if component_sizes is not None:
        feature_frames.append(component_sizes)

    logger.info("Calculating PageRank...")
    pagerank_feature = _pagerank_feature(listing_ids, contact_email_edges, contact_phone_edges)
    if pagerank_feature is not None:
        feature_frames.append(pagerank_feature)

    # Advanced features
    logger.info("Calculating isolation scores...")
    isolation = _calculate_isolation_scores(listing_ids, contact_email_edges, contact_phone_edges)
    if isolation is not None:
        feature_frames.append(isolation)

    logger.info("Calculating clustering/overlap metrics...")
    clustering = _calculate_clustering_coefficient(listing_ids, contact_email_edges, contact_phone_edges)
    if clustering is not None:
        feature_frames.append(clustering)

    if len(feature_frames) == 1:
        raise ValueError("No graph features were generated. Check that edge files exist and contain data.")

    # Join all features
    logger.info("Joining all features...")
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.join(frame, on="insertion_id", how="left")

    # Fill nulls with 0
    numerical_cols = [col for col in features.columns if col != "insertion_id"]
    if numerical_cols:
        features = features.with_columns([
            pl.col(col).fill_null(0) for col in numerical_cols
        ])

    # Save to disk if output_path is provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.write_parquet(output_path)
        logger.info(f"Saved graph features to {output_path}")
    
    logger.info(f"Generated {len(numerical_cols)} graph features for {len(listing_ids)} listings")
    return features


if __name__ == "__main__":
    generate_graph_features()

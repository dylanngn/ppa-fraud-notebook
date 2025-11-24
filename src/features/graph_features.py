from pathlib import Path
from typing import Dict, List, Optional

import polars as pl


ARTIFACTS_DIR = Path("artifacts")
LISTING_NODES = ARTIFACTS_DIR / "nodes_listing.parquet"
EDGE_LISTING_CONTACT_EMAIL = ARTIFACTS_DIR / "edges_listing_contact_email.parquet"
EDGE_LISTING_CONTACT_PHONE = ARTIFACTS_DIR / "edges_listing_contact_phone.parquet"
EDGE_USER_POSTS_LISTING = ARTIFACTS_DIR / "edges_user_posts_listing.parquet"
EDGE_USER_IP = ARTIFACTS_DIR / "edges_user_uses_ip.parquet"
OUTPUT_PATH = ARTIFACTS_DIR / "listing_graph_features.parquet"


def _groupby(df: pl.DataFrame, *args, **kwargs):
    method = getattr(df, "groupby", None)
    if method is None:
        method = getattr(df, "group_by", None)
    if method is None:
        raise AttributeError("DataFrame has no groupby/group_by method. Please update Polars.")
    return method(*args, **kwargs)


def _clip_zero(expr: pl.Expr) -> pl.Expr:
    """Clamp expression to zero without relying on version-specific APIs."""
    return pl.when(expr > 0).then(expr).otherwise(pl.lit(0))


def _ensure_artifact(path: Path) -> bool:
    if not path.exists():
        print(f"[graph-features] Skipping missing artifact: {path}")
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


def _contact_edge_features(edges: Optional[pl.DataFrame], prefix: str) -> Optional[pl.DataFrame]:
    if edges is None:
        return None

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


def _user_features() -> Optional[pl.DataFrame]:
    if not (_ensure_artifact(EDGE_USER_POSTS_LISTING) and _ensure_artifact(EDGE_USER_IP)):
        return None

    posts = pl.read_parquet(EDGE_USER_POSTS_LISTING).drop_nulls(["source", "target"])
    if posts.is_empty():
        return None

    posts = posts.rename({"source": "user_id", "target": "insertion_id"})
    listing_to_user = posts.select(["insertion_id", "user_id"])

    user_listing_counts = (
        _groupby(posts, "user_id")
        .agg(pl.len().alias("user_listing_count"))
    )

    user_ip = pl.read_parquet(EDGE_USER_IP).drop_nulls(["source", "target"])
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
    if not listing_ids:
        return None

    index_map = {lid: idx for idx, lid in enumerate(listing_ids)}
    dsu = DisjointSet(len(listing_ids))

    def union_from_edges(edges: Optional[pl.DataFrame]) -> None:
        if edges is None:
            return
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
    try:
        import networkx as nx
    except ImportError:
        print("[graph-features] networkx not installed; skipping PageRank feature.")
        return None

    if email_edges is None and phone_edges is None:
        return None

    G = nx.Graph()

    def add_edges(edges: Optional[pl.DataFrame], prefix: str) -> None:
        if edges is None:
            return
        for listing, target in edges.iter_rows():
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

    return pl.DataFrame(data, schema=["insertion_id", "listing_pagerank"])


def generate_graph_features(output_path: Path = OUTPUT_PATH) -> None:
    listings_df = _load_listing_ids()
    listing_ids = listings_df["insertion_id"].to_list()

    contact_email_edges = _safe_edges(EDGE_LISTING_CONTACT_EMAIL)
    contact_phone_edges = _safe_edges(EDGE_LISTING_CONTACT_PHONE)

    feature_frames = [listings_df]

    email_features = _contact_edge_features(contact_email_edges, "contact_email")
    if email_features is not None:
        feature_frames.append(email_features)

    phone_features = _contact_edge_features(contact_phone_edges, "contact_phone")
    if phone_features is not None:
        feature_frames.append(phone_features)

    user_features = _user_features()
    if user_features is not None:
        feature_frames.append(user_features)

    component_sizes = _component_sizes(listing_ids, contact_email_edges, contact_phone_edges)
    if component_sizes is not None:
        feature_frames.append(component_sizes)

    pagerank_feature = _pagerank_feature(listing_ids, contact_email_edges, contact_phone_edges)
    if pagerank_feature is not None:
        feature_frames.append(pagerank_feature)

    if len(feature_frames) == 1:
        print("[graph-features] No graph features were generated.")
        return

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
    print(f"[graph-features] Saved listing graph features to {output_path}")


if __name__ == "__main__":
    generate_graph_features()


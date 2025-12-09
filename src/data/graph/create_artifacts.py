"""
Creates graph artifacts (nodes and edges) from flattened and anonymized data.

This module creates the parquet files used by both:
- GNN models (via graph_builder.py)
- XGBoost models (via graph_features.py for graph-derived features)

Listing Node Strategy (2025-12-09):
    nodes_listing.parquet contains ALL columns from raw_insertions.parquet.
    This ensures consistency with XGBoost auto mode and fairness between models.
    Feature filtering for GNN is done in graph_builder.py.

Entity Naming Convention:
    - External name: listing_id (used in logs, configs, documentation)
    - Internal storage: insertion_id (database object_reference field)
    
Entity Identification (Source of Truth):
    - Listing: `object_reference` → stored as `insertion_id` (internal)
    - User: `owner_id` → aliased to `user_id`

Note: User-listing relationship is established via:
    i.listing->'legacy'->>'personId' = u.owner_id
This is handled in the ETL extract query (src/data/etl/extract.py).
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = resolve_path("artifacts")
RAW_INSERTIONS = resolve_path("artifacts/raw_insertions.parquet")

# Node file paths
NODE_FILES = {
    "user": ARTIFACTS_DIR / "nodes_user.parquet",
    "listing": ARTIFACTS_DIR / "nodes_listing.parquet",
    "ip": ARTIFACTS_DIR / "nodes_ip.parquet",
    "email": ARTIFACTS_DIR / "nodes_email.parquet",
    "phone": ARTIFACTS_DIR / "nodes_phone.parquet",
    "address": ARTIFACTS_DIR / "nodes_address.parquet",
}

# Edge file paths
EDGE_FILES = {
    "user_posts": ARTIFACTS_DIR / "edges_user_posts_listing.parquet",
    "user_ip": ARTIFACTS_DIR / "edges_user_uses_ip.parquet",
    "user_email": ARTIFACTS_DIR / "edges_user_has_email.parquet",
    "listing_contact_email": ARTIFACTS_DIR / "edges_listing_contact_email.parquet",
    "listing_billing_email": ARTIFACTS_DIR / "edges_listing_billing_email.parquet",
    "listing_phone": ARTIFACTS_DIR / "edges_listing_phone.parquet",  # Unified phone edge
    "listing_located_at": ARTIFACTS_DIR / "edges_listing_located_at.parquet",
    "listing_billing_addr": ARTIFACTS_DIR / "edges_listing_billing_addr.parquet",
}

# Email columns with coverage info
EMAIL_COLS = [
    "listing.lister.email.hash",           # 99.97% coverage - PRIMARY
    "listing.lister.billing.email.hash",   # 98.00% coverage - GOOD
]

# Phone columns with coverage info
# Coverage varies significantly - use billing phone as primary, coalesce with lister phone
#   billing.phoneDay.hash: 97.99% coverage - BEST (PRIMARY)
#   lister.phone.hash:     69.98% coverage - MODERATE (FALLBACK)
#   viewing.phone.hash:     3.41% coverage - SKIP (too sparse)
#   inquiry.phone.hash:     0.11% coverage - SKIP (too sparse)
PHONE_COL_PRIMARY = "listing.lister.billing.phoneDay.hash"   # 98% coverage
PHONE_COL_FALLBACK = "listing.lister.phone.hash"              # 70% coverage

# Legacy: Individual columns (kept for reference)
PHONE_COLS = [PHONE_COL_PRIMARY, PHONE_COL_FALLBACK]

# Columns to SKIP due to low coverage (<10%) - not used in graph
PHONE_COLS_LOW_COVERAGE = [
    "listing.lister.contacts.viewing.phone.hash",   # 3.4% - too sparse
    "listing.lister.contacts.inquiry.phone.hash",   # 0.1% - too sparse  
    "listing.lister.billing.phoneMobile.hash",      # 1.8% - too sparse
]


class ColumnHelper:
    """Helper class for safely accessing columns with fallback options."""
    
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.columns = set(df.columns)
    
    def get_col(self, col_name: str, fallback_col: Optional[str] = None) -> pl.Expr:
        """
        Get column expression, with optional fallback if primary doesn't exist.
        
        Args:
            col_name: Primary column name
            fallback_col: Optional fallback column name
            
        Returns:
            Polars expression for the column or None literal
        """
        if col_name in self.columns:
            return pl.col(col_name)
        elif fallback_col and fallback_col in self.columns:
            return pl.col(fallback_col)
        else:
            return pl.lit(None).cast(pl.Utf8)

def _create_user_nodes(df_listings: pl.DataFrame, _helper: ColumnHelper) -> pl.DataFrame:
    """Create user nodes."""
    return df_listings.select([
        pl.col("owner_id").alias("user_id"),
        pl.col("account_created_at"),
        pl.lit(None).cast(pl.Utf8).alias("email_domain")
    ]).drop_nulls(subset=["user_id"]).unique(subset=["user_id"])


def _create_listing_nodes(df_listings: pl.DataFrame, _helper: ColumnHelper) -> pl.DataFrame:
    """
    Create listing nodes with ALL fields from raw_insertions.
    
    Passes through all columns for consistency with XGBoost auto mode.
    Downstream code (graph_builder.py) uses raw ETL column names directly.
    
    Only adds essential derived columns:
    - insertion_id: alias of object_reference (graph ID)
    - user_id: alias of owner_id (graph relationship)
    - is_fraud: derived boolean from fraud_flag
    """
    return df_listings.with_columns([
        pl.col("object_reference").alias("insertion_id"),
        pl.col("owner_id").alias("user_id"),
        pl.col("fraud_flag").is_not_null().alias("is_fraud"),
    ]).unique(subset=["insertion_id"])


def _create_ip_nodes(df_listings: pl.DataFrame, helper: ColumnHelper) -> pl.DataFrame:
    """Create IP nodes."""
    return df_listings.select(
        helper.get_col("user_ip_address_hash", "user_ip_address").alias("user_ip_address")
    ).unique().drop_nulls()


def _create_email_nodes(df_listings: pl.DataFrame, helper: ColumnHelper) -> pl.DataFrame:
    """Create email nodes from user and listing email columns."""
    # Emails from users
    emails_from_users = df_listings.select(
        helper.get_col("contact_emails_hash", "contact_emails")
        .str.split(",")
        .explode()
        .str.strip_chars()
        .alias("email")
    ).drop_nulls()
    
    # Emails from listings
    emails_from_listings = []
    for col in EMAIL_COLS:
        if col in helper.columns:
            emails_from_listings.append(
                df_listings.select(pl.col(col).alias("email"))
            )
    
    if emails_from_listings:
        return pl.concat([emails_from_users] + emails_from_listings).unique().drop_nulls()
    else:
        return emails_from_users.unique().drop_nulls()


def _create_phone_nodes(df_listings: pl.DataFrame, helper: ColumnHelper) -> pl.DataFrame:
    """Create phone nodes from listing phone columns."""
    phones_from_listings = []
    for col in PHONE_COLS:
        if col in helper.columns:
            phones_from_listings.append(
                df_listings.select(pl.col(col).alias("phone"))
            )
    
    if phones_from_listings:
        return pl.concat(phones_from_listings).unique().drop_nulls()
    else:
        return pl.DataFrame({"phone": []})


def _create_address_df(
    df: pl.DataFrame,
    helper: ColumnHelper,
    street_col: str,
    zip_col: str,
    city_col: str,
    country_col: str,
    lat_col: Optional[str] = None,
    lon_col: Optional[str] = None
) -> pl.DataFrame:
    """Create address DataFrame with address_id."""
    cols = [
        helper.get_col(street_col).fill_null("").alias("street"),
        helper.get_col(zip_col).fill_null("").alias("zip"),
        helper.get_col(city_col).fill_null("").alias("city"),
        helper.get_col(country_col).fill_null("").alias("country")
    ]
    
    if lat_col and lon_col:
        cols.append(helper.get_col(lat_col).alias("latitude"))
        cols.append(helper.get_col(lon_col).alias("longitude"))
    else:
        cols.append(pl.lit(None).cast(pl.Float64).alias("latitude"))
        cols.append(pl.lit(None).cast(pl.Float64).alias("longitude"))
    
    return df.select(cols).with_columns(
        (pl.col("country") + "_" + pl.col("zip") + "_" + pl.col("city") + "_" + pl.col("street")).alias("address_id")
    ).drop_nulls(subset=["address_id"]).unique(subset=["address_id"])


def _create_address_nodes(df_listings: pl.DataFrame, helper: ColumnHelper) -> pl.DataFrame:
    """Create address nodes from property and billing addresses."""
    # Property address
    addr_property = _create_address_df(
        df_listings,
        helper,
        "listing.address.street.hash",
        "listing.address.postalCode.hash",
        "listing.address.city_hash",
        "listing.address.country_hash",
        "listing.address.geoCoordinates.latitude",
        "listing.address.geoCoordinates.longitude"
    )
    
    # Billing address
    addr_billing = _create_address_df(
        df_listings,
        helper,
        "listing.lister.billing.address.street_hash",
        "listing.lister.billing.address.zip_hash",
        "listing.lister.billing.address.city_hash",
        "listing.lister.billing.address.country_hash"
    )
    
    return pl.concat([addr_property, addr_billing]).unique(subset=["address_id"])


def _create_edge_df(
    df: pl.DataFrame,
    helper: ColumnHelper,
    src_col: str,
    dst_col: str,
    src_name: str = "source",
    dst_name: str = "target"
) -> pl.DataFrame:
    """Create edge DataFrame from source and destination columns."""
    return df.select([
        pl.col(src_col).alias(src_name),
        helper.get_col(dst_col).alias(dst_name)
    ]).drop_nulls().unique()


def _create_address_edge(
    df: pl.DataFrame,
    helper: ColumnHelper,
    country_col: str,
    zip_col: str,
    city_col: str,
    street_col: str
) -> pl.DataFrame:
    """Create address edge by concatenating address components."""
    return df.select([
        pl.col("object_reference").alias("source"),
        (
            helper.get_col(country_col).fill_null("") + "_" +
            helper.get_col(zip_col).fill_null("") + "_" +
            helper.get_col(city_col).fill_null("") + "_" +
            helper.get_col(street_col).fill_null("")
        ).alias("target")
    ]).drop_nulls().unique()


def _create_all_nodes(df_listings: pl.DataFrame, helper: ColumnHelper) -> Tuple[pl.DataFrame, ...]:
    """Create all node DataFrames."""
    nodes_user = _create_user_nodes(df_listings, helper)
    nodes_listing = _create_listing_nodes(df_listings, helper)
    nodes_ip = _create_ip_nodes(df_listings, helper)
    nodes_email = _create_email_nodes(df_listings, helper)
    nodes_phone = _create_phone_nodes(df_listings, helper)
    nodes_address = _create_address_nodes(df_listings, helper)
    
    return nodes_user, nodes_listing, nodes_ip, nodes_email, nodes_phone, nodes_address


def _create_all_edges(df_listings: pl.DataFrame, helper: ColumnHelper) -> Tuple[pl.DataFrame, ...]:
    """Create all edge DataFrames."""
    # User edges
    edges_user_posts = df_listings.select([
        pl.col("owner_id").alias("source"),
        pl.col("object_reference").alias("target")
    ]).drop_nulls().unique()
    
    edges_user_ip = df_listings.select([
        pl.col("owner_id").alias("source"),
        helper.get_col("user_ip_address_hash", "user_ip_address").alias("target")
    ]).drop_nulls().unique()
    
    edges_user_email = df_listings.select([
        pl.col("owner_id").alias("source"),
        helper.get_col("contact_emails_hash", "contact_emails")
        .str.split(",")
        .explode()
        .str.strip_chars()
        .alias("target")
    ]).drop_nulls().unique()
    
    # Listing-email edges
    edges_listing_contact_email = _create_edge_df(
        df_listings, helper, "object_reference", "listing.lister.email.hash"
    )
    edges_listing_billing_email = _create_edge_df(
        df_listings, helper, "object_reference", "listing.lister.billing.email.hash"
    )
    
    # Listing-phone edges
    # Use coalesce to prefer billing phone (98% coverage) over lister phone (70% coverage)
    edges_listing_phone = df_listings.select([
        pl.col("object_reference").alias("source"),
        pl.coalesce([
            helper.get_col(PHONE_COL_PRIMARY),   # billing.phoneDay.hash (98%)
            helper.get_col(PHONE_COL_FALLBACK),  # lister.phone.hash (70%)
        ]).alias("target")
    ]).drop_nulls().unique()
    
    # Listing-address edges
    edges_listing_located_at = _create_address_edge(
        df_listings, helper,
        "listing.address.country_hash",
        "listing.address.postalCode.hash",
        "listing.address.city_hash",
        "listing.address.street_hash"
    )
    edges_listing_billing_addr = _create_address_edge(
        df_listings, helper,
        "listing.lister.billing.address.country_hash",
        "listing.lister.billing.address.zip_hash",
        "listing.lister.billing.address.city_hash",
        "listing.lister.billing.address.street_hash"
    )
    
    return (
        edges_user_posts, edges_user_ip, edges_user_email,
        edges_listing_contact_email, edges_listing_billing_email,
        edges_listing_phone,  # Unified phone edge
        edges_listing_located_at, edges_listing_billing_addr
    )


def _save_artifacts(
    nodes: Tuple[pl.DataFrame, ...],
    edges: Tuple[pl.DataFrame, ...]
) -> None:
    """Save all node and edge DataFrames to parquet files."""
    logger.info("Saving parquet artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    node_names = ["user", "listing", "ip", "email", "phone", "address"]
    # Edge names
    edge_names = [
        "user_posts", "user_ip", "user_email",
        "listing_contact_email", "listing_billing_email",
        "listing_phone",  # Unified phone edge
        "listing_located_at", "listing_billing_addr"
    ]
    
    # Save nodes
    for name, df in zip(node_names, nodes):
        df.write_parquet(NODE_FILES[name])
    
    # Save edges
    for name, df in zip(edge_names, edges):
        df.write_parquet(EDGE_FILES[name])


def create_nodes_and_edges(df_listings: pl.DataFrame) -> Tuple[pl.DataFrame, ...]:
    """
    Creates the nodes and edges for the Heterogeneous Graph.
    
    Works directly with flattened data that uses dot-notation field names.
    The data should already be flattened and anonymized from fetch_raw_insertions().
    
    Args:
        df_listings: DataFrame with flattened and anonymized data (dot-notation columns)
        
    Returns:
        Tuple of all node and edge DataFrames
    """
    logger.info("Creating nodes and edges from flattened data...")
    
    helper = ColumnHelper(df_listings)
    
    # Create all nodes and edges
    nodes = _create_all_nodes(df_listings, helper)
    edges = _create_all_edges(df_listings, helper)
    
    # Save artifacts
    _save_artifacts(nodes, edges)
    
    logger.info("Graph artifacts created successfully!")
    
    return nodes + edges


def main():
    """Create graph artifacts from flattened and anonymized data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create graph artifacts (nodes and edges)")
    parser.add_argument(
        "--input-path", 
        default=None,
        help=f"Path to flattened and anonymized data parquet file (default: {RAW_INSERTIONS})"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_path) if args.input_path else RAW_INSERTIONS
    if not input_path.exists():
        logger.error(f"{input_path} not found. Run ETL first.")
        raise SystemExit(1)
    
    try:
        logger.info(f"Loading data from {input_path}...")
        df_insertions = pl.read_parquet(input_path)
        
        if len(df_insertions) == 0:
            logger.error("Input file is empty.")
            raise SystemExit(1)
        
        logger.info(f"Creating graph artifacts from {len(df_insertions):,} insertions...")
        create_nodes_and_edges(df_insertions)
        
        logger.info("Graph artifacts created successfully!")
    except Exception as e:
        logger.error(f"Error creating graph artifacts: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

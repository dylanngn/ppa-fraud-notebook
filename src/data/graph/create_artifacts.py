"""
Graph Artifact Creator

This module creates the parquet files used by both:
- GNN models (via graph_structure.py)
- XGBoost models (via graph_features.py for graph-derived features)

Listing Node Strategy (2025-12-09):
    nodes_listing.parquet contains ALL columns from raw_insertions.parquet.
    This ensures consistency with XGBoost auto mode and fairness between models.
    Feature filtering for GNN is done in graph_structure.py.

Entity Naming Convention:
    - External name: listing_id (used in logs, configs, documentation)
    - Internal storage: insertion_id (database object_reference field)
    - Graph node indexing: 0..N-1 based on full dataframe order (CRITICAL for embeddings)

Note: User-listing relationship is established via:
    i.listing->'legacy'->>'personId' = u.owner_id
This is handled in the ETL extract query (src/data/etl/extract.py).
"""

import logging
import pickle
from pathlib import Path

import polars as pl
import torch

from src.data.training_loader import load_data
from src.data.graph.graph_structure import build_graph
from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

# Artifact paths
ARTIFACTS_DIR = resolve_path("artifacts")
NODES_USER = resolve_path("artifacts/nodes_user.parquet")
NODES_LISTING = resolve_path("artifacts/nodes_listing.parquet")
NODES_IP = resolve_path("artifacts/nodes_ip.parquet")
NODES_EMAIL = resolve_path("artifacts/nodes_email.parquet")
NODES_PHONE = resolve_path("artifacts/nodes_phone.parquet")
NODES_ADDRESS = resolve_path("artifacts/nodes_address.parquet")
EDGES_USER_LISTING = resolve_path("artifacts/edges_user_listing.parquet")
EDGES_USER_IP = resolve_path("artifacts/edges_user_ip.parquet")
EDGES_USER_EMAIL = resolve_path("artifacts/edges_user_email.parquet")
EDGES_LISTING_EMAIL_CONTACT = resolve_path("artifacts/edges_listing_email_contact.parquet")
EDGES_LISTING_EMAIL_BILLING = resolve_path("artifacts/edges_listing_email_billing.parquet")
EDGES_LISTING_PHONE = resolve_path("artifacts/edges_listing_phone.parquet")
EDGES_LISTING_ADDRESS_LOCATION = resolve_path("artifacts/edges_listing_address_location.parquet")
EDGES_LISTING_ADDRESS_BILLING = resolve_path("artifacts/edges_listing_address_billing.parquet")
GRAPH_PT = resolve_path("artifacts/graph.pt")
MAPPINGS_PKL = resolve_path("artifacts/mappings.pkl")


def create_unique_entity_nodes(df: pl.DataFrame, id_col: str, entity_name: str):
    """
    Create unique entity nodes from hash columns.
    
    Args:
        df: DataFrame containing the hash column
        id_col: Name of the hash column (e.g., "user_ip_address_hash")
        entity_name: Name for logging (e.g., "IP")
            
        Returns:
        DataFrame with unique entity IDs
    """
    unique_entities = (
        df
        .select(pl.col(id_col))
        .unique()
        .drop_nulls()
        .with_row_index(name=f"{entity_name.lower()}_idx")
    )
    
    logger.info(f"  {entity_name}: {len(unique_entities):,} unique entities")
    return unique_entities


def create_user_nodes(df: pl.DataFrame):
    """Create user nodes from owner_id."""
    nodes = (
        df
        .select("owner_id")
        .unique()
        .drop_nulls()
        .with_row_index(name="user_idx")
    )
    logger.info(f"  User: {len(nodes):,} unique users")
    return nodes


def create_listing_nodes(df: pl.DataFrame):
    """
    Create listing nodes.
    
    Strategy: Include ALL columns from raw_insertions to maintain consistency
    with XGBoost auto mode. GNN will filter to relevant features in graph_structure.py.
    
    Index Strategy: Use row index as listing_idx (0..N-1) where N is total listings.
    This ensures embedding[i] corresponds to row i in nodes_listing.parquet.
    """
    # Keep all columns for consistency with XGBoost
    nodes = df.with_row_index(name="listing_idx")
    
    logger.info(f"  Listing: {len(nodes):,} listings with {len(nodes.columns)} columns")
    logger.info(f"    Index range: 0 to {len(nodes)-1}")
    logger.info(f"    Columns: {', '.join(nodes.columns[:10])}... (showing first 10)")
    
    return nodes


def create_edges(df: pl.DataFrame, nodes_dict: dict):
    """
    Create all edge tables with proper index mapping.
    
    Args:
        df: Raw insertions DataFrame
        nodes_dict: Dictionary of node DataFrames with indices
        
    Returns:
        Dictionary of edge DataFrames
    """
    edges = {}
    
    # =========================================================================
    # User -> Listing (posts)
    # =========================================================================
    edges["user_listing"] = (
        df
        .select(["owner_id", "object_reference"])
        .drop_nulls()
        .join(nodes_dict["user"], on="owner_id", how="inner")
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .select(["user_idx", "listing_idx"])
    )
    logger.info(f"  User -> Listing: {len(edges['user_listing']):,} edges")
    
    # =========================================================================
    # User -> IP (uses)
    # =========================================================================
    edges["user_ip"] = (
        df
        .select(["owner_id", "user_ip_address_hash"])
        .drop_nulls()
        .join(nodes_dict["user"], on="owner_id", how="inner")
        .join(nodes_dict["ip"], on="user_ip_address_hash", how="inner")
        .select(["user_idx", "ip_idx"])
    )
    logger.info(f"  User -> IP: {len(edges['user_ip']):,} edges")
    
    # =========================================================================
    # User -> Email (has_email)
    # =========================================================================
    edges["user_email"] = (
        df
        .select(["owner_id", "owner_email_hash"])
        .drop_nulls()
        .join(nodes_dict["user"], on="owner_id", how="inner")
        .join(nodes_dict["email"], on="owner_email_hash", how="inner")
        .select(["user_idx", "email_idx"])
    )
    logger.info(f"  User -> Email: {len(edges['user_email']):,} edges")
    
    # =========================================================================
    # Listing -> Email (has_contact_email)
    # =========================================================================
    edges["listing_email_contact"] = (
        df
        .select(["object_reference", "lister_email_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["email"], on="lister_email_hash", how="inner")
        .select(["listing_idx", "email_idx"])
    )
    logger.info(f"  Listing -> Email (contact): {len(edges['listing_email_contact']):,} edges")
    
    # =========================================================================
    # Listing -> Email (has_billing_email)
    # =========================================================================
    edges["listing_email_billing"] = (
        df
        .select(["object_reference", "billing_email_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["email"], on="billing_email_hash", how="inner")
        .select(["listing_idx", "email_idx"])
    )
    logger.info(f"  Listing -> Email (billing): {len(edges['listing_email_billing']):,} edges")
    
    # =========================================================================
    # Listing -> Phone (has_phone)
    # UNIFIED: Both billing_phone_hash and lister_phone_hash map to same edge type
    # =========================================================================
    
    # Billing phone edges
    billing_phone_edges = (
        df
        .select(["object_reference", "billing_phone_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["phone"], on="billing_phone_hash", how="inner")
        .select(["listing_idx", "phone_idx"])
    )
    
    # Lister phone edges
    lister_phone_edges = (
        df
        .select(["object_reference", "lister_phone_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["phone"], on="lister_phone_hash", how="inner")
        .select(["listing_idx", "phone_idx"])
    )
    
    # Combine and deduplicate
    edges["listing_phone"] = pl.concat([billing_phone_edges, lister_phone_edges]).unique()
    
    logger.info(f"  Listing -> Phone: {len(edges['listing_phone']):,} edges " +
                f"(billing: {len(billing_phone_edges):,}, lister: {len(lister_phone_edges):,}, unique: {len(edges['listing_phone']):,})")
    
    # =========================================================================
    # Listing -> Address (located_at)
    # =========================================================================
    edges["listing_address_location"] = (
        df
        .select(["object_reference", "location_address_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["address"], on="location_address_hash", how="inner")
        .select(["listing_idx", "address_idx"])
    )
    logger.info(f"  Listing -> Address (location): {len(edges['listing_address_location']):,} edges")
    
    # =========================================================================
    # Listing -> Address (has_billing_addr)
    # =========================================================================
    edges["listing_address_billing"] = (
        df
        .select(["object_reference", "billing_address_hash"])
        .drop_nulls()
        .join(
            nodes_dict["listing"].select(["object_reference", "listing_idx"]),
            on="object_reference",
            how="inner"
        )
        .join(nodes_dict["address"], on="billing_address_hash", how="inner")
        .select(["listing_idx", "address_idx"])
    )
    logger.info(f"  Listing -> Address (billing): {len(edges['listing_address_billing']):,} edges")
    
    return edges


def create_nodes_and_edges(df: pl.DataFrame):
    """
    Main function to create all node and edge artifacts.
    
    Strategy:
    1. Create unique entity nodes (user, ip, email, phone, address)
    2. Keep ALL columns for listing nodes (consistency with XGBoost)
    3. Create edges with proper index mapping
    4. Save parquet files
    5. Build and save PyTorch graph
    6. Cache metadata separately for fast access
    """
    logger.info("=" * 70)
    logger.info("Creating Node Artifacts")
    logger.info("=" * 70)
    
    # Create nodes
    nodes_user = create_user_nodes(df)
    nodes_listing = create_listing_nodes(df)
    nodes_ip = create_unique_entity_nodes(df, "user_ip_address_hash", "IP")
    nodes_email = create_unique_entity_nodes(df, "owner_email_hash", "Email")
    nodes_phone = create_unique_entity_nodes(df, "billing_phone_hash", "Phone")
    nodes_address = create_unique_entity_nodes(df, "location_address_hash", "Address")
    
    # Store in dict for edge creation
    nodes_dict = {
        "user": nodes_user,
        "listing": nodes_listing,
        "ip": nodes_ip,
        "email": nodes_email,
        "phone": nodes_phone,
        "address": nodes_address,
    }
    
    logger.info("=" * 70)
    logger.info("Creating Edge Artifacts")
    logger.info("=" * 70)
    
    edges_dict = create_edges(df, nodes_dict)
    
    logger.info("=" * 70)
    logger.info("Saving Artifacts")
    logger.info("=" * 70)
    
    # Save nodes
    nodes_user.write_parquet(NODES_USER)
    nodes_listing.write_parquet(NODES_LISTING)
    nodes_ip.write_parquet(NODES_IP)
    nodes_email.write_parquet(NODES_EMAIL)
    nodes_phone.write_parquet(NODES_PHONE)
    nodes_address.write_parquet(NODES_ADDRESS)
    logger.info(f"Saved 6 node files to {ARTIFACTS_DIR}")
    
    # Save edges
    edges_dict["user_listing"].write_parquet(EDGES_USER_LISTING)
    edges_dict["user_ip"].write_parquet(EDGES_USER_IP)
    edges_dict["user_email"].write_parquet(EDGES_USER_EMAIL)
    edges_dict["listing_email_contact"].write_parquet(EDGES_LISTING_EMAIL_CONTACT)
    edges_dict["listing_email_billing"].write_parquet(EDGES_LISTING_EMAIL_BILLING)
    edges_dict["listing_phone"].write_parquet(EDGES_LISTING_PHONE)
    edges_dict["listing_address_location"].write_parquet(EDGES_LISTING_ADDRESS_LOCATION)
    edges_dict["listing_address_billing"].write_parquet(EDGES_LISTING_ADDRESS_BILLING)
    logger.info(f"Saved 8 edge files to {ARTIFACTS_DIR}")
    
    # Build graph with PyTorch Geometric
    logger.info("Building PyTorch Geometric graph...")
    graph_data = build_graph()
    
    # Save graph
    logger.info(f"Saving graph to {GRAPH_PT}")
    torch.save(graph_data, GRAPH_PT)
    logger.info(f"Graph saved ({GRAPH_PT.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Cache metadata separately for fast access
    metadata = graph_data.metadata()
    metadata_path = str(GRAPH_PT).replace('.pt', '_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata cached to {metadata_path}")
    
    # Log summary
    node_types, edge_types = metadata
    logger.info("=" * 70)
    logger.info("Graph artifacts created successfully!")
    logger.info(f"Listings: {len(nodes_listing):,}")
    logger.info(f"Total nodes: {sum(graph_data[nt].num_nodes for nt in graph_data.node_types):,}")
    logger.info(f"Node types: {len(node_types)}, Edge types: {len(edge_types)}")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    logger.info("Starting graph artifact creation...")
    
    try:
        # Load data
        logger.info("Loading data...")
        df_insertions = load_data()
        logger.info(f"Loaded {len(df_insertions):,} insertions with {len(df_insertions.columns)} columns")
        
        # Create artifacts
        create_nodes_and_edges(df_insertions)
        
        logger.info("Graph artifacts created successfully!")
    except Exception as e:
        logger.error(f"Error creating graph artifacts: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

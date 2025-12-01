"""
Graph Builder for Fraud Detection GNN

Builds a heterogeneous graph from node/edge parquet artifacts for GNN training.

Graph Structure (SIMPLIFIED - Phase 2 cleanup):
- 6 Node Types: user, listing, ip, email, phone, address
- 8 Edge Types (reduced from 9):
  * user -> posts -> listing
  * user -> uses -> ip  
  * user -> has_email -> email
  * listing -> has_contact_email -> email
  * listing -> has_billing_email -> email
  * listing -> has_phone -> phone (UNIFIED: billing + lister phone coalesced)
  * listing -> located_at -> address
  * listing -> billing_address -> address

Data Quality Notes (2025-11-30):
- Phone: Using coalesced billing+lister phone (98%+70% -> ~99% coverage)
- Boolean characteristics: is_new (0%), has_elevator (40%) - low coverage, excluded
- Primary email and address: excellent coverage (98-100%)
"""

import logging
import os
import pickle

import numpy as np
import polars as pl
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)
from datetime import datetime
from typing import Optional
from sentence_transformers import SentenceTransformer

# Coverage thresholds for feature inclusion
MIN_FEATURE_COVERAGE = 0.5  # Require at least 50% non-null values

def load_node_mapping(df, id_col, node_type):
    """
    Creates a mapping from Raw ID -> PyG Index (0..N-1).
    Returns:
        mapping (dict): Raw ID -> Index
        x (torch.Tensor): Node features (if any)
    """
    logger.info(f"Processing {node_type} nodes...")
    
    # Ensure unique
    df = df.unique(subset=[id_col])
    
    # Create mapping
    ids = df[id_col].to_list()
    mapping = {raw_id: i for i, raw_id in enumerate(ids)}
    
    return mapping, df

def generate_embeddings(df_listings):
    """
    Generates text embeddings for listing descriptions.
    
    This function is called on-demand when building the graph for GNN training.
    Embeddings are NOT stored in parquet files - they are generated fresh each time.
    
    Args:
        df_listings: Polars DataFrame with 'description_text' column
        
    Returns:
        numpy array of shape (n_listings, 384) with embeddings
    """
    logger.info("Generating text embeddings for GNN (this may take a while)...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"Using device: {device}")

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Handle null descriptions
    if "description_text" not in df_listings.columns:
        logger.warning("description_text not found. Using empty strings.")
        texts = [""] * len(df_listings)
    else:
        texts = df_listings["description_text"].fill_null("").to_list()
    
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings

def build_graph(cutoff_date: Optional[datetime] = None):
    """
    Build graph from parquet artifacts with optional temporal filtering.
    
    Args:
        cutoff_date: If provided, only include listings and edges before this date.
                     This prevents temporal leakage during training.
    
    Returns:
        HeteroData graph object
    """
    logger.info("Loading parquet artifacts...")
    if cutoff_date is not None:
        logger.info(f"Filtering graph by cutoff_date: {cutoff_date}")
    
    # --- Load Nodes ---
    df_user = pl.read_parquet("artifacts/nodes_user.parquet")
    df_listing = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_ip = pl.read_parquet("artifacts/nodes_ip.parquet")
    df_email = pl.read_parquet("artifacts/nodes_email.parquet")
    df_phone = pl.read_parquet("artifacts/nodes_phone.parquet")
    df_address = pl.read_parquet("artifacts/nodes_address.parquet")
    
    # Filter listings by cutoff_date if provided
    if cutoff_date is not None:
        # Ensure submission_at is datetime
        df_listing = df_listing.with_columns(
            pl.col("submission_at").cast(pl.Datetime("ns"))
        )
        df_listing = df_listing.filter(pl.col("submission_at") < cutoff_date)
        logger.info(f"Filtered listings to {len(df_listing)} before cutoff_date")
    
    data = HeteroData()
    
    # --- Process Mappings & Features ---
    
    # 1. User
    user_map, df_user = load_node_mapping(df_user, "user_id", "user")
    # Features: Account Age (days)
    # We can compute account age relative to a fixed date or just use 1s for now.
    # Let's stick to dummy features for MVP to avoid complex date parsing here (handled in ETL/Features)
    data['user'].x = torch.ones(len(df_user), 1)
    data['user'].num_nodes = len(df_user)
    
    # 2. Listing
    listing_map, df_listing = load_node_mapping(df_listing, "insertion_id", "listing")
    
    # Features: Price, Size, Rooms, OfferType, Characteristics, Bundle, Payment, Location
    # 
    # DATA QUALITY NOTES (from 2025-11-30 analysis):
    # - High coverage (>80%): price, rooms, living_space, offer_type, location, bundle
    # - Moderate coverage (50-80%): has_balcony (71%), has_parking (55%)
    # - Low coverage (<50%): is_new (0.04%), has_elevator (40%) - CAUTION: fill_null masks missing data
    #
    # Handle Nulls - with explicit coverage tracking
    df_listing = df_listing.with_columns([
        # High coverage numerical features (>80%)
        pl.col("price_rent_gross").fill_null(0),
        pl.col("price_buy").fill_null(0),
        pl.col("living_space").fill_null(0),
        pl.col("rooms").fill_null(0),
        
        # High coverage categorical (>95%)
        pl.col("offer_type").fill_null("RENT"),
        
        # Bundle features (77-98% coverage)
        pl.col("bundle_period").fill_null(7),
        pl.col("bundle_tier").fill_null("basic").str.to_lowercase(),
        pl.col("payment_type").fill_null("INVOICE"),
        
        # Location (99.4% coverage)
        pl.col("latitude").fill_null(0.0),
        pl.col("longitude").fill_null(0.0),
        
        # Metadata (100% coverage)
        pl.col("customer_segment").fill_null("unknown").str.to_lowercase(),
        pl.col("language").fill_null("de").str.to_lowercase(),
        
        # CAUTION: Low-coverage boolean features
        # These are filled with False but 40-99% of data is actually missing
        # Consider removing from model or using special "unknown" encoding
        pl.col("is_new").fill_null(False).cast(pl.Int8),           # 0.04% coverage!
        pl.col("has_balcony").fill_null(False).cast(pl.Int8),      # 71% coverage
        pl.col("has_elevator").fill_null(False).cast(pl.Int8),     # 40% coverage - UNRELIABLE
        pl.col("has_parking").fill_null(False).cast(pl.Int8),      # 55% coverage
    ])
    
    # One-hot encode offer_type (RENT=0, BUY=1)
    offer_type_feat = (df_listing["offer_type"] == "BUY").cast(pl.Int8).to_numpy().reshape(-1, 1)
    
    # Encode Payment Type (INVOICE=0, DIRECT=1)
    payment_feat = (df_listing["payment_type"] == "DIRECT").cast(pl.Int8).to_numpy().reshape(-1, 1)
    
    # Encode Bundle Tier (Ordinal: basic=0, premium=1, top=2)
    tier_map = {"basic": 0, "premium": 1, "top": 2}
    tier_series = df_listing["bundle_tier"].replace(tier_map, default=0).cast(pl.Int64).to_numpy().reshape(-1, 1)

    # Encode Customer Segment (One-Hot)
    # Segments: tenant, owner, business, unknown
    segments = ["tenant", "owner", "business"]
    segment_feats = []
    for seg in segments:
        feat = (df_listing["customer_segment"] == seg).cast(pl.Int8).to_numpy().reshape(-1, 1)
        segment_feats.append(feat)
    segment_matrix = np.concatenate(segment_feats, axis=1)

    # Encode Language (One-Hot)
    # Languages: de, en, fr, it
    langs = ["de", "en", "fr", "it"]
    lang_feats = []
    for lang in langs:
        feat = (df_listing["language"] == lang).cast(pl.Int8).to_numpy().reshape(-1, 1)
        lang_feats.append(feat)
    lang_matrix = np.concatenate(lang_feats, axis=1)

    # Numerical Features
    num_feats = df_listing.select([
        "price_rent_gross", "price_buy", "living_space", "rooms",
        "is_new", "has_balcony", "has_elevator", "has_parking",
        "bundle_period", "latitude", "longitude"
    ]).to_numpy()
    
    # Embeddings - Generate on-the-fly if not present
    if "description_embedding" in df_listing.columns:
        logger.info("Using pre-computed embeddings from nodes_listing.parquet")
        embeddings = np.stack(df_listing["description_embedding"].to_numpy())
    else:
        logger.info("description_embedding not found. Generating embeddings on-the-fly...")
        embeddings = generate_embeddings(df_listing)
    
    # Concatenate
    x_listing = np.concatenate([
        num_feats, 
        offer_type_feat, 
        payment_feat, 
        tier_series, 
        segment_matrix, 
        lang_matrix, 
        embeddings
    ], axis=1)
    data['listing'].x = torch.from_numpy(x_listing).float()
    
    # Labels (Target)
    y = df_listing["is_fraud"].cast(pl.Int64).fill_null(0).to_numpy()
    data['listing'].y = torch.from_numpy(y).long()
    
    # Timestamps
    # Fix 1970 issue: Ensure nanoseconds (Polars defaults to us for some sources)
    timestamps = df_listing["submission_at"].cast(pl.Datetime("ns")).cast(pl.Int64).fill_null(0).to_numpy() # ns
    
    # Filter out invalid timestamps (e.g. 0 or very old)
    # We only want listings with valid submission_at
    # But we can't easily drop nodes here without re-indexing everything.
    # Instead, we'll set a mask or just ensure ETL provides valid data.
    # For now, let's just warn or use a default recent date if 0?
    # No, 0 means 1970. 
    # Better: In the future, ETL should filter these.
    # Here, let's just ensure we don't crash, but train_gnn should filter them.
    
    data['listing'].timestamp = torch.from_numpy(timestamps)
    data['listing'].num_nodes = len(df_listing)

    # 3. IP
    ip_map, _ = load_node_mapping(df_ip, "user_ip_address", "ip")
    data['ip'].num_nodes = len(ip_map)
    data['ip'].x = torch.ones(len(ip_map), 1)

    # 4. Email (Unified)
    email_map, _ = load_node_mapping(df_email, "email", "email")
    data['email'].num_nodes = len(email_map)
    data['email'].x = torch.ones(len(email_map), 1)

    # 5. Phone (Unified)
    phone_map, _ = load_node_mapping(df_phone, "phone", "phone")
    data['phone'].num_nodes = len(phone_map)
    data['phone'].x = torch.ones(len(phone_map), 1)

    # 6. Address (Granular)
    addr_map, df_address = load_node_mapping(df_address, "address_id", "address")
    
    # Address Features: Lat, Lon
    df_address = df_address.with_columns([
        pl.col("latitude").fill_null(0.0),
        pl.col("longitude").fill_null(0.0)
    ])
    addr_feats = df_address.select(["latitude", "longitude"]).to_numpy()
    data['address'].x = torch.from_numpy(addr_feats).float()
    data['address'].num_nodes = len(addr_map)

    # --- Process Edges ---
    
    # Mapping for edge timestamps (Listing Time)
    # Fix 1970 issue: Ensure nanoseconds
    listing_time_map = dict(zip(df_listing["insertion_id"], df_listing["submission_at"].cast(pl.Datetime("ns")).cast(pl.Int64)))
    
    # For user-based edges, compute user's earliest listing time
    # This is used to timestamp user->ip and user->email edges
    user_earliest_listing = None
    user_earliest_listing_map = {}
    if cutoff_date is not None:
        # Get user->listing edges to find earliest listing per user
        user_posts = pl.read_parquet("artifacts/edges_user_posts_listing.parquet")
        user_posts = user_posts.join(
            df_listing.select(["insertion_id", "submission_at"]),
            left_on="target",
            right_on="insertion_id",
            how="inner"
        )
        user_earliest_listing = (
            user_posts.group_by("source")
            .agg(pl.col("submission_at").min().alias("earliest_listing_time"))
            .with_columns(
                pl.col("earliest_listing_time").cast(pl.Datetime("ns")).cast(pl.Int64)
            )
        )
        user_earliest_listing_map = dict(
            zip(user_earliest_listing["source"], user_earliest_listing["earliest_listing_time"])
        )
    
    def add_edge(filename, src_col, dst_col, src_type, dst_type, rel_name, time_source_col=None, edge_type="listing_to_target"):
        """
        Add edge to graph with temporal filtering.
        
        Args:
            edge_type: How to filter edges temporally:
                - "listing_to_target": source is listing_id, filter by source
                - "target_to_listing": target is listing_id, filter by target
                - "user_based": edge involves user, filter by user's earliest listing
        """
        logger.info(f"Processing edge: {src_type} - {rel_name} - {dst_type}")
        if not os.path.exists(f"artifacts/{filename}"):
            logger.warning(f"{filename} not found. Skipping.")
            return

        df_edge = pl.read_parquet(f"artifacts/{filename}")
        
        # Apply temporal filtering if cutoff_date is provided
        if cutoff_date is not None:
            if edge_type == "listing_to_target":
                # source is listing_id, filter by source
                df_edge = df_edge.join(
                    df_listing.select(["insertion_id"]),
                    left_on="source",
                    right_on="insertion_id",
                    how="inner"
                ).select(["source", "target"])
            elif edge_type == "target_to_listing":
                # target is listing_id, filter by target
                df_edge = df_edge.join(
                    df_listing.select(["insertion_id"]),
                    left_on="target",
                    right_on="insertion_id",
                    how="inner"
                ).select(["source", "target"])
            elif edge_type == "user_based":
                # Filter by user's earliest listing time
                if user_earliest_listing is not None:
                    df_edge = df_edge.join(
                        user_earliest_listing,
                        left_on="source",
                        right_on="source",
                        how="inner"
                    ).select(["source", "target"])
                else:
                    # No users with listings before cutoff, skip this edge type
                    return
        
        if df_edge.is_empty():
            logger.info(f"  No edges remaining after temporal filtering")
            return
        
        # Map IDs to Indices
        src_indices = [src_map.get(i) for i in df_edge[src_col].to_list()]
        dst_indices = [dst_map.get(i) for i in df_edge[dst_col].to_list()]
        
        # Filter Nones
        valid_mask = [(s is not None and d is not None) for s, d in zip(src_indices, dst_indices)]
        src_indices = [s for s, v in zip(src_indices, valid_mask) if v]
        dst_indices = [d for d, v in zip(dst_indices, valid_mask) if v]
        
        if not src_indices:
            return

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data[src_type, rel_name, dst_type].edge_index = edge_index
        
        # Assign Timestamps
        edge_times = []
        if time_source_col:
            if dst_type == 'listing':
                valid_dst_ids = [i for i, v in zip(df_edge[dst_col].to_list(), valid_mask) if v]
                edge_times = [listing_time_map.get(i, 0) for i in valid_dst_ids]
            elif src_type == 'listing':
                valid_src_ids = [i for i, v in zip(df_edge[src_col].to_list(), valid_mask) if v]
                edge_times = [listing_time_map.get(i, 0) for i in valid_src_ids]
        elif edge_type == "user_based" and cutoff_date is not None:
            # For user-based edges, use user's earliest listing time
            valid_src_ids = [i for i, v in zip(df_edge[src_col].to_list(), valid_mask) if v]
            edge_times = [user_earliest_listing_map.get(i, 0) for i in valid_src_ids]
                
        if edge_times:
            data[src_type, rel_name, dst_type].timestamp = torch.tensor(edge_times, dtype=torch.long)

    # Define mappings
    maps = {
        "user": user_map,
        "listing": listing_map,
        "ip": ip_map,
        "email": email_map,
        "phone": phone_map,
        "address": addr_map
    }

    # 1. User -> Posts -> Listing
    src_map, dst_map = maps["user"], maps["listing"]
    add_edge("edges_user_posts_listing.parquet", "source", "target", "user", "listing", "posts", time_source_col="target", edge_type="target_to_listing")
    
    # 2. User -> Uses -> IP
    src_map, dst_map = maps["user"], maps["ip"]
    add_edge("edges_user_uses_ip.parquet", "source", "target", "user", "ip", "uses", edge_type="user_based")
    
    # 3. User -> Has -> Email
    src_map, dst_map = maps["user"], maps["email"]
    add_edge("edges_user_has_email.parquet", "source", "target", "user", "email", "has_email", edge_type="user_based")
    
    # 4. Listing -> Has -> Email (All types)
    src_map, dst_map = maps["listing"], maps["email"]
    add_edge("edges_listing_contact_email.parquet", "source", "target", "listing", "email", "has_contact_email", time_source_col="source", edge_type="listing_to_target")
    add_edge("edges_listing_billing_email.parquet", "source", "target", "listing", "email", "has_billing_email", time_source_col="source", edge_type="listing_to_target")
    
    # 5. Listing -> Has -> Phone (UNIFIED - Phase 2 simplification)
    # Single phone edge using coalesced billing+lister phone for ~99% coverage
    src_map, dst_map = maps["listing"], maps["phone"]
    add_edge("edges_listing_phone.parquet", "source", "target", "listing", "phone", "has_phone", time_source_col="source", edge_type="listing_to_target")
    
    # 8. Listing -> Located_At -> Address
    src_map, dst_map = maps["listing"], maps["address"]
    add_edge("edges_listing_located_at.parquet", "source", "target", "listing", "address", "located_at", time_source_col="source", edge_type="listing_to_target")
    add_edge("edges_listing_billing_addr.parquet", "source", "target", "listing", "address", "billing_address", time_source_col="source", edge_type="listing_to_target")

    # --- Reverse Edges ---
    transform = T.ToUndirected()
    data = transform(data)
    
    # After ToUndirected(), copy timestamps to reverse edges
    # Reverse edges should have the same timestamps as forward edges
    for edge_type in list(data.edge_types):
        if 'timestamp' in data[edge_type]:
            # Timestamp is already set for forward edges
            # ToUndirected() may have created reverse edges without timestamps
            # Check if reverse edge exists and copy timestamp if needed
            src_type, rel_name, dst_type = edge_type
            reverse_type = (dst_type, rel_name, src_type)
            if reverse_type in data.edge_types and 'timestamp' not in data[reverse_type]:
                # Copy timestamp from forward edge
                data[reverse_type].timestamp = data[edge_type].timestamp
    
    logger.info("Graph construction complete!")
    logger.info(data)
    
    # Only save if no cutoff_date (full graph for initial training)
    if cutoff_date is None:
        torch.save(data, "artifacts/graph.pt")
        
        # Save Mappings
        with open("artifacts/mappings.pkl", "wb") as f:
            pickle.dump(maps, f)
    else:
        logger.info("Graph built with cutoff_date, not saving to artifacts/graph.pt (temporary graph)")
    
    return data

if __name__ == "__main__":
    build_graph()

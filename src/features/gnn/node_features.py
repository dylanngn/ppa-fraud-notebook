"""
Graph Node Feature Engineering

Creates node features for the heterogeneous fraud detection graph.
This module handles all feature engineering logic, keeping it separate
from graph structure building.

Node Feature Engineering:
  • Listing: Numerical, categorical, boolean, text embeddings
  • User: Simple dummy features (could be extended)
  • Address: Lat/lon coordinates
  • IP, Email, Phone: Dummy features

Usage:
    from src.features.gnn.node_features import create_all_node_features
    
    node_features = create_all_node_features(
        df_listing=df_listing,
        df_user=df_user,
        df_address=df_address,
        ...
    )
"""

import logging
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def generate_text_embeddings(df_listings: pl.DataFrame) -> np.ndarray:
    """
    Generate text embeddings for listing descriptions using SentenceTransformer.
    
    This is called on-demand when building the graph if embeddings aren't
    already stored in the parquet file.
    
    Args:
        df_listings: DataFrame with 'description_text' column
        
    Returns:
        numpy array of shape (n_listings, 384) with embeddings
    """
    logger.info("Generating text embeddings for GNN (this may take a while)...")
    
    # Check if GPU is available for faster embedding generation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    
    logger.info(f"Using device: {device}")
    
    # Load pre-trained multilingual model
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
    
    # Get descriptions (handle nulls)
    descriptions = df_listings["description_text"].fill_null("").to_list()
    
    # Generate embeddings in batches for efficiency
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    logger.info(f"Generated {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    return embeddings


def create_listing_features(df_listing: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create feature tensor for listing nodes.
    
    Features include:
      • Numerical: price, size, rooms, location (lat/lon)
      • Categorical: offer_type, payment, bundle_tier, segment, language
      • Boolean: 18 property characteristics (balcony, parking, etc.)
      • Text: Embedding from description (384-dim)
    
    Args:
        df_listing: Listing DataFrame with raw ETL columns
        
    Returns:
        Tuple of (features, labels, timestamps):
        - features: torch.Tensor of shape (n_listings, n_features)
        - labels: torch.Tensor of shape (n_listings,) with fraud labels
        - timestamps: torch.Tensor of shape (n_listings,) with submission times
    """
    logger.info("Creating listing features...")
    
    # Extract and clean features using raw ETL column names
    df_listing = df_listing.with_columns([
        # High coverage numerical features (>80%)
        pl.col("listing.prices.rent.gross").fill_null(0).alias("price_rent_gross"),
        pl.col("listing.prices.buy.price").fill_null(0).alias("price_buy"),
        pl.col("listing.characteristics.livingSpace").fill_null(0).alias("living_space"),
        pl.col("listing.characteristics.numberOfRooms").fill_null(0).alias("rooms"),
        
        # High coverage categorical (>95%)
        pl.col("listing.offerType").fill_null("RENT").alias("offer_type"),
        
        # Bundle features (77-98% coverage)
        pl.col("bundle.period").fill_null(7).alias("bundle_period"),
        pl.col("bundle.tier").fill_null("basic").str.to_lowercase().alias("bundle_tier"),
        pl.col("listing.lister.billing.payment.paymentType").fill_null("INVOICE").alias("payment_type"),
        
        # Location (99.4% coverage)
        pl.col("listing.address.geoCoordinates.latitude").fill_null(0.0).alias("latitude"),
        pl.col("listing.address.geoCoordinates.longitude").fill_null(0.0).alias("longitude"),
        
        # Metadata (100% coverage)
        pl.col("customer_segment").fill_null("unknown").str.to_lowercase(),
        pl.col("listing.localization.primary").fill_null("de").str.to_lowercase().alias("language"),
        
        # Boolean indicator features (NULL = FALSE semantics = 100% semantic coverage)
        pl.col("listing.characteristics.hasBalcony").fill_null(False).cast(pl.Int8).alias("has_balcony"),
        pl.col("listing.characteristics.hasParking").fill_null(False).cast(pl.Int8).alias("has_parking"),
        pl.col("listing.characteristics.hasNiceView").fill_null(False).cast(pl.Int8).alias("has_nice_view"),
        pl.col("listing.characteristics.hasGarage").fill_null(False).cast(pl.Int8).alias("has_garage"),
        pl.col("listing.characteristics.isChildFriendly").fill_null(False).cast(pl.Int8).alias("is_child_friendly"),
        pl.col("listing.characteristics.isQuiet").fill_null(False).cast(pl.Int8).alias("is_quiet"),
        pl.col("listing.characteristics.hasElevator").fill_null(False).cast(pl.Int8).alias("has_elevator"),
        pl.col("listing.characteristics.hasWashingMachine").fill_null(False).cast(pl.Int8).alias("has_washing_machine"),
        pl.col("listing.characteristics.arePetsAllowed").fill_null(False).cast(pl.Int8).alias("are_pets_allowed"),
        pl.col("listing.characteristics.isWheelchairAccessible").fill_null(False).cast(pl.Int8).alias("is_wheelchair_accessible"),
        pl.col("listing.characteristics.isOldBuilding").fill_null(False).cast(pl.Int8).alias("is_old"),
        pl.col("listing.characteristics.isNewBuilding").fill_null(False).cast(pl.Int8).alias("is_new_building"),
        pl.col("listing.characteristics.hasCableTv").fill_null(False).cast(pl.Int8).alias("has_cable_tv"),
        pl.col("listing.characteristics.hasFireplace").fill_null(False).cast(pl.Int8).alias("has_fireplace"),
        pl.col("listing.characteristics.isMinergieGeneral").fill_null(False).cast(pl.Int8).alias("is_minergie_general"),
        pl.col("listing.characteristics.isMinergieCertified").fill_null(False).cast(pl.Int8).alias("is_minergie_certified"),
        pl.col("listing.characteristics.isSmokingAllowed").fill_null(False).cast(pl.Int8).alias("is_smoking_allowed"),
        pl.col("listing.characteristics.hasSwimmingPool").fill_null(False).cast(pl.Int8).alias("has_swimming_pool"),
    ])
    
    # === Categorical Encodings ===
    
    # One-hot encode offer_type (RENT=0, BUY=1)
    offer_type_feat = (df_listing["offer_type"] == "BUY").cast(pl.Int8).to_numpy().reshape(-1, 1)
    
    # Encode Payment Type (INVOICE=0, DIRECT=1)
    payment_feat = (df_listing["payment_type"] == "DIRECT").cast(pl.Int8).to_numpy().reshape(-1, 1)
    
    # Encode Bundle Tier (Ordinal: basic=0, premium=1, top=2)
    tier_map = {"basic": 0, "premium": 1, "top": 2}
    tier_series = df_listing["bundle_tier"].replace(tier_map, default=0).cast(pl.Int64).to_numpy().reshape(-1, 1)

    # Encode Customer Segment (One-Hot: tenant, owner, business, unknown)
    segments = ["tenant", "owner", "business"]
    segment_feats = []
    for seg in segments:
        feat = (df_listing["customer_segment"] == seg).cast(pl.Int8).to_numpy().reshape(-1, 1)
        segment_feats.append(feat)
    segment_matrix = np.concatenate(segment_feats, axis=1)

    # Encode Language (One-Hot: de, en, fr, it)
    langs = ["de", "en", "fr", "it"]
    lang_feats = []
    for lang in langs:
        feat = (df_listing["language"] == lang).cast(pl.Int8).to_numpy().reshape(-1, 1)
        lang_feats.append(feat)
    lang_matrix = np.concatenate(lang_feats, axis=1)

    # === Numerical + Boolean Features ===
    num_feats = df_listing.select([
        # Core numerical
        "price_rent_gross", "price_buy", "living_space", "rooms",
        "bundle_period", "latitude", "longitude",
        # Boolean indicators (NULL = FALSE, 100% semantic coverage)
        "has_balcony", "has_parking", "has_nice_view", "has_garage",
        "is_child_friendly", "is_quiet", "has_elevator",
        "has_washing_machine", "are_pets_allowed", "is_wheelchair_accessible",
        "is_old", "is_new_building",
        "has_cable_tv", "has_fireplace", "is_minergie_general",
        "is_minergie_certified", "is_smoking_allowed", "has_swimming_pool"
    ]).to_numpy()
    
    # === Text Embeddings ===
    # Use pre-computed if available, otherwise generate on-the-fly
    if "description_embedding" in df_listing.columns:
        logger.info("Using pre-computed embeddings from parquet")
        embeddings = np.stack(df_listing["description_embedding"].to_numpy())
    else:
        logger.info("Generating embeddings on-the-fly...")
        embeddings = generate_text_embeddings(df_listing)
    
    # === Concatenate All Features ===
    x_listing = np.concatenate([
        num_feats,           # Numerical + boolean features
        offer_type_feat,     # Categorical encodings
        payment_feat,
        tier_series,
        segment_matrix,
        lang_matrix,
        embeddings           # Text embeddings (384-dim)
    ], axis=1)
    
    features = torch.from_numpy(x_listing).float()
    
    # === Labels (Target) ===
    labels = df_listing["is_fraud"].cast(pl.Int64).fill_null(0).to_numpy()
    labels = torch.from_numpy(labels).long()
    
    # === Timestamps ===
    timestamps = df_listing["submission_at"].cast(pl.Datetime("ns")).cast(pl.Int64).fill_null(0).to_numpy()
    timestamps = torch.from_numpy(timestamps)
    
    logger.info(f"Created listing features: {features.shape}")
    logger.info(f"  Numerical: 25, Categorical: 10, Embeddings: 384")
    
    return features, labels, timestamps


def create_user_features(df_user: pl.DataFrame) -> torch.Tensor:
    """
    Create feature tensor for user nodes.
    
    Currently uses simple dummy features (ones).
    Could be extended with: account age, listing count, etc.
    
    Args:
        df_user: User DataFrame
        
    Returns:
        torch.Tensor of shape (n_users, 1) with dummy features
    """
    # Simple dummy features for now
    # Could extend with: account age, total listings, fraud history, etc.
    return torch.ones(len(df_user), 1)


def create_address_features(df_address: pl.DataFrame) -> torch.Tensor:
    """
    Create feature tensor for address nodes.
    
    Features: latitude, longitude
    
    Args:
        df_address: Address DataFrame
        
    Returns:
        torch.Tensor of shape (n_addresses, 2) with lat/lon features
    """
    df_address = df_address.with_columns([
        pl.col("latitude").fill_null(0.0),
        pl.col("longitude").fill_null(0.0)
    ])
    addr_feats = df_address.select(["latitude", "longitude"]).to_numpy()
    return torch.from_numpy(addr_feats).float()


def create_entity_features(n_entities: int) -> torch.Tensor:
    """
    Create simple dummy features for entity nodes (IP, Email, Phone).
    
    These are simple identifier nodes with no rich features.
    Uses ones as placeholder features.
    
    Args:
        n_entities: Number of entities
        
    Returns:
        torch.Tensor of shape (n_entities, 1) with dummy features
    """
    return torch.ones(n_entities, 1)


def create_all_node_features(
    df_listing: pl.DataFrame,
    df_user: pl.DataFrame,
    df_address: pl.DataFrame,
    n_ip: int,
    n_email: int,
    n_phone: int
) -> dict:
    """
    Create features for all node types.
    
    Convenience function to generate all node features at once.
    
    Args:
        df_listing: Listing DataFrame
        df_user: User DataFrame
        df_address: Address DataFrame
        n_ip: Number of IP nodes
        n_email: Number of email nodes
        n_phone: Number of phone nodes
        
    Returns:
        Dictionary with features for each node type:
        {
            'listing': {'x': Tensor, 'y': Tensor, 'timestamp': Tensor},
            'user': {'x': Tensor},
            'address': {'x': Tensor},
            'ip': {'x': Tensor},
            'email': {'x': Tensor},
            'phone': {'x': Tensor}
        }
    """
    logger.info("Creating features for all node types...")
    
    # Listing features (rich features)
    listing_x, listing_y, listing_t = create_listing_features(df_listing)
    
    # Other node features
    user_x = create_user_features(df_user)
    address_x = create_address_features(df_address)
    ip_x = create_entity_features(n_ip)
    email_x = create_entity_features(n_email)
    phone_x = create_entity_features(n_phone)
    
    return {
        'listing': {'x': listing_x, 'y': listing_y, 'timestamp': listing_t},
        'user': {'x': user_x},
        'address': {'x': address_x},
        'ip': {'x': ip_x},
        'email': {'x': email_x},
        'phone': {'x': phone_x}
    }

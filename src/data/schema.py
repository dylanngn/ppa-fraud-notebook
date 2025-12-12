"""
Data schemas and validation for the fraud detection pipeline.
Single source of truth for feature definitions.
UPDATED: Based on EDA of artifacts/raw_insertions.parquet.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class ModelVariant(Enum):
    VANILLA_XGBOOST = "vanilla_xgboost"
    HANDCRAFTED_XGBOOST = "handcrafted_xgboost"
    GRAPHSAGE_XGBOOST = "graphsage_xgboost"


@dataclass(frozen=True)
class FeatureSchema:
    """
    Immutable feature schema - single source of truth.
    """
    
    # === IDENTIFIERS ===
    # Using specific flattened fields found in EDA
    id_columns: Tuple[str, ...] = (
        "listing_id",
        "user_id", 
        "submission_at", # REPLACES created_at
    )
    
    # === GRAPH IDENTITY COLUMNS ===
    graph_identity_columns: Tuple[str, ...] = (
        "listing.lister.phone.hash",       # Contact Phone (70% coverage)
        "listing.lister.email.hash",       # Contact Email (99% coverage)
        "listing.lister.billing.phoneDay.hash", # Billing Phone (98% coverage) - NEW
        "listing.lister.billing.email.hash",    # Billing Email (98% coverage) - NEW
        "user_ip_address_hash",
    )
    
    # === RAW SOURCE COLUMNS (Used for Coalescence) ===
    # We define them here to inform loading, but the Model sees 'features'
    raw_price_cols: Tuple[str, ...] = (
        "listing.prices.rent.gross",
        "listing.prices.rent.net",
        "listing.prices.buy.price"
    )
    
    raw_area_cols: Tuple[str, ...] = (
        "listing.prices.rent.area",
        "listing.prices.buy.area", 
        "listing.characteristics.lotSize"
    )
    
    # === ENGINEERED FEATURES (Post-Coalescence) ===
    # These are the columns the Model will actually use
    base_numerical: Tuple[str, ...] = (
        "feature_price",      # Coalesced
        "feature_area",       # Coalesced
        "feature_rooms",      # listing.characteristics.numberOfRooms
        "feature_bathrooms",  # listing.characteristics.numberOfBathrooms
        "feature_year_built", # listing.characteristics.yearBuilt
        "feature_floors",     # listing.characteristics.numberOfFloors
    )
    
    # === BENCHMARKS (Not for Training) ===
    benchmark_features: Tuple[str, ...] = (
        "benchmark_seon_approved", # Production Binary Baseline (Approved=True means Legit)
    )
    
    # === BASE CATEGORICAL FEATURES ===
    base_categorical: Tuple[str, ...] = (
        "listing.lister.billing.address.city_hash",
        "listing.lister.billing.phoneDay.area_hash", # TOP PREDICTOR
        "listing.lister.email.domain_hash",          # TOP PREDICTOR
        "listing.type",
        "bundle.tier",
        "listing.platforms", # Size/list? Might need encoding or size extraction.
    )
    
    # === TEMPORAL FEATURES (Derived) ===
    temporal_features: Tuple[str, ...] = (
        "posting_hour",
        "posting_weekday",
    )
    
    # === TARGET ===
    target: str = "is_fraud" # Derived from fraud_flag
    raw_target_source: str = "fraud_flag" # Timestamp column
    
    # === GNN EMBEDDING ===
    gnn_embedding_dim: int = 64
    
    # === HANDCRAFTED ===
    handcrafted_graph: Tuple[str, ...] = (
        "degree_total",
        "neighbor_fraud_rate",
    )

    @property
    def all_base_features(self) -> Tuple[str, ...]:
        return self.base_numerical + self.base_categorical + self.temporal_features
    
    def get_features_for_variant(self, variant: ModelVariant) -> List[str]:
        base = list(self.all_base_features)
        if variant == ModelVariant.VANILLA_XGBOOST:
            return base
        elif variant == ModelVariant.HANDCRAFTED_XGBOOST:
            return base + list(self.handcrafted_graph)
        elif variant == ModelVariant.GRAPHSAGE_XGBOOST:
            return base + [f"gnn_emb_{i}" for i in range(self.gnn_embedding_dim)]
        return base
        
    def get_gnn_input_features(self) -> List[str]:
        return list(self.base_numerical)

FEATURE_SCHEMA = FeatureSchema()


@dataclass
class TemporalBoundary:
    cutoff_date: str
    split_name: str
    accumulation_id: Optional[str] = None

@dataclass 
class DataSplit:
    train: TemporalBoundary
    val: TemporalBoundary
    test: TemporalBoundary
    gap_days: int = 7
    
    def validate(self) -> bool:
        from datetime import datetime
        t = datetime.fromisoformat
        if not (t(self.train.cutoff_date) < t(self.val.cutoff_date) < t(self.test.cutoff_date)):
            raise ValueError("Invalid split ordering")
        return True

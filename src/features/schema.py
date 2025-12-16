"""
Data schemas and validation for the fraud detection pipeline.
Single source of truth for feature definitions.
"""

from dataclasses import dataclass
from typing import List, Tuple
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
    id_columns: Tuple[str, ...] = (
        "listing_id",
        "user_id", 
        "submission_at",
    )
    
    # === GRAPH IDENTITY COLUMNS ===
    graph_identity_columns: Tuple[str, ...] = (
        "listing.lister.phone.hash",
        "listing.lister.email.hash",
        "listing.lister.billing.phoneDay.hash",
        "listing.lister.billing.email.hash",
        "user_ip_address_hash",
    )
    
    # === RAW SOURCE COLUMNS ===
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
    
    # === ENGINEERED FEATURES ===
    base_numerical: Tuple[str, ...] = (
        "feature_price",
        "feature_area",
        "feature_rooms",
        "feature_bathrooms",
        "feature_year_built",
        "feature_floors",
    )
    
    # === BENCHMARKS ===
    benchmark_features: Tuple[str, ...] = (
        "benchmark_seon_approved",
    )
    
    # === BASE CATEGORICAL FEATURES ===
    base_categorical: Tuple[str, ...] = (
        "listing.lister.billing.address.city_hash",
        "listing.lister.billing.phoneDay.area_hash",
        "listing.lister.email.domain_hash",
        "listing.type",
        "bundle.tier",
        "listing.platforms",
    )
    
    # === TEMPORAL FEATURES (Derived) ===
    temporal_features: Tuple[str, ...] = (
        "posting_hour",
        "posting_weekday",
    )
    
    # === TARGET ===
    target: str = "is_fraud"
    raw_target_source: str = "fraud_flag"
    
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

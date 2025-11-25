"""
Feature Store with Temporal Validation

Prevents time-travel trap by ensuring features use only historical data
available at the time of prediction.
"""
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import polars as pl
import pickle
from dataclasses import dataclass


@dataclass
class FeatureSnapshot:
    """Features as they existed at a specific point in time."""
    listing_id: int
    as_of_time: datetime
    features: Dict[str, float]
    is_cold_start: bool
    confidence: float


class FeatureStore:
    """
    Manages feature retrieval with temporal consistency guarantees.
    
    For graph features, this ensures we only use edges/data that existed
    BEFORE the listing's submission time.
    """
    
    def __init__(self, artifacts_dir: Path = Path("artifacts")):
        self.artifacts_dir = artifacts_dir
        self.listing_nodes_path = artifacts_dir / "nodes_listing.parquet"
        self.graph_features_path = artifacts_dir / "listing_graph_features.parquet"
        self.advanced_features_path = artifacts_dir / "listing_advanced_features.parquet"
        
        # Load static data
        self._load_data()
        
    def _load_data(self):
        """Load parquet files into memory for fast access."""
        if self.listing_nodes_path.exists():
            self.listing_nodes = pl.read_parquet(self.listing_nodes_path)
        else:
            raise FileNotFoundError(f"Listing nodes not found: {self.listing_nodes_path}")
            
        # Graph features (optional)
        self.graph_features = None
        if self.graph_features_path.exists():
            self.graph_features = pl.read_parquet(self.graph_features_path)
            
        # Advanced features (optional)
        self.advanced_features = None
        if self.advanced_features_path.exists():
            self.advanced_features = pl.read_parquet(self.advanced_features_path)
    
    def get_features(
        self, 
        listing_id: int, 
        as_of_time: Optional[datetime] = None
    ) -> FeatureSnapshot:
        """
        Retrieve features for a listing as they existed at as_of_time.
        
        Args:
            listing_id: Listing to get features for
            as_of_time: Point in time to retrieve features (defaults to now)
            
        Returns:
            FeatureSnapshot with temporal guarantees
            
        Raises:
            ValueError: If listing not found or temporal violation detected
        """
        if as_of_time is None:
            as_of_time = datetime.now()
            
        # Get listing metadata
        listing = self.listing_nodes.filter(pl.col("insertion_id") == listing_id)
        
        if listing.is_empty():
            raise ValueError(f"Listing {listing_id} not found")
            
        listing_row = listing.row(0, named=True)
        submission_time = listing_row.get("submission_at")
        
        # Temporal validation: You cannot get features "as of" a time BEFORE the listing existed
        if submission_time and as_of_time < submission_time:
            raise ValueError(
                f"Time travel violation: as_of_time ({as_of_time}) is before "
                f"listing submission ({submission_time})"
            )
        
        # Build feature dict
        features = self._extract_tabular_features(listing_row)
        
        # Add graph features if available
        is_cold_start = False
        if self.graph_features is not None:
            graph_feats = self.graph_features.filter(
                pl.col("insertion_id") == listing_id
            )
            if not graph_feats.is_empty():
                features.update(self._extract_graph_features(graph_feats.row(0, named=True)))
            else:
                is_cold_start = True
                
        # Add advanced features if available
        if self.advanced_features is not None:
            adv_feats = self.advanced_features.filter(
                pl.col("insertion_id") == listing_id
            )
            if not adv_feats.is_empty():
                features.update(self._extract_advanced_features(adv_feats.row(0, named=True)))
        
        # Calculate confidence based on cold start status
        confidence = 0.5 if is_cold_start else 1.0
        
        return FeatureSnapshot(
            listing_id=listing_id,
            as_of_time=as_of_time,
            features=features,
            is_cold_start=is_cold_start,
            confidence=confidence
        )
    
    def _extract_tabular_features(self, listing: Dict) -> Dict[str, float]:
        """Extract base tabular features."""
        features = {}
        
        # Account age
        if listing.get("submission_at") and listing.get("account_created_at"):
            account_age = (listing["submission_at"] - listing["account_created_at"]).total_seconds() / 86400
            features["account_age_days"] = account_age
        else:
            features["account_age_days"] = 0.0
            
        # Price
        import numpy as np
        features["log_price"] = np.log1p(listing.get("price_rent_gross", 0) or 0)
        
        # Numeric features
        features["living_space"] = listing.get("living_space", 0) or 0
        features["rooms"] = listing.get("rooms", 0) or 0
        
        # Boolean features
        features["is_new"] = int(listing.get("is_new", False))
        features["has_balcony"] = int(listing.get("has_balcony", False))
        features["has_elevator"] = int(listing.get("has_elevator", False))
        features["has_parking"] = int(listing.get("has_parking", False))
        
        # Bundle
        features["bundle_period"] = listing.get("bundle_period", 7)
        bundle_tier = listing.get("bundle_tier", "basic")
        tier_map = {"basic": 0, "premium": 1, "top": 2}
        features["bundle_tier_score"] = tier_map.get(bundle_tier.lower() if bundle_tier else "basic", 0)
        
        # Payment/Offer type
        features["is_direct_payment"] = int(listing.get("payment_type") == "DIRECT")
        features["is_buy"] = int(listing.get("offer_type") == "BUY")
        
        # Location
        features["latitude"] = listing.get("latitude", 0.0) or 0.0
        features["longitude"] = listing.get("longitude", 0.0) or 0.0
        
        return features
    
    def _extract_graph_features(self, graph_row: Dict) -> Dict[str, float]:
        """Extract graph-derived features."""
        return {
            "contact_email_count": graph_row.get("contact_email_count", 0) or 0,
            "shared_contact_email_count": graph_row.get("shared_contact_email_count", 0) or 0,
            "max_shared_contact_email": graph_row.get("max_shared_contact_email", 0) or 0,
            "contact_phone_count": graph_row.get("contact_phone_count", 0) or 0,
            "shared_contact_phone_count": graph_row.get("shared_contact_phone_count", 0) or 0,
            "max_shared_contact_phone": graph_row.get("max_shared_contact_phone", 0) or 0,
            "user_listing_count": graph_row.get("user_listing_count", 0) or 0,
            "user_unique_ip_count": graph_row.get("user_unique_ip_count", 0) or 0,
            "shared_ip_user_count": graph_row.get("shared_ip_user_count", 0) or 0,
            "max_shared_ip_users": graph_row.get("max_shared_ip_users", 0) or 0,
            "listing_component_size": graph_row.get("listing_component_size", 0) or 0,
            "listing_pagerank": graph_row.get("listing_pagerank", 0.0) or 0.0,
        }
    
    def _extract_advanced_features(self, adv_row: Dict) -> Dict[str, float]:
        """Extract advanced graph features."""
        return {
            "degree_total": adv_row.get("degree_total", 0) or 0,
            "is_isolated": adv_row.get("is_isolated", 0) or 0,
            "unique_identifier_count": adv_row.get("unique_identifier_count", 0) or 0,
            "neighbor_overlap_score": adv_row.get("neighbor_overlap_score", 0) or 0,
            "avg_neighbor_degree": adv_row.get("avg_neighbor_degree", 0.0) or 0.0,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names for model input."""
        base_features = [
            "account_age_days", "log_price", "living_space", "rooms",
            "is_new", "has_balcony", "has_elevator", "has_parking",
            "bundle_period", "bundle_tier_score",
            "is_direct_payment", "is_buy",
            "latitude", "longitude"
        ]
        
        graph_features = [
            "contact_email_count", "shared_contact_email_count", "max_shared_contact_email",
            "contact_phone_count", "shared_contact_phone_count", "max_shared_contact_phone",
            "user_listing_count", "user_unique_ip_count", "shared_ip_user_count",
            "max_shared_ip_users", "listing_component_size", "listing_pagerank"
        ]
        
        advanced_features = [
            "degree_total", "is_isolated", "unique_identifier_count",
            "neighbor_overlap_score", "avg_neighbor_degree"
        ]
        
        # Only include graph/advanced features if they exist
        all_features = base_features.copy()
        if self.graph_features is not None:
            all_features.extend(graph_features)
        if self.advanced_features is not None:
            all_features.extend(advanced_features)
            
        return all_features

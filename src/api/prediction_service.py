"""
Prediction Service

Handles model loading, inference, and caching.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import pickle
import numpy as np
from dataclasses import dataclass

from src.features.store import FeatureStore, FeatureSnapshot


@dataclass
class PredictionResult:
    """Prediction response."""
    listing_id: int
    fraud_score: float
    fraud_probability: float
    is_cold_start: bool
    confidence: float
    model_version: str
    timestamp: datetime


class PredictionService:
    """
    Main prediction service that orchestrates feature retrieval and inference.
    """
    
    def __init__(
        self,
        model_dir: Path = Path("artifacts/models/baseline_graph"),
        feature_store: Optional[FeatureStore] = None
    ):
        self.model_dir = model_dir
        self.feature_store = feature_store or FeatureStore()
        
        # Load the latest model
        self.model, self.model_version = self._load_latest_model()
        self.feature_names = self.feature_store.get_feature_names()
        
    def _load_latest_model(self) -> Tuple:
        """Load the most recent trained model."""
        model_files = sorted(self.model_dir.glob("model_window_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No models found in {self.model_dir}")
            
        latest_model_path = model_files[-1]
        
        with open(latest_model_path, 'rb') as f:
            model_bundle = pickle.load(f)
            
        model = model_bundle['model']
        window_info = model_bundle.get('window_info', {})
        version = f"window_{window_info.get('window_idx', 'unknown')}"
        
        print(f"[PredictionService] Loaded model: {latest_model_path.name} (version: {version})")
        
        return model, version
    
    def predict(
        self,
        listing_id: int,
        as_of_time: Optional[datetime] = None
    ) -> PredictionResult:
        """
        Make a fraud prediction for a listing.
        
        Args:
            listing_id: Listing to predict
            as_of_time: Time to retrieve features (for temporal consistency)
            
        Returns:
            PredictionResult with score and metadata
        """
        # 1. Get features (with temporal validation)
        feature_snapshot = self.feature_store.get_features(listing_id, as_of_time)
        
        # 2. Prepare feature vector
        X = self._prepare_features(feature_snapshot.features)
        
        # 3. Handle cold start
        if feature_snapshot.is_cold_start:
            # Use model but flag low confidence
            fraud_prob = self.model.predict_proba(X)[0, 1]
            fraud_score = fraud_prob
            confidence = 0.5  # Low confidence for cold start
        else:
            # Normal prediction
            fraud_prob = self.model.predict_proba(X)[0, 1]
            fraud_score = fraud_prob
            confidence = feature_snapshot.confidence
            
        return PredictionResult(
            listing_id=listing_id,
            fraud_score=fraud_score,
            fraud_probability=fraud_prob,
            is_cold_start=feature_snapshot.is_cold_start,
            confidence=confidence,
            model_version=self.model_version,
            timestamp=datetime.now()
        )
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in correct order."""
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(features.get(name, 0.0))
        return np.array([feature_vector])
    
    def get_model_info(self) -> Dict:
        """Return model metadata."""
        return {
            "model_version": self.model_version,
            "model_dir": str(self.model_dir),
            "feature_count": len(self.feature_names),
            "features": self.feature_names
        }

"""
Abstract base classes for model components.
Ensures consistent interface across all model variants.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData
import polars as pl
from src.data.schema import DataSplit

class BaseEmbedder(ABC):
    """Abstract base for embedding generators (GNN or handcrafted)."""
    
    @abstractmethod
    def fit(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> "BaseEmbedder":
        """Fit the embedder on training data."""
        pass
    
    @abstractmethod
    def transform(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        """Generate embeddings for all nodes."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save embedder to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseEmbedder":
        """Load embedder from disk."""
        pass


class BaseClassifier(ABC):
    """Abstract base for final classifier."""
    
    @abstractmethod
    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> "BaseClassifier":
        """Fit the classifier."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict fraud probability."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save classifier to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load classifier from disk."""
        pass


class HybridFraudDetector(ABC):
    """Abstract base for complete fraud detection pipeline."""
    
    @abstractmethod
    def fit(
        self,
        df: pl.DataFrame,
        graph: HeteroData,
        split: DataSplit,
    ) -> "HybridFraudDetector":
        """Fit the complete pipeline."""
        pass

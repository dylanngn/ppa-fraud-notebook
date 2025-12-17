"""
Hybrid GNN + XGBoost Pipeline for Fraud Detection.

Orchestrates the flow: Data -> Heterogeneous Graph -> GNN Embeddings -> Classifier

Model Variants:
1. VANILLA_XGBOOST: Tabular features only (baseline)
2. GNN_XGBOOST: Tabular + GNN embeddings from heterogeneous graph

Key Design Principles:
- Temporal integrity: All graph operations respect time ordering
- Heterogeneous GNN: Multiple node types (listing, device, IP, user, email_domain)
- GNN embeddings capture multi-hop fraud patterns
"""

from typing import Optional, Tuple, Dict, Any, List
import logging

import polars as pl
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.features.schema import FEATURE_SCHEMA, ModelVariant
from src.features.graph_builder import HeterogeneousGraphBuilder, HeteroGraphConfig
from src.models.base import HybridFraudDetector, BaseEmbedder, BaseClassifier

logger = logging.getLogger(__name__)


class HybridPipeline(HybridFraudDetector):
    """
    Hybrid GNN+XGBoost pipeline for fraud detection.
    
    Two variants:
    - VANILLA: classifier only (baseline)
    - GNN: graph_builder + GNN embedder + classifier
    
    Usage:
        pipeline = HybridPipeline(
            variant=ModelVariant.GNN_XGBOOST,
            graph_builder=HeterogeneousGraphBuilder(config),
            embedder=HeteroGNNEmbedder(...),
            classifier=XGBoostClassifier(...),
        )
        pipeline.fit(train_df, train_graph)
        probs = pipeline.predict(test_df, inference_graph)
    """
    
    def __init__(
        self,
        variant: ModelVariant,
        graph_builder: Optional[HeterogeneousGraphBuilder] = None,
        embedder: Optional[BaseEmbedder] = None,
        classifier: BaseClassifier = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            variant: Model variant (VANILLA_XGBOOST or GNN_XGBOOST)
            graph_builder: For graph construction (required for GNN variant)
            embedder: GNN embedder (required for GNN variant)
            classifier: XGBoost classifier (required for all variants)
        """
        self.variant = variant
        self.graph_builder = graph_builder
        self.embedder = embedder
        self.classifier = classifier
        
        # Categorical encoding state
        self.train_categoricals: Dict[str, pd.CategoricalDtype] = {}
        self.input_example_ = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate that required components are provided."""
        if self.variant == ModelVariant.GNN_XGBOOST:
            if not (self.graph_builder and self.embedder and self.classifier):
                raise ValueError(
                    "GNN variant requires graph_builder, embedder, and classifier"
                )
        elif self.variant == ModelVariant.VANILLA_XGBOOST:
            if not self.classifier:
                raise ValueError("Vanilla variant requires classifier")
    
    def fit(
        self,
        train_df: pl.DataFrame,
        graph: Optional[HeteroData] = None,
    ) -> "HybridPipeline":
        """
        Fit the pipeline on training data.
        
        Args:
            train_df: Training data with features and labels
            graph: Training graph (for GNN variant)
            
        Returns:
            Fitted pipeline
        """
        logger.info(f"Fitting {self.variant.value} pipeline on {len(train_df)} samples")
        
        # Prepare GNN embeddings if using GNN variant
        train_embeddings = None
        
        if self.variant == ModelVariant.GNN_XGBOOST:
            if graph is None:
                raise ValueError("Graph required for GNN variant")
            train_embeddings = self._fit_gnn(train_df, graph)
        
        # Prepare base features
        X_train = self._prepare_base_features(train_df)
        y_train = train_df[FEATURE_SCHEMA.target].to_numpy()
        
        # Combine with GNN embeddings
        if train_embeddings is not None:
            emb_df = self._embeddings_to_dataframe(train_embeddings, X_train.index)
            X_train = pd.concat([X_train, emb_df], axis=1)
        
        # Store example for MLflow logging
        sample = X_train.iloc[[0]].copy()
        int_cols = sample.select_dtypes(include='integer').columns
        sample[int_cols] = sample[int_cols].astype('float64')
        self.input_example_ = sample
        
        # Fit classifier
        self.classifier.fit(X_train, y_train)
        
        logger.info(f"Pipeline fitted with {X_train.shape[1]} features")
        return self
    
    def _fit_gnn(
        self,
        train_df: pl.DataFrame,
        graph: HeteroData,
    ) -> np.ndarray:
        """Fit GNN embedder and get embeddings for training nodes."""
        # Get node features for listings
        listing_features = self._get_listing_features(train_df)
        
        # Fit GNN embedder
        self.embedder.fit(graph, listing_features, train_mask=None)
        
        # Get embeddings
        embeddings = self.embedder.transform(graph, listing_features)
        
        return embeddings
    
    def _get_listing_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract listing features as tensor for GNN."""
        gnn_cols = FEATURE_SCHEMA.get_gnn_input_features()
        available_cols = [c for c in gnn_cols if c in df.columns]
        
        if not available_cols:
            logger.warning("No GNN input features found, using zeros")
            return torch.zeros((len(df), 1), dtype=torch.float32)
        
        x_np = df.select(available_cols).to_numpy()
        x_tensor = torch.tensor(x_np, dtype=torch.float32)
        
        # Handle NaN with column medians
        if torch.isnan(x_tensor).any():
            for j in range(x_tensor.size(1)):
                col = x_tensor[:, j]
                mask = torch.isnan(col)
                if mask.any():
                    median = torch.nanmedian(col)
                    if torch.isnan(median):
                        median = torch.tensor(0.0)
                    x_tensor[mask, j] = median
        
        return x_tensor
    
    def _prepare_base_features(
        self,
        df: pl.DataFrame,
    ) -> pd.DataFrame:
        """Prepare base tabular features as pandas DataFrame."""
        base_cols = list(FEATURE_SCHEMA.all_base_features)
        available_cols = [c for c in base_cols if c in df.columns]
        
        X = df.select(available_cols).to_pandas()
        
        # Convert boolean columns to int8
        bool_cols = list(FEATURE_SCHEMA.all_seon_boolean)
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].astype("Int8")  # nullable int
        
        # Handle categorical columns (string types only)
        cat_cols = list(FEATURE_SCHEMA.all_categorical_features)
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")
                self.train_categoricals[col] = X[col].cat.categories
        
        return X
    
    def _embeddings_to_dataframe(
        self,
        embeddings: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        """Convert GNN embeddings to pandas DataFrame."""
        num_dims = embeddings.shape[1]
        col_names = [f"gnn_emb_{i}" for i in range(num_dims)]
        return pd.DataFrame(embeddings, columns=col_names, index=index)
    
    def predict(
        self,
        df: pl.DataFrame,
        graph: Optional[HeteroData] = None,
    ) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            df: Data to predict on
            graph: Inference graph (includes train + test nodes for GNN)
            
        Returns:
            Array of fraud probabilities
        """
        # Prepare base features
        X_base = self._prepare_base_features_for_prediction(df)
        
        # Get GNN embeddings if using GNN variant
        if self.variant == ModelVariant.GNN_XGBOOST:
            if graph is None:
                raise ValueError("Graph required for GNN prediction")
            
            embeddings = self._predict_gnn(df, graph)
            emb_df = self._embeddings_to_dataframe(embeddings, X_base.index)
            X = pd.concat([X_base, emb_df], axis=1)
        else:
            X = X_base
        
        # Predict
        return self.classifier.predict_proba(X)
    
    def _prepare_base_features_for_prediction(
        self,
        df: pl.DataFrame,
    ) -> pd.DataFrame:
        """Prepare base features for prediction (apply train categorical encoding)."""
        base_cols = list(FEATURE_SCHEMA.all_base_features)
        available_cols = [c for c in base_cols if c in df.columns]
        
        X = df.select(available_cols).to_pandas()
        
        # Convert boolean columns to int8
        bool_cols = list(FEATURE_SCHEMA.all_seon_boolean)
        for col in bool_cols:
            if col in X.columns:
                X[col] = X[col].astype("Int8")
        
        # Apply categorical encoding from training
        cat_cols = list(FEATURE_SCHEMA.all_categorical_features)
        for col in cat_cols:
            if col in X.columns and col in self.train_categoricals:
                known_cats = self.train_categoricals[col]
                X[col] = X[col].astype(
                    pd.CategoricalDtype(categories=known_cats, ordered=False)
                )
        
        return X
    
    def _predict_gnn(
        self,
        df: pl.DataFrame,
        graph: HeteroData,
    ) -> np.ndarray:
        """Get GNN embeddings for prediction (model is frozen)."""
        listing_features = self._get_listing_features(df)
        return self.embedder.transform(graph, listing_features)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the classifier."""
        if hasattr(self.classifier, 'model') and hasattr(self.classifier.model, 'feature_importances_'):
            importances = self.classifier.model.feature_importances_
            
            if self.input_example_ is not None:
                feature_names = list(self.input_example_.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        
        return None
    
    def get_gnn_embedding_importance(self) -> Optional[pd.DataFrame]:
        """
        Get importance of GNN embedding dimensions.
        
        Useful for understanding which graph patterns contribute most.
        """
        importance_df = self.get_feature_importance()
        if importance_df is None:
            return None
        
        # Filter to GNN embedding features
        gnn_features = importance_df[
            importance_df['feature'].str.startswith('gnn_emb_')
        ]
        
        return gnn_features

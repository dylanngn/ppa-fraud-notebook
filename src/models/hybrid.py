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
        elif self.variant in (
            ModelVariant.VANILLA_XGBOOST,
            ModelVariant.LOGISTIC_REGRESSION,
            ModelVariant.RANDOM_FOREST,
        ):
            if not self.classifier:
                raise ValueError(f"{self.variant.value} requires classifier")
    
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
        """
        Fit GNN embedder and get embeddings for training nodes.
        
        Uses the node features pre-computed by graph_builder (graph["listing"].x)
        and maps embeddings back to DataFrame rows.
        """
        # Use node features from graph (already built by graph_builder)
        listing_features = graph["listing"].x
        
        # Fit GNN embedder
        self.embedder.fit(graph, listing_features, train_mask=None)
        
        # Get embeddings for all nodes
        node_embeddings = self.embedder.transform(graph, listing_features)
        
        # Map node embeddings back to DataFrame rows using graph's stored mappings
        return self._map_embeddings_to_rows(train_df, graph, node_embeddings)
    
    def _map_embeddings_to_rows(
        self,
        df: pl.DataFrame,
        graph: HeteroData,
        node_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Map graph node embeddings back to DataFrame rows."""
        listing_id_col = FEATURE_SCHEMA.temporal_config.insertion_id_column
        
        # Use node mappings stored in graph (set by graph_builder.build_graph)
        node_mapping = getattr(graph, 'node_mappings', {}).get("listing", {})
        
        if not node_mapping:
            logger.warning("No node mapping found in graph, using zeros")
            return np.zeros((len(df), node_embeddings.shape[1]), dtype=np.float32)
        
        listing_ids = df[listing_id_col].to_list()
        row_embeddings = []
        
        default_embedding = np.zeros(node_embeddings.shape[1], dtype=np.float32)
        
        for lid in listing_ids:
            if lid in node_mapping:
                node_idx = node_mapping[lid]
                if node_idx < len(node_embeddings):
                    row_embeddings.append(node_embeddings[node_idx])
                else:
                    row_embeddings.append(default_embedding)
            else:
                # Listing not in graph (new listing in test), use zeros
                row_embeddings.append(default_embedding)
        
        return np.array(row_embeddings, dtype=np.float32)
    
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
        
        # Store for potential SHAP analysis
        self._last_prediction_features = X
        
        # Predict
        return self.classifier.predict_proba(X)
    
    def get_prediction_features(
        self,
        df: pl.DataFrame,
        graph: Optional[HeteroData] = None,
    ) -> pd.DataFrame:
        """
        Get the feature DataFrame used for prediction (for SHAP analysis).
        
        Args:
            df: Data to prepare
            graph: Inference graph (for GNN variant)
            
        Returns:
            Prepared feature DataFrame matching classifier input
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
        
        return X
    
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
        """
        Get GNN embeddings for prediction (model is frozen).
        
        Maps graph node embeddings back to DataFrame rows.
        Each row gets the embedding of its corresponding listing node.
        """
        # Get embeddings for all nodes in the graph
        # graph["listing"].x already has features from graph_builder
        node_embeddings = self.embedder.transform(graph, graph["listing"].x)
        
        # Map node embeddings back to DataFrame rows using graph's stored mappings
        return self._map_embeddings_to_rows(df, graph, node_embeddings)
    
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

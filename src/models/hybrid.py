"""
Hybrid GNN + XGBoost Pipeline.
Orchestrates the flow: Data -> Graph -> Embeddings -> Classifier.
"""

from typing import Optional
import polars as pl
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data.schema import FEATURE_SCHEMA, ModelVariant
from src.graph.builder import TemporalGraphBuilder
from src.graph.features import HandcraftedGraphFeatures
from src.models.base import HybridFraudDetector, BaseEmbedder, BaseClassifier

class HybridPipeline(HybridFraudDetector):
    """
    Implementation of the hybrid pipeline.
    Suitable for all 3 variants (by enabling/disabling components).
    """
    
    def __init__(
        self,
        variant: ModelVariant,
        graph_builder: Optional[TemporalGraphBuilder] = None,
        embedder: Optional[BaseEmbedder] = None,
        classifier: BaseClassifier = None,
        handcrafted_features: Optional[HandcraftedGraphFeatures] = None
    ):
        self.variant = variant
        self.graph_builder = graph_builder
        self.embedder = embedder
        self.classifier = classifier
        self.handcrafted_features = handcrafted_features
        self.train_categoricals = {}
        self.input_example_ = None
        self._train_labels = None
        self._train_node_count = None
        
        if variant == ModelVariant.GRAPHSAGE_XGBOOST:
            if not (graph_builder and embedder and classifier):
                raise ValueError("GraphSAGE variant requires builder, embedder, and classifier")
        elif variant == ModelVariant.HANDCRAFTED_XGBOOST:
            if not (graph_builder and handcrafted_features and classifier):
                raise ValueError("Handcrafted variant requires builder, feature extractor, and classifier")
        elif variant == ModelVariant.VANILLA_XGBOOST:
            if not classifier:
                raise ValueError("Vanilla variant requires classifier")
    
    def fit(
        self,
        train_df: pl.DataFrame,
        graph: Optional[HeteroData],
    ) -> "HybridPipeline":
        
        train_extra_features = None
        
        if self.variant == ModelVariant.GRAPHSAGE_XGBOOST:
            gnn_cols = FEATURE_SCHEMA.get_gnn_input_features()
            x_np = train_df.select(gnn_cols).to_numpy()
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            
            if torch.isnan(x_tensor).any():
                col_medians = torch.nanmedian(x_tensor, dim=0).values
                nan_mask = torch.isnan(x_tensor)
                x_tensor = torch.where(nan_mask, col_medians.unsqueeze(0).expand_as(x_tensor), x_tensor)
                x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
            
            self.embedder.fit(graph, x_tensor, train_mask=None)
            embeddings = self.embedder.transform(graph, x_tensor)
            train_extra_features = embeddings
            
        elif self.variant == ModelVariant.HANDCRAFTED_XGBOOST:
            y_train = train_df[FEATURE_SCHEMA.target].to_numpy()
            train_mask = np.ones(len(train_df), dtype=bool) 
            
            features_pl = self.handcrafted_features.compute_features(graph, y_train, train_mask)
            train_extra_features = features_pl.to_numpy()
            
            self._train_labels = y_train.copy()
            self._train_node_count = len(train_df)
            
        base_cols = FEATURE_SCHEMA.all_base_features
        X_base = train_df.select(base_cols).to_pandas()
        
        import pandas as pd
        for col in FEATURE_SCHEMA.base_categorical:
            if col in X_base.columns:
                X_base[col] = X_base[col].astype("category")
                self.train_categoricals[col] = X_base[col].cat.categories
        
        y_train = train_df[FEATURE_SCHEMA.target].to_numpy()
        
        if train_extra_features is not None:
             import pandas as pd
             if isinstance(train_extra_features, np.ndarray):
                 extra_cols = [f"extra_{i}" for i in range(train_extra_features.shape[1])]
                 df_extra = pd.DataFrame(train_extra_features, columns=extra_cols, index=X_base.index)
             else:
                 df_extra = train_extra_features
             X_train = pd.concat([X_base, df_extra], axis=1)
        else:
            X_train = X_base
        
        sample = X_train.iloc[[0]].copy()
        int_cols = sample.select_dtypes(include='integer').columns
        sample[int_cols] = sample[int_cols].astype('float64')
        self.input_example_ = sample
        
        self.classifier.fit(X_train, y_train)
        
        return self

    def predict(
        self,
        df: pl.DataFrame,
        graph: Optional[HeteroData]
    ) -> np.ndarray:
        base_cols = FEATURE_SCHEMA.all_base_features
        X_base = df.select(base_cols).to_pandas()
        
        import pandas as pd
        for col in FEATURE_SCHEMA.base_categorical:
            if col in X_base.columns and col in self.train_categoricals:
                known_cats = self.train_categoricals[col]
                X_base[col] = X_base[col].astype(pd.CategoricalDtype(categories=known_cats, ordered=False))
        
        if self.variant == ModelVariant.VANILLA_XGBOOST:
            X_test = X_base
        
        elif self.variant == ModelVariant.GRAPHSAGE_XGBOOST:
            gnn_cols = FEATURE_SCHEMA.get_gnn_input_features()
            x_np = df.select(gnn_cols).to_numpy()
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            
            if torch.isnan(x_tensor).any():
                col_medians = torch.nanmedian(x_tensor, dim=0).values
                nan_mask = torch.isnan(x_tensor)
                x_tensor = torch.where(nan_mask, col_medians.unsqueeze(0).expand_as(x_tensor), x_tensor)
                x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
            
            embeddings = self.embedder.transform(graph, x_tensor)
            
            extra_cols = [f"extra_{i}" for i in range(embeddings.shape[1])]
            df_extra = pd.DataFrame(embeddings, columns=extra_cols, index=X_base.index)
            X_test = pd.concat([X_base, df_extra], axis=1)
            
        elif self.variant == ModelVariant.HANDCRAFTED_XGBOOST:
            if self._train_labels is None or self._train_node_count is None:
                raise RuntimeError("Handcrafted model not fitted. Call fit() first.")
            
            num_nodes = graph["listing"].num_nodes
            full_labels = np.full(num_nodes, np.nan)
            full_labels[:self._train_node_count] = self._train_labels
            
            train_mask = np.zeros(num_nodes, dtype=bool)
            train_mask[:self._train_node_count] = True
            
            features_pl = self.handcrafted_features.compute_features(graph, full_labels, train_mask)
            extra_features = features_pl.to_numpy()
            
            extra_cols = [f"extra_{i}" for i in range(extra_features.shape[1])]
            df_extra = pd.DataFrame(extra_features, columns=extra_cols, index=X_base.index)
            X_test = pd.concat([X_base, df_extra], axis=1)
            
        else:
            raise ValueError("Unknown variant")
            
        return self.classifier.predict_proba(X_test)

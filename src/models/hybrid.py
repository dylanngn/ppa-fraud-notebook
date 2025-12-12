"""
Hybrid GNN + XGBoost Pipeline.
Orchestrates the flow: Data -> Graph -> Embeddings -> Classifier.
"""

from typing import Optional, Dict, List
import polars as pl
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data.schema import DataSplit, FEATURE_SCHEMA, ModelVariant
from src.graph.builder import TemporalGraphBuilder
from src.graph.features import HandcraftedGraphFeatures
from src.models.base import HybridFraudDetector, BaseEmbedder, BaseClassifier
from src.data.feature_store import FeatureStore

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
        self.train_categoricals = {} # To store allowed categories per column
        
        # Checking dependencies based on variant
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
        df: pl.DataFrame,
        graph: Optional[HeteroData], 
        split: DataSplit,
    ) -> "HybridPipeline":
        
        # 1. Prepare Data
        # Assume df contains TRAIN data only (from Orchestrator/accumulated.py refactor)
        train_df = df
        
        # 2. Generate Graph Features / Embeddings (Train Only)
        train_extra_features = None
        
        if self.variant == ModelVariant.GRAPHSAGE_XGBOOST:
            # Fit Embedder
            gnn_cols = FEATURE_SCHEMA.get_gnn_input_features()
            x_np = train_df.select(gnn_cols).to_numpy()
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            
            # HANDLE MISSING DATA FOR GNN: Feature Propagation
            # XGBoost can handle NaNs, but GraphSAGE cannot.
            # We impute using graph structure.
            if torch.isnan(x_tensor).any():
                from torch_geometric.transforms import FeaturePropagation
                # We need to temporarily modify the graph to apply propagation
                # Clone graph structure or work on reference if safe?
                # Propagation requires 'x' in graph.
                prop_data = graph.clone() 
                prop_data["listing"].x = x_tensor
                
                # Apply propagation
                # Note: FeaturePropagation infers missing_mask from NaNs in x
                propagator = FeaturePropagation(missing_mask=torch.isnan(x_tensor), num_iterations=40)
                prop_data = propagator(prop_data)
                x_tensor = prop_data["listing"].x
            
            self.embedder.fit(graph, x_tensor, train_mask=None)
            
            # Generate Embeddings (for Train)
            embeddings = self.embedder.transform(graph, x_tensor)
            train_extra_features = embeddings
            
        elif self.variant == ModelVariant.HANDCRAFTED_XGBOOST:
            # Compute handcrafted (using simplified fill for now or robust?)
            # Handcrafted features like Neighbor Mean need to handle NaNs in aggregation.
            # Our implementation might need checking. 
            # For now, let's assume nanmean in numpy or polars handles it.
            
            y_train = train_df[FEATURE_SCHEMA.target].to_numpy()
            train_mask = np.ones(len(train_df), dtype=bool) 
            
            features_pl = self.handcrafted_features.compute_features(
                graph, y_train, train_mask
            )
            train_extra_features = features_pl.to_numpy()
            
        # 3. Combine Features for Classifier
        # Convert to Pandas for XGBoost categorical support
        base_cols = FEATURE_SCHEMA.all_base_features
        X_base_pl = train_df.select(base_cols)
        X_base = X_base_pl.to_pandas()
        
        # Cast categoricals explicitly and STORE schema
        # XGBoost requires 'category' dtype, not object
        import pandas as pd
        for col in FEATURE_SCHEMA.base_categorical:
            if col in X_base.columns:
                # Let Pandas infer categories from Train data
                X_base[col] = X_base[col].astype("category")
                self.train_categoricals[col] = X_base[col].dtype.categories
        
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
            
        # 4. Train Classifier
        # XGBoost handles NaNs in X_train natively.
        self.classifier.fit(X_train, y_train)
        
        return self

    def predict(
        self,
        df: pl.DataFrame,
        graph: Optional[HeteroData] # Inference graph (extended)
    ) -> np.ndarray:
        
        # 1. Base Features
        base_cols = FEATURE_SCHEMA.all_base_features
        X_base_pl = df.select(base_cols)
        X_base = X_base_pl.to_pandas()
        
        # Cast categoricals ENFORCING Train Schema
        # Any value appearing in Test but not Train will be mapped to NaN
        # This prevents "XGBoostError: Found a category not in the training set"
        import pandas as pd
        for col in FEATURE_SCHEMA.base_categorical:
            if col in X_base.columns and col in self.train_categoricals:
                known_cats = self.train_categoricals[col]
                # Enforce known categories. Unknowns becomes NaN/Null.
                X_base[col] = X_base[col].astype(pd.CategoricalDtype(categories=known_cats, ordered=False))
        
        # 2. Extra Features
        import pandas as pd
        
        if self.variant == ModelVariant.VANILLA_XGBOOST:
            X_test = X_base
        
        elif self.variant == ModelVariant.GRAPHSAGE_XGBOOST:
            gnn_cols = FEATURE_SCHEMA.get_gnn_input_features()
            x_np = df.select(gnn_cols).to_numpy()
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            
            # HANDLE MISSING DATA FOR GNN INFERENCE
            if torch.isnan(x_tensor).any():
                from torch_geometric.transforms import FeaturePropagation
                prop_data = graph.clone()
                prop_data["listing"].x = x_tensor
                propagator = FeaturePropagation(missing_mask=torch.isnan(x_tensor), num_iterations=40)
                prop_data = propagator(prop_data)
                x_tensor = prop_data["listing"].x
            
            embeddings = self.embedder.transform(graph, x_tensor)
            
            extra_cols = [f"extra_{i}" for i in range(embeddings.shape[1])]
            df_extra = pd.DataFrame(embeddings, columns=extra_cols, index=X_base.index)
            X_test = pd.concat([X_base, df_extra], axis=1)
            
        elif self.variant == ModelVariant.HANDCRAFTED_XGBOOST:
            raise NotImplementedError("Handcrafted inference requires historical label state")
            
        else:
            raise ValueError("Unknown variant")
            
        return self.classifier.predict_proba(X_test)

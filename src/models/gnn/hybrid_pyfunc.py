"""
Example implementation of MLflow "Models from Code" pattern for hybrid GNN + XGBoost models.

This demonstrates how to package a complex hybrid model (GNN embeddings + XGBoost)
as a custom Python function model in MLflow, following the models-from-code pattern.

Reference: https://mlflow.org/docs/latest/ml/model/models-from-code/
"""
import os
import torch
import numpy as np
import polars as pl
import mlflow
from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict, Any, Optional

from src.models.utils.common import get_device, filter_graph_by_time
from src.models.feature_engineering import feature_engineering


class HybridFraudDetectionModel(PythonModel):
    """
    Custom Python model for hybrid GNN + XGBoost fraud detection.
    
    This model encapsulates:
    1. GNN model for generating embeddings
    2. XGBoost model for final predictions
    3. Feature engineering pipeline
    
    This follows MLflow's "models from code" pattern, allowing the model
    to be deployed as a single unit with all inference logic bundled.
    """
    
    def load_context(self, context: PythonModelContext):
        """
        Load model artifacts from MLflow context.
        
        The context.artifacts dictionary contains paths to:
        - gnn_model: Path to the GNN model directory
        - xgb_model: Path to the XGBoost model directory
        - graph: Path to the graph artifact (if needed)
        """
        import mlflow.pytorch
        import mlflow.xgboost
        
        # Load GNN model
        gnn_path = context.artifacts.get("gnn_model")
        if gnn_path:
            # Handle both file paths and MLflow URIs
            if gnn_path.startswith("runs:/") or gnn_path.startswith("models:/"):
                self.gnn_model = mlflow.pytorch.load_model(gnn_path)
            elif os.path.exists(gnn_path):
                self.gnn_model = mlflow.pytorch.load_model(gnn_path)
            else:
                self.gnn_model = None
            if self.gnn_model:
                self.gnn_model.eval()
        else:
            self.gnn_model = None
        
        # Load XGBoost model
        xgb_path = context.artifacts.get("xgb_model")
        if not xgb_path:
            raise ValueError("XGBoost model path not provided in artifacts")
        
        # Handle both file paths and MLflow URIs
        if xgb_path.startswith("runs:/") or xgb_path.startswith("models:/"):
            self.xgb_model = mlflow.xgboost.load_model(xgb_path)
        elif os.path.exists(xgb_path):
            self.xgb_model = mlflow.xgboost.load_model(xgb_path)
        else:
            raise ValueError(f"XGBoost model not found at {xgb_path}")
        
        # Extract feature names from XGBoost model (if available)
        try:
            if hasattr(self.xgb_model, 'get_booster'):
                self.feature_names = self.xgb_model.get_booster().feature_names
            else:
                self.feature_names = None
        except Exception:
            self.feature_names = None
        
        # Load graph if needed for embedding generation
        graph_path = context.artifacts.get("graph")
        if graph_path and os.path.exists(graph_path):
            self.graph_data = torch.load(graph_path, weights_only=False)
        else:
            self.graph_data = None
        
        # Load listing ID to index mapping for embedding lookup
        listing_mapping_path = context.artifacts.get("listing_mapping", "artifacts/nodes_listing.parquet")
        if os.path.exists(listing_mapping_path):
            try:
                df_listing = pl.read_parquet(listing_mapping_path)
                self.listing_id_to_idx = {
                    row["insertion_id"]: idx 
                    for idx, row in enumerate(df_listing.iter_rows(named=True))
                }
            except Exception:
                self.listing_id_to_idx = {}
        else:
            self.listing_id_to_idx = {}
        
        self.device = get_device()
        if self.gnn_model:
            self.gnn_model = self.gnn_model.to(self.device)
    
    def predict(self, context: PythonModelContext, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Generate predictions for input data.
        
        Args:
            context: MLflow model context
            model_input: Input data (pandas DataFrame or dict)
            params: Optional inference parameters
        
        Returns:
            Predictions (fraud probabilities)
        """
        # Convert input to Polars DataFrame if needed
        if isinstance(model_input, dict):
            df = pl.DataFrame(model_input)
        else:
            df = pl.from_pandas(model_input) if hasattr(model_input, 'to_pandas') else model_input
        
        # Feature engineering
        # Use the latest timestamp in the data as cutoff date
        if "submission_at" not in df.columns:
            raise ValueError("DataFrame must contain 'submission_at' column for temporal filtering")
        cutoff_date = df["submission_at"].max()
        
        # Create default experiment config if not stored in model artifacts
        from src.models.config.experiment_config import ExperimentConfig
        config = ExperimentConfig()  # Uses all feature categories by default
        df = feature_engineering(df, cutoff_date=cutoff_date, config=config)
        
        # Generate GNN embeddings if GNN model is available
        if self.gnn_model and self.graph_data:
            embeddings = self._generate_embeddings(df)
            # Add embeddings to dataframe
            for i, emb_col in enumerate([f"embed_{i}" for i in range(embeddings.shape[1])]):
                df = df.with_columns(pl.Series(emb_col, embeddings[:, i]))
        
        # Extract features for XGBoost
        # Use feature names from model if available, otherwise infer from columns
        if self.feature_names:
            # Use exact feature names from training
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                # Fill missing features with zeros
                for feat in missing_features:
                    df = df.with_columns(pl.Series(feat, [0.0] * len(df)))
            feature_cols = [col for col in self.feature_names if col in df.columns]
        else:
            # Fallback: exclude non-feature columns
            feature_cols = [col for col in df.columns 
                           if col not in ["insertion_id", "submission_at", "is_fraud"]]
        
        # Ensure columns are in the correct order
        X = df.select(feature_cols).to_numpy()
        
        # Get predictions from XGBoost
        predictions = self.xgb_model.predict_proba(X)[:, 1]
        
        return predictions
    
    def _generate_embeddings(self, df: pl.DataFrame) -> np.ndarray:
        """
        Generate GNN embeddings for listings in the dataframe.
        
        Args:
            df: DataFrame with insertion_id column
        
        Returns:
            Array of embeddings (n_samples, embedding_dim)
        """
        if self.gnn_model is None or self.graph_data is None:
            # Return zero embeddings if GNN not available
            return np.zeros((len(df), 64))
        
        # Filter graph to temporal cutoff (prevent leakage)
        # In production, use the cutoff_date from feature engineering
        cutoff_ns = int(df["submission_at"].max().timestamp() * 1e9) if "submission_at" in df.columns else None
        
        # Check if model expects edge_time_dict (HGT models)
        import inspect
        forward_sig = inspect.signature(self.gnn_model.forward)
        has_edge_time = 'edge_time_dict' in forward_sig.parameters
        
        if cutoff_ns:
            # Filter graph with edge times if needed for HGT
            if has_edge_time:
                filtered_data, filtered_edge_times = filter_graph_by_time(
                    self.graph_data, cutoff_ns, return_edge_times=True
                )
                # Move edge times to device
                filtered_edge_times = {
                    k: v.to(self.device) if v is not None else None 
                    for k, v in filtered_edge_times.items()
                }
            else:
                filtered_data = filter_graph_by_time(self.graph_data, cutoff_ns)
                filtered_edge_times = None
        else:
            filtered_data = self.graph_data
            filtered_edge_times = None
        
        filtered_data = filtered_data.to(self.device)
        
        # Generate embeddings
        self.gnn_model.eval()
        with torch.no_grad():
            if has_edge_time:
                # Use edge times from filtered graph
                z_listing = self.gnn_model(
                    filtered_data.x_dict,
                    filtered_data.edge_index_dict,
                    edge_time_dict=filtered_edge_times
                )
            else:
                # SAGE models don't use edge times
                z_listing = self.gnn_model(
                    filtered_data.x_dict,
                    filtered_data.edge_index_dict
                )
        
        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        
        embeddings = z_listing.cpu().numpy()
        
        # Map embeddings to insertion_ids using the mapping
        n_samples = len(df)
        embedding_dim = embeddings.shape[1]
        result_embeddings = np.zeros((n_samples, embedding_dim))
        
        if "insertion_id" in df.columns and self.listing_id_to_idx:
            insertion_ids = df["insertion_id"].to_list()
            for i, insertion_id in enumerate(insertion_ids):
                idx = self.listing_id_to_idx.get(insertion_id, -1)
                if idx >= 0 and idx < len(embeddings):
                    result_embeddings[i] = embeddings[idx]
        else:
            # Fallback: use first N embeddings if mapping not available
            n_use = min(n_samples, len(embeddings))
            result_embeddings[:n_use] = embeddings[:n_use]
        
        return result_embeddings


def log_hybrid_model_as_pyfunc(
    gnn_model_uri: str,
    xgb_model_uri: str,
    graph_path: str = "artifacts/graph.pt",
    registered_model_name: Optional[str] = None
) -> str:
    """
    Log a hybrid model using the "models from code" pattern.
    
    This function packages both the GNN and XGBoost models together
    as a single MLflow Python function model.
    
    Args:
        gnn_model_uri: URI to the logged GNN model (e.g., "runs:/run_id/gnn_model")
        xgb_model_uri: URI to the logged XGBoost model (e.g., "runs:/run_id/model")
        graph_path: Path to graph artifact file
        registered_model_name: Optional name for model registry
    
    Returns:
        Model URI of the logged hybrid model
    
    Example:
        ```python
        # After training both models
        gnn_uri = "runs:/abc123/gnn_model"
        xgb_uri = "runs:/def456/model"
        
        hybrid_uri = log_hybrid_model_as_pyfunc(
            gnn_model_uri=gnn_uri,
            xgb_model_uri=xgb_uri,
            registered_model_name="fraud-detection-hybrid"
        )
        ```
    """
    from src.models.utils.mlflow_helpers import get_model_dependencies
    
    # Create artifacts dictionary
    artifacts = {
        "gnn_model": gnn_model_uri,
        "xgb_model": xgb_model_uri,
    }
    
    if os.path.exists(graph_path):
        artifacts["graph"] = graph_path
    
    # Include listing mapping for embedding lookup
    listing_mapping_path = "artifacts/nodes_listing.parquet"
    if os.path.exists(listing_mapping_path):
        artifacts["listing_mapping"] = listing_mapping_path
    
    # Get dependencies
    deps = get_model_dependencies()
    
    # Log as Python function model
    model_uri = mlflow.pyfunc.log_model(
        name="hybrid_model",
        python_model=HybridFraudDetectionModel(),
        artifacts=artifacts,
        registered_model_name=registered_model_name,
        **deps
    )
    
    return model_uri


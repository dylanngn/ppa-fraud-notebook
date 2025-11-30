"""
SAGE Hybrid Model Training

Optimized SAGE (GraphSAGE) implementation for fraud detection.
Trains GNN embeddings and hybrid XGBoost model with model-specific optimizations.
"""
import os
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from torch_geometric.nn import SAGEConv, Linear, to_hetero

from src.models.training_window import train_accumulating_window
from src.models.feature_engineering import load_data, add_base_tabular_features
from src.models.utils.common import get_device, setup_mlflow, filter_graph_by_time
from src.models.utils.mlflow_helpers import (
    create_gnn_signature,
    create_input_example_for_gnn,
    get_model_dependencies
)
from src.utils.metrics import calculate_metrics


class GraphSAGE(nn.Module):
    """GraphSAGE model for heterogeneous graphs."""
    
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)


class SAGEWrapper(nn.Module):
    """
    SAGE wrapper for heterogeneous graphs.
    Optimized for fraud detection with skip connections to prevent over-smoothing.
    """
    
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_layers=2):
        super().__init__()
        # Input projections for all node types
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        # Core SAGE model (mean aggregation works best for heterogeneous graphs)
        model = GraphSAGE(hidden_channels, hidden_channels, num_layers)
        self.gnn = to_hetero(model, metadata, aggr='mean')
        
        # Output projection with skip connection (preserve self-features)
        self.lin_out = Linear(hidden_channels * 2, out_channels)
        
        # Classifier for training
        self.classifier = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Project inputs
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
        
        # Cache listing self-representation before message passing
        listing_self = x_dict_proj['listing']
        
        # Apply SAGE
        x_dict_out = self.gnn(x_dict_proj, edge_index_dict)
        
        # Concatenate self features with aggregated message
        listing_out = x_dict_out['listing']
        z_listing = torch.cat([listing_self, listing_out], dim=-1)
        z_listing = self.lin_out(z_listing)
        return z_listing

    def predict(self, x_dict, edge_index_dict):
        z = self.forward(x_dict, edge_index_dict)
        return self.classifier(z)


def train_sage_embeddings(epochs=25, split_percent=0.8, window_days=90, step_days=14):
    """
    Train SAGE embeddings with optimized parameters.
    
    Optimized settings:
    - 25 epochs (more than default for better convergence)
    - Learning rate: 0.001 (standard for GNNs)
    - 2 layers, 64 hidden channels (optimal for this graph size)
    
    Returns:
        str: Model URI of the logged GNN model (e.g., "runs:/run_id/gnn_model")
    """
    setup_mlflow()
    mlflow.pytorch.autolog()
    
    # Build or load graph
    if not os.path.exists("artifacts/graph.pt"):
        print("Graph not found. Building full graph...")
        from src.data.graph.graph_builder import build_graph
        data = build_graph(cutoff_date=None)  # Build full graph for initial training
    else:
        data = torch.load("artifacts/graph.pt", weights_only=False)
        # Safeguard: Remove person nodes if they exist (legacy from old graph artifacts)
        if 'person' in data.node_types:
            print("Warning: Found 'person' node type in loaded graph. Removing it...")
            # Remove person node data
            if hasattr(data['person'], 'x'):
                del data['person']
            # Remove any edges involving person nodes
            edge_types_to_remove = [et for et in data.edge_types if 'person' in et]
            for edge_type in edge_types_to_remove:
                del data[edge_type]
            print("Removed person nodes and edges from graph.")
    device = get_device()

    # Temporal split
    timestamps = data['listing'].timestamp.numpy()
    start_threshold = datetime(2023, 1, 1).timestamp() * 1e9
    valid_mask = timestamps >= start_threshold
    valid_timestamps = timestamps[valid_mask]
    
    if len(valid_timestamps) == 0:
        valid_timestamps = timestamps
        
    min_time = valid_timestamps.min()
    max_time = valid_timestamps.max()
    split_time = np.percentile(valid_timestamps, split_percent * 100)
    split_date = datetime.fromtimestamp(split_time / 1e9)

    # Create training subgraph
    train_data = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)
    
    # Get metadata
    filtered_metadata = train_data.metadata()
    
    # Initialize model with optimized parameters
    model = SAGEWrapper(
        metadata=filtered_metadata,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # MLflow tracking
    mlflow.start_run(run_name="gnn_sage", tags={"model_type": "gnn", "gnn_variant": "sage"})
    gnn_model_uri = None
    try:
        mlflow.log_params({
            "epochs": epochs,
            "split_percent": split_percent,
            "window_days": window_days,
            "step_days": step_days,
            "hidden_channels": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
        })
    
        train_mask = ((train_data['listing'].timestamp <= split_time) & 
                     (train_data['listing'].timestamp >= start_threshold)).to(device)
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            out = model.predict(train_data.x_dict, train_data.edge_index_dict)
            loss = F.binary_cross_entropy_with_logits(
                out[train_mask], 
                train_data['listing'].y[train_mask].float().view(-1, 1)
            )
            
            loss.backward()
            optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "artifacts/model_sage_best.pt")
                mlflow.log_metric("best_loss", best_loss.item())

        # Load best model
        model.load_state_dict(torch.load("artifacts/model_sage_best.pt", weights_only=False))
        
        # Evaluate on test split (before loading full graph to save memory)
        test_mask = ((data['listing'].timestamp > split_time) & 
                    (data['listing'].timestamp >= start_threshold))
        
        if test_mask.sum() > 0:
            # Use full graph for evaluation (will reuse for embeddings)
            full_data = data.to(device)
            test_mask_device = test_mask.to(device)
            model.eval()
            with torch.no_grad():
                test_out = model.predict(full_data.x_dict, full_data.edge_index_dict)
                test_pred = test_out[test_mask_device].sigmoid().cpu().numpy().flatten()
                test_y = data['listing'].y[test_mask].cpu().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(test_y, test_pred)
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "gnn_test_auc_pr": metrics["auc_pr"],
                "gnn_test_auc_roc": metrics["auc_roc"],
                "gnn_test_p_at_100": metrics["p@100"],
                "gnn_test_lift_at_100": metrics["lift@100"],
                "gnn_test_fraud_count": metrics["fraud_count"],
            })
        else:
            full_data = data.to(device)
        
        # Generate embeddings for all nodes (reuse full_data from evaluation)
        model.eval()
        with torch.no_grad():
            z_listing = model(full_data.x_dict, full_data.edge_index_dict)

        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        z_listing = z_listing.cpu()
            
        # Save embeddings
        save_path = "artifacts/embeddings_sage.pt"
        torch.save(z_listing, save_path)
        mlflow.log_artifact(save_path)
        
        # Register GNN model to Model Registry
        try:
            model.eval()
            with torch.no_grad():
                # Create sample input (using first few nodes)
                sample_x_dict = {k: v[:5] if v.numel() > 0 else v for k, v in train_data.x_dict.items()}
                sample_edge_index_dict = {}
                for edge_type, edge_index in train_data.edge_index_dict.items():
                    if edge_index.numel() > 0:
                        # Take first few edges
                        sample_edge_index_dict[edge_type] = edge_index[:, :min(10, edge_index.size(1))]
                    else:
                        sample_edge_index_dict[edge_type] = edge_index
                
                # Get sample output
                sample_output = model(sample_x_dict, sample_edge_index_dict)
                if isinstance(sample_output, dict):
                    sample_output = sample_output['listing']
                
                # Create input example using helper (properly serializable)
                input_example = create_input_example_for_gnn(
                    sample_x_dict, 
                    sample_edge_index_dict, 
                    edge_time_dict=None,  # SAGE doesn't use edge times
                    max_nodes=5,
                    max_edges=10
                )
                output_example = sample_output.cpu().numpy() if isinstance(sample_output, torch.Tensor) else sample_output
                
                # Create signature using helper (handles complex GNN inputs)
                signature = create_gnn_signature(input_example, output_example)
            
            # Get explicit dependencies
            deps = get_model_dependencies()
            
            gnn_model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="gnn_model",
                registered_model_name="fraud-detection-gnn-sage",
                signature=signature,
                input_example=input_example,
                **deps,  # Add explicit dependencies
                metadata={
                    "model_type": "GraphSAGE (SAGE)",
                    "task": "fraud_detection",
                    "framework": "pytorch",
                    "graph_type": "heterogeneous",
                    "hidden_channels": 64,
                    "out_channels": 64,
                    "num_layers": 2,
                    "training_epochs": epochs,
                },
                params={
                    "hidden_channels": 64,
                    "out_channels": 64,
                    "num_layers": 2,
                    "learning_rate": 0.001,
                    "epochs": epochs,
                    "split_percent": split_percent,
                }
            )
        except Exception:
            # Try alternative: register from autologged model
            try:
                run_id = mlflow.active_run().info.run_id
                gnn_model_uri = f"runs:/{run_id}/gnn_model"
                mlflow.register_model(
                    model_uri=gnn_model_uri,
                    name="fraud-detection-gnn-sage"
                )
            except Exception:
                # Fallback: use run ID directly
                run_id = mlflow.active_run().info.run_id
                gnn_model_uri = f"runs:/{run_id}/gnn_model"
        
    finally:
        mlflow.end_run()
    
    return gnn_model_uri


def create_sage_embedding_generator():
    """
    Creates an embedding generator function for per-window embedding generation.
    This ensures temporal fairness by filtering the graph before generating embeddings.
    
    Returns:
        Callback function(train_data, test_data, train_end) -> (train_embeddings_df, test_embeddings_df, embed_cols)
    """
    # Load the trained model and full graph once
    if not os.path.exists("artifacts/model_sage_best.pt"):
        raise FileNotFoundError(
            "SAGE model not found. Run train_sage_embeddings() first."
        )
    
    # Load model (graph will be built dynamically per window)
    device = get_device()
    
    # Get metadata for model initialization (without building full graph)
    # Use saved graph.pt if available, otherwise build graph for metadata only
    from src.data.graph.graph_builder import build_graph
    if os.path.exists("artifacts/graph.pt"):
        # Use saved graph (already built during initial training)
        full_data = torch.load("artifacts/graph.pt", weights_only=False)
        print("Using saved graph.pt for metadata")
    else:
        # Build full graph only if needed (for metadata)
        print("Building graph for metadata (this is a one-time cost)")
        full_data = build_graph(cutoff_date=None)
    
    # Get metadata
    filtered_metadata = full_data.metadata()
    
    # Initialize model
    model = SAGEWrapper(
        metadata=filtered_metadata,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load("artifacts/model_sage_best.pt", weights_only=False))
    model.eval()
    
    # Load listing node mapping (will be used to map insertion_ids to graph indices)
    df_listing_all = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    def generate_embeddings_for_window(train_data, test_data, train_end):
        """
        Generate embeddings for a specific window with temporal filtering.
        
        Args:
            train_data: Training DataFrame (already filtered by time)
            test_data: Test DataFrame (already filtered by time)
            train_end: Cutoff datetime for temporal filtering
        
        Returns:
            (train_embeddings_df, test_embeddings_df, embed_cols)
        """
        # Build graph dynamically with temporal filtering
        # This ensures only edges between entities that existed before train_end are included
        filtered_data = build_graph(cutoff_date=train_end)
        
        # Create listing_id to index mapping for this filtered graph
        df_listing_filtered = pl.read_parquet("artifacts/nodes_listing.parquet")
        df_listing_filtered = df_listing_filtered.with_columns(
            pl.col("submission_at").cast(pl.Datetime("ns"))
        )
        df_listing_filtered = df_listing_filtered.filter(
            pl.col("submission_at") < train_end
        )
        listing_id_to_idx = {
            row["insertion_id"]: idx 
            for idx, row in enumerate(df_listing_filtered.iter_rows(named=True))
        }
        
        filtered_data = filtered_data.to(device)
        
        # Generate embeddings on filtered graph
        with torch.no_grad():
            z_listing = model(filtered_data.x_dict, filtered_data.edge_index_dict)
        
        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        
        embeddings = z_listing.cpu().numpy()
        
        # Create embedding DataFrame
        embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
        
        # Map embeddings to insertion_ids
        train_insertion_ids = train_data["insertion_id"].to_list()
        test_insertion_ids = test_data["insertion_id"].to_list()
        
        # Get indices for train and test listings
        train_indices = [listing_id_to_idx.get(insertion_id, -1) for insertion_id in train_insertion_ids]
        test_indices = [listing_id_to_idx.get(insertion_id, -1) for insertion_id in test_insertion_ids]
        
        # Extract embeddings for train and test
        train_embeddings = []
        test_embeddings = []
        
        for idx in train_indices:
            if idx >= 0 and idx < len(embeddings):
                train_embeddings.append(embeddings[idx])
            else:
                # Listing not in graph, use zero embeddings
                train_embeddings.append([0.0] * len(embed_cols))
        
        for idx in test_indices:
            if idx >= 0 and idx < len(embeddings):
                test_embeddings.append(embeddings[idx])
            else:
                # Listing not in graph, use zero embeddings
                test_embeddings.append([0.0] * len(embed_cols))
        
        # Create DataFrames
        train_embeddings_df = pl.DataFrame({
            "insertion_id": train_insertion_ids,
            **{col: [emb[i] for emb in train_embeddings] for i, col in enumerate(embed_cols)}
        })
        
        test_embeddings_df = pl.DataFrame({
            "insertion_id": test_insertion_ids,
            **{col: [emb[i] for emb in test_embeddings] for i, col in enumerate(embed_cols)}
        })
        
        return train_embeddings_df, test_embeddings_df, embed_cols
    
    return generate_embeddings_for_window


def main(
    experiment_name: str = "ppa-fraud-detection",
    initial_window_days: int = 180,
    step_days: int = 7,
    epochs: int = 25,
    max_windows: Optional[int] = None
):
    """
    Train SAGE hybrid model: embeddings + XGBoost.
    
    Pipeline:
    1. Train SAGE model on graph (temporal split)
    2. Create embedding generator for per-window embedding generation
    3. Train hybrid XGBoost model with accumulating window (embeddings generated per window with temporal filtering)
    
    Args:
        experiment_name: MLflow experiment name
        initial_window_days: Initial training window size in days
        step_days: Step size between evaluation windows in days
        epochs: Number of GNN training epochs
        max_windows: Optional maximum number of windows to evaluate (for faster training)
    """
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        raise FileNotFoundError("Artifacts not found. Please run ETL.py first.")
    
    # Step 1: Train SAGE model (once, on training split)
    gnn_model_uri = train_sage_embeddings(epochs=epochs)
    
    # Step 2: Create embedding generator (will generate embeddings per window)
    embedding_generator = create_sage_embedding_generator()
    
    # Step 3: Load base data (without embeddings - they'll be generated per window)
    df = load_data()
    
    # Step 4: Add base tabular features (graph features will be added per window)
    df = add_base_tabular_features(df)
    
    # Step 5: Create config and train hybrid model with per-window embeddings
    from src.models.config.experiment_config import ExperimentConfig
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        initial_window_days=initial_window_days,
        step_days=step_days
    )
    
    result = train_accumulating_window(
        df,
        embedding_generator=embedding_generator,
        model_name="hybrid_sage",
        config=config,
        max_windows=max_windows
    )
    
    # Step 6: Log hybrid model using "models from code" pattern (pyfunc)
    if gnn_model_uri and result.get("best_run_id"):
        try:
            from src.models.gnn.hybrid_pyfunc import log_hybrid_model_as_pyfunc
            
            xgb_model_uri = f"runs:/{result['best_run_id']}/model"
            
            # Log hybrid model in the parent run
            with mlflow.start_run(run_id=result["run_id"]):
                hybrid_model_uri = log_hybrid_model_as_pyfunc(
                    gnn_model_uri=gnn_model_uri,
                    xgb_model_uri=xgb_model_uri,
                    graph_path="artifacts/graph.pt",
                    registered_model_name="fraud-detection-hybrid-sage"
                )
                
                mlflow.log_param("hybrid_model_uri", hybrid_model_uri)
                mlflow.log_param("gnn_model_uri", gnn_model_uri)
                mlflow.log_param("xgb_model_uri", xgb_model_uri)
                
                print(f"✓ Logged hybrid model as pyfunc: {hybrid_model_uri}")
        except Exception as e:
            print(f"Warning: Could not log hybrid pyfunc model: {e}")
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(main)

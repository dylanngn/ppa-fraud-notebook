"""
SAGE Hybrid Model Training

Optimized SAGE (GraphSAGE) implementation for fraud detection.
Trains GNN embeddings and hybrid XGBoost model with model-specific optimizations.
"""
import os
from datetime import datetime

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from torch_geometric.nn import SAGEConv, Linear, to_hetero

from src.models.train_baseline import feature_engineering, train_accumulating_window
from src.utils.metrics import calculate_metrics
from src.utils.mlflow_init import init_mlflow


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


def filter_graph_by_time(data, max_time_ns):
    """Returns a subgraph containing only edges and nodes visible at max_time_ns."""
    from torch_geometric.data import HeteroData
    
    new_data = HeteroData()
    
    # Copy node features
    for node_type, x in data.x_dict.items():
        new_data[node_type].x = x
        new_data[node_type].num_nodes = data[node_type].num_nodes
    
    # Copy listing labels & timestamps
    new_data['listing'].y = data['listing'].y
    new_data['listing'].timestamp = data['listing'].timestamp
    
    # Filter edges by timestamp
    for edge_type, edge_index in data.edge_index_dict.items():
        if 'timestamp' in data[edge_type]:
            edge_times = data[edge_type].timestamp
            mask = edge_times <= max_time_ns
            new_data[edge_type].edge_index = edge_index[:, mask]
            new_data[edge_type].timestamp = edge_times[mask]
        else:
            # Static edges (keep all)
            new_data[edge_type].edge_index = edge_index
            
    return new_data


def train_sage_embeddings(epochs=25, split_percent=0.8, window_days=90, step_days=14):
    """
    Train SAGE embeddings with optimized parameters.
    
    Optimized settings:
    - 25 epochs (more than default for better convergence)
    - Learning rate: 0.001 (standard for GNNs)
    - 2 layers, 64 hidden channels (optimal for this graph size)
    """
    print("Training SAGE Embeddings...")
    
    # Initialize MLflow with database backend
    init_mlflow()
    mlflow.set_experiment("ppa-fraud-detection")
    mlflow.pytorch.autolog()
    
    if not os.path.exists("artifacts/graph.pt"):
        print("Graph not found. Run graph_builder.py first.")
        return
    
    data = torch.load("artifacts/graph.pt", weights_only=False)
    # Device selection: Prioritize MPS (Mac GPU) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device} (MacBook Pro compatible)")

    # Temporal split
    timestamps = data['listing'].timestamp.numpy()
    start_threshold = datetime(2023, 1, 1).timestamp() * 1e9
    valid_mask = timestamps >= start_threshold
    valid_timestamps = timestamps[valid_mask]
    
    if len(valid_timestamps) == 0:
        print("Warning: No valid timestamps found >= 2023. Using all data.")
        valid_timestamps = timestamps
        
    min_time = valid_timestamps.min()
    max_time = valid_timestamps.max()
    split_time = np.percentile(valid_timestamps, split_percent * 100)
    split_date = datetime.fromtimestamp(split_time / 1e9)
    print(f"Splitting at: {split_date}")

    # Create training subgraph
    print("Creating Training Subgraph...")
    train_data = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)

    # Initialize model with optimized parameters
    model = SAGEWrapper(
        metadata=train_data.metadata(),
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # MLflow tracking
    mlflow.start_run(run_name="gnn_sage", tags={"model_type": "gnn", "gnn_variant": "sage"})
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
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), "artifacts/model_sage_best.pt")
                mlflow.log_metric("best_loss", best_loss.item())

        # Load best model
        model.load_state_dict(torch.load("artifacts/model_sage_best.pt", weights_only=False))
        
        # Evaluate on test split (before loading full graph to save memory)
        print("\nEvaluating SAGE on test split...")
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
            
            print(f"Test Metrics - AUC-PR: {metrics['auc_pr']:.4f}, "
                  f"AUC-ROC: {metrics['auc_roc']:.4f}, "
                  f"P@100: {metrics['p@100']:.4f}, "
                  f"Lift@100: {metrics['lift@100']:.2f}")
        else:
            print("Warning: No test samples found for evaluation")
            full_data = data.to(device)
        
        # Generate embeddings for all nodes (reuse full_data from evaluation)
        print("\nGenerating Full Graph Embeddings...")
        model.eval()
        with torch.no_grad():
            z_listing = model(full_data.x_dict, full_data.edge_index_dict)

        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        z_listing = z_listing.cpu()
            
        # Save embeddings
        save_path = "artifacts/embeddings_sage.pt"
        torch.save(z_listing, save_path)
        print(f"Saved embeddings to {save_path}")
        mlflow.log_artifact(save_path)
        
        # Register GNN model to Model Registry
        # Note: autolog may have already logged the model, but we explicitly log and register
        try:
            print("\nRegistering SAGE GNN model to Model Registry...")
            # Log model explicitly (autolog may have logged it, but this ensures it's registered)
            # Note: Using 'name' parameter instead of deprecated 'artifact_path' for MLflow 3.0+
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="gnn_model",  # Replaces deprecated 'artifact_path'
                registered_model_name="fraud-detection-gnn-sage"
            )
            print(f"✓ SAGE GNN model registered successfully: {model_uri}")
        except Exception as e:
            print(f"Warning: GNN model registration failed: {e}")
            # Try alternative: register from autologged model
            try:
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="fraud-detection-gnn-sage"
                )
                print(f"✓ SAGE GNN model registered via autolog: {registered_model.name} v{registered_model.version}")
            except Exception as e2:
                print(f"Warning: Alternative registration also failed: {e2}")
        
    finally:
        mlflow.end_run()


def load_sage_embeddings():
    """Load SAGE embeddings and merge with DataFrame."""
    embedding_path = "artifacts/embeddings_sage.pt"
    print(f"Loading SAGE Embeddings from {embedding_path}...")
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"Embeddings not found at {embedding_path}. "
            "Run train_sage_embeddings() first."
        )
        
    embeddings = torch.load(embedding_path).numpy()
    df_listing = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
    df_embed = pl.DataFrame(embeddings, schema=embed_cols)
    
    if len(df_listing) != len(df_embed):
        raise ValueError(f"Mismatch: Listings {len(df_listing)} vs Embeddings {len(df_embed)}")
        
    df_listing = df_listing.hstack(df_embed)
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    df = df_listing.join(df_users, on="user_id", how="left")
    
    return df, embed_cols


def main():
    """
    Train SAGE hybrid model: embeddings + XGBoost.
    
    Pipeline:
    1. Train SAGE embeddings on graph
    2. Load embeddings and merge with tabular features
    3. Train hybrid XGBoost model with accumulating window
    """
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
        return
    
    # Step 1: Train embeddings
    train_sage_embeddings()
    
    # Step 2: Load data & embeddings
    df, embed_cols = load_sage_embeddings()
    
    # Step 3: Feature engineering
    df = feature_engineering(df)
    
    # Step 4: Train hybrid model
    print(f"Training SAGE hybrid model with {len(embed_cols)} embedding features...")
    result = train_accumulating_window(
        df,
        extra_features=embed_cols,
        model_name="hybrid_sage"
    )
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(lambda: main())


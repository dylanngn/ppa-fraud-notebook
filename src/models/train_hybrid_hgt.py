"""
HGT Hybrid Model Training

Optimized HGT (Heterogeneous Graph Transformer) with RTE implementation for fraud detection.
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
from torch_geometric.nn import HGTConv, Linear

from src.models.train_baseline import feature_engineering, train_accumulating_window
from src.utils.metrics import calculate_metrics


class TemporalEncoding(nn.Module):
    """
    Sinusoidal + learnable temporal encoding for HGT with RTE.
    Encodes time differences between edge timestamps.
    """
    
    def __init__(self, hidden_channels, max_time_scale=1e12):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.max_time_scale = max_time_scale
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, time_diff: torch.Tensor) -> torch.Tensor:
        time_norm = time_diff / self.max_time_scale
        freqs = torch.arange(
            0, self.hidden_channels, 2, device=time_diff.device, dtype=torch.float
        )
        freqs = 1.0 / (10000 ** (freqs / self.hidden_channels))
        encoding = torch.zeros(time_diff.size(0), self.hidden_channels, device=time_diff.device)
        encoding[:, 0::2] = torch.sin(time_norm.unsqueeze(-1) * freqs)
        encoding[:, 1::2] = torch.cos(time_norm.unsqueeze(-1) * freqs)
        return self.mlp(encoding)


class HGTWrapper(nn.Module):
    """
    HGT wrapper with RTE (Relative Temporal Encoding) for heterogeneous graphs.
    Optimized for fraud detection with temporal awareness and skip connections.
    """
    
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_heads=4, num_layers=2):
        super().__init__()
        # Input projections for all node types
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        # HGT layers (HGTConv is single layer, stack multiple)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
        
        # Temporal encoder (always enabled for HGT)
        self.temporal_encoder = TemporalEncoding(hidden_channels)
        
        # Output projection with skip connection
        self.lin_out = Linear(hidden_channels * 2, out_channels)
        
        # Classifier for training
        self.classifier = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        # Project inputs
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
        
        # Cache listing self-representation before message passing
        listing_self = x_dict_proj['listing']
        
        # Compute temporal embeddings if edge times provided
        temporal_embeddings = None
        if edge_time_dict is not None:
            temporal_embeddings = self._compute_temporal_embeddings(edge_time_dict)
        
        # Apply HGT layers
        x_dict_out = x_dict_proj
        for conv in self.convs:
            x_dict_out = conv(x_dict_out, edge_index_dict)
            if temporal_embeddings:
                x_dict_out = self._apply_temporal_bias(x_dict_out, edge_index_dict, temporal_embeddings)
        
        # Concatenate self features with aggregated message
        listing_out = x_dict_out['listing']
        z_listing = torch.cat([listing_self, listing_out], dim=-1)
        z_listing = self.lin_out(z_listing)
        return z_listing

    def predict(self, x_dict, edge_index_dict, edge_time_dict=None):
        z = self.forward(x_dict, edge_index_dict, edge_time_dict=edge_time_dict)
        return self.classifier(z)

    def _compute_temporal_embeddings(self, edge_time_dict):
        """Compute temporal embeddings for all edge types."""
        embeddings = {}
        for edge_type, timestamps in edge_time_dict.items():
            if timestamps is None or timestamps.numel() == 0:
                continue
            min_time = timestamps.min()
            time_diff = (timestamps - min_time).float()
            embeddings[edge_type] = self.temporal_encoder(time_diff)
        return embeddings

    def _apply_temporal_bias(self, x_dict_out, edge_index_dict, temporal_embeddings):
        """Apply temporal bias to node representations."""
        for node_type, node_repr in x_dict_out.items():
            temporal_bias = None
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if dst_type != node_type:
                    continue
                temp_emb = temporal_embeddings.get(edge_type)
                if temp_emb is None:
                    continue
                targets = edge_index[1]
                if temporal_bias is None:
                    temporal_bias = torch.zeros_like(node_repr)
                temporal_bias.index_add_(0, targets, temp_emb)
            if temporal_bias is not None:
                # Small weight to avoid overwhelming structural signals
                x_dict_out[node_type] = node_repr + 0.1 * temporal_bias
        return x_dict_out


def filter_graph_by_time(data, max_time_ns):
    """Returns a subgraph containing only edges and nodes visible at max_time_ns."""
    from torch_geometric.data import HeteroData
    
    new_data = HeteroData()
    edge_time_dict = {}
    
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
            edge_time_dict[edge_type] = edge_times[mask]
        else:
            # Static edges (keep all)
            new_data[edge_type].edge_index = edge_index
            edge_time_dict[edge_type] = None
            
    return new_data, edge_time_dict


def train_hgt_embeddings(epochs=30, split_percent=0.8, window_days=90, step_days=14):
    """
    Train HGT embeddings with optimized parameters.
    
    Optimized settings:
    - 30 epochs (HGT benefits from longer training)
    - Learning rate: 0.001 (standard for GNNs)
    - 2 layers, 64 hidden channels, 4 heads (optimal for heterogeneous graphs)
    - RTE always enabled for temporal awareness
    """
    print("Training HGT Embeddings (with RTE)...")
    
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
    train_data, train_edge_times = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)
    
    # Move edge times to device
    train_edge_times_device = {
        k: v.to(device) if v is not None else None 
        for k, v in train_edge_times.items()
    }

    # Initialize model with optimized parameters
    model = HGTWrapper(
        metadata=train_data.metadata(),
        hidden_channels=64,
        out_channels=64,
        num_heads=4,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # MLflow tracking
    mlflow.start_run(run_name="gnn_hgt", tags={"model_type": "gnn", "gnn_variant": "hgt_rte"})
    try:
        mlflow.log_params({
            "epochs": epochs,
            "split_percent": split_percent,
            "window_days": window_days,
            "step_days": step_days,
            "hidden_channels": 64,
            "num_heads": 4,
            "num_layers": 2,
            "learning_rate": 0.001,
            "rte_enabled": True,
        })
    
        train_mask = ((train_data['listing'].timestamp <= split_time) & 
                     (train_data['listing'].timestamp >= start_threshold)).to(device)
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            out = model.predict(
                train_data.x_dict, 
                train_data.edge_index_dict, 
                train_edge_times_device
            )
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
                torch.save(model.state_dict(), "artifacts/model_hgt_best.pt")
                mlflow.log_metric("best_loss", best_loss.item())

        # Load best model
        model.load_state_dict(torch.load("artifacts/model_hgt_best.pt", weights_only=False))
        
        # Prepare edge times for full data (needed for evaluation and embedding generation)
        full_edge_times_device = {}
        for edge_type in data.edge_index_dict.keys():
            if 'timestamp' in data[edge_type]:
                full_edge_times_device[edge_type] = data[edge_type].timestamp.to(device)
            else:
                full_edge_times_device[edge_type] = None
        
        # Evaluate on test split (before generating embeddings to save memory)
        print("\nEvaluating HGT on test split...")
        test_mask = ((data['listing'].timestamp > split_time) & 
                    (data['listing'].timestamp >= start_threshold))
        
        if test_mask.sum() > 0:
            # Load full graph for evaluation (will reuse for embeddings)
            full_data = data.to(device)
            test_mask_device = test_mask.to(device)
            model.eval()
            with torch.no_grad():
                test_out = model.predict(
                    full_data.x_dict, 
                    full_data.edge_index_dict, 
                    full_edge_times_device
                )
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
            z_listing = model(
                full_data.x_dict, 
                full_data.edge_index_dict, 
                full_edge_times_device
            )

        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        z_listing = z_listing.cpu()
            
        # Save embeddings
        save_path = "artifacts/embeddings_hgt.pt"
        torch.save(z_listing, save_path)
        print(f"Saved embeddings to {save_path}")
        mlflow.log_artifact(save_path)
        
        # Register GNN model to Model Registry
        try:
            print("\nRegistering HGT GNN model to Model Registry...")
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="gnn_model",
                registered_model_name="fraud-detection-gnn-hgt"
            )
            print(f"✓ HGT GNN model registered successfully: {model_uri}")
        except Exception as e:
            print(f"Warning: GNN model registration failed: {e}")
            # Try alternative: register from autologged model
            try:
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="fraud-detection-gnn-hgt"
                )
                print(f"✓ HGT GNN model registered via autolog: {registered_model.name} v{registered_model.version}")
            except Exception as e2:
                print(f"Warning: Alternative registration also failed: {e2}")
        
    finally:
        mlflow.end_run()


def load_hgt_embeddings():
    """Load HGT embeddings and merge with DataFrame."""
    embedding_path = "artifacts/embeddings_hgt.pt"
    print(f"Loading HGT Embeddings from {embedding_path}...")
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"Embeddings not found at {embedding_path}. "
            "Run train_hgt_embeddings() first."
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
    Train HGT hybrid model: embeddings + XGBoost.
    
    Pipeline:
    1. Train HGT embeddings on graph (with RTE)
    2. Load embeddings and merge with tabular features
    3. Train hybrid XGBoost model with accumulating window
    """
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
        return
    
    # Step 1: Train embeddings
    train_hgt_embeddings()
    
    # Step 2: Load data & embeddings
    df, embed_cols = load_hgt_embeddings()
    
    # Step 3: Feature engineering
    df = feature_engineering(df)
    
    # Step 4: Train hybrid model
    print(f"Training HGT hybrid model with {len(embed_cols)} embedding features...")
    result = train_accumulating_window(
        df,
        extra_features=embed_cols,
        model_name="hybrid_hgt"
    )
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(lambda: main())


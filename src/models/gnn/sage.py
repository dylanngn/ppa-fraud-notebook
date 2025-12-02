"""
SAGE Hybrid Model Training

Optimized SAGE (GraphSAGE) implementation for fraud detection.
Trains GNN embeddings and hybrid XGBoost model with model-specific optimizations.
"""
import logging
import pickle
from datetime import datetime

import hydra
import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch_geometric.nn import Linear, SAGEConv, to_hetero

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

GRAPH_PT = resolve_path("artifacts/graph.pt")
MODEL_SAGE_BEST = resolve_path("artifacts/model_sage_best.pt")
EMBEDDINGS_SAGE = resolve_path("artifacts/embeddings_sage.pt")
NODES_LISTING = resolve_path("artifacts/nodes_listing.parquet")
MAPPINGS_PKL = resolve_path("artifacts/mappings.pkl")

from src.models.xgboost.trainer import train_accumulating_window
from src.models.utils.common import get_device, setup_mlflow, filter_graph_by_time
from src.models.utils.mlflow_helpers import get_model_dependencies
from src.utils.metrics import calculate_metrics
from src.data.graph.graph_builder import build_graph
from src.data.loader import load_data


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
    if not GRAPH_PT.exists():
        logger.info("Graph not found. Building full graph...")
        data = build_graph(cutoff_date=None)  # Build full graph for initial training
    else:
        data = torch.load(GRAPH_PT, weights_only=False)
    device = get_device()

    # Temporal split
    timestamps = data['listing'].timestamp.numpy()
    
    # Use 10th percentile as start threshold to filter out very old/invalid timestamps
    # This is more robust than a hardcoded date and adapts to the data range
    start_threshold = np.percentile(timestamps[timestamps > 0], 10)
    valid_mask = timestamps >= start_threshold
    valid_timestamps = timestamps[valid_mask]
    
    if len(valid_timestamps) == 0:
        valid_timestamps = timestamps
    
    logger.info(f"Using start_threshold: {datetime.fromtimestamp(start_threshold / 1e9)}")
        
    split_time = np.percentile(valid_timestamps, split_percent * 100)

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
                torch.save(model.state_dict(), MODEL_SAGE_BEST)
                mlflow.log_metric("best_loss", best_loss.item())

        # Load best model
        model.load_state_dict(torch.load(MODEL_SAGE_BEST, weights_only=False))
        
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
        torch.save(z_listing, EMBEDDINGS_SAGE)
        mlflow.log_artifact(str(EMBEDDINGS_SAGE))
        
        # Register GNN model to Model Registry
        # 
        # NOTE: GNN models cannot have MLflow signatures because:
        # 1. GNN inputs are Dict[str, Tensor] (x_dict, edge_index_dict)
        # 2. MLflow PyTorch flavor doesn't support Dict input types
        # 3. Attempting to create a signature causes: "The PyTorch flavor does not support List or Dict input types"
        #
        # This is fine because:
        # - GNN models aren't served via MLflow serving (graph structure required)
        # - Use embeddings file (embeddings_sage.pt) for inference instead
        # - The model IS logged and CAN be loaded, just without signature metadata
        #
        # The warning "Model logged without a signature" is expected and benign.
        try:
            deps = get_model_dependencies()
            
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                name="gnn_model",
                registered_model_name="fraud-detection-gnn-sage",
                **deps,
                metadata={
                    "model_type": "GraphSAGE (SAGE)",
                    "task": "fraud_detection",
                    "framework": "pytorch",
                    "graph_type": "heterogeneous",
                    "hidden_channels": 64,
                    "out_channels": 64,
                    "num_layers": 2,
                    "training_epochs": epochs,
                    "inference_note": "Use embeddings_sage.pt for inference, not MLflow serving",
                },
            )
            # Ensure we return a string URI, not ModelInfo object
            gnn_model_uri = model_info.model_uri if hasattr(model_info, 'model_uri') else str(model_info)
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
    
    OPTIMIZATION: Loads full graph once, then filters by time per window
    (instead of rebuilding the entire graph for each window).
    
    Returns:
        Callback function(train_data, test_data, train_end) -> (train_embeddings_df, test_embeddings_df, embed_cols)
    """
    # Load the trained model and full graph once
    if not MODEL_SAGE_BEST.exists():
        raise FileNotFoundError(
            "SAGE model not found. Run train_sage_embeddings() first."
        )
    
    device = get_device()
    
    # Load full graph ONCE (cached for all windows)
    # IMPORTANT: Must load graph FIRST to get correct metadata with reverse edges
    logger.info("Loading full graph for embedding generation (cached for all windows)...")
    if GRAPH_PT.exists():
        full_graph = torch.load(GRAPH_PT, weights_only=False)
    else:
        logger.info("Graph not found. Building full graph...")
        full_graph = build_graph(cutoff_date=None)
    
    # Get metadata from the actual graph (includes reverse edges from T.ToUndirected())
    metadata = full_graph.metadata()
    
    # Initialize model with the correct metadata
    model = SAGEWrapper(
        metadata=metadata,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
    ).to(device)
    
    # Load trained weights (trained with same metadata including reverse edges)
    model.load_state_dict(torch.load(MODEL_SAGE_BEST, weights_only=False))
    model.eval()
    
    # Load the EXACT mapping used when graph was built
    # This is critical: graph node indices are determined by graph_builder.py,
    # and Polars unique() may reorder rows. Using mappings.pkl guarantees correctness.
    if not MAPPINGS_PKL.exists():
        raise FileNotFoundError(
            f"Mappings file not found at {MAPPINGS_PKL}. Run 'python -m src.data.graph.build' first."
        )
    
    with open(MAPPINGS_PKL, "rb") as f:
        maps = pickle.load(f)
    
    listing_id_to_idx = maps["listing"]
    logger.info(f"Loaded listing_id → graph index mapping with {len(listing_id_to_idx)} entries from mappings.pkl")
    
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
        # Convert train_end to nanoseconds timestamp for filtering
        cutoff_ns = int(train_end.timestamp() * 1e9)
        
        # Filter graph by time using cached full graph
        # Note: filter_graph_by_time filters EDGES but keeps ALL nodes with same indices
        filtered_data = filter_graph_by_time(full_graph, cutoff_ns)
        
        filtered_data = filtered_data.to(device)
        
        # Generate embeddings on filtered graph
        with torch.no_grad():
            z_listing = model(filtered_data.x_dict, filtered_data.edge_index_dict)
        
        if isinstance(z_listing, dict):
            z_listing = z_listing['listing']
        
        embeddings = z_listing.cpu().numpy()
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Clean up GPU memory after embedding generation
        del filtered_data, z_listing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Create embedding DataFrame
        embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
        
        # Map embeddings to insertion_ids
        train_insertion_ids = train_data["insertion_id"].to_list()
        test_insertion_ids = test_data["insertion_id"].to_list()
        
        # Get indices for train and test listings (using FULL graph mapping)
        train_indices = [listing_id_to_idx.get(insertion_id, -1) for insertion_id in train_insertion_ids]
        test_indices = [listing_id_to_idx.get(insertion_id, -1) for insertion_id in test_insertion_ids]
        
        # Log index mapping stats
        train_found = sum(1 for idx in train_indices if idx >= 0)
        test_found = sum(1 for idx in test_indices if idx >= 0)
        logger.info(f"Index mapping: train {train_found}/{len(train_indices)}, test {test_found}/{len(test_indices)}")
        
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
        
        # Log non-zero embedding stats
        non_zero_train = sum(1 for emb in train_embeddings if any(abs(v) > 1e-10 for v in emb))
        non_zero_test = sum(1 for emb in test_embeddings if any(abs(v) > 1e-10 for v in emb))
        logger.info(f"Non-zero embeddings: train {non_zero_train}/{len(train_embeddings)}, test {non_zero_test}/{len(test_embeddings)}")
        
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


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Train SAGE hybrid model: embeddings + XGBoost.
    
    Pipeline:
    1. Train SAGE model on graph (temporal split)
    2. Create embedding generator for per-window embedding generation
    3. Train hybrid XGBoost model with accumulating window
    """
    if not NODES_LISTING.exists():
        raise FileNotFoundError("Artifacts not found. Run 'make etl' first.")
    
    # Get config values
    epochs = cfg.get("gnn", {}).get("epochs", 25)
    max_windows = cfg.model.training.get("max_windows", None)
    
    logger.info(f"Training SAGE hybrid model...")
    logger.info(f"  Experiment: {cfg.experiment_name}")
    logger.info(f"  GNN epochs: {epochs}")
    logger.info(f"  Feature categories: {cfg.features.categories}")
    
    # Step 1: Train SAGE model (once, on training split)
    gnn_model_uri = train_sage_embeddings(epochs=epochs)
    
    # Step 2: Create embedding generator (will generate embeddings per window)
    embedding_generator = create_sage_embedding_generator()
    
    # Step 3: Load base data (without embeddings - they'll be generated per window)
    df = load_data()
    
    # NOTE: Base features are computed by FeatureProcessor inside trainer.py
    # per-window with the correct temporal cutoff. Removed duplicate call here
    # that used datetime.now() which was semantically incorrect.
    
    # Step 4: Override model name for hybrid
    hybrid_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    hybrid_cfg.model.name = "hybrid_sage"
    
    # Step 6: Train hybrid model with per-window embeddings
    result = train_accumulating_window(
        df,
        config=hybrid_cfg,
        max_windows=max_windows,
        embedding_generator=embedding_generator
    )
    
    logger.info(f"SAGE hybrid training complete. Mean AUC-PR: {result['mean_auc_pr']:.4f}")
    
    return result


if __name__ == "__main__":
    main()

"""
SAGE (GraphSAGE) Model Training

GraphSAGE implementation for fraud detection with inductive learning.
Trains GNN embeddings that can be used standalone or with XGBoost.
"""
import logging
import pickle
from datetime import datetime

import hydra
import mlflow
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch_geometric.nn import Linear, SAGEConv, to_hetero

from src.utils.hydra_utils import resolve_path

logger = logging.getLogger(__name__)

GRAPH_PT = resolve_path("artifacts/graph.pt")
MODEL_SAGE_BEST = resolve_path("artifacts/model_sage_best.pt")
EMBEDDINGS_SAGE = resolve_path("artifacts/embeddings_sage.pt")
NODES_LISTING = resolve_path("artifacts/nodes_listing.parquet")
MAPPINGS_PKL = resolve_path("artifacts/mappings.pkl")

from src.models.xgb_trainer.trainer import train_single_window
from src.models.utils.common import get_device, setup_mlflow, filter_graph_by_time
from src.models.utils.mlflow_helpers import get_model_dependencies
from src.utils.metrics import calculate_metrics
from src.data.graph.graph_structure import build_graph, get_graph_metadata, validate_graph_metadata
from src.data.training_loader import load_data
from src.features.xgboost.processor import FeatureProcessor
from src.utils.temporal_split import AccumulatingWindowSplitter, TemporalTrainTestSplitter


class GraphSAGE(nn.Module):
    """
    GraphSAGE (homogeneous graph) model.
    
    Uses mean aggregation to combine neighbor features.
    Converted to heterogeneous via to_hetero() wrapper.
    """
    
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)


class SAGEWrapper(nn.Module):
    """
    GraphSAGE wrapper for heterogeneous fraud detection graphs.
    
    Key features:
    - Inductive learning (can handle unseen nodes)
    - Skip connections to prevent over-smoothing
    - Mean aggregation for stable neighbor combination
    - Converts homogeneous SAGE to heterogeneous via to_hetero()
    """
    
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_layers=2):
        super().__init__()
        
        # Input projections for all node types (listing, user, location, etc.)
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        
        # Core GraphSAGE model (homogeneous) converted to heterogeneous
        model = GraphSAGE(hidden_channels, hidden_channels, num_layers)
        self.gnn = to_hetero(model, metadata, aggr='mean')
        
        # Output projection with skip connection
        # Concatenates self-features with neighbor aggregation to prevent over-smoothing
        self.lin_out = Linear(hidden_channels * 2, out_channels)
        
        # Binary classifier for fraud prediction
        self.classifier = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        """
        Generate node embeddings.
        
        Args:
            x_dict: Dict of node type -> feature tensor
            edge_index_dict: Dict of edge type -> edge index tensor
            
        Returns:
            Listing node embeddings (Tensor)
        """
        # Project all node types to hidden dimension
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:
                x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
        
        # Cache listing self-representation before message passing
        listing_self = x_dict_proj['listing']
        
        # Apply GraphSAGE message passing
        x_dict_out = self.gnn(x_dict_proj, edge_index_dict)
        
        # Skip connection: concatenate self features with aggregated neighbors
        listing_out = x_dict_out['listing']
        z_listing = torch.cat([listing_self, listing_out], dim=-1)
        z_listing = self.lin_out(z_listing)
        
        return z_listing

    def predict(self, x_dict, edge_index_dict):
        """Generate fraud predictions (for training)."""
        z = self.forward(x_dict, edge_index_dict)
        return self.classifier(z)


def train_sage_embeddings(epochs=25, split_percent=0.8):
    """
    Train SAGE embeddings with optimized parameters.
    
    Optimized settings:
    - 25 epochs (more than default for better convergence)
    - Learning rate: 0.001 (standard for GNNs)
    - 2 layers, 64 hidden channels (optimal for this graph size)
    
    Returns:
        str: Path to saved embeddings file (e.g., "artifacts/embeddings_sage.pt")
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

    # Temporal split using utility from features
    timestamps = data['listing'].timestamp.numpy()
    splitter = TemporalTrainTestSplitter(
        timestamps=timestamps,
        split_percent=split_percent,
        start_threshold_percentile=10
    )
    train_mask, test_mask = splitter.split()
    split_info = splitter.get_split_info()
    
    logger.info(f"Temporal split: {split_info['train_size']} train, {split_info['test_size']} test")
    split_time = split_info['split_time']

    # Create training subgraph
    train_data = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)
    
    # Get and validate metadata from filtered graph
    filtered_metadata = train_data.metadata()
    validate_graph_metadata(filtered_metadata, strict=False)
    
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
    gnn_embeddings_path = None
    try:
        mlflow.log_params({
            "epochs": epochs,
            "split_percent": split_percent,
            "hidden_channels": 64,
            "num_layers": 2,
            "learning_rate": 0.001,
            **{f"split_{k}": v for k, v in split_info.items() if k not in ["split_time_readable", "start_threshold_readable"]}
        })
    
        train_mask_device = torch.from_numpy(train_mask).to(device)
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            out = model.predict(train_data.x_dict, train_data.edge_index_dict)
            loss = F.binary_cross_entropy_with_logits(
                out[train_mask_device], 
                train_data['listing'].y[train_mask_device].float().view(-1, 1)
            )
            
            loss.backward()
            optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), MODEL_SAGE_BEST)
                mlflow.log_metric("best_loss", best_loss.item())

        # Load best model
        model.load_state_dict(torch.load(MODEL_SAGE_BEST, weights_only=False))
        
        # Evaluate on test split
        if test_mask.sum() > 0:
            # Use full graph for evaluation (will reuse for embeddings)
            full_data = data.to(device)
            test_mask_device = torch.from_numpy(test_mask).to(device)
            model.eval()
            with torch.no_grad():
                test_out = model.predict(full_data.x_dict, full_data.edge_index_dict)
                test_pred = test_out[test_mask_device].sigmoid().cpu().numpy().flatten()
                test_y = data['listing'].y[test_mask_device].cpu().numpy()
            
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
            
        # Save embeddings to disk (this is what downstream XGBoost uses)
        torch.save(z_listing, EMBEDDINGS_SAGE)
        mlflow.log_artifact(str(EMBEDDINGS_SAGE))
        
        logger.info(f"GNN embeddings saved to {EMBEDDINGS_SAGE}")
        logger.info(f"  Shape: {z_listing.shape}")
        logger.info(f"  Note: GNN acts as feature extractor - embeddings.pt is the output artifact")
        
    finally:
        mlflow.end_run()
    
    # Return embeddings path instead of model URI (since we don't register the model)
    return str(EMBEDDINGS_SAGE)


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
    
    # Get metadata efficiently (uses cached metadata.pkl if available)
    metadata = get_graph_metadata()
    validate_graph_metadata(metadata, strict=True)
    
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
    # This is critical: graph node indices are determined by graph_structure.py,
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
    
    Orchestrates:
    1. Train SAGE model on graph (temporal split)
    2. Create embedding generator for per-window embedding generation
    3. Load dataset and create temporal splits
    4. Process features and embeddings per window
    5. Train hybrid XGBoost model with GNN embeddings
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
    gnn_embeddings_path = train_sage_embeddings(epochs=epochs)
    
    # Step 2: Create embedding generator (will generate embeddings per window)
    embedding_generator = create_sage_embedding_generator()
    
    # Step 3: Load dataset
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # Step 4: Create temporal splitter
    splitter = AccumulatingWindowSplitter(
        df=df,
        initial_window_days=cfg.model.training.initial_window_days,
        step_days=cfg.model.training.step_days,
        test_days=cfg.model.training.get("test_days", 14),
        max_windows=max_windows
    )
    
    # Step 5: Setup MLflow for hybrid training
    setup_mlflow(cfg.experiment_name)
    
    with mlflow.start_run(
        run_name=f"hybrid_sage_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": "hybrid_sage", "training_mode": "accumulating_window"}
    ) as parent_run:
        
        # Log configuration
        mlflow.log_params({
            "model_name": "hybrid_sage",
            "initial_window_days": cfg.model.training.initial_window_days,
            "step_days": cfg.model.training.step_days,
            "feature_categories": ",".join(cfg.features.categories),
            "uses_gnn_embeddings": True,
            "gnn_embeddings_path": gnn_embeddings_path,
        })
        
        # Step 6: Train per window with embeddings
        results = []
        best_auc_pr = 0
        best_run_id = None
        best_model_uri = None
        
        feature_processor = FeatureProcessor.from_config(cfg.features)
        
        for window_idx, train_data, test_data, window_info in splitter.split():
            logger.info(
                f"Window {window_idx}: "
                f"train up to {window_info['train_end'].date()}, "
                f"test {window_info['test_start'].date()} → {window_info['test_end'].date()}"
            )
            
            # Process base features
            train_processed, train_feature_cols = feature_processor.process(
                train_data,
                cutoff_date=window_info['train_end']
            )
            test_processed, test_feature_cols = feature_processor.process(
                test_data,
                cutoff_date=window_info['train_end'],
                expected_columns=train_feature_cols
            )
            
            # Generate embeddings for this window
            try:
                logger.info(f"Generating embeddings for window {window_idx}...")
                train_embed_df, test_embed_df, embed_cols = embedding_generator(
                    train_processed, test_processed, window_info['train_end']
                )
                
                if not embed_cols:
                    logger.warning("Embedding generator returned empty embed_cols, skipping embeddings")
                    all_feature_cols = train_feature_cols
                else:
                    # Merge embeddings with base features
                    train_processed = train_processed.join(train_embed_df, on="insertion_id", how="left")
                    test_processed = test_processed.join(test_embed_df, on="insertion_id", how="left")
                    
                    # Fill nulls with zeros
                    for col in embed_cols:
                        train_processed = train_processed.with_columns(pl.col(col).fill_null(0.0))
                        test_processed = test_processed.with_columns(pl.col(col).fill_null(0.0))
                    
                    # Combine feature columns
                    all_feature_cols = train_feature_cols + embed_cols
                    logger.info(f"Using {len(all_feature_cols)} features ({len(train_feature_cols)} base + {len(embed_cols)} embeddings)")
                
            except (KeyError, AttributeError) as e:
                logger.error(f"Embedding generation failed due to data structure issue: {e}", exc_info=True)
                all_feature_cols = train_feature_cols
                logger.warning("Continuing without embeddings")
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {type(e).__name__}: {e}", exc_info=True)
                all_feature_cols = train_feature_cols
                logger.warning("Continuing without embeddings")
            
            # Convert to pandas
            train_df = train_processed.to_pandas()
            test_df = test_processed.to_pandas()
            
            # Train model
            result = train_single_window(
                train_df=train_df,
                test_df=test_df,
                feature_cols=all_feature_cols,
                target_col="is_fraud",
                xgb_params=dict(cfg.model.params),
                window_idx=window_idx,
                log_model=True,
                nested=True
            )
            
            if result.get("skipped"):
                continue
            
            # Track best model
            if result["auc_pr"] > best_auc_pr:
                best_auc_pr = result["auc_pr"]
                best_run_id = result.get("run_id")
                best_model_uri = result.get("model_uri")
            
            results.append({
                "window_idx": window_idx,
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "auc_pr": result["auc_pr"],
                "auc_roc": result["auc_roc"],
                "p@100": result["p@100"],
            })
        
        # Step 7: Aggregate results
        if results:
            results_df = pl.DataFrame(results)
            mean_auc_pr = float(results_df["auc_pr"].mean())
            mean_auc_roc = float(results_df["auc_roc"].mean())
            
            mlflow.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "mean_auc_roc": mean_auc_roc,
                "best_auc_pr": best_auc_pr,
                "num_windows": float(len(results)),
            })
    
            logger.info(f"SAGE hybrid training complete. Mean AUC-PR: {mean_auc_pr:.4f}")
            
            # Register best model
            if best_model_uri is not None:
                registered_model = mlflow.register_model(
                    model_uri=best_model_uri,
                    name="fraud-detection-hybrid_sage"
                )
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=registered_model.name,
                    version=registered_model.version,
                    description=f"Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
                )
                logger.info(f"Registered {registered_model.name} version {registered_model.version}")
    
            return {
                "run_id": parent_run.info.run_id,
                "best_run_id": best_run_id,
                "results": results,
                "mean_auc_pr": mean_auc_pr,
                "best_auc_pr": best_auc_pr,
                "num_windows": len(results),
            }
        else:
            logger.warning("No windows processed!")
            return {
                "run_id": parent_run.info.run_id,
                "results": [],
                "mean_auc_pr": 0,
                "best_auc_pr": 0,
                "num_windows": 0,
            }


if __name__ == "__main__":
    main()

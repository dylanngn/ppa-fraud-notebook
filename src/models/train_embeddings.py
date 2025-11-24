import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import numpy as np
import polars as pl
from datetime import datetime
from src.models.gnn_variants import UnifiedGNNWrapper
from src.models.hgt_with_rte import HGTWrapperWithRTE

def filter_graph_by_time(data, max_time_ns):
    """
    Returns a subgraph containing only edges and nodes visible at max_time_ns.
    """
    from torch_geometric.data import HeteroData
    
    new_data = HeteroData()
    
    # Copy Node Features
    for node_type, x in data.x_dict.items():
        new_data[node_type].x = x
        new_data[node_type].num_nodes = data[node_type].num_nodes
        
    # Copy Listing Labels & Timestamps
    new_data['listing'].y = data['listing'].y
    new_data['listing'].timestamp = data['listing'].timestamp
    
    # Filter Edges by timestamp
    edge_time_dict = {}
    for edge_type, edge_index in data.edge_index_dict.items():
        src, rel, dst = edge_type
        
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

def train_embeddings(model_name="hgt", epochs=20, split_percent=0.8, window_days=90, step_days=14):
    print(f"Training Embeddings for Model: {model_name.upper()}")
    
    if not os.path.exists("artifacts/graph.pt"):
        print("Graph not found. Run graph_builder.py first.")
        return

    data = torch.load("artifacts/graph.pt", weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # Temporal Split
    timestamps = data['listing'].timestamp.numpy()
    
    # Filter valid timestamps (>= 2023)
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

    # Create Training Graph
    print("Creating Training Subgraph...")
    train_data, train_edge_times = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)
    
    # Initialize Model
    if model_name == "hgt_rte":
        # RTE requires edge times
        train_edge_times_device = {k: v.to(device) if v is not None else None for k, v in train_edge_times.items()}
        model = HGTWrapperWithRTE(
            hidden_channels=64, out_channels=64, num_heads=4, num_layers=2, data=train_data
        ).to(device)
    else:
        # Standard Models (GAT, GCN, HGT)
        model = UnifiedGNNWrapper(
            model_name=model_name, metadata=train_data.metadata(), 
            hidden_channels=64, out_channels=64, num_heads=4, num_layers=2
        ).to(device)
        train_edge_times_device = None # Not used for non-RTE

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Masks
    train_mask = ((train_data['listing'].timestamp <= split_time) & (train_data['listing'].timestamp >= start_threshold)).to(device)
    
    # Training Loop
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        if model_name == "hgt_rte":
            out = model.predict(train_data.x_dict, train_data.edge_index_dict, train_edge_times_device)
        else:
            out = model.predict(train_data.x_dict, train_data.edge_index_dict)
            
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask], 
            train_data['listing'].y[train_mask].float().view(-1, 1)
        )
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f"artifacts/model_{model_name}_best.pt")

    # Load Best Model
    model.load_state_dict(torch.load(f"artifacts/model_{model_name}_best.pt", weights_only=False))
    
    # Generate Embeddings for ALL nodes (Full Graph)
    print("Generating Full Graph Embeddings...")
    full_data = data.to(device)
    
    # Edge times for full data (needed for RTE)
    full_edge_times_device = {}
    if model_name == "hgt_rte":
        for edge_type in data.edge_index_dict.keys():
            if 'timestamp' in data[edge_type]:
                full_edge_times_device[edge_type] = data[edge_type].timestamp.to(device)
            else:
                full_edge_times_device[edge_type] = None
    
    model.eval()
    with torch.no_grad():
        if model_name == "hgt_rte":
            z_dict = model(full_data.x_dict, full_data.edge_index_dict, full_edge_times_device)
        else:
            z_dict = model(full_data.x_dict, full_data.edge_index_dict) # UnifiedWrapper returns z_listing directly? 
            # Wait, UnifiedWrapper.forward returns z_listing directly.
            # But HGTWrapperWithRTE.forward returns z_dict.
            # Let's check UnifiedWrapper.forward again.
            # It returns z_listing.
            # So for non-RTE, z_dict is actually z_listing tensor.
            pass

    # Handle return type difference
    if isinstance(z_dict, dict):
        z_listing = z_dict['listing'].cpu()
    else:
        z_listing = z_dict.cpu() # It's already the tensor
        
    # Save
    save_path = f"artifacts/embeddings_{model_name}.pt"
    torch.save(z_listing, save_path)
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    import typer
    typer.run(train_embeddings)

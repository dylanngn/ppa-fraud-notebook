import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import numpy as np
import polars as pl
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


def train_with_rte(epochs=20, split_percent=0.8, window_days=90, step_days=14):
    """
    Train HGT with Relative Temporal Encoding.
    """
    print("Loading Graph...")
    if not os.path.exists("artifacts/graph.pt"):
        print("Graph not found. Run graph_builder.py first.")
        return
        
    data = torch.load("artifacts/graph.pt", weights_only=False)
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")
    print("Training with Relative Temporal Encoding (RTE)")
    
    # Temporal Split
    timestamps = data['listing'].timestamp.numpy()
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    from datetime import datetime, timedelta
    
    # Filter valid timestamps for range calculation (fix 1970 issue)
    start_threshold = datetime(2023, 1, 1).timestamp() * 1e9
    valid_mask = timestamps >= start_threshold
    valid_timestamps = timestamps[valid_mask]
    
    if len(valid_timestamps) == 0:
        print("Warning: No valid timestamps found >= 2023. Using all data.")
        valid_timestamps = timestamps
        
    min_time = valid_timestamps.min()
    max_time = valid_timestamps.max()
    
    # Convert nanoseconds to seconds for datetime
    min_date = datetime.fromtimestamp(min_time / 1e9)
    max_date = datetime.fromtimestamp(max_time / 1e9)
    print(f"Data Range (Valid): {min_date} to {max_date}")
    
    split_time = np.percentile(valid_timestamps, split_percent * 100)
    split_date = datetime.fromtimestamp(split_time / 1e9)
    print(f"Splitting at: {split_date} ({split_percent*100}%)")
    
    # Create Training Graph with RTE info
    print("Creating Training Subgraph (Dynamic Slicing with RTE)...")
    train_data, train_edge_times = filter_graph_by_time(data, split_time)
    train_data = train_data.to(device)
    
    # Move edge times to device
    train_edge_times_device = {}
    for edge_type, times in train_edge_times.items():
        if times is not None:
            train_edge_times_device[edge_type] = times.to(device)
        else:
            train_edge_times_device[edge_type] = None
    
    # Test Graph
    test_data = data.to(device)
    test_edge_times_device = {}
    for edge_type in data.edge_index_dict.keys():
        if 'timestamp' in data[edge_type]:
            test_edge_times_device[edge_type] = data[edge_type].timestamp.to(device)
        else:
            test_edge_times_device[edge_type] = None
    
    # Initialize Model with RTE
    model = HGTWrapperWithRTE(
        hidden_channels=64, 
        out_channels=64, 
        num_heads=4, 
        num_layers=2, 
        data=train_data
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Masks
    # Masks (Exclude < 2023 from training)
    train_mask = ((train_data['listing'].timestamp <= split_time) & (train_data['listing'].timestamp >= start_threshold)).to(device)
    test_mask = (test_data['listing'].timestamp > split_time).to(device)
    
    print(f"Train Nodes: {train_mask.sum().item()}, Test Nodes: {test_mask.sum().item()}")
    
    # Training Loop
    train_history = []
    best_auc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass WITH temporal information
        out = model.predict(train_data.x_dict, train_data.edge_index_dict, train_edge_times_device)
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask], 
            train_data['listing'].y[train_mask].float().view(-1, 1)
        )
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model.predict(test_data.x_dict, test_data.edge_index_dict, test_edge_times_device)
            pred = out[test_mask].sigmoid().cpu().numpy()
            y_true = test_data['listing'].y[test_mask].cpu().numpy()
            
            if len(np.unique(y_true)) > 1:
                auc_pr = average_precision_score(y_true, pred)
                auc_roc = roc_auc_score(y_true, pred)
                
                # Calculate Precision@K (Lift) - operational metric
                # Shows fraud detection rate in top K predictions
                precisions_at_k = {}
                for k in [50, 100, 200]:
                    if len(pred) >= k:
                        top_k_indices = np.argsort(pred.flatten())[-k:][::-1]
                        precisions_at_k[f'p@{k}'] = y_true[top_k_indices].mean()
                    else:
                        precisions_at_k[f'p@{k}'] = 0.0
            else:
                auc_pr = 0.0
                auc_roc = 0.0
                precisions_at_k = {'p@50': 0.0, 'p@100': 0.0, 'p@200': 0.0}
            
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, "
                  f"P@100: {precisions_at_k['p@100']:.4f}")
            
            # Save epoch results
            train_history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "test_auc_pr": auc_pr,
                "test_auc_roc": auc_roc,
                "p@50": precisions_at_k['p@50'],
                "p@100": precisions_at_k['p@100'],
                "p@200": precisions_at_k['p@200']
            })
            
            # Save best model
            if auc_pr > best_auc:
                best_auc = auc_pr
                torch.save(model.state_dict(), "artifacts/model_hgt_rte_best.pt")
                
    # Load best model
    model.load_state_dict(torch.load("artifacts/model_hgt_rte_best.pt", weights_only=False))
    
    # ===== PHASE 2: Sliding Window Evaluation =====
    print(f"\n{'='*60}")
    print(f"PHASE 2: Sliding Window Evaluation ({window_days} days train, {step_days} days test)")
    print(f"{'='*60}\n")
    
    # Move full data to device
    full_data = data.to(device)
    
    # Edge times for full data
    full_edge_times_device = {}
    for edge_type in data.edge_index_dict.keys():
        if 'timestamp' in data[edge_type]:
            full_edge_times_device[edge_type] = data[edge_type].timestamp.to(device)
        else:
            full_edge_times_device[edge_type] = None

    window_size_ns = window_days * 24 * 60 * 60 * 1e9
    step_size_ns = step_days * 24 * 60 * 60 * 1e9
    
    current_time = max(min_time + window_size_ns, start_threshold)
    
    results = []
    
    model.eval()
    with torch.no_grad():
        while current_time + step_size_ns <= max_time:
            test_start_time = current_time
            test_end_time = current_time + step_size_ns
            
            test_mask = (
                (full_data['listing'].timestamp >= test_start_time) & 
                (full_data['listing'].timestamp < test_end_time)
            ).to(device)
            
            if test_mask.sum() == 0:
                current_time += step_size_ns
                continue
                
            out = model.predict(full_data.x_dict, full_data.edge_index_dict, full_edge_times_device)
            pred = out[test_mask].sigmoid().cpu().numpy()
            y_true = full_data['listing'].y[test_mask].cpu().numpy()
            
            if len(np.unique(y_true)) > 1:
                auc_pr = average_precision_score(y_true, pred)
                precisions_at_k = {}
                for k in [50, 100, 200]:
                    if len(pred) >= k:
                        top_k_indices = np.argsort(pred.flatten())[-k:][::-1]
                        precisions_at_k[f'p@{k}'] = y_true[top_k_indices].mean()
                    else:
                        precisions_at_k[f'p@{k}'] = 0.0
            else:
                auc_pr = 0.0
                precisions_at_k = {'p@50': 0.0, 'p@100': 0.0, 'p@200': 0.0}
            
            window_start_date = datetime.fromtimestamp(test_start_time / 1e9)
            window_end_date = datetime.fromtimestamp(test_end_time / 1e9)
            
            print(f"Window {window_start_date.date()} - {window_end_date.date()}: "
                  f"AUC-PR = {auc_pr:.4f}, P@100 = {precisions_at_k['p@100']:.4f}, "
                  f"Fraud Count = {y_true.sum()}")
            
            results.append({
                "window_start": window_start_date,
                "window_end": window_end_date,
                "auc_pr": auc_pr,
                "p@100": precisions_at_k['p@100'],
                "fraud_count": int(y_true.sum())
            })
            
            current_time += step_size_ns
    
    # Save final embeddings
    print("Saving Embeddings with RTE...")
    model.eval()
    with torch.no_grad():
        z_dict = model(test_data.x_dict, test_data.edge_index_dict, test_edge_times_device)
        z_listing = z_dict['listing'].cpu()
        
        torch.save(z_listing, "artifacts/embeddings_listing_rte.pt")
        torch.save(model.state_dict(), "artifacts/model_hgt_rte.pt")
    
    # Save training history
    os.makedirs("artifacts/results", exist_ok=True)
    history_df = pl.DataFrame(train_history)
    history_df.write_csv("artifacts/results/gnn_rte_results.csv")
    print(f"\nSaved results to artifacts/results/gnn_rte_results.csv")
    print(f"Best Test AUC-PR: {best_auc:.4f}")
    print(f"Final Test AUC-PR: {train_history[-1]['test_auc_pr']:.4f}")
    
    return train_history


if __name__ == "__main__":
    train_with_rte(epochs=20, split_percent=0.8)

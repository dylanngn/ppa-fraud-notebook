import torch
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            # We need to know input dimension for each node type
            # This will be handled dynamically or we need to pass it.
            # For simplicity, let's assume we project all inputs to hidden_channels first
            # But we don't know input dim here easily without data.
            # Let's assume we pass a dict of input dims.
            pass

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict

class HGTWrapper(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        
        # 1. Input Projections
        self.lin_dict = torch.nn.ModuleDict()
        for node_type, x in data.x_dict.items():
            self.lin_dict[node_type] = Linear(x.size(-1), hidden_channels)
            
        # 2. HGT Core
        self.hgt = HGT(hidden_channels, out_channels, num_heads, num_layers, data.metadata())
        
        # 3. Classifier (for training only)
        self.classifier = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Project inputs
        x_dict_out = {}
        for node_type, x in x_dict.items():
            x_dict_out[node_type] = self.lin_dict[node_type](x).relu()
            
        # HGT
        x_dict_out = self.hgt(x_dict_out, edge_index_dict)
        
        return x_dict_out

    def predict(self, x_dict, edge_index_dict):
        z_dict = self(x_dict, edge_index_dict)
        return self.classifier(z_dict['listing'])

def filter_graph_by_time(data, max_time_ns):
    """
    Returns a subgraph containing only edges and nodes visible at max_time_ns.
    """
    new_data = HeteroData()
    
    # Copy Node Features (Nodes are always present, but we mask them later)
    # Ideally we should also mask nodes, but for now we mask via train/test masks.
    for node_type, x in data.x_dict.items():
        new_data[node_type].x = x
        new_data[node_type].num_nodes = data[node_type].num_nodes
        
    # Copy Listing Labels & Timestamps
    new_data['listing'].y = data['listing'].y
    new_data['listing'].timestamp = data['listing'].timestamp
    
    # Filter Edges
    for edge_type, edge_index in data.edge_index_dict.items():
        src, rel, dst = edge_type
        
        # Check if this edge type has timestamps
        if 'timestamp' in data[edge_type]:
            edge_times = data[edge_type].timestamp
            mask = edge_times <= max_time_ns
            new_data[edge_type].edge_index = edge_index[:, mask]
        else:
            # Static edges (keep all)
            # CAUTION: This assumes static edges don't leak future info.
            # In our graph_builder, we only skipped timestamp for User->IP (assumed static/always valid)
            # and User->Location.
            new_data[edge_type].edge_index = edge_index
            
    return new_data

def train(epochs=20, split_percent=0.8, window_days=90, step_days=14):
    """
    Train HGT model and evaluate on sliding windows.
    
    Two-phase approach:
    1. Train model once on historical data (first split_percent of data)
    2. Evaluate on sliding windows (like baseline XGBoost)
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
    
    # Temporal Split for TRAINING
    timestamps = data['listing'].timestamp.numpy()
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    from datetime import datetime, timedelta
    # Convert nanoseconds to seconds for datetime
    min_date = datetime.fromtimestamp(min_time / 1e9)
    max_date = datetime.fromtimestamp(max_time / 1e9)
    print(f"Data Range: {min_date} to {max_date}")
    
    # ===== PHASE 1: Initial Training =====
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training HGT on historical data ({split_percent*100}%)")
    print(f"{'='*60}\n")
    
    # Use first split_percent for training
    train_split_time = np.percentile(timestamps, split_percent * 100)
    train_split_date = datetime.fromtimestamp(train_split_time / 1e9)
    print(f"Training on data up to: {train_split_date}")
    
    # Create Training Graph
    train_data = filter_graph_by_time(data, train_split_time)
    train_data = train_data.to(device)
    
    # Initialize Model
    model = HGTWrapper(
        hidden_channels=64, 
        out_channels=64, 
        num_heads=4, 
        num_layers=2, 
        data=train_data
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training mask
    train_mask = (train_data['listing'].timestamp <= train_split_time).to(device)
    
    # Quick validation split (last 20% of training data)
    val_split_time = np.percentile(train_data['listing'].timestamp.cpu().numpy(), 80)
    val_mask = (train_data['listing'].timestamp > val_split_time).to(device)
    
    print(f"Train Nodes: {train_mask.sum().item()}, Val Nodes: {val_mask.sum().item()}")
    
    # Training Loop
    best_val_auc = 0.0
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
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model.predict(train_data.x_dict, train_data.edge_index_dict)
            pred = out[val_mask].sigmoid().cpu().numpy()
            y_true = train_data['listing'].y[val_mask].cpu().numpy()
            
            if len(np.unique(y_true)) > 1:
                val_auc = average_precision_score(y_true, pred)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), "artifacts/model_hgt_best.pt")
            else:
                val_auc = 0.0
                
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val AUC-PR: {val_auc:.4f}")
    
    print(f"\nBest Validation AUC-PR: {best_val_auc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load("artifacts/model_hgt_best.pt", weights_only=False))
    
    # ===== PHASE 2: Sliding Window Evaluation =====
    print(f"\n{'='*60}")
    print(f"PHASE 2: Sliding Window Evaluation ({window_days} days train, {step_days} days test)")
    print(f"{'='*60}\n")
    
    # Move full data to device for evaluation
    full_data = data.to(device)
    
    # Define sliding windows
    window_size_ns = window_days * 24 * 60 * 60 * 1e9  # days to nanoseconds
    step_size_ns = step_days * 24 * 60 * 60 * 1e9
    
    # Start from min_time + window_size
    # Ensure we don't start before 2023 (fix 1970 issue)
    start_threshold = datetime(2023, 1, 1).timestamp() * 1e9
    current_time = max(min_time + window_size_ns, start_threshold)
    
    results = []
    window_idx = 0
    
    model.eval()
    with torch.no_grad():
        while current_time + step_size_ns <= max_time:
            window_end_time = current_time
            test_start_time = current_time
            test_end_time = current_time + step_size_ns
            
            # Test mask: listings in this test window
            test_mask = (
                (full_data['listing'].timestamp >= test_start_time) & 
                (full_data['listing'].timestamp < test_end_time)
            ).to(device)
            
            if test_mask.sum() == 0:
                current_time += step_size_ns
                continue
            
            # Predict
            out = model.predict(full_data.x_dict, full_data.edge_index_dict)
            pred = out[test_mask].sigmoid().cpu().numpy()
            y_true = full_data['listing'].y[test_mask].cpu().numpy()
            
            # Evaluate
            if len(np.unique(y_true)) > 1:
                auc_pr = average_precision_score(y_true, pred)
                
                # Precision@K
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
                "p@50": precisions_at_k['p@50'],
                "p@100": precisions_at_k['p@100'],
                "p@200": precisions_at_k['p@200'],
                "fraud_count": int(y_true.sum()),
                "test_count": int(test_mask.sum().item())
            })
            
            current_time += step_size_ns
            window_idx += 1
    
    # Save Embeddings (from full graph)
    print("\nSaving Embeddings...")
    z_dict = model(full_data.x_dict, full_data.edge_index_dict)
    z_listing = z_dict['listing'].cpu()
    
    torch.save(z_listing, "artifacts/embeddings_listing.pt")
    torch.save(model.state_dict(), "artifacts/model_hgt.pt")
    
    # Save sliding window results
    os.makedirs("artifacts/results", exist_ok=True)
    results_df = pl.DataFrame(results)
    results_df.write_csv("artifacts/results/gnn_results.csv")
    
    print(f"\nSaved results to artifacts/results/gnn_results.csv")
    print(f"Total Windows Evaluated: {len(results)}")
    print(f"Mean AUC-PR: {results_df['auc_pr'].mean():.4f}")
    print(f"Mean P@100: {results_df['p@100'].mean():.4f}")
    
    return results

if __name__ == "__main__":
    train()

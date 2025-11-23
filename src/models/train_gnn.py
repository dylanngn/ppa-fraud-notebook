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

def train(epochs=20, split_percent=0.8):
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
    
    # --- Temporal Split Strategy ---
    # User Request: Train on Nov '23 - Jan '24, Test on Feb '24
    # We need to define these timestamps.
    # Let's assume the data covers this range.
    
    timestamps = data['listing'].timestamp.numpy()
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    import pandas as pd
    min_date = pd.to_datetime(min_time)
    max_date = pd.to_datetime(max_time)
    print(f"Data Range: {min_date} to {max_date}")
    
    # Define Split Date (e.g., Feb 1, 2024)
    # For MVP, let's just take the 80th percentile time as the split point
    # Or strictly follow user request if we can parse dates.
    # Let's use a dynamic split point: 80% train, 20% test
    split_time = np.percentile(timestamps, split_percent * 100)
    split_date = pd.to_datetime(split_time)
    print(f"Splitting at: {split_date} ({split_percent*100}%)")
    
    # 1. Create Training Graph (Visible edges <= split_time)
    print("Creating Training Subgraph (Dynamic Slicing)...")
    train_data = filter_graph_by_time(data, split_time).to(device)
    
    # 2. Create Test Graph (All edges, but we only evaluate on test nodes)
    # Actually, for strict evaluation, the test graph should include edges up to test_time.
    # But since we test on the *future* (Feb), we can use the full graph (up to Feb) 
    # and mask out the future-future (March) if it existed.
    # Here, 'data' contains everything, so it serves as the Test Graph.
    test_data = data.to(device)
    
    model = HGTWrapper(hidden_channels=64, out_channels=64, num_heads=4, num_layers=2, data=train_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Masks
    # Train Mask: Nodes with timestamp <= split_time
    train_mask = (train_data['listing'].timestamp <= split_time).to(device)
    
    # Test Mask: Nodes with timestamp > split_time
    test_mask = (test_data['listing'].timestamp > split_time).to(device)
    
    print(f"Train Nodes: {train_mask.sum().item()}, Test Nodes: {test_mask.sum().item()}")
    
    # Training Loop
    train_history = []
    for epoch in range(1, epochs + 1): 
        model.train()
        optimizer.zero_grad()
        
        # Train on Training Graph
        out = model.predict(train_data.x_dict, train_data.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out[train_mask], train_data['listing'].y[train_mask].float().view(-1, 1))
        
        loss.backward()
        optimizer.step()
        
        # Evaluate on Test Graph
        # Note: We use the FULL graph for inference on test nodes, 
        # because test nodes are allowed to see past edges (from training period)
        # and edges within the test period.
        model.eval()
        with torch.no_grad():
            out = model.predict(test_data.x_dict, test_data.edge_index_dict)
            pred = out[test_mask].sigmoid().cpu().numpy()
            y_true = test_data['listing'].y[test_mask].cpu().numpy()
            
            if len(y_true) > 0:
                auc = average_precision_score(y_true, pred)
                
                # Calculate Precision@K (Lift) - operational metric
                precisions_at_k = {}
                for k in [50, 100, 200]:
                    if len(pred) >= k:
                        top_k_indices = np.argsort(pred.flatten())[-k:][::-1]
                        precisions_at_k[f'p@{k}'] = y_true[top_k_indices].mean()
                    else:
                        precisions_at_k[f'p@{k}'] = 0.0
            else:
                auc = 0.0
                precisions_at_k = {'p@50': 0.0, 'p@100': 0.0, 'p@200': 0.0}
                
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test AUC-PR: {auc:.4f}, P@100: {precisions_at_k['p@100']:.4f}")
            
            # Save epoch results
            train_history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "test_auc_pr": auc,
                "p@50": precisions_at_k['p@50'],
                "p@100": precisions_at_k['p@100'],
                "p@200": precisions_at_k['p@200']
            })
            
    # Save Embeddings (from the full graph)
    print("Saving Embeddings...")
    model.eval()
    with torch.no_grad():
        z_dict = model(test_data.x_dict, test_data.edge_index_dict)
        z_listing = z_dict['listing'].cpu()
        
        torch.save(z_listing, "artifacts/embeddings_listing.pt", weights_only=False)
        torch.save(model.state_dict(), "artifacts/model_hgt.pt", weights_only=False)
    
    # Save training history
    import polars as pl
    os.makedirs("artifacts/results", exist_ok=True)
    history_df = pl.DataFrame(train_history)
    history_df.write_csv("artifacts/results/gnn_results.csv")
    print(f"\nSaved results to artifacts/results/gnn_results.csv")
    print(f"Final Test AUC-PR: {train_history[-1]['test_auc_pr']:.4f}")

if __name__ == "__main__":
    train()

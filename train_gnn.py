import torch
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import os
from tqdm import tqdm
from sklearn.metrics import average_precision_score

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
                           num_heads, group='sum')
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

def train():
    print("Loading Graph...")
    if not os.path.exists("artifacts/graph.pt"):
        print("Graph not found. Run graph_builder.py first.")
        return
        
    data = torch.load("artifacts/graph.pt")
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    print(f"Using device: {device}")
    
    data = data.to(device)
    
    model = HGTWrapper(hidden_channels=64, out_channels=64, num_heads=4, num_layers=2, data=data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train/Test Split (Simple time-based mask for now)
    # We stored timestamp in data['listing'].timestamp
    # Let's take last 20% as test
    timestamps = data['listing'].timestamp
    split_idx = int(len(timestamps) * 0.8)
    sorted_idx = torch.argsort(timestamps)
    
    train_idx = sorted_idx[:split_idx]
    test_idx = sorted_idx[split_idx:]
    
    train_mask = torch.zeros(data['listing'].num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    
    test_mask = torch.zeros(data['listing'].num_nodes, dtype=torch.bool, device=device)
    test_mask[test_idx] = True
    
    # Training Loop
    for epoch in range(1, 21): # 20 epochs
        model.train()
        optimizer.zero_grad()
        
        out = model.predict(data.x_dict, data.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out[train_mask], data['listing'].y[train_mask].float().view(-1, 1))
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model.predict(data.x_dict, data.edge_index_dict)
            pred = out[test_mask].sigmoid().cpu().numpy()
            y_true = data['listing'].y[test_mask].cpu().numpy()
            
            auc = average_precision_score(y_true, pred)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test AUC-PR: {auc:.4f}")
            
    # Save Embeddings
    print("Saving Embeddings...")
    model.eval()
    with torch.no_grad():
        z_dict = model(data.x_dict, data.edge_index_dict)
        z_listing = z_dict['listing'].cpu()
        
        torch.save(z_listing, "artifacts/embeddings_listing.pt")
        torch.save(model.state_dict(), "artifacts/model_hgt.pt")

if __name__ == "__main__":
    train()

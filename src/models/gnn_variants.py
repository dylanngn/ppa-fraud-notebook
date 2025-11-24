import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, HGTConv, Linear, to_hetero
from src.models.hgt_with_rte import HGTWithRTE

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, add_self_loops=False)
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

class UnifiedGNNWrapper(nn.Module):
    """
    Unified wrapper for all GNN variants.
    Handles input projection and output classification/embedding.
    """
    def __init__(self, model_name, metadata, hidden_channels=64, out_channels=64, num_heads=4, num_layers=2):
        super().__init__()
        self.model_name = model_name
        
        # 1. Input Projections (for all node types)
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            
        # 2. Core GNN
        if model_name == "gat":
            model = GAT(hidden_channels, hidden_channels, num_heads, num_layers)
            self.gnn = to_hetero(model, metadata, aggr='sum')
        elif model_name == "gcn":
            model = GCN(hidden_channels, hidden_channels, num_layers)
            self.gnn = to_hetero(model, metadata, aggr='sum')
        elif model_name == "hgt":
            self.gnn = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            # HGTConv is a single layer, we need multiple
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
        elif model_name == "hgt_rte":
            # Handled separately due to custom RTE logic
            pass
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # 3. Output Projection (to embedding dim)
        self.lin_out = Linear(hidden_channels, out_channels)
        
        # 4. Classifier (for training)
        self.classifier = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Project Inputs
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
            
        # Apply GNN
        if self.model_name in ["gat", "gcn"]:
            x_dict_out = self.gnn(x_dict_proj, edge_index_dict)
        elif self.model_name == "hgt":
            x_dict_out = x_dict_proj
            for conv in self.convs:
                x_dict_out = conv(x_dict_out, edge_index_dict)
                
        # Output Projection
        z_listing = self.lin_out(x_dict_out['listing'])
        return z_listing

    def predict(self, x_dict, edge_index_dict):
        z = self.forward(x_dict, edge_index_dict)
        return self.classifier(z)

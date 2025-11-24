import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv, HGTConv, Linear, to_hetero


class TemporalEncoding(nn.Module):
    """
    Sinusoidal + learnable temporal encoding used by HGT with RTE.
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

class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels)
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
        self.use_rte = model_name == "hgt_rte"
        
        # 1. Input Projections (for all node types)
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
            
        # 2. Core GNN
        if model_name == "gat":
            model = GAT(hidden_channels, hidden_channels, num_heads, num_layers)
            self.gnn = to_hetero(model, metadata, aggr='sum')
        elif model_name == "sage":
            model = GraphSAGE(hidden_channels, hidden_channels, num_layers)
            self.gnn = to_hetero(model, metadata, aggr='mean')
        elif model_name in ["hgt", "hgt_rte"]:
            self.gnn = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            # HGTConv is a single layer, we need multiple
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))
            if self.use_rte:
                self.temporal_encoder = TemporalEncoding(hidden_channels)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        # 3. Output Projection (preserve self-features via skip connection)
        self.lin_out = Linear(hidden_channels * 2, out_channels)
        
        # 4. Classifier (for training)
        self.classifier = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        # Project Inputs
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
            
        # Cache listing self-representation before message passing to avoid over-smoothing
        listing_self = x_dict_proj['listing']
        
        # Apply GNN
        temporal_embeddings = None
        if self.use_rte and edge_time_dict is not None:
            temporal_embeddings = self._compute_temporal_embeddings(edge_time_dict)

        if self.model_name in ["gat", "sage"]:
            x_dict_out = self.gnn(x_dict_proj, edge_index_dict)
        elif self.model_name in ["hgt", "hgt_rte"]:
            x_dict_out = x_dict_proj
            for conv in self.convs:
                x_dict_out = conv(x_dict_out, edge_index_dict)
                if temporal_embeddings:
                    x_dict_out = self._apply_temporal_bias(x_dict_out, edge_index_dict, temporal_embeddings)
                
        # Concatenate self features with aggregated message before projection
        listing_out = x_dict_out['listing']
        z_listing = torch.cat([listing_self, listing_out], dim=-1)
        z_listing = self.lin_out(z_listing)
        return z_listing

    def predict(self, x_dict, edge_index_dict, edge_time_dict=None):
        z = self.forward(x_dict, edge_index_dict, edge_time_dict=edge_time_dict)
        return self.classifier(z)

    def _compute_temporal_embeddings(self, edge_time_dict):
        embeddings = {}
        for edge_type, timestamps in edge_time_dict.items():
            if timestamps is None or timestamps.numel() == 0:
                continue
            min_time = timestamps.min()
            time_diff = (timestamps - min_time).float()
            embeddings[edge_type] = self.temporal_encoder(time_diff)
        return embeddings

    def _apply_temporal_bias(self, x_dict_out, edge_index_dict, temporal_embeddings):
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
                x_dict_out[node_type] = node_repr + 0.1 * temporal_bias
        return x_dict_out

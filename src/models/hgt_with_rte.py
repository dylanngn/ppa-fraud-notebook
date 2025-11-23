import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData
import math


class TemporalEncoding(nn.Module):
    """
    Encodes temporal differences (ΔT) into learnable representations.
    Uses a combination of fixed sinusoidal encoding and learnable MLP.
    """
    def __init__(self, hidden_channels, max_time_scale=1e12):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.max_time_scale = max_time_scale
        
        # Learnable MLP to transform time differences
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def sinusoidal_encoding(self, time_diff):
        """
        Sinusoidal positional encoding for time differences.
        Similar to Transformer temporal encoding.
        """
        # Normalize time to [0, 1] range
        time_normalized = time_diff / self.max_time_scale
        
        # Create frequencies
        freqs = torch.arange(0, self.hidden_channels, 2, device=time_diff.device, dtype=torch.float)
        freqs = 1.0 / (10000 ** (freqs / self.hidden_channels))
        
        # Compute sinusoidal encoding
        encoding = torch.zeros(time_diff.size(0), self.hidden_channels, device=time_diff.device)
        encoding[:, 0::2] = torch.sin(time_normalized.unsqueeze(-1) * freqs)
        encoding[:, 1::2] = torch.cos(time_normalized.unsqueeze(-1) * freqs)
        
        return encoding
    
    def forward(self, time_diff):
        """
        Args:
            time_diff: Tensor of shape [num_edges] containing time differences in nanoseconds
        Returns:
            Temporal embeddings of shape [num_edges, hidden_channels]
        """
        # Get sinusoidal encoding
        sin_encoding = self.sinusoidal_encoding(time_diff)
        
        # Apply learnable transformation
        time_embedding = self.time_mlp(sin_encoding)
        
        return time_embedding


class HGTWithRTE(nn.Module):
    """
    Heterogeneous Graph Transformer with Relative Temporal Encoding.
    
    Key Enhancement: Incorporates ΔT (time difference) into attention mechanism
    to distinguish between long-established patterns and rapid fraud attacks.
    """
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, metadata):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Temporal encoding module
        self.temporal_encoder = TemporalEncoding(hidden_channels)
        
        # HGT convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
        
        # Output projection
        self.lin = Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        """
        Args:
            x_dict: Dictionary of node features {node_type: features}
            edge_index_dict: Dictionary of edge indices {edge_type: edge_index}
            edge_time_dict: Dictionary of edge timestamps {edge_type: timestamps}
                           (optional, if None, operates without RTE)
        """
        # Apply temporal encoding if timestamps are provided
        if edge_time_dict is not None:
            # Compute temporal embeddings for each edge type
            temporal_embeddings = {}
            for edge_type, timestamps in edge_time_dict.items():
                if timestamps is not None and len(timestamps) > 0:
                    # Compute ΔT: difference from earliest timestamp
                    # This represents "age" of the edge
                    min_time = timestamps.min()
                    time_diff = timestamps - min_time
                    temporal_embeddings[edge_type] = self.temporal_encoder(time_diff.float())
        else:
            temporal_embeddings = None
        
        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            # Standard HGT convolution
            x_dict_out = conv(x_dict, edge_index_dict)
            
            # Inject temporal information if available
            if temporal_embeddings is not None:
                # Add temporal bias to node embeddings based on incoming edges
                # This is a simplified approach - more sophisticated versions
                # would modify the attention mechanism directly
                for node_type in x_dict_out.keys():
                    # Find edges that target this node type
                    for edge_type, edge_index in edge_index_dict.items():
                        src_type, rel_type, dst_type = edge_type
                        if dst_type == node_type and edge_type in temporal_embeddings:
                            # Aggregate temporal information to target nodes
                            temp_emb = temporal_embeddings[edge_type]
                            target_nodes = edge_index[1]
                            
                            # Simple aggregation: add temporal bias
                            # More sophisticated: attention-weighted aggregation
                            temporal_bias = torch.zeros_like(x_dict_out[node_type])
                            temporal_bias.index_add_(0, target_nodes, temp_emb)
                            
                            # Scale and add
                            x_dict_out[node_type] = x_dict_out[node_type] + 0.1 * temporal_bias
            
            x_dict = x_dict_out
        
        return x_dict


class HGTWrapperWithRTE(nn.Module):
    """
    Complete wrapper for HGT with RTE, including input projections and classifier.
    """
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()
        
        # Input projection layers for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type, x in data.x_dict.items():
            self.lin_dict[node_type] = Linear(x.size(-1), hidden_channels)
        
        # HGT core with RTE
        self.hgt = HGTWithRTE(hidden_channels, out_channels, num_heads, num_layers, data.metadata())
        
        # Classifier head for fraud prediction
        self.classifier = Linear(hidden_channels, 1)
    
    def forward(self, x_dict, edge_index_dict, edge_time_dict=None):
        # Project inputs
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()
        
        # Apply HGT with RTE
        x_dict_out = self.hgt(x_dict_proj, edge_index_dict, edge_time_dict)
        
        return x_dict_out
    
    def predict(self, x_dict, edge_index_dict, edge_time_dict=None):
        """Predict fraud labels for listing nodes."""
        z_dict = self.forward(x_dict, edge_index_dict, edge_time_dict)
        return self.classifier(z_dict['listing'])

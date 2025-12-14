"""
GraphSAGE implementation with self-supervised link prediction.
Uses full-batch training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

from src.models.base import BaseEmbedder

logger = logging.getLogger(__name__)

class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for heterogeneous graphs.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        edge_types: List[str],
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            
            conv_dict = {
                ("listing", edge_type, "listing"): SAGEConv(in_ch, out_ch, aggr="mean")
                for edge_type in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple, torch.Tensor]):
        x = x_dict
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_dict)
            
            if i < self.num_layers - 1:
                x = {key: F.relu(val) for key, val in x.items()}
                x = {key: F.dropout(val, p=self.dropout, training=self.training) for key, val in x.items()}
                
        return x["listing"]


class GraphSAGEEmbedder(BaseEmbedder):
    """
    Wrapper for GraphSAGE training and inference.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 1024, # Unused in full-batch but kept for interface
        device: str = "cpu"
    ):
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        self.device = device
        self.model: Optional[GraphSAGEEncoder] = None
        
    def fit(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
        train_mask: torch.Tensor, # Unused
    ) -> "GraphSAGEEmbedder":
        
        edge_types = [e[1] for e in graph.edge_types]
        
        self.model = GraphSAGEEncoder(
            in_channels=self.config["in_channels"],
            hidden_channels=self.config["hidden_channels"],
            out_channels=self.config["out_channels"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            edge_types=edge_types
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        graph["listing"].x = node_features
        graph = graph.to(self.device)
        
        target_edge_type = graph.edge_types[0]
        pos_edge_index = graph[target_edge_type].edge_index
        
        self.model.train()
        for epoch in range(self.config["epochs"]):
            optimizer.zero_grad()
            z = self.model(graph.x_dict, graph.edge_index_dict)
            
            # Link prediction loss
            src, dst = pos_edge_index
            pos_score = (z[src] * z[dst]).sum(dim=-1)
            
            neg_src = torch.randint(0, z.size(0), (src.size(0),), device=self.device)
            neg_dst = torch.randint(0, z.size(0), (dst.size(0),), device=self.device)
            neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()
            
            logger.info(f"Epoch {epoch}: Loss {loss.item():.4f}")
            
        return self

    def transform(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        
        if self.model is None:
            raise RuntimeError("Model not fit")
            
        self.model.eval()
        graph["listing"].x = node_features
        graph = graph.to(self.device)
        
        with torch.no_grad():
            z = self.model(graph.x_dict, graph.edge_index_dict)
            return z.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config
        }, path)

    @classmethod
    def load(cls, path: str) -> "GraphSAGEEmbedder":
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        instance = cls(**config)
        # Note: Model not fully instantiated without edge_types (prototype limitation)
        return instance

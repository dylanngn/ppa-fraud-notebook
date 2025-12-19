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
        y: Optional[torch.Tensor] = None,  # Added labels for supervised training
        train_mask: Optional[torch.Tensor] = None, 
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
        
        # Find listing-to-listing edges for link prediction
        # These are the edges we use to learn embeddings (shared attributes between listings)
        listing_edges = [
            et for et in graph.edge_types 
            if et[0] == "listing" and et[2] == "listing"
        ]
        
        if y is None:
            # Self-supervised Link Prediction Setup
            if not listing_edges:
                logger.warning("No listing-to-listing edges found, using all edges")
                target_edge_type = graph.edge_types[0]
            else:
                target_edge_type = listing_edges[0]
            
            pos_edge_index = graph[target_edge_type].edge_index
        else:
            # Supervised Node Classification Setup
            # Create a temporary classification head
            self.predictor = nn.Linear(self.config["out_channels"], 1).to(self.device)
            # Add predictor parameters to optimizer
            optimizer.add_param_group({'params': self.predictor.parameters()})
            
            # Ensure y is float for BCEWithLogits
            y = y.float().to(self.device)
            if train_mask is not None:
                train_mask = train_mask.bool().to(self.device)
            else:
                 # Default to using all labeled data if no mask provided
                train_mask = torch.ones(y.size(0), dtype=torch.bool, device=self.device)
        
        self.model.train()
        for epoch in range(self.config["epochs"]):
            optimizer.zero_grad()
            z = self.model(graph.x_dict, graph.edge_index_dict)
            
            if y is None:
                # ---------------------------
                # Self-Supervised Link Prediction
                # ---------------------------
                src, dst = pos_edge_index
                pos_score = (z[src] * z[dst]).sum(dim=-1)
                
                neg_src = torch.randint(0, z.size(0), (src.size(0),), device=self.device)
                neg_dst = torch.randint(0, z.size(0), (dst.size(0),), device=self.device)
                neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
                
                scores = torch.cat([pos_score, neg_score])
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                
                loss = F.binary_cross_entropy_with_logits(scores, labels)
            else:
                # ---------------------------
                # Supervised Node Classification
                # ---------------------------
                # Project embeddings to logits
                logits = self.predictor(z).squeeze(-1)
                
                # Compute loss only on training nodes
                loss = F.binary_cross_entropy_with_logits(
                    logits[train_mask], 
                    y[train_mask]
                )
            
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

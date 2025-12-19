"""
Heterogeneous Graph Transformer (HGT) implementation.
Drop-in replacement for GraphSAGE - uses attention mechanism for heterogeneous graphs.

Reference: "Heterogeneous Graph Transformer" (WWW 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, Optional, List
import numpy as np
import logging

from src.models.base import BaseEmbedder

logger = logging.getLogger(__name__)


class HGTEncoder(nn.Module):
    """
    Heterogeneous Graph Transformer encoder.
    Uses multi-head attention to learn edge-type-specific patterns.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Linear projection for input features (using standard PyTorch Linear)
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        # HGT convolution layers
        # Note: HGTConv doesn't take dropout as parameter, we apply it between layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels if i < num_layers - 1 else out_channels,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
        
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor]
    ) -> torch.Tensor:
        # Project input features for listing nodes
        x_dict = {
            node_type: self.lin_in(x) if x.size(-1) == self.in_channels else x
            for node_type, x in x_dict.items()
        }
        
        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            if i < self.num_layers - 1:
                x_dict = {
                    key: F.relu(val) 
                    for key, val in x_dict.items()
                }
                x_dict = {
                    key: F.dropout(val, p=self.dropout, training=self.training) 
                    for key, val in x_dict.items()
                }
                
        return x_dict["listing"]


class HGTEmbedder(BaseEmbedder):
    """
    Wrapper for HGT training and inference.
    Drop-in replacement for GraphSAGEEmbedder.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 1024,  # Unused in full-batch
        device: str = "cpu"
    ):
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        self.device = device
        self.model: Optional[HGTEncoder] = None
        
    def fit(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
        y: Optional[torch.Tensor] = None,  # Added labels for supervised training
        train_mask: Optional[torch.Tensor] = None,
    ) -> "HGTEmbedder":
        
        # Filter to listing-only subgraph (HGT requires features for all node types)
        # We only have features for 'listing' nodes, so filter edges accordingly
        listing_edge_types = [
            et for et in graph.edge_types 
            if et[0] == "listing" and et[2] == "listing"
        ]
        
        if not listing_edge_types:
            raise ValueError("No listing-to-listing edges found in graph")
        
        # Create simplified metadata with only listing nodes
        metadata = (["listing"], listing_edge_types)
        logger.info(f"HGT metadata - Node types: {metadata[0]}, Edge types: {len(metadata[1])}")
        
        self.model = HGTEncoder(
            in_channels=self.config["in_channels"],
            hidden_channels=self.config["hidden_channels"],
            out_channels=self.config["out_channels"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            dropout=self.config["dropout"],
            metadata=metadata,
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Set node features
        graph["listing"].x = node_features
        graph = graph.to(self.device)
        
        # Find listing-to-listing edges for link prediction
        listing_edges = [
            et for et in graph.edge_types 
            if et[0] == "listing" and et[2] == "listing"
        ]
        
        if y is None:
            # Self-Supervised Setup
            if not listing_edges:
                logger.warning("No listing-to-listing edges found, using all edges")
                target_edge_type = graph.edge_types[0]
            else:
                target_edge_type = listing_edges[0]
            
            pos_edge_index = graph[target_edge_type].edge_index
        else:
            # Supervised Setup
            self.predictor = nn.Linear(self.config["out_channels"], 1).to(self.device)
            optimizer.add_param_group({'params': self.predictor.parameters()})
            
            y = y.float().to(self.device)
            if train_mask is not None:
                train_mask = train_mask.bool().to(self.device)
            else:
                train_mask = torch.ones(y.size(0), dtype=torch.bool, device=self.device)
        
        # Filter to only listing nodes and listing-to-listing edges
        x_dict_filtered = {"listing": graph["listing"].x}
        edge_index_dict_filtered = {
            et: graph[et].edge_index 
            for et in listing_edge_types 
            if et in graph.edge_types
        }
        
        # Store for transform
        self._listing_edge_types = listing_edge_types
        
        self.model.train()
        for epoch in range(self.config["epochs"]):
            optimizer.zero_grad()
            z = self.model(x_dict_filtered, edge_index_dict_filtered)
            
            if y is None:
                # Link prediction loss
                src, dst = pos_edge_index
                pos_score = (z[src] * z[dst]).sum(dim=-1)
                
                neg_src = torch.randint(0, z.size(0), (src.size(0),), device=self.device)
                neg_dst = torch.randint(0, z.size(0), (dst.size(0),), device=self.device)
                neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
                
                scores = torch.cat([pos_score, neg_score])
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                
                loss = F.binary_cross_entropy_with_logits(scores, labels)
            else:
                # Supervised loss
                logits = self.predictor(z).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(
                    logits[train_mask], 
                    y[train_mask]
                )
            
            loss.backward()
            optimizer.step()
            
            logger.info(f"HGT Epoch {epoch}: Loss {loss.item():.4f}")
            
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
        
        # Filter to only listing nodes and listing-to-listing edges
        x_dict_filtered = {"listing": graph["listing"].x}
        edge_index_dict_filtered = {
            et: graph[et].edge_index 
            for et in self._listing_edge_types 
            if et in graph.edge_types
        }
        
        with torch.no_grad():
            z = self.model(x_dict_filtered, edge_index_dict_filtered)
            return z.cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config
        }, path)

    @classmethod
    def load(cls, path: str) -> "HGTEmbedder":
        checkpoint = torch.load(path)
        config = checkpoint["config"]
        instance = cls(**config)
        return instance

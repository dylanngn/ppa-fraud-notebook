"""
CARE-GNN (CAmouflage-REsistant Graph Neural Network) implementation.

Designed for fraud detection in graphs with camouflaged fraudsters.
Key features:
1. Similarity-based neighbor weighting (simplified from RL-based selection)
2. Relation-aware aggregation
3. Attention mechanism to down-weight dissimilar neighbors

Reference: https://arxiv.org/abs/2008.08692
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

from src.models.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SimilarityAwareConv(MessagePassing):
    """
    Efficient similarity-aware convolution layer.
    
    Uses cosine similarity for neighbor weighting instead of expensive
    attention computation. This is much faster while still providing
    the camouflage-resistant property.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        similarity_dim: int = 32,
    ):
        super().__init__(aggr='mean')  # Use mean aggregation for efficiency
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Transform for node embeddings
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Projection for similarity computation (simpler than MLP)
        self.sim_proj = nn.Linear(in_channels, similarity_dim)
        
        # Gate to control neighbor influence
        self.gate = nn.Linear(2 * similarity_dim, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.sim_proj.reset_parameters()
        self.gate.reset_parameters()
    
    def forward(self, x, edge_index):
        # Project to similarity space
        sim_emb = F.normalize(self.sim_proj(x), dim=-1)
        
        # Transform node features
        x_transformed = self.lin(x)
        
        # Propagate with similarity weighting
        return self.propagate(
            edge_index, 
            x=x_transformed, 
            sim_emb=sim_emb
        )
    
    def message(self, x_j, sim_emb_i, sim_emb_j, index):
        # Compute similarity via dot product (both are normalized)
        sim = (sim_emb_i * sim_emb_j).sum(dim=-1, keepdim=True)
        
        # Apply sigmoid gate to get weight in [0, 1]
        # High similarity = high weight, low similarity = low weight
        gate_weight = torch.sigmoid(sim * 2)  # Scale for sharper gating
        
        # Weight messages by similarity
        return gate_weight * x_j
    
    def update(self, aggr_out):
        return aggr_out


class CAREGNNEncoder(nn.Module):
    """
    CARE-GNN encoder for heterogeneous graphs.
    
    Key differences from standard GNN:
    1. Uses similarity-aware convolution instead of mean aggregation
    2. Learns to down-weight dissimilar (potentially camouflaged) neighbors
    3. Supports multiple edge types with relation-specific processing
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        similarity_dim: int = 32,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # CARE convolution layers (applied per relation, then combined)
        self.convs = nn.ModuleList()
        self.relation_transforms = nn.ModuleList()
        
        for i in range(num_layers):
            ch = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(
                SimilarityAwareConv(hidden_channels, ch, similarity_dim)
            )
            # Relation-specific transformation
            self.relation_transforms.append(
                nn.Linear(ch, ch)
            )
        
        # Layer normalization for stability
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for transform in self.relation_transforms:
            transform.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
    
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        edge_index_dict: Dict[Tuple, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass for heterogeneous graph."""
        
        # Get listing node features
        x = x_dict.get("listing")
        if x is None:
            raise ValueError("No 'listing' node features found")
        
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Filter to listing-to-listing edges only
        listing_edges = {
            k: v for k, v in edge_index_dict.items()
            if k[0] == "listing" and k[2] == "listing"
        }
        
        if not listing_edges:
            logger.warning("No listing-to-listing edges, returning projected features")
            return x
        
        # Apply CARE convolutions
        for i, (conv, transform, norm) in enumerate(
            zip(self.convs, self.relation_transforms, self.norms)
        ):
            # Determine output dimension for this layer
            layer_out_dim = self.convs[i].out_channels
            
            # Aggregate across all relations
            out = torch.zeros(x.size(0), layer_out_dim, device=x.device)
            
            relation_count = 0
            for edge_type, edge_index in listing_edges.items():
                if edge_index.numel() == 0:
                    continue
                
                # Apply similarity-aware convolution
                rel_out = conv(x, edge_index)
                
                # Apply relation-specific transform
                rel_out = transform(rel_out)
                
                out = out + rel_out
                relation_count += 1
            
            if relation_count > 0:
                out = out / relation_count  # Average across relations
            else:
                # No edges - use a linear transform to maintain dimensions
                out = conv.lin(x)
            
            # Normalization and activation
            out = norm(out)
            
            if i < self.num_layers - 1:
                out = F.relu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
            
            x = out
        
        return x


class CAREGNNEmbedder(BaseEmbedder):
    """
    Wrapper for CARE-GNN training and inference.
    
    Uses self-supervised link prediction for training,
    similar to GraphSAGE but with similarity-aware convolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 2,
        dropout: float = 0.3,
        similarity_dim: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "similarity_dim": similarity_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
        }
        self.device = device
        self.model: Optional[CAREGNNEncoder] = None
    
    def fit(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
        train_mask: torch.Tensor = None,
    ) -> "CAREGNNEmbedder":
        """Train CARE-GNN using self-supervised link prediction."""
        
        self.model = CAREGNNEncoder(
            in_channels=self.config["in_channels"],
            hidden_channels=self.config["hidden_channels"],
            out_channels=self.config["out_channels"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            similarity_dim=self.config["similarity_dim"],
        ).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=1e-5,
        )
        
        # Prepare graph
        graph["listing"].x = node_features
        graph = graph.to(self.device)
        
        # Find listing-to-listing edges for link prediction
        listing_edges = [
            et for et in graph.edge_types
            if et[0] == "listing" and et[2] == "listing"
        ]
        
        if not listing_edges:
            logger.warning("No listing-to-listing edges, skipping training")
            return self
        
        target_edge_type = listing_edges[0]
        pos_edge_index = graph[target_edge_type].edge_index
        
        if pos_edge_index.numel() == 0:
            logger.warning("Empty edge index, skipping training")
            return self
        
        # Training loop
        self.model.train()
        for epoch in range(self.config["epochs"]):
            optimizer.zero_grad()
            
            # Forward pass
            z = self.model(graph.x_dict, graph.edge_index_dict)
            
            # Link prediction loss
            src, dst = pos_edge_index
            pos_score = (z[src] * z[dst]).sum(dim=-1)
            
            # Negative sampling
            neg_src = torch.randint(0, z.size(0), (src.size(0),), device=self.device)
            neg_dst = torch.randint(0, z.size(0), (dst.size(0),), device=self.device)
            neg_score = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            
            # Binary cross-entropy loss
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([
                torch.ones_like(pos_score),
                torch.zeros_like(neg_score)
            ])
            
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            
            # L2 regularization on embeddings
            reg_loss = 0.001 * (z ** 2).mean()
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0 or epoch == self.config["epochs"] - 1:
                logger.info(
                    f"CARE-GNN Epoch {epoch}: Loss {loss.item():.4f}, "
                    f"Reg {reg_loss.item():.4f}"
                )
        
        return self
    
    def transform(
        self,
        graph: HeteroData,
        node_features: torch.Tensor,
    ) -> np.ndarray:
        """Generate node embeddings."""
        
        if self.model is None:
            raise RuntimeError("Model not fit")
        
        self.model.eval()
        graph["listing"].x = node_features
        graph = graph.to(self.device)
        
        with torch.no_grad():
            z = self.model(graph.x_dict, graph.edge_index_dict)
            return z.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "config": self.config,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CAREGNNEmbedder":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        instance = cls(**config, device=device)
        
        if checkpoint["model_state"]:
            instance.model = CAREGNNEncoder(
                in_channels=config["in_channels"],
                hidden_channels=config["hidden_channels"],
                out_channels=config["out_channels"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                similarity_dim=config["similarity_dim"],
            ).to(device)
            instance.model.load_state_dict(checkpoint["model_state"])
        
        return instance

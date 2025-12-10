"""
GNN Models for Fraud Detection.

Currently supports:
- GraphSAGE: Inductive graph neural network for fraud detection
"""
from src.models.gnn.sage import SAGEWrapper, train_sage_embeddings, create_sage_embedding_generator

__all__ = [
    "SAGEWrapper",
    "train_sage_embeddings", 
    "create_sage_embedding_generator",
]

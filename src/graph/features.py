"""
Compute graph-based features without GNN (Handcrafted).
Used for Variant 2 (Manual Graph Features + XGBoost).
"""

import networkx as nx
import polars as pl
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple

class HandcraftedGraphFeatures:
    """
    Compute graph-based features using NetworkX.
    """
    
    def compute_features(
        self,
        graph: HeteroData,
        train_labels: np.ndarray,
        train_mask: np.ndarray,
    ) -> pl.DataFrame:
        """
        Compute handcrafted graph features.
        
        Args:
            graph: PyG HeteroData object
            train_labels: Array of labels (must align with graph nodes)
            train_mask: Boolean array indicating which nodes are training examples
            
        Returns:
            Polars DataFrame with feature columns
        """
        
        num_nodes = graph["listing"].num_nodes
        
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        for edge_type_tuple in graph.edge_types:
            edge_type_name = edge_type_tuple[1]
            edge_index = graph[edge_type_tuple].edge_index
            
            if edge_index.numel() > 0:
                src = edge_index[0].numpy()
                dst = edge_index[1].numpy()
                G.add_edges_from(list(zip(src, dst)), edge_type=edge_type_name)
        
        degree_total = dict(G.degree())
        
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)
        except nx.PowerIterationFailedConvergence:
            pagerank = {i: 0.0 for i in range(num_nodes)}
            
        clustering = nx.clustering(G.to_undirected())
        neighbor_fraud_stats = self._compute_neighbor_fraud_stats(G, train_labels, train_mask)
        
        features = {
            "degree_total": [degree_total.get(i, 0) for i in range(num_nodes)],
            "pagerank": [pagerank.get(i, 0.0) for i in range(num_nodes)],
            "clustering_coefficient": [clustering.get(i, 0.0) for i in range(num_nodes)],
        }
        
        # Merge stats
        features.update(neighbor_fraud_stats)
        
        return pl.DataFrame(features)
    
    def _compute_neighbor_fraud_stats(
        self,
        G: nx.DiGraph,
        labels: np.ndarray,
        train_mask: np.ndarray,
    ) -> Dict[str, List[float]]:
        
        num_nodes = G.number_of_nodes()
        neighbor_fraud_rate = []
        neighbor_count = []
        two_hop_fraud_rate = []
        
        visible_labels = np.full(num_nodes, np.nan)
        visible_labels[train_mask] = labels[train_mask]
        
        for node in range(num_nodes):
            neighbors = list(G.predecessors(node))
            
            n_count = len(neighbors)
            neighbor_count.append(n_count)
            
            if n_count > 0:
                n_labels = visible_labels[neighbors]
                valid_labels = n_labels[~np.isnan(n_labels)]
                
                if len(valid_labels) > 0:
                    neighbor_fraud_rate.append(float(np.mean(valid_labels)))
                else:
                    neighbor_fraud_rate.append(0.0)
                    
                # 2-hop fraud (neighbors of neighbors)
                two_hop_neighbors = []
                for n in neighbors:
                    two_hop_neighbors.extend(list(G.predecessors(n)))
                
                two_hop_neighbors = list(set(two_hop_neighbors)) # Unique
                if two_hop_neighbors:
                    n2_labels = visible_labels[two_hop_neighbors]
                    valid_n2_labels = n2_labels[~np.isnan(n2_labels)]
                    if len(valid_n2_labels) > 0:
                        two_hop_fraud_rate.append(float(np.mean(valid_n2_labels)))
                    else:
                        two_hop_fraud_rate.append(0.0)
                else:
                    two_hop_fraud_rate.append(0.0)
                    
            else:
                neighbor_fraud_rate.append(0.0)
                two_hop_fraud_rate.append(0.0)
        
        return {
            "neighbor_count": neighbor_count,
            "neighbor_fraud_rate": neighbor_fraud_rate,
            "two_hop_fraud_rate": two_hop_fraud_rate,
        }

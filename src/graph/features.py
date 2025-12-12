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
        
        # 1. Convert to NetworkX for structural features
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Add edges from all types
        for edge_type_tuple in graph.edge_types:
            edge_type_name = edge_type_tuple[1]
            edge_index = graph[edge_type_tuple].edge_index
            
            if edge_index.numel() > 0:
                src = edge_index[0].numpy()
                dst = edge_index[1].numpy()
                edges = list(zip(src, dst))
                # Add edges with attributes
                G.add_edges_from(edges, edge_type=edge_type_name)
        
        # 2. Structural Features
        # Degree
        degree_total = dict(G.degree())
        out_degree = dict(G.out_degree())
        in_degree = dict(G.in_degree())
        
        # PageRank (computational expense warning for large graphs)
        # Using a small alpha to emphasize local structure
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=50) # Reduced iter for speed
        except nx.PowerIterationFailedConvergence:
            pagerank = {i: 0.0 for i in range(num_nodes)}
            
        # Clustering Coefficient (needs undirected conversion)
        clustering = nx.clustering(G.to_undirected())
        
        # 3. Label-based Features (Leakage Safe)
        # Neighbor Fraud Rates
        # CRITICAL: Only use Training labels!
        # If a node is in Test set, it can only see fraud rates from its Train neighbors
        
        neighbor_fraud_stats = self._compute_neighbor_fraud_stats(
            G, train_labels, train_mask
        )
        
        # 4. Assemble DataFrame
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
        neighbor_mean_price = [] # TODO: Need prices passed in? skipping price for now
        neighbor_fraud_rate = []
        neighbor_count = []
        
        # 2-hop stats
        two_hop_fraud_rate = []
        
        # Pre-compute valid label map (only train labels are visible)
        # Test labels are treated as Unknown/NaN
        visible_labels = np.full(num_nodes, np.nan)
        visible_labels[train_mask] = labels[train_mask]
        
        for node in range(num_nodes):
            # Get neighbors (Incoming edges = "Who pointed to me?" i.e. Older items)
            # Or Outgoing?
            # Build logic: Older -> Newer.
            # If I am a new node, I have incoming edges from older nodes.
            # So I should look at my PREDECESSORS (incoming edges).
            
            # G is constructed with Older -> Newer. 
            # So `u -> v` means `u` is older than `v`.
            # `v` (current node) wants to know about its history (`u`).
            # So we look at predecessors.
            
            neighbors = list(G.predecessors(node))
            
            n_count = len(neighbors)
            neighbor_count.append(n_count)
            
            if n_count > 0:
                # 1-hop fraud
                n_labels = visible_labels[neighbors]
                # Ignore NaNs (which would be test neighbors if any, though build_graph shouldn't allow test->train or test->test)
                # Actually, build_graph only allows Older -> Newer.
                # If 'node' is Test, neighbors can be Train or Test (if Test is older? but Test is usually later).
                # Splitting ensures Test comes AFTER Train. So all neighbors of Test must be Train or earlier Test.
                # But we masked ALL Test labels. So we only see Train labels.
                
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

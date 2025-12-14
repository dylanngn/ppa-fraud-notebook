"""
Temporal graph construction with strict leakage prevention.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import polars as pl
import torch
from torch_geometric.data import HeteroData

from src.data.schema import FEATURE_SCHEMA


@dataclass
class GraphConfig:
    """Configuration for graph construction."""
    
    identity_columns: List[str]
    min_edge_weight: int = 1          # Minimum shared occurrences to create edge
    max_neighbors: Optional[int] = 50  # Cap on edges per node (prevents super-nodes)
    directed: bool = True              # MUST be True for temporal safety


class TemporalGraphBuilder:
    """
    Builds graphs with strict temporal ordering.
    
    CRITICAL INVARIANT: All edges are directed from OLDER to NEWER nodes.
    This ensures information only flows from past to future.
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        
        if not config.directed:
            raise ValueError(
                "Directed edges are REQUIRED for temporal graphs to prevent leakage. "
                "Set config.directed = True"
            )
    
    def build_graph(
        self,
        df: pl.DataFrame,
        cutoff_date: datetime,
        node_features: Optional[torch.Tensor] = None,
    ) -> HeteroData:
        """Build a temporal graph from data up to cutoff_date."""
        df_filtered = df.filter(pl.col("submission_at") <= cutoff_date)
        
        if df_filtered.height == 0:
            # Return empty graph with standard structure instead of erroring
            data = HeteroData()
            data["listing"].num_nodes = 0
            for col in self.config.identity_columns:
                 data["listing", f"shares_{col}", "listing"].edge_index = torch.empty((2, 0), dtype=torch.long)
            return data
        
        listing_ids = df_filtered["listing_id"].to_list()
        timestamps = df_filtered["submission_at"].to_list()
        
        id_to_idx = {lid: idx for idx, lid in enumerate(listing_ids)}
        idx_to_timestamp = {idx: ts for idx, ts in enumerate(timestamps)}
        
        edges_by_type = {}
        
        for identity_col in self.config.identity_columns:
            edges = self._build_edges_for_identity(
                df_filtered,
                identity_col,
                id_to_idx,
                idx_to_timestamp,
            )
            
            if edges is not None:
                edges_by_type[f"shares_{identity_col}"] = edges
        
        data = HeteroData()
        data["listing"].num_nodes = len(listing_ids)
        
        if node_features is not None:
            if node_features.size(0) != len(listing_ids):
                raise ValueError(f"Node features size {node_features.size(0)} != num nodes {len(listing_ids)}")
            data["listing"].x = node_features
        
        for edge_type, (src, dst) in edges_by_type.items():
            if len(src) > 0:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                data["listing", edge_type, "listing"].edge_index = edge_index
        
        data.cutoff_date = cutoff_date.isoformat()
        
        return data
    
    def _build_edges_for_identity(
        self,
        df: pl.DataFrame,
        identity_col: str,
        id_to_idx: Dict[str, int],
        idx_to_timestamp: Dict[int, datetime],
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Build directed edges for a single identity column."""
        if identity_col not in df.columns:
            return None
            
        groups = (
            df
            .filter(pl.col(identity_col).is_not_null())
            .filter(pl.col(identity_col) != "")  # Exclude empty strings
            .group_by(identity_col)
            .agg([
                pl.col("listing_id").alias("listing_ids"),
                pl.col("submission_at").alias("timestamps"),
            ])
            .filter(pl.col("listing_ids").list.len() >= 2)  # At least 2 to form edge
        )
        
        if groups.height == 0:
            return None
        
        edge_src, edge_dst = [], []
        
        for row in groups.iter_rows(named=True):
            listings = list(zip(row["listing_ids"], row["timestamps"]))
            listings.sort(key=lambda x: x[1])
            
            for i in range(len(listings)):
                src_id, src_time = listings[i]
                neighbors_added = 0
                
                for j in range(i + 1, len(listings)):
                    dst_id, dst_time = listings[j]
                    
                    if src_time < dst_time:
                        if src_id in id_to_idx and dst_id in id_to_idx:
                            edge_src.append(id_to_idx[src_id])
                            edge_dst.append(id_to_idx[dst_id])
                            neighbors_added += 1
                    
                    if (self.config.max_neighbors and neighbors_added >= self.config.max_neighbors):
                        break
        
        return (edge_src, edge_dst) if edge_src else None
    
    def validate_no_leakage(
        self,
        data: HeteroData,
        timestamps: List[datetime],
    ) -> bool:
        """Validate that NO edge points from future to past."""
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue
                
            src_nodes = edge_index[0].tolist()
            dst_nodes = edge_index[1].tolist()
            
            for src, dst in zip(src_nodes, dst_nodes):
                if timestamps[src] > timestamps[dst]:
                     raise ValueError(
                        f"LEAKAGE DETECTED: Edge from node {src} to {dst} points backwards in time!"
                    )
        
        return True

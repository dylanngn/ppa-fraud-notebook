"""
Heterogeneous Temporal Graph Construction for Fraud Detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from collections import defaultdict

import polars as pl
import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.features.schema import FEATURE_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class HeteroGraphConfig:
    """Configuration for heterogeneous graph construction."""
    
    primary_node_type: str = "listing"
    primary_id_column: str = "INSERTION_ID_hash"
    
    entity_node_types: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "user": {
            "id_columns": ["user_id_hash", "CREATEDATUSERNAME_hash"],
            "feature_columns": [],
        },
        "device": {
            "id_columns": ["session/device_hash"],
            "feature_columns": ["session/os", "session/device_type", "session/browser"],
        },
        "ip": {
            "id_columns": ["ip_hash"],
            "feature_columns": ["ip_type", "ip_country", "ip_isp_name", 
                               "data_center_proxy", "residential_proxy", "public_proxy"],
        },
    })
    
    # Only use columns with POSITIVE fraud correlation
    # (high connectivity = more fraud)
    # Based on analysis: IP and Browser FP have positive signal
    # Device, Email, Phone, User have INVERSE signal (removed)
    temporal_identity_columns: List[str] = field(default_factory=lambda: [
        # Expanded identity set to let model learn weights
        "ip_hash",
        "session/device_hash",
        "session/similarity_hash", 
        "user_id_hash",
        "LISTING_LISTER_EMAIL_hash",
        "LISTING_LISTER_PHONE_hash",
    ])
    
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    max_neighbors_per_edge_type: int = 100
    min_entity_occurrences: int = 1
    add_reverse_edges: bool = True


class HeterogeneousGraphBuilder:
    """Builds a heterogeneous graph with multiple node and edge types."""
    
    def __init__(self, config: HeteroGraphConfig):
        self.config = config
        self._node_mappings: Dict[str, Dict[str, int]] = {}
    
    def build_graph(self, df: pl.DataFrame, cutoff_date: datetime) -> HeteroData:
        """Build heterogeneous graph from data."""
        df_filtered = df.filter(pl.col(self.config.time_column) <= cutoff_date)
        
        if df_filtered.height == 0:
            logger.warning("No data before cutoff_date")
            return self._create_empty_graph()
        
        df_filtered = df_filtered.sort(self.config.time_column)
        logger.info(f"Building heterogeneous graph from {df_filtered.height} listings")
        
        self._build_node_mappings(df_filtered)
        
        data = HeteroData()
        self._add_nodes_to_graph(data, df_filtered)
        self._add_listing_to_entity_edges(data, df_filtered)
        self._add_temporal_listing_edges(data, df_filtered)
        
        if self.config.add_reverse_edges:
            self._add_reverse_edges(data)
        
        data.cutoff_date = cutoff_date.isoformat()
        
        # Store node mappings in graph for later use in predictions
        data.node_mappings = dict(self._node_mappings)
        
        self._log_graph_summary(data)
        
        return data
    
    def _build_node_mappings(self, df: pl.DataFrame):
        """Create ID → index mappings for all node types."""
        self._node_mappings = {}
        
        if self.config.primary_id_column in df.columns:
            listing_ids = df[self.config.primary_id_column].unique().to_list()
            self._node_mappings["listing"] = {lid: idx for idx, lid in enumerate(listing_ids) if lid}
        else:
            self._node_mappings["listing"] = {}
        
        for entity_type, entity_config in self.config.entity_node_types.items():
            entity_ids = set()
            
            for id_col in entity_config["id_columns"]:
                if id_col in df.columns:
                    values = df[id_col].drop_nulls().unique().to_list()
                    entity_ids.update(v for v in values if v and v != "")
            
            if self.config.min_entity_occurrences > 1:
                occurrence_counts = defaultdict(int)
                for id_col in entity_config["id_columns"]:
                    if id_col in df.columns:
                        for val in df[id_col].to_list():
                            if val:
                                occurrence_counts[val] += 1
                
                entity_ids = {
                    eid for eid in entity_ids 
                    if occurrence_counts[eid] >= self.config.min_entity_occurrences
                }
            
            self._node_mappings[entity_type] = {
                eid: idx for idx, eid in enumerate(sorted(entity_ids))
            }
    
    def _add_nodes_to_graph(self, data: HeteroData, df: pl.DataFrame):
        """Add nodes with features to the graph."""
        num_listings = len(self._node_mappings.get("listing", {}))
        data["listing"].num_nodes = num_listings
        
        listing_feature_cols = list(FEATURE_SCHEMA.get_gnn_input_features())
        available_cols = [c for c in listing_feature_cols if c in df.columns]
        
        if available_cols and self.config.primary_id_column in df.columns:
            # Smart Aggregation Strategy
            # 1. SEON Booleans (signals) -> MAX (did it ever happen?)
            # 2. SEON Numerics (counts) -> MEAN (average behavior)
            # 3. Listing Numerics (static) -> FIRST (property of listing)
            
            agg_exprs = []
            seon_bools = set(FEATURE_SCHEMA.all_seon_boolean)
            seon_nums = set(FEATURE_SCHEMA.all_seon_numeric)
            
            for col in available_cols:
                if col in seon_bools:
                    agg_exprs.append(pl.col(col).max().alias(col))
                elif col in seon_nums:
                    agg_exprs.append(pl.col(col).mean().alias(col))
                else:
                    # Default/Listing features -> First
                    agg_exprs.append(pl.col(col).first().alias(col))
            
            # Additional feature: Event count (proxy for velocity/updates)
            agg_exprs.append(pl.len().alias("event_count"))
            
            listing_features = (
                df.group_by(self.config.primary_id_column)
                .agg(agg_exprs)
                .sort(self.config.primary_id_column)
            )
            
            # Base features
            x = listing_features.select(available_cols).to_numpy().astype(np.float32)
            for j in range(x.shape[1]):
                mask = np.isnan(x[:, j])
                if mask.any():
                    # If column is all NaNs, fill with 0.0 directly to avoid RuntimeWarning from nanmedian
                    if mask.all():
                        x[:, j] = 0.0
                    else:
                        median = np.nanmedian(x[:, j])
                        x[mask, j] = median if not np.isnan(median) else 0.0
            
            # Add temporal encoding features
            temporal_features = self._compute_temporal_features(df, listing_features)
            if temporal_features is not None:
                x = np.concatenate([x, temporal_features], axis=1)
                logger.info(f"Added {temporal_features.shape[1]} temporal features to node embeddings")
            
            data["listing"].x = torch.tensor(x, dtype=torch.float32)
        
        for entity_type, entity_config in self.config.entity_node_types.items():
            num_entities = len(self._node_mappings.get(entity_type, {}))
            if num_entities == 0:
                continue
            
            data[entity_type].num_nodes = num_entities
    
    def _compute_temporal_features(
        self, 
        df: pl.DataFrame, 
        listing_features: pl.DataFrame
    ) -> Optional[np.ndarray]:
        """
        Compute temporal encoding features for each listing node.
        
        Features:
        1. time_since_start: Days since first event in dataset (normalized)
        2. time_until_cutoff: Days until cutoff date (normalized)
        3. relative_position: Position in timeline [0, 1]
        4. hour_sin/cos: Cyclical encoding of hour of day
        5. weekday_sin/cos: Cyclical encoding of day of week
        """
        time_col = self.config.time_column
        id_col = self.config.primary_id_column
        
        if time_col not in df.columns or id_col not in listing_features.columns:
            return None
        
        try:
            # Get first event time per listing (already sorted by primary_id in listing_features)
            time_df = (
                df.group_by(id_col)
                .agg(pl.col(time_col).first().alias("first_event_time"))
                .sort(id_col)
            )
            
            # Extract timestamps
            timestamps = time_df["first_event_time"].to_numpy()
            
            # Convert to numeric (seconds since epoch)
            # Handle both datetime and string types
            if hasattr(timestamps[0], 'timestamp'):
                ts_numeric = np.array([t.timestamp() for t in timestamps])
            else:
                # Already numeric or needs conversion
                ts_numeric = timestamps.astype(np.float64)
            
            # Handle NaNs
            valid_mask = ~np.isnan(ts_numeric)
            if not valid_mask.any():
                return None
            
            min_ts = np.nanmin(ts_numeric)
            max_ts = np.nanmax(ts_numeric)
            ts_range = max_ts - min_ts if max_ts > min_ts else 1.0
            
            # Feature 1: Days since start (normalized to [0, 1])
            days_since_start = (ts_numeric - min_ts) / ts_range
            days_since_start = np.nan_to_num(days_since_start, nan=0.5)
            
            # Feature 2: Relative position in timeline
            relative_position = days_since_start  # Same as normalized time
            
            # Features 3-4: Cyclical hour encoding
            # Convert to datetime for hour/weekday extraction
            try:
                hours = np.array([
                    (t.hour if hasattr(t, 'hour') else 
                     (np.datetime64(int(t), 's').astype('datetime64[h]').astype(int) % 24))
                    for t in timestamps
                ])
            except:
                hours = np.zeros(len(timestamps))
            
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)
            
            # Features 5-6: Cyclical weekday encoding
            try:
                weekdays = np.array([
                    (t.weekday() if hasattr(t, 'weekday') else 0)
                    for t in timestamps
                ])
            except:
                weekdays = np.zeros(len(timestamps))
            
            weekday_sin = np.sin(2 * np.pi * weekdays / 7)
            weekday_cos = np.cos(2 * np.pi * weekdays / 7)
            
            # Feature 7: Recency (inverse of time since start, more recent = higher)
            recency = 1.0 - days_since_start
            
            # Stack all temporal features
            temporal_features = np.column_stack([
                days_since_start,
                relative_position,
                hour_sin,
                hour_cos,
                weekday_sin,
                weekday_cos,
                recency,
                # Feature 8: Event Count (log normalized)
                np.log1p(listing_features["event_count"].to_numpy()),
            ]).astype(np.float32)
            
            return temporal_features
            
        except Exception as e:
            logger.warning(f"Could not compute temporal features: {e}")
            return None
    
    def _add_listing_to_entity_edges(self, data: HeteroData, df: pl.DataFrame):
        """Add edges from listings to entity nodes."""
        listing_mapping = self._node_mappings.get("listing", {})
        
        for entity_type, entity_config in self.config.entity_node_types.items():
            entity_mapping = self._node_mappings.get(entity_type, {})
            if not entity_mapping:
                continue
            
            edge_src, edge_dst = [], []
            
            for id_col in entity_config["id_columns"]:
                if id_col not in df.columns:
                    continue
                
                for row in df.select([self.config.primary_id_column, id_col]).iter_rows(named=True):
                    listing_id = row[self.config.primary_id_column]
                    entity_id = row[id_col]
                    
                    if listing_id in listing_mapping and entity_id in entity_mapping:
                        edge_src.append(listing_mapping[listing_id])
                        edge_dst.append(entity_mapping[entity_id])
            
            if edge_src:
                edges = list(set(zip(edge_src, edge_dst)))
                edge_src = [e[0] for e in edges]
                edge_dst = [e[1] for e in edges]
                
                edge_type = f"used_{entity_type}"
                data["listing", edge_type, entity_type].edge_index = torch.tensor(
                    [edge_src, edge_dst], dtype=torch.long
                )
    
    def _add_temporal_listing_edges(self, data: HeteroData, df: pl.DataFrame):
        """Add temporal edges between listings sharing identities."""
        listing_mapping = self._node_mappings.get("listing", {})
        time_col = self.config.time_column
        
        if self.config.primary_id_column not in df.columns:
            return
        
        listing_times = {}
        for row in df.select([self.config.primary_id_column, time_col]).iter_rows(named=True):
            lid = row[self.config.primary_id_column]
            ts = row[time_col]
            if lid and (lid not in listing_times or ts < listing_times[lid]):
                listing_times[lid] = ts
        
        for identity_col in self.config.temporal_identity_columns:
            if identity_col not in df.columns:
                continue
            
            edge_type = self._get_temporal_edge_type(identity_col)
            
            groups = (
                df.filter(pl.col(identity_col).is_not_null())
                .filter(pl.col(identity_col) != "")
                .group_by(identity_col)
                .agg([pl.col(self.config.primary_id_column).alias("listing_ids")])
                .filter(pl.col("listing_ids").list.len() >= 2)
            )
            
            edge_src, edge_dst = [], []
            
            for row in groups.iter_rows(named=True):
                listing_ids = row["listing_ids"]
                
                timed_listings = [
                    (lid, listing_times.get(lid, datetime.max))
                    for lid in listing_ids if lid in listing_mapping
                ]
                timed_listings.sort(key=lambda x: x[1])
                
                for i, (src_lid, src_time) in enumerate(timed_listings):
                    neighbors_added = 0
                    for j in range(i + 1, len(timed_listings)):
                        dst_lid, dst_time = timed_listings[j]
                        if src_time < dst_time:
                            edge_src.append(listing_mapping[src_lid])
                            edge_dst.append(listing_mapping[dst_lid])
                            neighbors_added += 1
                        if neighbors_added >= self.config.max_neighbors_per_edge_type:
                            break
            
            if edge_src:
                data["listing", edge_type, "listing"].edge_index = torch.tensor(
                    [edge_src, edge_dst], dtype=torch.long
                )
    
    def _get_temporal_edge_type(self, identity_col: str) -> str:
        col_lower = identity_col.lower()
        if "device" in col_lower or "session" in col_lower:
            return "shares_device"
        elif "ip" in col_lower:
            return "shares_ip"
        elif "email" in col_lower:
            return "shares_email"
        elif "phone" in col_lower:
            return "shares_phone"
        elif "user" in col_lower:
            return "shares_user"
            return f"shares_{identity_col.replace('/', '_')}"
    
    def _add_reverse_edges(self, data: HeteroData):
        """Add reverse edges for bidirectional message passing."""
        for edge_type in list(data.edge_types):
            src_type, relation, dst_type = edge_type
            if src_type == dst_type:
                continue
            
            reverse_relation = f"rev_{relation}"
            edge_index = data[edge_type].edge_index
            
            if edge_index.numel() > 0:
                rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                data[dst_type, reverse_relation, src_type].edge_index = rev_edge_index
    
    def _create_empty_graph(self) -> HeteroData:
        data = HeteroData()
        data["listing"].num_nodes = 0
        for entity_type in self.config.entity_node_types.keys():
            data[entity_type].num_nodes = 0
        return data
    
    def _log_graph_summary(self, data: HeteroData):
        node_info = [f"{nt}={data[nt].num_nodes}" for nt in data.node_types]
        edge_info = []
        total_edges = 0
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            num = edge_index.size(1) if edge_index.numel() > 0 else 0
            total_edges += num
            edge_info.append(f"{edge_type[1]}={num}")
        
        logger.info(
            f"Graph built: Nodes={{{', '.join(node_info)}}}, "
            f"Edges={{{', '.join(edge_info)}}}, Total={total_edges}"
        )
    
    def get_node_mapping(self, node_type: str) -> Dict[str, int]:
        return self._node_mappings.get(node_type, {})


@dataclass
class GraphConfig:
    """Configuration for simple temporal graph construction."""
    identity_columns: List[str] = field(
        default_factory=lambda: list(FEATURE_SCHEMA.graph_config.high_signal_columns)
    )
    max_neighbors: Optional[int] = 50
    time_column: str = "DATAPIPELINE_EVENT_SENT_AT"
    node_id_column: str = "INSERTION_ID_hash"


class TemporalGraphBuilder:
    """Simple temporal graph builder (homogeneous, listing nodes only)."""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.hetero_config = HeteroGraphConfig(
            primary_id_column=config.node_id_column,
            time_column=config.time_column,
            temporal_identity_columns=config.identity_columns,
            max_neighbors_per_edge_type=config.max_neighbors or 50,
            entity_node_types={},
            add_reverse_edges=False,
        )
        self.builder = HeterogeneousGraphBuilder(self.hetero_config)
    
    def build_graph(
        self,
        df: pl.DataFrame,
        cutoff_date: datetime,
        node_features: Optional[torch.Tensor] = None,
    ) -> HeteroData:
        data = self.builder.build_graph(df, cutoff_date)
        if node_features is not None:
            data["listing"].x = node_features
        return data
    
    def get_graph_statistics(self, data: HeteroData) -> Dict:
        stats = {
            "num_nodes": data["listing"].num_nodes,
            "edge_types": {},
            "total_edges": 0,
        }
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            num = edge_index.size(1) if edge_index.numel() > 0 else 0
            stats["edge_types"][str(edge_type)] = num
            stats["total_edges"] += num
        return stats

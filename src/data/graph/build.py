"""
Graph Build Pipeline

Builds graph artifacts in two stages:
1. Parquet artifacts (nodes and edges) from raw insertions
2. PyTorch Geometric HeteroData graph from parquet artifacts

Usage:
    python -m src.data.graph.build
    
Entity Naming Convention:
    - External name: listing_id (used in logs, configs, documentation)
    - Internal storage: insertion_id (database object_reference field)
"""
import logging

import hydra
from omegaconf import DictConfig

from src.data.graph.create_artifacts import create_nodes_and_edges, RAW_INSERTIONS
from src.data.graph.graph_structure import build_graph

import polars as pl

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(_cfg: DictConfig) -> None:
    """
    Build graph artifacts with Hydra configuration.
    
    Steps:
        1. Load raw insertions from parquet
        2. Create node/edge parquet files (graph artifacts)
        3. Build PyTorch Geometric HeteroData graph
    """
    # Check for raw data
    if not RAW_INSERTIONS.exists():
        logger.error(f"Raw insertions not found at {RAW_INSERTIONS}")
        logger.error("Run ETL first: python -m src.data.etl.pipeline")
        raise SystemExit(1)
    
    # Step 1: Load raw data
    logger.info(f"Loading raw insertions from {RAW_INSERTIONS}...")
    df_insertions = pl.read_parquet(RAW_INSERTIONS)
    logger.info(f"Loaded {len(df_insertions):,} listings")
    
    # Step 2: Create parquet artifacts (nodes and edges)
    logger.info("Creating graph artifacts (parquet files)...")
    create_nodes_and_edges(df_insertions)
    
    # Step 3: Build PyG graph
    logger.info("Building PyTorch Geometric HeteroData graph...")
    data = build_graph(cutoff_date=None)  # Full graph, saved to artifacts/graph.pt
    
    # Log graph statistics
    logger.info("Graph build complete!")
    logger.info(f"  Node types: {len(data.node_types)}")
    logger.info(f"  Edge types: {len(data.edge_types)}")
    for node_type in data.node_types:
        if hasattr(data[node_type], 'num_nodes'):
            logger.info(f"  {node_type}: {data[node_type].num_nodes:,} nodes")
    
    total_edges = sum(
        data[et].edge_index.size(1) 
        for et in data.edge_types 
        if hasattr(data[et], 'edge_index')
    )
    logger.info(f"  Total edges: {total_edges:,}")
    
    logger.info("Graph build pipeline complete!")


if __name__ == "__main__":
    main()


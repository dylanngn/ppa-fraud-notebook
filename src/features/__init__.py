"""
Features module.

Structure:
- definitions/: Feature computation functions for training pipeline
  - base.py: compute_base_features() - tabular features from raw data
  - graph.py: compute_graph_features() - features from graph artifacts

- generators/: Standalone scripts that create parquet artifacts
  - graph_features.py: Generate basic graph features
  - advanced_graph_features.py: Generate advanced graph features

- processor.py: FeatureProcessor - orchestrates feature generation
- registry.py: FeatureRegistry - maps category names to generators
"""
from src.features import generators
from src.features import definitions
from src.features.registry import FeatureRegistry
from src.features.processor import FeatureProcessor

__all__ = [
    "definitions",
    "generators",
    "FeatureProcessor",
    "FeatureRegistry",
]

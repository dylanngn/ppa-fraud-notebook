"""
Features module.

Structure:
- definitions/: Feature computation functions for training pipeline
  - base.py: compute_base_features() - tabular features from raw data
  - graph.py: compute_graph_features() - features from graph artifacts
  - text.py: compute_text_features() - NLP-based features

- generators/: Standalone scripts that create parquet artifacts
  - graph_features.py: Generate listing_graph_features.parquet
  - advanced_graph_features.py: Generate listing_advanced_features.parquet
  - time_weighted_features.py: Generate time-weighted features
  - interaction_features.py: Generate interaction features
  - text_features.py: Generate text-based features

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


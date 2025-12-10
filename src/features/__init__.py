"""
Features Module.

Structure:
- xgboost/: XGBoost feature engineering
  - processor.py: FeatureProcessor - orchestrates feature generation
  - registry.py: FeatureRegistry - maps category names to generators
  - definitions/: Feature computation functions
  - generators/: Standalone scripts that create artifacts
  
- gnn/: GNN feature engineering
  - node_features.py: Graph node feature engineering
"""
from src.features.xgboost import generators
from src.features.xgboost import definitions
from src.features.xgboost.registry import FeatureRegistry
from src.features.xgboost.processor import FeatureProcessor

__all__ = [
    "definitions",
    "generators",
    "FeatureProcessor",
    "FeatureRegistry",
]

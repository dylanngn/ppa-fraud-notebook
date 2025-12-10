"""
Feature generators module.

Exports graph feature generator:
    from src.features.xgboost import generators
    generators.graph_features.generate_graph_features(...)
"""
from src.features.xgboost.generators import graph_features

__all__ = [
    "graph_features",
]

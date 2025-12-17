"""
Feature engineering module for fraud detection.
"""

from src.features.schema import (
    FEATURE_SCHEMA,
    FeatureSchema,
    ModelVariant,
    TemporalSplitStrategy,
    LabelPropagation,
    GraphIdentityConfig,
    TemporalConfig,
)
from src.features.store import FeatureStore
from src.features.graph_builder import (
    HeterogeneousGraphBuilder,
    HeteroGraphConfig,
    TemporalGraphBuilder,
    GraphConfig,
)
from src.features.temporal_split import (
    TemporalSplitter,
    TemporalSplitConfig,
    ExpandingWindowSplitter,
    validate_temporal_integrity,
    compute_pit_label_stats,
    LabelPropagation,
    OverlapMode,
)

__all__ = [
    "FEATURE_SCHEMA",
    "FeatureSchema",
    "ModelVariant",
    "TemporalSplitStrategy",
    "LabelPropagation",
    "GraphIdentityConfig",
    "TemporalConfig",
    "FeatureStore",
    "HeterogeneousGraphBuilder",
    "HeteroGraphConfig",
    "TemporalGraphBuilder",
    "GraphConfig",
    "TemporalSplitter",
    "TemporalSplitConfig",
    "ExpandingWindowSplitter",
    "validate_temporal_integrity",
    "compute_pit_label_stats",
    "LabelPropagation",
    "OverlapMode",
]

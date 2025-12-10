"""
Feature Registry.
Allows registering feature generators and retrieving them by category.
"""
from typing import Callable, Dict, List, Any
import polars as pl
from datetime import datetime

FeatureGenerator = Callable[[pl.DataFrame, datetime, Any], pl.DataFrame]

class FeatureRegistry:
    _registry: Dict[str, FeatureGenerator] = {}
    
    @classmethod
    def register(cls, category: str):
        """Decorator to register a feature generator function."""
        def decorator(func: FeatureGenerator):
            cls._registry[category] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, category: str) -> FeatureGenerator:
        """Get a feature generator by category."""
        if category not in cls._registry:
            raise ValueError(f"Feature category '{category}' not found in registry")
        return cls._registry[category]
    
    @classmethod
    def list_categories(cls) -> List[str]:
        """List all registered categories."""
        return list(cls._registry.keys())

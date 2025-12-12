"""
PPA Fraud Detection - Main Package

Quick imports for common operations:
    from src import load_data, FeatureProcessor, calculate_metrics
    
    # Load raw data
    df = load_data()
"""
# from src.data.paths import load_data
# from src.features.xgboost.processor import FeatureProcessor
from src.utils.metrics import calculate_metrics

__all__ = [
#    "load_data",
#    "FeatureProcessor", 
    "calculate_metrics",
]


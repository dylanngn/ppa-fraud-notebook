"""
PPA Fraud Detection - Main Package

Quick imports for common operations:
    from src import load_data, FeatureProcessor, calculate_metrics
"""
from src.data.loader import load_data
from src.features.processor import FeatureProcessor
from src.utils.metrics import calculate_metrics, measure_inference_latency

__all__ = [
    "load_data",
    "FeatureProcessor", 
    "calculate_metrics",
    "measure_inference_latency",
]


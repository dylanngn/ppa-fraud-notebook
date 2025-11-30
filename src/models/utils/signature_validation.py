"""
Signature validation and comparison utilities for model experiments.

Integrates MLflow signature best practices with ExperimentConfig to ensure
fair and reproducible model comparisons across baseline and hybrid models.

Accounts for:
- Shared preprocessing pipeline (create_graph_artifacts.py)
- Shared training pipeline (training_window.py)
- Dynamic feature addition (base + graph + optional embeddings)
"""
import mlflow
from mlflow.models import infer_signature, ModelSignature
from mlflow.models.signature import set_signature
from typing import Dict, List, Optional, Any, Set
import pandas as pd
import numpy as np
import polars as pl

from src.models.config.experiment_config import ExperimentConfig
from src.models.config.constants import (
    BASE_FEATURES,
    ALL_GRAPH_FEATURE_COLUMNS,
)


def create_signature_from_config(
    config: ExperimentConfig,
    sample_data: pl.DataFrame,
    model_output: Optional[np.ndarray] = None,
    model_type: Optional[str] = None
) -> ModelSignature:
    """
    Create a model signature based on ExperimentConfig feature categories.
    
    Accounts for shared preprocessing and training pipeline:
    - All models use same base + graph features
    - Hybrid models add embeddings (embed_0 to embed_63)
    
    Args:
        config: ExperimentConfig with feature categories
        sample_data: Sample DataFrame with features (should match config features)
        model_output: Optional sample predictions for output schema
        model_type: Optional model type ("baseline_graph", "hybrid_hgt", "hybrid_sage")
                    If provided, includes embeddings for hybrid models
    
    Returns:
        ModelSignature with input/output schemas
    """
    # Get expected features for this model type
    if model_type:
        expected_features = get_expected_features_for_model_type(model_type, config)
    else:
        # Default: use config features (assumes baseline, no embeddings)
        expected_features = set(config.get_feature_columns())
    
    # Filter sample data to only include expected features
    available_features = [f for f in expected_features if f in sample_data.columns]
    
    if len(available_features) != len(expected_features):
        missing = set(expected_features) - set(available_features)
        print(f"Warning: Missing features in sample data: {missing}")
    
    # Convert to pandas for signature inference (MLflow works better with pandas)
    sample_df = sample_data.select(available_features).to_pandas()
    
    # Infer signature
    if model_output is not None:
        signature = infer_signature(sample_df, model_output)
    else:
        # Create dummy output for signature inference
        dummy_output = np.zeros(len(sample_df))
        signature = infer_signature(sample_df, dummy_output)
    
    return signature


def validate_model_signature(
    model_uri: str,
    expected_features: List[str],
    sample_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Validate that a logged model's signature matches expected features.
    
    Args:
        model_uri: MLflow model URI (e.g., "runs:/run_id/model")
        expected_features: List of expected feature names
        sample_data: Optional sample data to test prediction
    
    Returns:
        Dictionary with validation results:
        - valid: bool
        - input_features: List of features in signature
        - missing_features: List of expected but missing features
        - extra_features: List of features in signature but not expected
        - type_mismatches: Dict of feature name to type mismatch info
    """
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        signature = model.metadata.signature
        
        if signature is None:
            return {
                "valid": False,
                "error": "Model has no signature",
                "input_features": [],
                "missing_features": expected_features,
                "extra_features": [],
                "type_mismatches": {}
            }
        
        # Extract input features from signature
        input_schema = signature.inputs
        input_features = []
        
        if hasattr(input_schema, 'input_names'):
            input_features = input_schema.input_names()
        elif hasattr(input_schema, 'columns'):
            input_features = [col.name for col in input_schema.columns]
        else:
            # Try to extract from schema string representation
            input_features = []
        
        # Compare with expected features
        missing_features = set(expected_features) - set(input_features)
        extra_features = set(input_features) - set(expected_features)
        
        # Test prediction if sample data provided
        prediction_error = None
        if sample_data is not None:
            try:
                # Ensure sample data has all required features
                sample_subset = sample_data[[f for f in input_features if f in sample_data.columns]]
                _ = model.predict(sample_subset)
            except Exception as e:
                prediction_error = str(e)
        
        return {
            "valid": len(missing_features) == 0 and prediction_error is None,
            "input_features": input_features,
            "missing_features": list(missing_features),
            "extra_features": list(extra_features),
            "type_mismatches": {},  # Could be enhanced to check types
            "prediction_error": prediction_error
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "input_features": [],
            "missing_features": expected_features,
            "extra_features": [],
            "type_mismatches": {}
        }


def get_expected_features_for_model_type(
    model_type: str,
    config: ExperimentConfig
) -> Set[str]:
    """
    Get expected features for a specific model type, accounting for:
    - Shared preprocessing (create_graph_artifacts.py)
    - Shared training pipeline (training_window.py)
    - Model-specific embeddings (baseline has none, hybrid models have 64)
    
    Args:
        model_type: Model type ("baseline_graph", "hybrid_hgt", "hybrid_sage")
        config: ExperimentConfig with feature categories
    
    Returns:
        Set of expected feature names
    """
    # All models share: BASE_FEATURES + graph features from config
    config_features = config.get_feature_columns()
    
    # Filter to actual feature categories (exclude embeddings from config)
    base_and_graph = set(BASE_FEATURES) | set(ALL_GRAPH_FEATURE_COLUMNS)
    expected = set(config_features) & base_and_graph
    
    # Add embeddings for hybrid models (64 dimensions, embed_0 to embed_63)
    if model_type.startswith("hybrid"):
        embedding_features = {f"embed_{i}" for i in range(64)}
        expected |= embedding_features
    
    return expected


def compare_model_signatures(
    model_uris: Dict[str, str],
    config: ExperimentConfig
) -> Dict[str, Any]:
    """
    Compare signatures across multiple models to ensure compatibility.
    
    Accounts for shared preprocessing and training pipeline:
    - All models use same base features (from add_base_tabular_features)
    - All models use same graph features (from feature_engineering per window)
    - Hybrid models add embeddings (embed_0 to embed_63)
    
    Args:
        model_uris: Dictionary mapping model names to MLflow URIs
                   e.g., {"baseline_graph": "runs:/run1/model", "hybrid_hgt": "runs:/run2/hybrid_model"}
        config: ExperimentConfig used for training
    
    Returns:
        Dictionary with comparison results:
        - common_features: Features present in all models (base + graph, no embeddings)
        - unique_features: Dict mapping model name to unique features (embeddings for hybrid)
        - compatibility: bool indicating if all models are compatible
        - signature_details: Dict mapping model name to signature info
        - shared_preprocessing: Confirmation that all models use same preprocessing
    """
    results = {
        "common_features": set(),  # Will be intersection of all models
        "unique_features": {},
        "compatibility": True,
        "signature_details": {},
        "shared_preprocessing": True,  # All models use create_graph_artifacts.py
        "shared_training_pipeline": True,  # All models use training_window.py
    }
    
    all_model_features = []
    model_type_to_features = {}
    
    # Get expected features for each model type
    for model_name, model_uri in model_uris.items():
        # Extract model type from name (baseline_graph, hybrid_hgt, hybrid_sage)
        model_type = model_name
        
        # Get expected features for this model type
        expected_features = get_expected_features_for_model_type(model_type, config)
        model_type_to_features[model_name] = expected_features
        
        # Validate model signature
        validation = validate_model_signature(model_uri, list(expected_features))
        results["signature_details"][model_name] = validation
        
        model_features = set(validation["input_features"])
        all_model_features.append(model_features)
        
        # Track unique features (should be embeddings for hybrid models)
        unique = model_features - expected_features
        if unique:
            results["unique_features"][model_name] = list(unique)
        
        # Check if model has expected embeddings
        if model_type.startswith("hybrid"):
            embedding_features = {f"embed_{i}" for i in range(64)}
            has_embeddings = embedding_features & model_features
            if not has_embeddings:
                results["compatibility"] = False
                validation["missing_features"].extend(list(embedding_features))
        elif model_type == "baseline_graph":
            # Baseline should NOT have embeddings
            embedding_features = {f"embed_{i}" for i in range(64)}
            has_embeddings = embedding_features & model_features
            if has_embeddings:
                results["compatibility"] = False
                validation["extra_features"].extend(list(has_embeddings))
        
        # Check compatibility
        if not validation["valid"]:
            results["compatibility"] = False
    
    # Find common features (intersection of all models)
    # This should be BASE_FEATURES + graph features (no embeddings)
    if all_model_features:
        results["common_features"] = set.intersection(*all_model_features)
        
        # Verify common features match expected base + graph features
        expected_common = set(BASE_FEATURES) | (set(config.get_feature_columns()) & set(ALL_GRAPH_FEATURE_COLUMNS))
        embedding_features = {f"embed_{i}" for i in range(64)}
        
        # Common features should NOT include embeddings
        if embedding_features & results["common_features"]:
            results["compatibility"] = False
            print(f"Warning: Embeddings found in common features - models may not be properly separated")
        
        # Check if common features match expected
        missing_common = expected_common - results["common_features"]
        if missing_common:
            print(f"Warning: Missing expected common features: {missing_common}")
    
    return results


def update_model_signature(
    model_uri: str,
    config: ExperimentConfig,
    sample_data: pl.DataFrame,
    model_output: Optional[np.ndarray] = None
) -> bool:
    """
    Update a model's signature based on ExperimentConfig.
    
    Useful for fixing models that were logged without proper signatures.
    
    Args:
        model_uri: MLflow model URI to update
        config: ExperimentConfig with feature categories
        sample_data: Sample DataFrame with features
        model_output: Optional sample predictions
    
    Returns:
        True if signature was updated successfully
    """
    try:
        signature = create_signature_from_config(config, sample_data, model_output)
        set_signature(model_uri, signature)
        return True
    except Exception as e:
        print(f"Error updating signature: {e}")
        return False


def create_experiment_signature_report(
    experiment_name: str,
    model_types: List[str] = ["baseline_graph", "hybrid_hgt", "hybrid_sage"],
    config: Optional[ExperimentConfig] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive signature report for all models in an experiment.
    
    This helps ensure fair comparison across model types by validating
    that all models have compatible signatures.
    
    Args:
        experiment_name: MLflow experiment name
        model_types: List of model type names to compare
        config: Optional ExperimentConfig (if None, uses default)
    
    Returns:
        Dictionary with signature comparison report
    """
    if config is None:
        config = ExperimentConfig(experiment_name=experiment_name)
    
    # Find latest runs for each model type
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    model_uris = {}
    for model_type in model_types:
        try:
            # Find latest run for this model type
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                continue
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.model_type = '{model_type}'",
                max_results=1,
                order_by=["start_time DESC"]
            )
            
            if runs:
                run_id = runs[0].info.run_id
                # Try to find model artifact
                artifacts = client.list_artifacts(run_id)
                model_path = None
                for artifact in artifacts:
                    if artifact.path in ["model", "hybrid_model"]:
                        model_path = artifact.path
                        break
                
                if model_path:
                    model_uris[model_type] = f"runs:/{run_id}/{model_path}"
        except Exception as e:
            print(f"Error finding model for {model_type}: {e}")
    
    if not model_uris:
        return {
            "error": "No models found for comparison",
            "model_uris": {}
        }
    
    # Compare signatures
    comparison = compare_model_signatures(model_uris, config)
    
    # Add preprocessing pipeline info
    preprocessing_info = {
        "shared_preprocessing": "create_graph_artifacts.py",
        "shared_training_pipeline": "training_window.py",
        "base_features_count": len(BASE_FEATURES),
        "graph_features_count": len(ALL_GRAPH_FEATURE_COLUMNS),
        "embedding_dimensions": 64,  # Both HGT and SAGE use 64-dim embeddings
    }
    
    return {
        "experiment_name": experiment_name,
        "model_uris": model_uris,
        "comparison": comparison,
        "config_features": config.get_feature_columns(),
        "preprocessing_info": preprocessing_info,
        "timestamp": pd.Timestamp.now().isoformat()
    }


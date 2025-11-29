"""
Utility script to compare signatures across model types for research validation.

Run this after training models to ensure fair comparison.

Accounts for:
- Shared preprocessing (create_graph_artifacts.py)
- Shared training pipeline (training_window.py)
- Model-specific embeddings (baseline has none, hybrid models have 64)
"""
from typing import List
import mlflow
from mlflow.tracking import MlflowClient
from src.models.experiment_config import ExperimentConfig
from src.models.signature_validation import create_experiment_signature_report
from src.models.constants import BASE_FEATURES, ALL_GRAPH_FEATURE_COLUMNS
import json


def compare_all_models(
    experiment_name: str = "ppa-fraud-detection",
    model_types: List[str] = None
) -> None:
    """
    Compare signatures for all model types in an experiment.
    
    This ensures fair comparison by validating that all models:
    1. Use the same feature set
    2. Have compatible input schemas
    3. Can process the same input data
    
    Args:
        experiment_name: MLflow experiment name
        model_types: List of model types to compare (default: all three)
    """
    if model_types is None:
        model_types = ["baseline_graph", "hybrid_hgt", "hybrid_sage"]
    
    print(f"Comparing signatures for models: {model_types}")
    print(f"Experiment: {experiment_name}\n")
    
    # Create config (use default for now, could be customized)
    config = ExperimentConfig(experiment_name=experiment_name)
    
    # Generate signature report
    report = create_experiment_signature_report(
        experiment_name=experiment_name,
        model_types=model_types,
        config=config
    )
    
    if "error" in report:
        print(f"Error: {report['error']}")
        return
    
    # Print results
    print("=" * 80)
    print("SIGNATURE COMPARISON REPORT")
    print("=" * 80)
    print(f"\nExperiment: {experiment_name}")
    print(f"Models compared: {', '.join(model_types)}")
    print(f"Expected features: {len(report['config_features'])}")
    print(f"\nModel URIs:")
    for model_type, uri in report["model_uris"].items():
        print(f"  {model_type}: {uri}")
    
    comparison = report["comparison"]
    
    print(f"\n{'=' * 80}")
    print("COMPATIBILITY RESULTS")
    print("=" * 80)
    print(f"All models compatible: {comparison['compatibility']}")
    print(f"Common features (base + graph, shared across all): {len(comparison['common_features'])}")
    
    # Show common features breakdown
    if comparison["common_features"]:
        common_base = sum(1 for f in comparison["common_features"] if f in BASE_FEATURES)
        common_graph = sum(1 for f in comparison["common_features"] if f in ALL_GRAPH_FEATURE_COLUMNS)
        print(f"  - Base features: {common_base}")
        print(f"  - Graph features: {common_graph}")
    
    if comparison["unique_features"]:
        print(f"\nUnique features by model (should be embeddings for hybrid models):")
        for model_name, features in comparison["unique_features"].items():
            embed_features = [f for f in features if f.startswith("embed_")]
            other_features = [f for f in features if not f.startswith("embed_")]
            if embed_features:
                print(f"  {model_name}: {len(embed_features)} embeddings (embed_0 to embed_{len(embed_features)-1})")
            if other_features:
                print(f"    ⚠️  Unexpected features: {other_features}")
    
    print(f"\n{'=' * 80}")
    print("PREPROCESSING PIPELINE INFO")
    print("=" * 80)
    if "preprocessing_info" in report:
        info = report["preprocessing_info"]
        print(f"Shared preprocessing: {info['shared_preprocessing']}")
        print(f"Shared training pipeline: {info['shared_training_pipeline']}")
        print(f"Base features: {info['base_features_count']}")
        print(f"Graph features: {info['graph_features_count']}")
        print(f"Embedding dimensions (hybrid models): {info['embedding_dimensions']}")
    
    print(f"\n{'=' * 80}")
    print("DETAILED SIGNATURE INFO")
    print("=" * 80)
    
    for model_name, details in comparison["signature_details"].items():
        print(f"\n{model_name}:")
        print(f"  Valid: {details['valid']}")
        print(f"  Input features: {len(details['input_features'])}")
        
        # Show feature breakdown
        if details["input_features"]:
            base_count = sum(1 for f in details["input_features"] if f in BASE_FEATURES)
            graph_count = sum(1 for f in details["input_features"] if f in ALL_GRAPH_FEATURE_COLUMNS)
            embed_count = sum(1 for f in details["input_features"] if f.startswith("embed_"))
            print(f"    - Base features: {base_count}")
            print(f"    - Graph features: {graph_count}")
            print(f"    - Embeddings: {embed_count}")
        
        if details["missing_features"]:
            print(f"  Missing features: {details['missing_features']}")
        if details["extra_features"]:
            print(f"  Extra features: {details['extra_features']}")
        if details.get("prediction_error"):
            print(f"  Prediction error: {details['prediction_error']}")
    
    # Save report to file
    report_path = f"artifacts/signature_comparison_{experiment_name}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print(f"Report saved to: {report_path}")
    print("=" * 80)
    
    # Recommendations
    if not comparison["compatibility"]:
        print("\n⚠️  WARNING: Models have incompatible signatures!")
        print("   This may affect fair comparison. Consider:")
        print("   1. Retraining models with consistent feature sets")
        print("   2. Updating signatures using update_model_signature()")
        print("   3. Using only common features for comparison")
    else:
        print("\n✓ All models have compatible signatures - fair comparison ensured!")


if __name__ == "__main__":
    import sys
    
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "ppa-fraud-detection"
    compare_all_models(experiment_name=experiment_name)


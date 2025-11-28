"""
MLflow Feature Store Integration

Logs feature store metadata to MLflow runs for traceability and lineage.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import mlflow
from mlflow.tracking import MlflowClient

from src.features.store import FeatureStore, FeatureSnapshot


def log_feature_store_metadata(
    feature_store: FeatureStore,
    feature_names: List[str],
    artifacts_dir: Path = Path("artifacts"),
    context: str = "training"
) -> None:
    """
    Log feature store metadata to the active MLflow run.
    
    This includes:
    - Feature names and categories
    - Feature store file paths and timestamps
    - Feature availability status
    - Feature store configuration
    
    Args:
        feature_store: FeatureStore instance
        feature_names: List of feature names used in training
        artifacts_dir: Directory containing feature artifacts
        context: Context for logging (e.g., "training", "inference")
    """
    if mlflow.active_run() is None:
        print("Warning: No active MLflow run. Skipping feature store metadata logging.")
        return
    
    try:
        # 1. Log feature names and categories
        feature_categories = _categorize_features(feature_names, feature_store)
        mlflow.log_dict(
            {
                "feature_names": feature_names,
                "feature_categories": feature_categories,
                "total_features": len(feature_names),
                "base_features": len(feature_categories.get("base", [])),
                "graph_features": len(feature_categories.get("graph", [])),
                "advanced_features": len(feature_categories.get("advanced", [])),
                "time_weighted_features": len(feature_categories.get("time_weighted", [])),
                "interaction_features": len(feature_categories.get("interaction", [])),
            },
            artifact_file="feature_store/metadata.json"
        )
        
        # 2. Log feature store file metadata
        file_metadata = _get_feature_file_metadata(artifacts_dir)
        mlflow.log_dict(
            file_metadata,
            artifact_file="feature_store/files.json"
        )
        
        # 3. Log feature store paths as parameters
        mlflow.log_params({
            "feature_store.artifacts_dir": str(artifacts_dir),
            "feature_store.has_graph_features": str(feature_store.graph_features is not None),
            "feature_store.has_advanced_features": str(feature_store.advanced_features is not None),
            "feature_store.context": context,
        })
        
        # 4. Log feature store configuration
        config = {
            "listing_nodes_path": str(feature_store.listing_nodes_path),
            "graph_features_path": str(feature_store.graph_features_path),
            "advanced_features_path": str(feature_store.advanced_features_path),
            "feature_store_type": "temporal_validation",
            "temporal_validation_enabled": True,
        }
        mlflow.log_dict(config, artifact_file="feature_store/config.json")
        
        # 5. Log feature statistics (sample from training data if available)
        # This is optional and can be done separately if needed
        
        print(f"✓ Logged feature store metadata to MLflow run")
        
    except Exception as e:
        print(f"Warning: Failed to log feature store metadata: {e}")


def log_feature_store_statistics(
    feature_store: FeatureStore,
    sample_listing_ids: Optional[List[int]] = None,
    as_of_time: Optional[datetime] = None
) -> None:
    """
    Log feature store statistics by sampling listings.
    
    Computes:
    - Cold start rate
    - Average confidence scores
    - Feature availability statistics
    
    Args:
        feature_store: FeatureStore instance
        sample_listing_ids: List of listing IDs to sample (if None, uses first 1000)
        as_of_time: Point in time for feature retrieval
    """
    if mlflow.active_run() is None:
        return
    
    try:
        # Get sample listing IDs
        if sample_listing_ids is None:
            # Sample first 1000 listings
            listings = feature_store.listing_nodes.head(1000)
            sample_listing_ids = listings["insertion_id"].to_list()
        
        # Sample features
        cold_start_count = 0
        confidence_scores = []
        feature_availability = {}
        
        for listing_id in sample_listing_ids[:100]:  # Limit to 100 for performance
            try:
                snapshot = feature_store.get_features(listing_id, as_of_time=as_of_time)
                
                if snapshot.is_cold_start:
                    cold_start_count += 1
                
                confidence_scores.append(snapshot.confidence)
                
                # Track feature availability
                for feat_name in snapshot.features.keys():
                    if feat_name not in feature_availability:
                        feature_availability[feat_name] = {"available": 0, "total": 0}
                    feature_availability[feat_name]["total"] += 1
                    if snapshot.features[feat_name] != 0.0:  # Non-zero indicates available
                        feature_availability[feat_name]["available"] += 1
                        
            except (ValueError, KeyError):
                continue
        
        # Compute statistics
        n_samples = len(sample_listing_ids[:100])
        cold_start_rate = cold_start_count / n_samples if n_samples > 0 else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Log statistics
        mlflow.log_metrics({
            "feature_store.cold_start_rate": cold_start_rate,
            "feature_store.avg_confidence": avg_confidence,
            "feature_store.samples_analyzed": float(n_samples),
        })
        
        # Log feature availability
        availability_dict = {
            name: {
                "available": stats["available"],
                "total": stats["total"],
                "availability_rate": stats["available"] / stats["total"] if stats["total"] > 0 else 0.0
            }
            for name, stats in feature_availability.items()
        }
        mlflow.log_dict(
            availability_dict,
            artifact_file="feature_store/availability.json"
        )
        
        print(f"✓ Logged feature store statistics (cold_start_rate={cold_start_rate:.2%})")
        
    except Exception as e:
        print(f"Warning: Failed to log feature store statistics: {e}")


def log_feature_lineage(
    feature_names: List[str],
    feature_sources: Dict[str, str]
) -> None:
    """
    Log feature lineage (where features come from).
    
    Args:
        feature_names: List of feature names
        feature_sources: Dict mapping feature names to their sources
                      (e.g., {"account_age_days": "tabular", "contact_email_count": "graph"})
    """
    if mlflow.active_run() is None:
        return
    
    try:
        lineage = {
            "features": {
                name: {
                    "source": feature_sources.get(name, "unknown"),
                    "category": _infer_category(name)
                }
                for name in feature_names
            }
        }
        
        mlflow.log_dict(lineage, artifact_file="feature_store/lineage.json")
        print("✓ Logged feature lineage")
        
    except Exception as e:
        print(f"Warning: Failed to log feature lineage: {e}")


def _categorize_features(
    feature_names: List[str],
    feature_store: FeatureStore
) -> Dict[str, List[str]]:
    """Categorize features into base, graph, advanced, etc."""
    categories = {
        "base": [],
        "graph": [],
        "advanced": [],
        "time_weighted": [],
        "interaction": [],
        "other": []
    }
    
    # Base features
    base_features = feature_store.get_feature_names()
    base_set = set(base_features[:14])  # First 14 are base tabular features
    
    # Graph features
    graph_features = [
        "contact_email_count", "shared_contact_email_count", "max_shared_contact_email",
        "contact_phone_count", "shared_contact_phone_count", "max_shared_contact_phone",
        "user_listing_count", "user_unique_ip_count", "shared_ip_user_count",
        "max_shared_ip_users", "listing_component_size", "listing_pagerank"
    ]
    graph_set = set(graph_features)
    
    # Advanced features
    advanced_features = [
        "degree_total", "is_isolated", "unique_identifier_count",
        "neighbor_overlap_score", "avg_neighbor_degree"
    ]
    advanced_set = set(advanced_features)
    
    # Time-weighted features (common patterns)
    time_weighted_patterns = ["_7d", "_30d", "_velocity", "_acceleration", "_burst", "_recency", "_historical"]
    time_weighted_set = {f for f in feature_names if any(pattern in f for pattern in time_weighted_patterns)}
    
    # Interaction features (common patterns)
    interaction_patterns = ["_reuse", "_combo", "_risk_score", "new_account_", "in_large_component"]
    interaction_set = {f for f in feature_names if any(pattern in f for pattern in interaction_patterns)}
    
    # Categorize
    for feat in feature_names:
        if feat in base_set:
            categories["base"].append(feat)
        elif feat in graph_set:
            categories["graph"].append(feat)
        elif feat in advanced_set:
            categories["advanced"].append(feat)
        elif feat in time_weighted_set:
            categories["time_weighted"].append(feat)
        elif feat in interaction_set:
            categories["interaction"].append(feat)
        else:
            categories["other"].append(feat)
    
    return categories


def _get_feature_file_metadata(artifacts_dir: Path) -> Dict[str, Any]:
    """Get metadata about feature files (paths, existence, modification times)."""
    metadata = {}
    
    feature_files = {
        "listing_nodes": artifacts_dir / "nodes_listing.parquet",
        "graph_features": artifacts_dir / "listing_graph_features.parquet",
        "advanced_features": artifacts_dir / "listing_advanced_features.parquet",
        "time_weighted_features": artifacts_dir / "listing_time_weighted_features.parquet",
        "interaction_features": artifacts_dir / "listing_interaction_features.parquet",
    }
    
    for name, path in feature_files.items():
        if path.exists():
            stat = path.stat()
            metadata[name] = {
                "path": str(path),
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        else:
            metadata[name] = {
                "path": str(path),
                "exists": False,
            }
    
    return metadata


def _infer_category(feature_name: str) -> str:
    """Infer feature category from name."""
    if any(x in feature_name for x in ["email", "phone", "ip", "component", "pagerank"]):
        return "graph"
    elif any(x in feature_name for x in ["_7d", "_30d", "velocity", "acceleration", "burst"]):
        return "time_weighted"
    elif any(x in feature_name for x in ["reuse", "combo", "risk_score", "interaction"]):
        return "interaction"
    elif feature_name in ["degree_total", "is_isolated", "unique_identifier_count", "neighbor_overlap_score", "avg_neighbor_degree"]:
        return "advanced"
    else:
        return "tabular"


def get_feature_store_from_mlflow_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve feature store metadata from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dict with feature store metadata, or None if not found
    """
    client = MlflowClient()
    
    try:
        run = client.get_run(run_id)
        
        # Try to get feature store metadata artifact
        artifacts = client.list_artifacts(run_id, "feature_store")
        
        metadata = {}
        for artifact in artifacts:
            if artifact.path.endswith(".json"):
                artifact_path = f"feature_store/{artifact.path.split('/')[-1]}"
                try:
                    content = client.download_artifacts(run_id, artifact_path)
                    with open(content, 'r') as f:
                        metadata[artifact.path.split('/')[-1].replace('.json', '')] = json.load(f)
                except Exception:
                    continue
        
        # Get feature store params
        params = run.data.params
        feature_store_params = {
            k: v for k, v in params.items() 
            if k.startswith("feature_store.")
        }
        
        return {
            "metadata": metadata,
            "params": feature_store_params,
            "run_id": run_id,
        }
        
    except Exception as e:
        print(f"Error retrieving feature store metadata: {e}")
        return None


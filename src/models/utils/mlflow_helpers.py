"""
MLflow helper utilities for following best practices.

Provides utilities for:
- Model signature creation (especially for complex models like GNNs)
- Dependency management
- Input example creation
"""
from typing import Any, Dict, Optional
import numpy as np
import torch
from mlflow.models import infer_signature, ModelSignature
from mlflow.types import Schema, TensorSpec


def create_gnn_signature(
    input_example: Dict[str, np.ndarray],
    output_example: np.ndarray,
    input_schema: Optional[Schema] = None,
    output_schema: Optional[Schema] = None
) -> Optional[ModelSignature]:
    """
    Create a model signature for GNN models with dictionary inputs.
    
    For complex models like GNNs with dictionary inputs, infer_signature() often fails.
    This function creates a proper signature manually using TensorSpec.
    
    Args:
        input_example: Dictionary of input arrays (node features, edge indices, etc.)
        output_example: Output array (embeddings or predictions)
        input_schema: Optional pre-defined input schema
        output_schema: Optional pre-defined output schema
    
    Returns:
        ModelSignature or None if creation fails
    """
    try:
        # Try standard inference first (works for simple cases)
        return infer_signature(input_example, output_example)
    except Exception:
        # Fall back to manual TensorSpec creation for complex inputs
        try:
            # Create input schema from dictionary
            if input_schema is None:
                input_specs = []
                for key, value in input_example.items():
                    if isinstance(value, np.ndarray):
                        shape = list(value.shape)
                        # Use -1 for batch dimension to allow variable batch sizes
                        if len(shape) > 0:
                            shape[0] = -1
                        input_specs.append(
                            TensorSpec(type=np.dtype(value.dtype).name, shape=tuple(shape), name=key)
                        )
                input_schema = Schema(input_specs)
            
            # Create output schema
            if output_schema is None:
                if isinstance(output_example, np.ndarray):
                    output_shape = list(output_example.shape)
                    if len(output_shape) > 0:
                        output_shape[0] = -1  # Variable batch size
                    output_schema = Schema([
                        TensorSpec(
                            type=np.dtype(output_example.dtype).name,
                            shape=tuple(output_shape),
                            name="output"
                        )
                    ])
            
            return ModelSignature(inputs=input_schema, outputs=output_schema)
        except Exception:
            return None


def create_input_example_for_gnn(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[str, torch.Tensor],
    edge_time_dict: Optional[Dict[str, torch.Tensor]] = None,
    max_nodes: int = 5,
    max_edges: int = 10
) -> Dict[str, np.ndarray]:
    """
    Create a serializable input example for GNN models.
    
    Args:
        x_dict: Dictionary of node feature tensors
        edge_index_dict: Dictionary of edge index tensors
        edge_time_dict: Optional dictionary of edge time tensors
        max_nodes: Maximum number of nodes to include in example
        max_edges: Maximum number of edges to include in example
    
    Returns:
        Dictionary of numpy arrays suitable for MLflow input_example
    """
    input_example = {}
    
    # Convert node features
    for node_type, features in x_dict.items():
        if features.numel() > 0:
            sample_size = min(max_nodes, features.size(0))
            input_example[f"{node_type}_features"] = features[:sample_size].cpu().numpy()
        else:
            input_example[f"{node_type}_features"] = features.cpu().numpy()
    
    # Convert edge indices
    for edge_type, edge_index in edge_index_dict.items():
        if edge_index.numel() > 0:
            sample_size = min(max_edges, edge_index.size(1))
            input_example[f"{edge_type}_edges"] = edge_index[:, :sample_size].cpu().numpy()
        else:
            input_example[f"{edge_type}_edges"] = edge_index.cpu().numpy()
    
    # Convert edge times if provided
    if edge_time_dict is not None:
        for edge_type, edge_times in edge_time_dict.items():
            if edge_times is not None and edge_times.numel() > 0:
                sample_size = min(max_edges, edge_times.size(0))
                input_example[f"{edge_type}_times"] = edge_times[:sample_size].cpu().numpy()
    
    return input_example


def get_model_dependencies() -> Dict[str, Any]:
    """
    Get explicit dependency requirements for logged models.
    
    Returns:
        Dictionary with pip_requirements or conda_env for model logging
    """
    # Read from requirements.txt to ensure consistency
    try:
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # Filter to ML-relevant packages (MLflow will auto-add framework-specific deps)
        ml_packages = [
            req for req in requirements
            if any(pkg in req.lower() for pkg in [
                "torch", "xgboost", "scikit-learn", "numpy", "pandas",
                "polars", "torch_geometric", "sentence-transformers"
            ])
        ]
        
        return {"pip_requirements": ml_packages}
    except FileNotFoundError:
       raise FileNotFoundError("requirements.txt not found")
    except Exception as e:
        raise Exception(f"Failed to get model dependencies: {e}")

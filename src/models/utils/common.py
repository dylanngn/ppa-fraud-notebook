"""
Common utilities shared across model training scripts.
"""
import torch
import mlflow
from torch_geometric.data import HeteroData


def get_device():
    """
    Select the best available device for PyTorch operations.
    Prioritizes MPS (Mac GPU) > CUDA > CPU.
    
    Returns:
        torch.device: The selected device
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def setup_mlflow(experiment_name: str = "ppa-fraud-detection"):
    """
    Initialize MLflow tracking with SQLite backend.
    
    Args:
        experiment_name: Name of the MLflow experiment
    """
    mlflow.set_tracking_uri("sqlite:///fraud-detection-mlflow.db")
    mlflow.set_experiment(experiment_name)


def filter_graph_by_time(data, max_time_ns):
    """
    Returns a subgraph containing only edges and nodes visible at max_time_ns.
    
    Filters edges by timestamp to prevent data leakage during temporal training.
    Node features and indices remain unchanged (GraphSAGE is inductive).
    
    Args:
        data: HeteroData graph object
        max_time_ns: Maximum timestamp in nanoseconds
        
    Returns:
        Filtered HeteroData graph
    """
    new_data = HeteroData()
    
    # Copy node features (all nodes, all features)
    # GraphSAGE is inductive - nodes can have features even without edges
    for node_type, x in data.x_dict.items():
        new_data[node_type].x = x
        new_data[node_type].num_nodes = data[node_type].num_nodes
    
    # Copy listing labels & timestamps
    new_data['listing'].y = data['listing'].y
    new_data['listing'].timestamp = data['listing'].timestamp
    
    # Filter edges by timestamp (key for temporal fairness)
    for edge_type, edge_index in data.edge_index_dict.items():
        if 'timestamp' in data[edge_type]:
            edge_times = data[edge_type].timestamp
            mask = edge_times <= max_time_ns
            new_data[edge_type].edge_index = edge_index[:, mask]
            new_data[edge_type].timestamp = edge_times[mask]
        else:
            # Static edges (e.g., listing-location, keep all)
            new_data[edge_type].edge_index = edge_index
    
    return new_data


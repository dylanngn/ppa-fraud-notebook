"""
MLflow initialization utility to ensure consistent configuration across the project.

MLflow Architecture:
- **MLflow Server**: Centralized service that manages metadata, metrics, and artifacts.
  Started via `mlflow server` command. Clients connect via HTTP URI (e.g., http://localhost:5000).
  
- **MLflow Client**: Library used in code to interact with MLflow (server or direct DB).
  For local development, can connect directly to SQLite DB (client mode).
  For production/shared environments, should connect to MLflow server.

This module provides client-side initialization. The tracking URI can be set via:
1. Environment variable MLFLOW_TRACKING_URI (recommended)
2. Explicit call to init_mlflow() with tracking_uri parameter
3. Default: SQLite database (sqlite:///mlflow.db) for local development

Best Practice: Set MLFLOW_TRACKING_URI environment variable once rather than
calling init_mlflow() multiple times.
"""
import os
from pathlib import Path
import mlflow


def init_mlflow(tracking_uri: str = None, force: bool = False):
    """
    Initialize MLflow tracking URI (idempotent).
    
    This function is idempotent - it won't change the tracking URI if already set,
    unless force=True. Best practice is to set MLFLOW_TRACKING_URI environment
    variable instead of calling this function.
    
    Args:
        tracking_uri: Optional custom tracking URI. If None:
            - Checks MLFLOW_TRACKING_URI environment variable first
            - Falls back to SQLite database (sqlite:///mlflow.db) for local dev
        force: If True, override existing tracking URI. Default: False.
    
    Returns:
        The tracking URI that is now active (may be existing or newly set)
    
    Examples:
        # Client mode (direct DB connection) - local development
        init_mlflow()  # Uses sqlite:///mlflow.db
        
        # Server mode - connect to MLflow server
        init_mlflow("http://localhost:5000")
        
        # Use environment variable (recommended)
        # export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
        # Then no need to call init_mlflow() in code
    """
    # Check if tracking URI is already set
    current_uri = mlflow.get_tracking_uri()
    
    # If already set and not forcing, return existing
    if not force and current_uri and current_uri != "file:./mlruns":
        return current_uri
    
    # Determine tracking URI
    if tracking_uri is None:
        # Check environment variable first (recommended approach)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        
        if tracking_uri is None:
            # Default: Use SQLite database backend instead of filesystem
            # This is client mode - direct DB connection for local development
            db_path = Path("mlflow.db").absolute()
            tracking_uri = f"sqlite:///{db_path}"
    
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def get_mlflow_tracking_uri() -> str:
    """
    Get the current MLflow tracking URI.
    
    Returns:
        Current tracking URI string
    """
    return mlflow.get_tracking_uri()


"""
Model Registry wrapper for MLflow.
"""
import mlflow
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Wrapper for MLflow Model Registry operations."""
    
    @staticmethod
    def register_model(
        run_id: str,
        model_name: str,
        description: Optional[str] = None
    ):
        """
        Register a model from a run to the Model Registry.
        
        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            description: Optional description for the model version
            
        Returns:
            Registered model version info
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            if description:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=registered_model.name,
                    version=registered_model.version,
                    description=description
                )
            
            logger.info(f"Registered model {model_name} version {registered_model.version}")
            return registered_model
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

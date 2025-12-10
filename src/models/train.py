"""
Main training entry point using Hydra.

Orchestrates:
- Dataset loading based on config
- Temporal splitting strategy
- Feature processing per window
- Model training
"""
import hydra
from omegaconf import DictConfig
import logging
import mlflow
import polars as pl
from datetime import datetime

from src.data.training_loader import load_data
from src.features.xgboost.processor import FeatureProcessor
from src.utils.temporal_split import AccumulatingWindowSplitter
from src.models.xgb_trainer.trainer import train_single_window
from src.models.utils.common import setup_mlflow

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function with orchestration.
    
    Orchestrates:
    1. Load dataset (based on config)
    2. Create temporal splits (based on config)
    3. Process features per window
    4. Train model per window
    5. Aggregate and log results
    """
    logger.info("Starting training...")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Features: {cfg.features.categories}")
    
    # 1. Load data (orchestrator decides which dataset)
    logger.info("Loading dataset...")
    df = load_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # 2. Create temporal splitter (orchestrator decides strategy)
    logger.info("Setting up temporal splitter...")
    splitter = AccumulatingWindowSplitter(
        df=df,
        initial_window_days=cfg.model.training.initial_window_days,
        step_days=cfg.model.training.step_days,
        test_days=cfg.model.training.get("test_days", 14),
        time_column="submission_at",
        min_train_samples=cfg.model.training.get("min_train_samples", 1000),
        min_test_samples=cfg.model.training.get("min_test_samples", 50),
        max_windows=cfg.model.training.get("max_windows", None)
    )
    
    # 3. Setup MLflow
    setup_mlflow(cfg.experiment_name)
    
    # Start parent run
    with mlflow.start_run(
        run_name=f"{cfg.model.name}_accumulating_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": cfg.model.name, "training_mode": "accumulating_window"}
    ) as parent_run:
        
        # Log configuration
        mlflow.log_params({
            "model_name": cfg.model.name,
            "initial_window_days": cfg.model.training.initial_window_days,
            "step_days": cfg.model.training.step_days,
            "total_samples": len(df),
            "fraud_rate": float(df["is_fraud"].mean()),
            "feature_categories": ",".join(cfg.features.categories),
        })
        
        # Log full Hydra config
        mlflow.log_dict(cfg, "hydra_config.yaml")
        
        # 4. Train per window
        results = []
        best_auc_pr = 0
        best_run_id = None
        best_model_uri = None
        
        # Create feature processor once (reused across windows)
        feature_processor = FeatureProcessor.from_config(cfg.features)
        
        for window_idx, train_data, test_data, window_info in splitter.split():
            logger.info(
                f"Window {window_idx}: "
                f"train up to {window_info['train_end'].date()}, "
                f"test {window_info['test_start'].date()} → {window_info['test_end'].date()}"
            )
            
            # Process features with temporal cutoff
            train_processed, train_feature_cols = feature_processor.process(
                train_data, 
                cutoff_date=window_info['train_end']
            )
            test_processed, test_feature_cols = feature_processor.process(
                test_data,
                cutoff_date=window_info['train_end'],
                expected_columns=train_feature_cols  # Ensure consistency
            )
            
            # Convert to pandas for XGBoost
            train_df = train_processed.to_pandas()
            test_df = test_processed.to_pandas()
            
            # Train model
            result = train_single_window(
                train_df=train_df,
                test_df=test_df,
                feature_cols=train_feature_cols,
                target_col="is_fraud",
                xgb_params=dict(cfg.model.params),
                window_idx=window_idx,
                log_model=True,
                nested=True
            )
            
            if result.get("skipped"):
                continue
            
            # Track best model
            if result["auc_pr"] > best_auc_pr:
                best_auc_pr = result["auc_pr"]
                best_run_id = result.get("run_id")
                best_model_uri = result.get("model_uri")
            
            results.append({
                "window_idx": window_idx,
                "train_size": result["train_size"],
                "test_size": result["test_size"],
                "auc_pr": result["auc_pr"],
                "auc_roc": result["auc_roc"],
                "p@100": result["p@100"],
            })
        
        # 5. Aggregate and log results
        if results:
            results_df = pl.DataFrame(results)
            mean_auc_pr = float(results_df["auc_pr"].mean())
            mean_auc_roc = float(results_df["auc_roc"].mean())
            
            mlflow.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "mean_auc_roc": mean_auc_roc,
                "best_auc_pr": best_auc_pr,
                "num_windows": float(len(results)),
            })
            
            logger.info(f"Training complete: {len(results)} windows")
            logger.info(f"Mean AUC-PR: {mean_auc_pr:.4f}")
            logger.info(f"Best AUC-PR: {best_auc_pr:.4f}")
            
            # Register best model
            if best_model_uri is not None:
                registered_model = mlflow.register_model(
                    model_uri=best_model_uri,
                    name=f"fraud-detection-{cfg.model.name}"
                )
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=registered_model.name,
                    version=registered_model.version,
                    description=f"Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}"
                )
                logger.info(f"Registered {registered_model.name} version {registered_model.version}")
    
            return {
                "run_id": parent_run.info.run_id,
                "best_run_id": best_run_id,
                "results": results,
                "mean_auc_pr": mean_auc_pr,
                "best_auc_pr": best_auc_pr,
                "num_windows": len(results),
            }
        else:
            logger.warning("No windows processed!")
            return {
                "run_id": parent_run.info.run_id,
                "results": [],
                "mean_auc_pr": 0,
                "best_auc_pr": 0,
                "num_windows": 0,
            }


if __name__ == "__main__":
    main()

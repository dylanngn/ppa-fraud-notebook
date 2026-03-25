"""
Main Training Entry Point.

Supports two training modes:
  single     — Fixed train/test split (default). Suitable for controlled A/B comparisons.
  expanding  — Expanding window backtesting for concept drift validation (RQ3).

Supports Optuna HPO sweeps via hydra-optuna-sweeper (see configs/hpo_*.yaml).

MLflow experiment structure
---------------------------
  fraud-detection           — All single-mode training runs
  fraud-detection-expanding — All expanding-window runs
  fraud-detection-hpo       — All HPO sweep trials (set in hpo_*.yaml)

Each run is tagged with model.variant, model.encoder, and training.mode so
that individual runs can be filtered without needing separate experiments.
"""

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
import os
import mlflow

from src.training.pipeline import SingleTrainingPipeline, ExpandingWindowPipeline
from src.features.store import FeatureStore

logger = logging.getLogger(__name__)


def _setup_mlflow(cfg: DictConfig, training_mode: str, original_cwd: str) -> None:
    """Configure MLflow tracking URI and experiment from cfg.mlflow."""
    mlflow_cfg = cfg.get("mlflow", {})

    # --- Tracking URI ---
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///ppa-fraud-detection-mlflow.db")
    # Resolve relative SQLite paths against the project root (not Hydra's output dir)
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_path = tracking_uri.replace("sqlite:///", "")
        tracking_uri = f"sqlite:///{os.path.join(original_cwd, db_path)}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # --- Artifact Location ---
    artifact_location = mlflow_cfg.get("artifact_location", "mlruns")
    if not os.path.isabs(artifact_location):
        artifact_location = os.path.join(original_cwd, artifact_location)

    # --- Experiment Name ---
    # Expanding-window runs go to a dedicated experiment so they don't pollute
    # the main single-run experiment in the MLflow UI.
    base_name = mlflow_cfg.get("experiment_name", "fraud-detection")
    if training_mode == "expanding":
        experiment_name = f"{base_name}-expanding"
    else:
        experiment_name = base_name

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")


@hydra.main(version_base=None, config_path="../../configs", config_name="experiment")
def main(cfg: DictConfig) -> float:
    """
    Main training entry point.

    Returns:
        float: The primary optimisation metric (AUC-PR) for Optuna HPO.
               Direction is 'maximize' (set in the sweeper config).
    """
    original_cwd = get_original_cwd()

    variant = cfg.model.variant
    training_mode = cfg.training.get("mode", "single")
    encoder = (
        cfg.model.get("gnn", {}).get("encoder", "none")
        if variant == "gnn_xgboost"
        else "none"
    )

    _setup_mlflow(cfg, training_mode, original_cwd)

    # --- Data ---
    data_path = cfg.data.path
    if not os.path.isabs(data_path):
        data_path = os.path.join(original_cwd, data_path)
    feature_store = FeatureStore(data_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # --- Pipeline ---
    if training_mode == "expanding":
        pipeline = ExpandingWindowPipeline(cfg_dict, feature_store)
    else:
        pipeline = SingleTrainingPipeline(cfg_dict, feature_store)

    # --- Run name: concise and human-readable ---
    # e.g. "vanilla_xgboost", "gnn_xgboost/graphsage", "gnn_xgboost/expanding"
    if training_mode == "expanding":
        run_name = f"{variant}/expanding"
    elif encoder != "none":
        run_name = f"{variant}/{encoder}"
    else:
        run_name = variant

    with mlflow.start_run(run_name=run_name):
        # Tags — primary mechanism for filtering runs in the MLflow UI
        mlflow.set_tags({
            "model.variant": variant,
            "model.encoder": encoder,
            "training.mode": training_mode,
        })

        # Params — logged flat for easy comparison across runs
        mlflow.log_params({
            "model.variant": variant,
            "model.n_estimators": cfg.model.xgboost.n_estimators,
            "model.max_depth": cfg.model.xgboost.max_depth,
            "model.learning_rate": cfg.model.xgboost.learning_rate,
            "model.min_child_weight": cfg.model.xgboost.min_child_weight,
            "model.subsample": cfg.model.xgboost.subsample,
            "model.colsample_bytree": cfg.model.xgboost.colsample_bytree,
            "training.gap_days": cfg.training.gap_days,
            "training.mode": training_mode,
            "data.train_end_date": (
                cfg.data.test_end_date if training_mode == "expanding"
                else cfg.data.train_end_date
            ),
            "data.test_end_date": cfg.data.test_end_date,
        })

        if training_mode == "expanding":
            mlflow.log_params({
                "expanding.window_days": cfg.get("expanding_window", {}).get("window_days", 30),
                "expanding.min_train_windows": cfg.get("expanding_window", {}).get("min_train_windows", 2),
            })

        if variant == "gnn_xgboost":
            gnn_cfg = cfg.model.get("gnn", {})
            mlflow.log_params({
                "gnn.encoder": gnn_cfg.get("encoder", "graphsage"),
                "gnn.hidden_dim": gnn_cfg.get("hidden_dim", 32),
                "gnn.output_dim": gnn_cfg.get("output_dim", 16),
                "gnn.num_layers": gnn_cfg.get("num_layers", 2),
                "gnn.epochs": gnn_cfg.get("epochs", 10),
                "gnn.supervised": gnn_cfg.get("supervised", True),
            })

        result = pipeline.run()

        if training_mode == "expanding":
            optimization_metric = (
                result.get("aggregate", {}).get("mean_auc_pr", 0.0) if result else 0.0
            )
        else:
            optimization_metric = result.get("auc_pr", 0.0) if result else 0.0

        logger.info(f"Optimisation metric (auc_pr): {optimization_metric}")
        return optimization_metric


if __name__ == "__main__":
    main()

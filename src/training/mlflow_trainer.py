"""
MLflow Integration for Fraud Detection Training

Provides experiment tracking, model versioning, and artifact management.

Features:
- Automatic experiment tracking (hyperparams, metrics, artifacts)
- Model registry for staging/production management
- SHAP integration for model explanations
- Adaptation report logging

Usage:
    from src.training.mlflow_trainer import MLflowTrainer
    
    trainer = MLflowTrainer(experiment_name="fraud-detection")
    with trainer.start_run(run_name="baseline_v1"):
        trainer.log_params(hyperparams)
        model = train_model(...)
        trainer.log_metrics(metrics)
        trainer.log_model(model, feature_names)
        trainer.log_shap_summary(model, X_sample)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import json

import numpy as np
import mlflow
from mlflow.models import infer_signature
import xgboost as xgb

# Optional imports for SHAP/plots
try:
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class MLflowTrainer:
    """
    MLflow integration for fraud detection model training.
    
    Handles:
    - Experiment tracking
    - Model versioning
    - Artifact management
    - SHAP explanations
    """
    
    def __init__(
        self,
        experiment_name: str = "ppa-fraud-detection",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow trainer.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Where to store artifacts (default: ./mlruns)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self._active_run = None
        self._run_id = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Human-readable run name
            tags: Additional tags for the run
            nested: Whether this is a nested run
            
        Returns:
            self (for context manager usage)
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        default_tags = {
            "framework": "xgboost",
            "task": "fraud_detection",
            "timestamp": datetime.now().isoformat(),
        }
        
        if tags:
            default_tags.update(tags)
        
        self._active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=default_tags,
            nested=nested,
        )
        self._run_id = self._active_run.info.run_id
        
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
    
    def end_run(self):
        """End the current run."""
        if self._active_run:
            mlflow.end_run()
            self._active_run = None
            self._run_id = None
    
    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run_id
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        # MLflow has limits on param value length, so stringify long values
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                value = json.dumps(value)[:500]
            mlflow.log_param(key, value)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (for time series metrics)
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        X_sample: Optional[np.ndarray] = None,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> str:
        """
        Log trained model to MLflow.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            X_sample: Sample input for signature inference
            artifact_path: Path within artifacts
            registered_model_name: If provided, register in Model Registry
            
        Returns:
            Model URI
        """
        # Infer signature from sample data
        signature = None
        if X_sample is not None:
            try:
                y_pred = model.predict_proba(X_sample)
                signature = infer_signature(X_sample, y_pred)
            except Exception as e:
                print(f"Warning: Could not infer signature: {e}")
        
        # Log model with XGBoost flavor
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=X_sample[:5] if X_sample is not None else None,
            registered_model_name=registered_model_name,
        )
        
        # Log feature names as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"feature_names": feature_names}, f)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="metadata")
            os.unlink(f.name)
        
        return model_info.model_uri
    
    def log_feature_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
    ) -> None:
        """
        Log feature importance as metrics and artifact.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return
        
        # Log as metrics (top 10) - convert to Python floats for JSON serialization
        importance_dict = {
            name: float(val)
            for name, val in zip(feature_names, importances)
        }
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feat, imp) in enumerate(sorted_importance[:10]):
            mlflow.log_metric(f"feature_importance_{i+1}_{feat}", imp)
        
        # Log full importance as artifact
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(importance_dict, f, indent=2)
            f.flush()
            mlflow.log_artifact(f.name, artifact_path="importance")
            os.unlink(f.name)
    
    def log_shap_summary(
        self,
        model: xgb.XGBClassifier,
        X_sample: np.ndarray,
        feature_names: List[str],
        max_samples: int = 500,
    ) -> None:
        """
        Generate and log SHAP summary plot.
        
        Args:
            model: Trained XGBoost model
            X_sample: Sample data for SHAP analysis
            feature_names: List of feature names
            max_samples: Maximum samples for SHAP (for speed)
        """
        if not SHAP_AVAILABLE:
            print("Warning: SHAP not available, skipping SHAP summary")
            return
        
        try:
            # Sample for speed
            if len(X_sample) > max_samples:
                indices = np.random.choice(len(X_sample), max_samples, replace=False)
                X_shap = X_sample[indices]
            else:
                X_shap = X_sample
            
            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            # Create summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_shap,
                feature_names=feature_names,
                show=False,
                plot_type="bar",
            )
            plt.tight_layout()
            
            # Save and log
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                plt.savefig(f.name, dpi=100, bbox_inches='tight')
                mlflow.log_artifact(f.name, artifact_path="shap")
                os.unlink(f.name)
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
    
    def log_adaptation_report(
        self,
        report_path: str,
    ) -> None:
        """
        Log adaptation report as artifact.
        
        Args:
            report_path: Path to adaptation report (markdown or JSON)
        """
        if os.path.exists(report_path):
            mlflow.log_artifact(report_path, artifact_path="reports")
    
    def log_training_data_info(
        self,
        train_size: int,
        test_size: int,
        fraud_rate: float,
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> None:
        """
        Log training data statistics.
        
        Args:
            train_size: Number of training samples
            test_size: Number of test samples
            fraud_rate: Fraud rate in training data
            window_start: Training window start date
            window_end: Training window end date
        """
        mlflow.log_params({
            "train_size": train_size,
            "test_size": test_size,
            "fraud_rate": f"{fraud_rate:.4f}",
        })
        
        if window_start:
            mlflow.log_param("window_start", str(window_start))
        if window_end:
            mlflow.log_param("window_end", str(window_end))
    
    def log_figure(
        self,
        fig: Any,
        filename: str,
        artifact_path: str = "figures",
    ) -> None:
        """
        Log matplotlib figure as artifact.
        
        Args:
            fig: Matplotlib figure
            filename: Filename for the figure
            artifact_path: Path within artifacts
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig.savefig(f.name, dpi=100, bbox_inches='tight')
            mlflow.log_artifact(f.name, artifact_path=artifact_path)
            os.unlink(f.name)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        await_registration: bool = True,
    ) -> str:
        """
        Register model in the Model Registry.
        
        Args:
            model_uri: URI of the logged model
            name: Name for the registered model
            await_registration: Whether to wait for registration to complete
            
        Returns:
            Model version
        """
        result = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            await_registration_for=300 if await_registration else 0,
        )
        return result.version
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """
        Transition model to a new stage.
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
    
    def load_production_model(
        self,
        name: str = "fraud-detection",
    ) -> xgb.XGBClassifier:
        """
        Load the production model from registry.
        
        Args:
            name: Registered model name
            
        Returns:
            Loaded XGBoost model
        """
        model_uri = f"models:/{name}/Production"
        return mlflow.xgboost.load_model(model_uri)
    
    def compare_with_production(
        self,
        new_metrics: Dict[str, float],
        name: str = "fraud-detection",
        improvement_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compare new model metrics with production model.
        
        Args:
            new_metrics: Metrics from new model
            name: Registered model name
            improvement_threshold: Minimum improvement to recommend deployment
            
        Returns:
            Comparison results with recommendation
        """
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Get production model version
            prod_versions = client.get_latest_versions(name, stages=["Production"])
            
            if not prod_versions:
                return {
                    "has_production": False,
                    "recommendation": "deploy",
                    "reason": "No production model exists",
                }
            
            prod_version = prod_versions[0]
            prod_run = client.get_run(prod_version.run_id)
            prod_metrics = prod_run.data.metrics
            
            # Compare primary metric (AUC-PR)
            new_auc = new_metrics.get("auc_pr", 0)
            prod_auc = prod_metrics.get("auc_pr", 0)
            
            improvement = new_auc - prod_auc
            improvement_pct = (improvement / prod_auc * 100) if prod_auc > 0 else 100
            
            result = {
                "has_production": True,
                "production_auc_pr": prod_auc,
                "new_auc_pr": new_auc,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "production_version": prod_version.version,
                "production_run_id": prod_version.run_id,
            }
            
            if improvement >= improvement_threshold:
                result["recommendation"] = "deploy"
                result["reason"] = f"New model is {improvement_pct:.2f}% better"
            else:
                result["recommendation"] = "keep_current"
                result["reason"] = f"Improvement ({improvement_pct:.2f}%) below threshold ({improvement_threshold*100:.1f}%)"
            
            return result
            
        except Exception as e:
            return {
                "has_production": False,
                "recommendation": "deploy",
                "reason": f"Could not compare: {e}",
            }


def train_with_mlflow(
    experiment_name: str = "ppa-fraud-detection",
    model_type: str = "baseline",
    window_days: int = 90,
    step_days: int = 7,
    register_model: bool = False,
) -> Dict[str, Any]:
    """
    Train fraud detection model with full MLflow tracking.
    
    This is the recommended entry point for tracked training.
    
    Args:
        experiment_name: MLflow experiment name
        model_type: Type of model (baseline, baseline_graph, etc.)
        window_days: Training window size
        step_days: Sliding window step size
        register_model: Whether to register in Model Registry
        
    Returns:
        Training results and run info
    """
    from src.models.train_baseline import (
        load_data,
        feature_engineering,
        get_base_features,
        GRAPH_FEATURE_COLUMNS,
        ADVANCED_GRAPH_FEATURE_COLUMNS,
        TIME_WEIGHTED_FEATURE_COLUMNS,
        INTERACTION_FEATURE_COLUMNS,
    )
    from src.utils.metrics import calculate_metrics
    from datetime import timedelta
    import polars as pl
    
    include_graph = model_type in ["baseline_graph", "graph"]
    
    trainer = MLflowTrainer(experiment_name=experiment_name)
    
    # Load and prepare data
    print("Loading data...")
    df = load_data()
    df = feature_engineering(df, include_graph_features=include_graph)
    df = df.sort("submission_at")
    
    # Build feature list
    all_graph_cols = (
        GRAPH_FEATURE_COLUMNS +
        ADVANCED_GRAPH_FEATURE_COLUMNS +
        TIME_WEIGHTED_FEATURE_COLUMNS +
        INTERACTION_FEATURE_COLUMNS
    )
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + (graph_columns if include_graph else [])
    
    # Define window parameters
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=14)
    
    # Hyperparameters (optimized from Experiment 11)
    hyperparams = {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "gamma": 0.4,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
    }
    
    all_results = []
    best_auc = 0
    best_model = None
    best_metrics = None
    best_X_test = None
    
    current_date = start_date + window_size
    window_idx = 0
    
    with trainer.start_run(
        run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d')}",
        tags={"model_type": model_type}
    ):
        # Log configuration
        trainer.log_params({
            "model_type": model_type,
            "window_days": window_days,
            "step_days": step_days,
            "feature_count": len(features),
            "include_graph_features": include_graph,
            **hyperparams,
        })
        
        print(f"Training {model_type} with {len(features)} features...")
        
        while current_date + test_size <= end_date:
            train_end = current_date
            test_end = current_date + test_size
            
            # Split data
            train_data = df.filter(
                (pl.col("submission_at") < train_end) &
                (pl.col("submission_at") >= train_end - window_size)
            )
            test_data = df.filter(
                (pl.col("submission_at") >= train_end) &
                (pl.col("submission_at") < test_end)
            )
            
            if len(test_data) == 0 or len(train_data) == 0:
                current_date += step_size
                continue
            
            X_train = train_data.select(features).to_numpy()
            y_train = train_data.select("is_fraud").to_numpy().flatten()
            X_test = test_data.select(features).to_numpy()
            y_test = test_data.select("is_fraud").to_numpy().flatten()
            
            # Train model
            scale_pos = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)
            
            model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos,
                n_jobs=-1,
                random_state=42,
                **hyperparams,
            )
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            proba = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, proba)
            
            # Log per-window metrics (sanitize names - @ not allowed in MLflow)
            trainer.log_metrics({
                f"window_{window_idx}_auc_pr": metrics["auc_pr"],
                f"window_{window_idx}_p_at_100": metrics["p@100"],
            }, step=window_idx)
            
            all_results.append({
                "window_idx": window_idx,
                "window_start": str(train_end),
                **metrics,
            })
            
            # Track best model
            if metrics["auc_pr"] > best_auc:
                best_auc = metrics["auc_pr"]
                best_model = model
                best_metrics = metrics
                best_X_test = X_test
            
            if window_idx % 20 == 0:
                print(f"Window {window_idx}: AUC-PR = {metrics['auc_pr']:.4f}")
            
            window_idx += 1
            current_date += step_size
        
        # Compute aggregate metrics
        if all_results:
            mean_auc_pr = np.mean([r["auc_pr"] for r in all_results])
            mean_p100 = np.mean([r["p@100"] for r in all_results])
            
            trainer.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "mean_p_at_100": mean_p100,  # Sanitized for MLflow
                "num_windows": float(len(all_results)),
                "best_auc_pr": best_auc,
            })
            
            print(f"\nTraining complete!")
            print(f"  Windows evaluated: {len(all_results)}")
            print(f"  Mean AUC-PR: {mean_auc_pr:.4f}")
            print(f"  Best AUC-PR: {best_auc:.4f}")
        
        # Log best model
        if best_model is not None:
            model_uri = trainer.log_model(
                best_model,
                feature_names=features,
                X_sample=best_X_test[:100] if best_X_test is not None else None,
                registered_model_name="fraud-detection" if register_model else None,
            )
            
            # Log feature importance
            trainer.log_feature_importance(best_model, features)
            
            # Log SHAP summary (for best model)
            if best_X_test is not None:
                trainer.log_shap_summary(
                    best_model,
                    best_X_test[:500],
                    features,
                )
            
            # Compare with production
            if register_model:
                comparison = trainer.compare_with_production(
                    {"auc_pr": best_auc},
                    name="fraud-detection",
                )
                print(f"\nProduction comparison: {comparison['recommendation']}")
                print(f"  Reason: {comparison['reason']}")
        
        return {
            "run_id": trainer.run_id,
            "results": all_results,
            "mean_auc_pr": mean_auc_pr if all_results else 0,
            "best_auc_pr": best_auc,
            "model_uri": model_uri if best_model else None,
        }


if __name__ == "__main__":
    # Example usage
    result = train_with_mlflow(
        experiment_name="ppa-fraud-detection",
        model_type="baseline",
        window_days=90,
        step_days=14,
        register_model=False,
    )
    
    print(f"\nRun ID: {result['run_id']}")
    print(f"Mean AUC-PR: {result['mean_auc_pr']:.4f}")
    print(f"Model URI: {result['model_uri']}")


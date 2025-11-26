"""
Continuous Pipeline Orchestrator for Fraud Detection.

Orchestrates the complete continuous learning lifecycle:
- Daily: Drift monitoring
- Weekly: Model retraining and evaluation
- Monthly: Hyperparameter optimization

Usage:
    from src.orchestration import ContinuousPipeline, PipelineConfig
    
    config = PipelineConfig(
        experiment_name="ppa-fraud-detection",
        model_type="baseline_graph",
    )
    
    pipeline = ContinuousPipeline(config)
    
    # Run individual steps
    pipeline.run_daily_drift_check()
    pipeline.run_weekly_retrain()
    
    # Or start scheduler
    pipeline.start_scheduler()
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import os

import numpy as np
import polars as pl

from .drift_detector import DriftDetector, DriftReport
from .notifier import Notifier, ConsoleNotifier, create_notifier_from_config


@dataclass
class PipelineConfig:
    """Configuration for the continuous pipeline."""
    # MLflow settings
    experiment_name: str = "ppa-fraud-detection"
    model_registry_name: str = "fraud-detection"
    tracking_uri: Optional[str] = None
    
    # Training settings
    model_type: str = "baseline_graph"
    window_days: int = 90
    step_days: int = 14
    
    # Drift detection
    drift_method: str = "psi"
    drift_threshold: float = 0.2
    critical_feature_count: int = 3
    
    # Retraining
    improvement_threshold: float = 0.01
    force_retrain_on_drift: bool = True
    
    # Hyperparameter optimization
    hyperopt_n_trials: int = 50
    hyperopt_n_windows: int = 5
    
    # Notifications
    notification_config: Dict = field(default_factory=lambda: {"type": "console"})
    
    # Paths
    artifacts_dir: str = "artifacts"
    reports_dir: str = "artifacts/reports"
    models_dir: str = "artifacts/models"
    
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return {
            "experiment_name": self.experiment_name,
            "model_registry_name": self.model_registry_name,
            "model_type": self.model_type,
            "window_days": self.window_days,
            "step_days": self.step_days,
            "drift_method": self.drift_method,
            "drift_threshold": self.drift_threshold,
            "improvement_threshold": self.improvement_threshold,
        }


@dataclass
class PipelineRunResult:
    """Result from a pipeline run."""
    job_type: str  # "daily", "weekly", "monthly"
    timestamp: datetime
    success: bool
    duration_seconds: float
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "job_type": self.job_type,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary,
            "details": self.details,
        }


class ContinuousPipeline:
    """
    Main orchestrator for the continuous fraud detection pipeline.
    
    Coordinates:
    - Feature generation
    - Model training (via MLflow)
    - Drift detection
    - Adaptation analysis
    - Notifications
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        notifier: Optional[Notifier] = None,
    ):
        """
        Initialize the continuous pipeline.
        
        Args:
            config: Pipeline configuration
            notifier: Notification service (default: console)
        """
        self.config = config or PipelineConfig()
        
        # Initialize notifier
        if notifier:
            self.notifier = notifier
        else:
            self.notifier = create_notifier_from_config(
                self.config.notification_config
            )
        
        # Initialize drift detector
        self.drift_detector = DriftDetector(
            method=self.config.drift_method,
            psi_threshold=self.config.drift_threshold,
        )
        
        # Initialize MLflow trainer (lazy import)
        self._trainer = None
        
        # Ensure directories exist
        Path(self.config.reports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        
        # Load training stats if available
        stats_path = Path(self.config.artifacts_dir) / "training_stats.json"
        if stats_path.exists():
            self.drift_detector.load_stats(str(stats_path))
    
    @property
    def trainer(self):
        """Lazy-load MLflow trainer."""
        if self._trainer is None:
            from src.training.mlflow_trainer import MLflowTrainer
            self._trainer = MLflowTrainer(
                experiment_name=self.config.experiment_name,
                tracking_uri=self.config.tracking_uri,
            )
        return self._trainer
    
    def run_daily_drift_check(self) -> PipelineRunResult:
        """
        Daily job: Check for feature drift.
        
        Compares current production predictions with training distribution.
        Alerts if critical drift is detected.
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"[{start_time}] Running Daily Drift Check")
        print(f"{'='*60}\n")
        
        try:
            # Load recent data (last 24 hours)
            # In production, this would query your production database
            recent_data = self._load_recent_data(hours=24)
            
            if recent_data is None or len(recent_data) == 0:
                return PipelineRunResult(
                    job_type="daily",
                    timestamp=start_time,
                    success=True,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    summary="No recent data available for drift check",
                    details={"data_count": 0},
                )
            
            # Get feature columns
            feature_columns = self._get_feature_columns()
            
            # Detect drift
            if not self.drift_detector.training_stats:
                # First run - compute training stats
                print("Computing training statistics...")
                training_data = self._load_training_data()
                self.drift_detector.compute_training_stats(training_data, feature_columns)
                self.drift_detector.save_stats(
                    str(Path(self.config.artifacts_dir) / "training_stats.json")
                )
            
            report = self.drift_detector.detect_drift(recent_data, feature_columns)
            
            # Save report
            report_path = f"{self.config.reports_dir}/drift_report_{start_time.strftime('%Y%m%d')}.json"
            report.save_json(report_path)
            
            # Alert if critical drift
            if report.has_critical_drift:
                self.notifier.send_critical(
                    "Critical Drift Detected",
                    report.summary,
                    drifted_features=report.drifted_features,
                    critical_count=report.critical_count,
                )
            elif report.drifted_features > 0:
                self.notifier.send_warning(
                    "Feature Drift Detected",
                    report.summary,
                    drifted_features=report.drifted_features,
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\nDrift check complete in {duration:.1f}s")
            print(f"  Total features: {report.total_features}")
            print(f"  Drifted: {report.drifted_features}")
            print(f"  Critical: {report.critical_count}")
            
            return PipelineRunResult(
                job_type="daily",
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                summary=report.summary,
                details={
                    "total_features": report.total_features,
                    "drifted_features": report.drifted_features,
                    "critical_count": report.critical_count,
                    "has_critical_drift": report.has_critical_drift,
                },
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.notifier.send_critical(
                "Daily Drift Check Failed",
                str(e),
            )
            return PipelineRunResult(
                job_type="daily",
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                summary=f"Error: {e}",
                details={"error": str(e)},
            )
    
    def run_weekly_retrain(self, force: bool = False) -> PipelineRunResult:
        """
        Weekly job: Retrain model and compare with production.
        
        Deploys new model if improvement exceeds threshold.
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"[{start_time}] Running Weekly Retrain")
        print(f"{'='*60}\n")
        
        try:
            from src.training.mlflow_trainer import train_with_mlflow
            
            # Train new model
            print(f"Training {self.config.model_type} model...")
            result = train_with_mlflow(
                experiment_name=self.config.experiment_name,
                model_type=self.config.model_type,
                window_days=self.config.window_days,
                step_days=self.config.step_days,
                register_model=True,
            )
            
            # Compare with production
            comparison = self.trainer.compare_with_production(
                {"auc_pr": result["best_auc_pr"]},
                name=self.config.model_registry_name,
                improvement_threshold=self.config.improvement_threshold,
            )
            
            should_deploy = (
                comparison["recommendation"] == "deploy" or 
                force or
                (self.config.force_retrain_on_drift and self._check_recent_drift())
            )
            
            if should_deploy:
                # Promote to production
                if comparison.get("has_production"):
                    new_version = self._get_latest_version()
                    self.trainer.transition_model_stage(
                        name=self.config.model_registry_name,
                        version=str(new_version),
                        stage="Production",
                    )
                    
                    self.notifier.send_info(
                        "New Model Deployed",
                        f"Model v{new_version} promoted to Production. "
                        f"AUC-PR: {result['best_auc_pr']:.4f}",
                        model_version=new_version,
                        auc_pr=result["best_auc_pr"],
                        improvement=comparison.get("improvement_pct", 0),
                    )
            
            # Run adaptation analysis
            self._run_adaptation_analysis(result["run_id"])
            
            duration = (datetime.now() - start_time).total_seconds()
            
            summary = (
                f"Training complete. AUC-PR: {result['best_auc_pr']:.4f}. "
                f"{'Deployed.' if should_deploy else 'Not deployed.'} "
                f"Reason: {comparison['reason']}"
            )
            
            print(f"\n{summary}")
            print(f"Duration: {duration:.1f}s")
            
            return PipelineRunResult(
                job_type="weekly",
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                summary=summary,
                details={
                    "run_id": result["run_id"],
                    "mean_auc_pr": result["mean_auc_pr"],
                    "best_auc_pr": result["best_auc_pr"],
                    "deployed": should_deploy,
                    "comparison": comparison,
                },
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.notifier.send_critical(
                "Weekly Retrain Failed",
                str(e),
            )
            return PipelineRunResult(
                job_type="weekly",
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                summary=f"Error: {e}",
                details={"error": str(e)},
            )
    
    def run_monthly_hyperopt(self) -> PipelineRunResult:
        """
        Monthly job: Run hyperparameter optimization.
        
        Updates best parameters if improvement found.
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"[{start_time}] Running Monthly Hyperparameter Optimization")
        print(f"{'='*60}\n")
        
        try:
            from src.experiments.staged_hyperopt import main as run_staged_hyperopt
            
            # Run staged hyperopt
            print(f"Running staged hyperopt with {self.config.hyperopt_n_windows} windows...")
            run_staged_hyperopt(n_windows=self.config.hyperopt_n_windows)
            
            # Load results
            results_path = Path(self.config.artifacts_dir) / "results" / "staged_hyperopt_summary.csv"
            if results_path.exists():
                results_df = pl.read_csv(str(results_path))
                best_row = results_df.sort("composite", descending=True).head(1)
                best_auc = float(best_row["auc_pr"][0])
                best_config = best_row["config_name"][0]
            else:
                best_auc = 0
                best_config = "unknown"
            
            # Force retrain with best params
            print("\nRetraining with optimized parameters...")
            retrain_result = self.run_weekly_retrain(force=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            summary = (
                f"Hyperopt complete. Best config: {best_config} (AUC-PR: {best_auc:.4f}). "
                f"Retrained and deployed."
            )
            
            self.notifier.send_info(
                "Monthly Hyperopt Complete",
                summary,
                best_config=best_config,
                best_auc_pr=best_auc,
                duration_minutes=duration / 60,
            )
            
            return PipelineRunResult(
                job_type="monthly",
                timestamp=start_time,
                success=True,
                duration_seconds=duration,
                summary=summary,
                details={
                    "best_config": best_config,
                    "best_auc_pr": best_auc,
                    "retrain_result": retrain_result.to_dict(),
                },
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.notifier.send_critical(
                "Monthly Hyperopt Failed",
                str(e),
            )
            return PipelineRunResult(
                job_type="monthly",
                timestamp=start_time,
                success=False,
                duration_seconds=duration,
                summary=f"Error: {e}",
                details={"error": str(e)},
            )
    
    def start_scheduler(self, blocking: bool = True) -> None:
        """
        Start the pipeline scheduler.
        
        Args:
            blocking: Whether to block (True) or run in background (False)
        """
        from .scheduler import PipelineScheduler
        
        scheduler = PipelineScheduler(
            pipeline=self,
            blocking=blocking,
        )
        scheduler.setup_default_schedule()
        scheduler.start()
    
    def _load_recent_data(self, hours: int = 24) -> Optional[pl.DataFrame]:
        """Load recent production data for drift detection."""
        # In production, this would query your production database
        # For now, use test data from artifacts
        nodes_path = Path(self.config.artifacts_dir) / "nodes_listing.parquet"
        
        if not nodes_path.exists():
            return None
        
        df = pl.read_parquet(str(nodes_path))
        
        # Filter to recent data
        cutoff = datetime.now() - pl.duration(hours=hours)
        if "submission_at" in df.columns:
            # For demo, just take last N rows
            df = df.tail(min(1000, len(df)))
        
        return df
    
    def _load_training_data(self) -> pl.DataFrame:
        """Load training data for statistics computation."""
        nodes_path = Path(self.config.artifacts_dir) / "nodes_listing.parquet"
        return pl.read_parquet(str(nodes_path))
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns to monitor."""
        return [
            "account_age_days", "log_price", "living_space", "rooms",
            "is_new", "has_balcony", "has_elevator", "has_parking",
            "bundle_period", "bundle_tier_score",
            "is_direct_payment", "is_buy",
            "latitude", "longitude",
        ]
    
    def _check_recent_drift(self) -> bool:
        """Check if critical drift was detected recently."""
        reports_dir = Path(self.config.reports_dir)
        drift_reports = sorted(reports_dir.glob("drift_report_*.json"))
        
        if not drift_reports:
            return False
        
        # Check last report
        with open(drift_reports[-1]) as f:
            report = json.load(f)
        
        return report.get("has_critical_drift", False)
    
    def _get_latest_version(self) -> int:
        """Get latest registered model version."""
        import mlflow
        client = mlflow.tracking.MlflowClient()
        
        try:
            versions = client.search_model_versions(
                f"name='{self.config.model_registry_name}'"
            )
            if versions:
                return max(int(v.version) for v in versions)
        except Exception:
            pass
        
        return 1
    
    def _run_adaptation_analysis(self, run_id: str) -> None:
        """Run adaptation analysis on the trained model."""
        try:
            from src.explainability.adaptation_engine import AdaptationEngine
            
            # Load the latest trained model
            import mlflow
            import glob
            
            # Find latest model file
            model_files = sorted(glob.glob(
                f"{self.config.models_dir}/{self.config.model_type}/model_window_*.pkl"
            ))
            
            if model_files:
                from src.utils.explainability import load_saved_model
                bundle = load_saved_model(model_files[-1])
                
                engine = AdaptationEngine(
                    feature_names=bundle["features"],
                    history_window=5,
                )
                
                report = engine.analyze_window(
                    model=bundle["model"],
                    X_test=bundle["X_test"],
                    y_test=bundle["y_test"],
                    y_pred=bundle["y_pred"],
                    window_info=bundle.get("window_info", {}),
                )
                
                # Save report
                report_path = (
                    f"{self.config.reports_dir}/"
                    f"adaptation_report_weekly_{datetime.now().strftime('%Y%m%d')}.md"
                )
                report.save_markdown(report_path)
                
                print(f"Adaptation report saved to {report_path}")
                
        except Exception as e:
            print(f"Warning: Adaptation analysis failed: {e}")


def run_pipeline_job(
    job_type: str,
    config_path: Optional[str] = None,
    **kwargs,
) -> PipelineRunResult:
    """
    Run a pipeline job from CLI.
    
    Args:
        job_type: "daily", "weekly", or "monthly"
        config_path: Optional path to YAML config
        **kwargs: Override config options
    """
    if config_path:
        config = PipelineConfig.from_yaml(config_path)
    else:
        config = PipelineConfig(**kwargs)
    
    pipeline = ContinuousPipeline(config)
    
    if job_type == "daily":
        return pipeline.run_daily_drift_check()
    elif job_type == "weekly":
        return pipeline.run_weekly_retrain()
    elif job_type == "monthly":
        return pipeline.run_monthly_hyperopt()
    else:
        raise ValueError(f"Unknown job type: {job_type}")


if __name__ == "__main__":
    # Test the pipeline
    config = PipelineConfig(
        model_type="baseline",
        window_days=90,
        step_days=56,  # Faster for testing
    )
    
    pipeline = ContinuousPipeline(config)
    
    print("Testing Daily Drift Check...")
    result = pipeline.run_daily_drift_check()
    print(f"\nResult: {result.summary}")


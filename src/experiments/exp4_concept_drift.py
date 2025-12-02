"""
Experiment 4: Concept Drift Evaluation (RQ3)

Compares:
- Part A: Accumulating window (periodic retraining) - existing MLflow runs
- Part B: Static model (no retraining) - trains once, evaluates on all windows

Research Question: To what extent does periodic retraining maintain the model's
predictive performance against concept drift over sequential time windows?

Run with: python -m src.experiments.exp4_concept_drift experiment_name=concept-drift-rq3 features=production
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import hydra
import mlflow
import pandas as pd
import polars as pl
import xgboost as xgb
from omegaconf import DictConfig

from src.data.loader import load_data
from src.features.definitions.base import compute_base_features
from src.features.processor import FeatureProcessor
from src.models.utils.common import setup_mlflow
from src.models.xgboost.utils import get_optimal_tree_method, validate_features, get_categorical_features
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate_static_model(
    df: pl.DataFrame,
    config: DictConfig,
    max_windows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train a static model on initial window, then evaluate on ALL future windows
    WITHOUT retraining. This simulates what happens without periodic retraining.
    
    Args:
        df: DataFrame with features and target
        config: Hydra configuration
        max_windows: Optional limit on windows (for debugging)
        
    Returns:
        Dictionary with per-window results and degradation analysis
    """
    # Extract config
    initial_window_days = config.model.training.initial_window_days
    step_days = config.model.training.step_days
    xgb_params = dict(config.model.params)
    feature_categories = config.features.categories
    experiment_name = config.experiment_name
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Define window parameters
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    target = "is_fraud"
    
    # Setup MLflow
    setup_mlflow(experiment_name)
    
    # Compute features using processor
    processor = FeatureProcessor.from_config(config)
    
    training_start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"static_model_drift_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"model_type": "static", "training_mode": "concept_drift_evaluation"}
    ) as parent_run:
        
        mlflow.log_params({
            "model_name": "xgboost_static",
            "initial_window_days": initial_window_days,
            "step_days": step_days,
            "total_samples": len(df),
            "fraud_rate": float(df[target].mean()),
            "training_mode": "static_no_retrain",
        })
        
        # =====================================================
        # STEP 1: Train ONCE on initial window
        # =====================================================
        initial_train_end = start_date + timedelta(days=initial_window_days)
        initial_train_data = df.filter(pl.col("submission_at") < initial_train_end)
        
        logger.info(f"Training static model on initial window: {start_date.date()} → {initial_train_end.date()}")
        logger.info(f"  Training samples: {len(initial_train_data)}")
        
        # Compute features for initial training data
        initial_features = processor.process(initial_train_data, initial_train_end)
        
        # Prepare training data
        train_pdf = initial_features.to_pandas()
        
        # Validate features
        valid_features, _ = validate_features(train_pdf, strict=False)
        feature_cols = valid_features
        
        # Handle categorical features (like trainer.py does)
        categorical_features = get_categorical_features(valid_features)
        for cat_feat in categorical_features:
            unique_vals = train_pdf[cat_feat].dropna().unique()
            train_pdf[cat_feat] = pd.Categorical(train_pdf[cat_feat], categories=unique_vals)
        
        X_train = train_pdf[valid_features]
        y_train = initial_features[target].to_pandas()
        
        # Store categorical info for test data processing
        categorical_mappings = {
            cat_feat: train_pdf[cat_feat].cat.categories.tolist()
            for cat_feat in categorical_features
        }
        
        # Train the STATIC model
        tree_method = get_optimal_tree_method()
        xgb_params["tree_method"] = tree_method
        
        # Enable categorical support if we have categorical features
        if categorical_features:
            xgb_params["enable_categorical"] = True
        
        static_model = xgb.XGBClassifier(**xgb_params)
        static_model.fit(X_train, y_train)
        
        logger.info(f"Static model trained. Now evaluating on ALL future windows...")
        
        mlflow.log_param("static_model_train_end", str(initial_train_end.date()))
        mlflow.log_param("static_model_train_samples", len(initial_train_data))
        
        # =====================================================
        # STEP 2: Evaluate on ALL future windows (NO retraining)
        # =====================================================
        current_date = start_date + timedelta(days=initial_window_days)
        test_size = timedelta(days=14)
        step_size = timedelta(days=step_days)
        
        results = []
        window_idx = 0
        
        while current_date + test_size <= end_date:
            if max_windows and window_idx >= max_windows:
                break
                
            train_end = current_date
            test_end = current_date + test_size
            
            # Get test data for this window
            test_data = df.filter(
                (pl.col("submission_at") >= train_end) & 
                (pl.col("submission_at") < test_end)
            )
            
            if len(test_data) == 0 or test_data[target].sum() == 0:
                current_date += step_size
                window_idx += 1
                continue
            
            # Compute features for test data (using train_end as cutoff)
            test_features = processor.process(test_data, train_end)
            test_pdf = test_features.to_pandas()
            
            # Apply same categorical mappings from training
            for cat_feat in categorical_features:
                test_pdf[cat_feat] = pd.Categorical(
                    test_pdf[cat_feat], 
                    categories=categorical_mappings[cat_feat]
                )
            
            X_test = test_pdf[feature_cols]
            y_test = test_features[target].to_pandas()
            
            # Predict using the STATIC (frozen) model
            with mlflow.start_run(run_name=f"window_{window_idx}", nested=True):
                mlflow.log_params({
                    "window_index": window_idx,
                    "test_start": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    "test_samples": len(test_data),
                    "test_fraud_count": int(test_data[target].sum()),
                    "days_since_training": (train_end - initial_train_end).days,
                })
                
                proba = static_model.predict_proba(X_test)[:, 1]
                metrics = calculate_metrics(y_test, proba)
                
                mlflow.log_metrics({
                    "auc_pr": metrics["auc_pr"],
                    "auc_roc": metrics["auc_roc"],
                    "p_at_100": metrics["p@100"],
                })
                
                days_since_train = (train_end - initial_train_end).days
                
                results.append({
                    "window_idx": window_idx,
                    "test_start": train_end,
                    "test_end": test_end,
                    "days_since_training": days_since_train,
                    "auc_pr": metrics["auc_pr"],
                    "auc_roc": metrics["auc_roc"],
                    "p@100": metrics["p@100"],
                    "test_samples": len(test_data),
                    "fraud_count": int(test_data[target].sum()),
                })
                
                if window_idx % 20 == 0:
                    logger.info(f"  Window {window_idx}: days_since_train={days_since_train}, AUC-PR={metrics['auc_pr']:.4f}")
            
            current_date += step_size
            window_idx += 1
        
        # =====================================================
        # STEP 3: Analyze degradation
        # =====================================================
        if results:
            results_df = pd.DataFrame(results)
            
            # Calculate summary metrics
            mean_auc_pr = results_df["auc_pr"].mean()
            std_auc_pr = results_df["auc_pr"].std()
            first_window_auc = results_df.iloc[0]["auc_pr"]
            last_window_auc = results_df.iloc[-1]["auc_pr"]
            
            # Calculate degradation
            total_degradation = last_window_auc - first_window_auc
            total_days = results_df.iloc[-1]["days_since_training"]
            degradation_per_month = (total_degradation / total_days) * 30 if total_days > 0 else 0
            
            mlflow.log_metrics({
                "mean_auc_pr": mean_auc_pr,
                "std_auc_pr": std_auc_pr,
                "first_window_auc_pr": first_window_auc,
                "last_window_auc_pr": last_window_auc,
                "total_degradation": total_degradation,
                "degradation_per_month": degradation_per_month,
                "num_windows": len(results),
            })
            
            logger.info(f"\n{'='*60}")
            logger.info("STATIC MODEL DEGRADATION ANALYSIS")
            logger.info(f"{'='*60}")
            logger.info(f"First window AUC-PR:  {first_window_auc:.4f}")
            logger.info(f"Last window AUC-PR:   {last_window_auc:.4f}")
            logger.info(f"Total degradation:    {total_degradation:+.4f}")
            logger.info(f"Degradation/month:    {degradation_per_month:+.4f}")
            logger.info(f"Mean AUC-PR:          {mean_auc_pr:.4f} ± {std_auc_pr:.4f}")
            logger.info(f"Total windows:        {len(results)}")
            logger.info(f"Total days evaluated: {total_days}")
        
        training_time = time.time() - training_start_time
        mlflow.log_metric("total_training_time_seconds", training_time)
        
        return {
            "run_id": parent_run.info.run_id,
            "results": results,
            "mean_auc_pr": mean_auc_pr if results else 0,
            "degradation_per_month": degradation_per_month if results else 0,
            "num_windows": len(results),
        }


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 4: Concept Drift Evaluation
    
    Run with: python -m src.experiments.exp4_concept_drift experiment_name=concept-drift-rq3 features=production
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 4: CONCEPT DRIFT EVALUATION (RQ3)")
    logger.info("="*60)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Feature categories: {cfg.features.categories}")
    logger.info("")
    logger.info("This will train a STATIC model (no retraining) and evaluate")
    logger.info("on all future windows to measure concept drift impact.")
    logger.info("="*60)
    
    # Load and prepare data
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(), config=None)
    
    # Run static model evaluation
    result = evaluate_static_model(
        df=df,
        config=cfg,
        max_windows=cfg.model.training.get("max_windows", None)
    )
    
    logger.info(f"\nStatic model evaluation complete!")
    logger.info(f"  Run ID: {result['run_id']}")
    logger.info(f"  Mean AUC-PR: {result['mean_auc_pr']:.4f}")
    logger.info(f"  Degradation/month: {result['degradation_per_month']:+.4f}")
    logger.info(f"  Windows evaluated: {result['num_windows']}")
    
    logger.info("\nNext: Compare with accumulating window results in MLflow")
    
    return result


if __name__ == "__main__":
    main()


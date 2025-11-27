"""
Experiment 11: Hyperparameter Optimization

Tunes XGBoost hyperparameters using Optuna to maximize a composite objective:
    0.7 * AUC-PR + 0.3 * P@100

Hypothesis: Default XGBoost parameters are suboptimal for imbalanced fraud detection.

Target: Push AUC-PR from 0.6713 (current best) to 0.70+
"""
import os
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import mlflow
import polars as pl
import xgboost as xgb
from optuna.samplers import TPESampler

from src.utils.mlflow_init import init_mlflow
from src.models.train_baseline import (
    ADVANCED_GRAPH_FEATURE_COLUMNS,
    GRAPH_FEATURE_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
    TIME_WEIGHTED_FEATURE_COLUMNS,
    feature_engineering,
    get_base_features,
    load_data,
    load_graph_features,
)
from src.utils.metrics import calculate_metrics


def objective(trial: optuna.Trial, df: pl.DataFrame, features: list, n_windows: int = 5, initial_window_days: int = 180) -> float:
    """
    Optuna objective function. Evaluates XGBoost with suggested hyperparameters
    using accumulating window cross-validation (production-realistic approach).
    
    Args:
        trial: Optuna trial object
        df: Preprocessed DataFrame with features
        features: List of feature column names
        n_windows: Number of accumulating windows for CV
        initial_window_days: Initial training window size (data accumulates from this point)
    
    Returns:
        Composite score: 0.7 * AUC-PR + 0.3 * P@100
    """
    # Hyperparameter search space (from experiment journal)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
    }
    
    # Sort by time for temporal splitting
    df = df.sort("submission_at")
    
    # Define window parameters (accumulating window approach)
    step_days = 28  # Larger steps for faster optimization (less overlap)
    test_days = 14
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=test_days)
    initial_window = timedelta(days=initial_window_days)
    
    # Start from initial window point
    current_date = start_date + initial_window
    
    # Calculate how many windows we can evaluate
    total_days = (end_date - current_date).days
    max_windows = total_days // step_days
    actual_windows = min(n_windows, max_windows)
    
    auc_pr_scores = []
    p_at_100_scores = []
    
    target = "is_fraud"
    
    windows_evaluated = 0
    while current_date + test_size <= end_date and windows_evaluated < actual_windows:
        train_end = current_date
        test_end = current_date + test_size
        
        # ACCUMULATING WINDOW: Use ALL data from start to train_end
        train_data = df.filter(pl.col("submission_at") < train_end)
        test_data = df.filter(
            (pl.col("submission_at") >= train_end) & 
            (pl.col("submission_at") < test_end)
        )
        
        if len(test_data) < 50 or len(train_data) < 1000:
            current_date += step_size
            continue
        
        X_train = train_data.select(features).to_numpy()
        y_train = train_data.select(target).to_numpy().flatten()
        X_test = test_data.select(features).to_numpy()
        y_test = test_data.select(target).to_numpy().flatten()
        
        # Skip if no fraud in test set
        if sum(y_test) == 0:
            current_date += step_size
            continue
        
        # Compute class weight
        n_neg = len(y_train[y_train == 0])
        n_pos = len(y_train[y_train == 1])
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Train model with trial parameters
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            **params
        )
        
        model.fit(X_train, y_train, verbose=False)
        
        # Predict
        proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, proba)
        
        auc_pr_scores.append(metrics["auc_pr"])
        p_at_100_scores.append(metrics["p@100"])
        
        current_date += step_size
        windows_evaluated += 1
    
    if not auc_pr_scores:
        return 0.0  # No valid windows
    
    # Compute composite score
    mean_auc_pr = np.mean(auc_pr_scores)
    mean_p_at_100 = np.mean(p_at_100_scores)
    
    composite_score = 0.7 * mean_auc_pr + 0.3 * mean_p_at_100
    
    # Log intermediate results
    trial.set_user_attr("mean_auc_pr", mean_auc_pr)
    trial.set_user_attr("mean_p@100", mean_p_at_100)
    trial.set_user_attr("n_windows", windows_evaluated)
    
    # Log to MLflow (nested run for this trial)
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        # Enable XGBoost autologging
        mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
        
        mlflow.log_params(params)
        mlflow.log_metrics({
            "composite_score": composite_score,
            "mean_auc_pr": mean_auc_pr,
            "mean_p_at_100": mean_p_at_100,
            "n_windows": float(windows_evaluated)
        })
        
        # Log trial metadata
        mlflow.set_tag("trial_number", str(trial.number))
        mlflow.set_tag("optuna_trial", "true")
    
    return composite_score


def run_hyperparameter_optimization(
    n_trials: int = 100,
    n_windows: int = 5,
    timeout_minutes: Optional[int] = None,
    study_name: str = "xgboost_fraud_detection",
    results_dir: str = "artifacts/results",
    initial_window_days: int = 180,
) -> dict:
    """
    Run Optuna hyperparameter optimization for XGBoost fraud detection.
    
    Args:
        n_trials: Number of optimization trials
        n_windows: Number of sliding windows for cross-validation per trial
        timeout_minutes: Optional timeout in minutes
        study_name: Name for the Optuna study
        results_dir: Directory to save results
    
    Returns:
        Dictionary with best parameters and results
    """
    print("=" * 70)
    print("EXPERIMENT 11: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Trials: {n_trials}")
    print(f"  CV Windows per trial: {n_windows}")
    print(f"  Window strategy: Accumulating (production-realistic)")
    print(f"  Initial window: {initial_window_days} days")
    print(f"  Objective: 0.7 * AUC-PR + 0.3 * P@100")
    print(f"  Target: AUC-PR > 0.70 (current best: 0.6713)")
    
    # Load and prepare data
    print("\n[1/4] Loading data...")
    df = load_data()
    df = feature_engineering(df, include_graph_features=True)
    
    # Build feature list
    all_graph_cols = (
        GRAPH_FEATURE_COLUMNS + 
        ADVANCED_GRAPH_FEATURE_COLUMNS + 
        TIME_WEIGHTED_FEATURE_COLUMNS + 
        INTERACTION_FEATURE_COLUMNS
    )
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + graph_columns
    
    print(f"  Total features: {len(features)}")
    print(f"  Samples: {len(df):,}")
    print(f"  Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # Create Optuna study
    print("\n[2/4] Creating Optuna study...")
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )
    
    # Run optimization
    print(f"\n[3/4] Running optimization ({n_trials} trials)...")
    print("-" * 70)
    
    timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
    
    # Start parent MLflow run for the study
    init_mlflow()
    mlflow.set_experiment("ppa-fraud-detection")
    with mlflow.start_run(run_name=f"hyperopt_{study_name}", tags={"type": "hyperopt"}):
        mlflow.log_params({
            "n_trials": n_trials,
            "n_windows": n_windows,
            "study_name": study_name,
            "initial_window_days": initial_window_days,
            "window_strategy": "accumulating",
        })
        
        study.optimize(
            lambda trial: objective(trial, df, features, n_windows, initial_window_days),
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=True,
            gc_after_trial=True,
        )
        
        # Log best results to parent run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metrics({
            "best_value": study.best_value,
            "best_auc_pr": study.best_trial.user_attrs.get("mean_auc_pr", 0),
            "best_p_at_100": study.best_trial.user_attrs.get("mean_p@100", 0)
        })
        
        # Extract results while still in run context
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        best_auc_pr = best_trial.user_attrs.get("mean_auc_pr", 0)
        best_p_at_100 = best_trial.user_attrs.get("mean_p@100", 0)
        
        # Save results
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best parameters
        params_path = os.path.join(results_dir, "best_hyperparams.pkl")
        with open(params_path, "wb") as f:
            pickle.dump({
                "best_params": best_params,
                "best_score": best_score,
                "best_auc_pr": best_auc_pr,
                "best_p_at_100": best_p_at_100,
                "baseline_auc_pr": 0.6713,
                "improvement_pct": ((best_auc_pr - 0.6713) / 0.6713) * 100,
                "n_trials": n_trials,
                "n_windows": n_windows,
            }, f)
        
        # Save all trial results
        trials_df = pl.DataFrame([
            {
                "trial_number": t.number,
                "composite_score": t.value,
                "mean_auc_pr": t.user_attrs.get("mean_auc_pr", 0),
                "mean_p@100": t.user_attrs.get("mean_p@100", 0),
                "n_windows": t.user_attrs.get("n_windows", 0),
                **t.params,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        
        trials_path = os.path.join(results_dir, "hyperopt_trials.csv")
        trials_df.write_csv(trials_path)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(params_path, artifact_path="results")
        mlflow.log_artifact(trials_path, artifact_path="results")
        
        # Log study summary as artifact
        import json
        import tempfile
        study_summary = {
            "n_trials": len(study.trials),
            "n_complete_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(study_summary, f, indent=2, default=str)
            mlflow.log_artifact(f.name, artifact_path="study_summary")
            os.unlink(f.name)
        
        # Store parent run ID for model registration
        parent_run_id = mlflow.active_run().info.run_id
    
    # Extract results for return (outside run context)
    print("\n" + "-" * 70)
    print("\n[4/4] Results")
    print("=" * 70)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value
    best_auc_pr = best_trial.user_attrs.get("mean_auc_pr", 0)
    best_p_at_100 = best_trial.user_attrs.get("mean_p@100", 0)
    
    print(f"\n✅ BEST TRIAL: #{best_trial.number}")
    print(f"   Composite Score: {best_score:.4f}")
    print(f"   Mean AUC-PR: {best_auc_pr:.4f}")
    print(f"   Mean P@100: {best_p_at_100:.4f}")
    
    print("\n📊 BEST HYPERPARAMETERS:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    # Compare with baseline
    baseline_auc_pr = 0.6713
    improvement = ((best_auc_pr - baseline_auc_pr) / baseline_auc_pr) * 100
    
    print(f"\n📈 IMPROVEMENT OVER BASELINE:")
    print(f"   Baseline AUC-PR: {baseline_auc_pr:.4f}")
    print(f"   Optimized AUC-PR: {best_auc_pr:.4f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    if best_auc_pr >= 0.70:
        print("\n🎯 TARGET ACHIEVED: AUC-PR >= 0.70!")
    else:
        gap = 0.70 - best_auc_pr
        print(f"\n⚠️ Gap to 0.70 target: {gap:.4f}")
    
    print(f"\n💾 Saved best params to: {params_path}")
    print(f"💾 Saved trial history to: {trials_path}")
    
    # Register best model from best trial
    try:
        # Find the best trial's run ID
        best_trial_run_id = None
        # The best trial was logged as a nested run, we need to find it
        # For now, we'll register a model after validation instead
        print("\n📝 Note: Model will be registered after validation completes")
    except Exception as e:
        print(f"Warning: Could not register model from best trial: {e}")
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_auc_pr": best_auc_pr,
        "best_p_at_100": best_p_at_100,
        "improvement_pct": improvement,
        "study": study,
        "parent_run_id": parent_run_id if 'parent_run_id' in locals() else None,
    }


def validate_best_params(best_params: dict, full_evaluation: bool = True, mlflow_run_id: Optional[str] = None, initial_window_days: int = 180) -> dict:
    """
    Validate the best parameters using accumulating window evaluation
    (same as train_baseline.py for fair comparison).
    
    Args:
        best_params: Best hyperparameters from optimization
        full_evaluation: Whether to run full evaluation (vs quick validation)
        mlflow_run_id: Optional MLflow run ID to log validation results to
        initial_window_days: Initial training window size for accumulating windows
    
    Returns:
        Dictionary with validation metrics
    """
    print("\n" + "=" * 70)
    print("VALIDATING BEST HYPERPARAMETERS")
    print("=" * 70)
    print(f"Window strategy: Accumulating (initial: {initial_window_days} days)")
    
    # Start MLflow run for validation
    init_mlflow()
    mlflow.set_experiment("ppa-fraud-detection")
    with mlflow.start_run(run_name="hyperopt_validation", tags={"type": "validation", "experiment": "hyperopt", "window_strategy": "accumulating"}):
        # Load data
        df = load_data()
        df = feature_engineering(df, include_graph_features=True)
        
        # Build feature list
        all_graph_cols = (
            GRAPH_FEATURE_COLUMNS + 
            ADVANCED_GRAPH_FEATURE_COLUMNS + 
            TIME_WEIGHTED_FEATURE_COLUMNS + 
            INTERACTION_FEATURE_COLUMNS
        )
        graph_columns = [col for col in all_graph_cols if col in df.columns]
        features = get_base_features() + graph_columns
        
        # Sort by time
        df = df.sort("submission_at")
        
        # Window parameters (accumulating window, same as train_baseline.py)
        step_days = 7 if full_evaluation else 28
        test_days = 14
        
        start_date = df["submission_at"].min()
        end_date = df["submission_at"].max()
        
        step_size = timedelta(days=step_days)
        test_size = timedelta(days=test_days)
        initial_window = timedelta(days=initial_window_days)
        
        current_date = start_date + initial_window
        
        mlflow.log_params({
            "initial_window_days": initial_window_days,
            "step_days": step_days,
            "window_strategy": "accumulating",
        })
        
        results = []
        target = "is_fraud"
        best_auc_pr = 0
        best_model = None
        best_run_id = None
        window_idx = 0
        
        print(f"\nRunning accumulating window validation (step={step_days} days)...")
        
        while current_date + test_size <= end_date:
            train_end = current_date
            test_end = current_date + test_size
            
            # ACCUMULATING WINDOW: Use ALL data from start to train_end
            train_data = df.filter(pl.col("submission_at") < train_end)
            test_data = df.filter(
                (pl.col("submission_at") >= train_end) & 
                (pl.col("submission_at") < test_end)
            )
            
            if len(test_data) < 50 or len(train_data) < 1000:
                current_date += step_size
                continue
            
            X_train = train_data.select(features).to_numpy()
            y_train = train_data.select(target).to_numpy().flatten()
            X_test = test_data.select(features).to_numpy()
            y_test = test_data.select(target).to_numpy().flatten()
            
            if sum(y_test) == 0:
                current_date += step_size
                continue
            
            # Nested run for this validation window
            with mlflow.start_run(run_name=f"validation_window_{window_idx}", nested=True):
                # Enable XGBoost autologging
                mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
                
                # Log window info
                mlflow.log_params({
                    "window_index": window_idx,
                    "window_start": str(start_date.date()),
                    "window_end": str(train_end.date()),
                    "test_start": str(train_end.date()),
                    "test_end": str(test_end.date()),
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "fraud_rate_train": float(y_train.mean()),
                    "fraud_rate_test": float(y_test.mean()),
                })
                
                # Compute class weight
                n_neg = len(y_train[y_train == 0])
                n_pos = len(y_train[y_train == 1])
                scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
                
                # Train with optimized params
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    random_state=42,
                    **best_params
                )
                
                model.fit(X_train, y_train, verbose=False)
                
                # Predict
                proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                metrics = calculate_metrics(y_test, proba)
                
                # Log metrics
                mlflow.log_metrics({
                    "auc_pr": metrics["auc_pr"],
                    "auc_roc": metrics["auc_roc"],
                    "p_at_100": metrics["p@100"],
                    "lift_at_100": metrics["lift@100"],
                })
                
                # Track best model
                if metrics["auc_pr"] > best_auc_pr:
                    best_auc_pr = metrics["auc_pr"]
                    best_model = model
                    best_run_id = mlflow.active_run().info.run_id
                
                results.append({
                    "window_start": train_end,
                    **metrics
                })
            
            window_idx += 1
            current_date += step_size
        
        # Compute aggregates
        results_df = pl.DataFrame(results)
        
        mean_auc_pr = float(results_df["auc_pr"].mean())
        mean_auc_roc = float(results_df["auc_roc"].mean())
        mean_p_at_100 = float(results_df["p@100"].mean())
        mean_lift = float(results_df["lift@100"].mean())
        
        print(f"\n✅ VALIDATION RESULTS ({len(results)} windows):")
        print(f"   Mean AUC-PR: {mean_auc_pr:.4f}")
        print(f"   Mean AUC-ROC: {mean_auc_roc:.4f}")
        print(f"   Mean P@100: {mean_p_at_100:.4f}")
        print(f"   Mean Lift@100: {mean_lift:.2f}")
        
        # Save validation results
        os.makedirs("artifacts/results", exist_ok=True)
        results_path = "artifacts/results/hyperopt_validation_results.csv"
        results_df.write_csv(results_path)
        print(f"\n💾 Saved validation results to: {results_path}")
        
        # Log validation results to MLflow
        mlflow.log_artifact(results_path, artifact_path="validation")
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "mean_auc_pr": mean_auc_pr,
            "mean_auc_roc": mean_auc_roc,
            "mean_p_at_100": mean_p_at_100,
            "mean_lift_at_100": mean_lift,
            "best_auc_pr": best_auc_pr,
            "n_windows": len(results),
        })
        
        # Register best model to Model Registry
        if best_model is not None and best_run_id is not None:
            print(f"\n📦 Registering best model (AUC-PR={best_auc_pr:.4f}) to Model Registry...")
            try:
                model_uri = f"runs:/{best_run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri=model_uri,
                    name="fraud-detection-optimized-xgboost"
                )
                
                print(f"✓ Registered as: {registered_model.name} (version {registered_model.version})")
                
                # Add description to model version
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                client.update_model_version(
                    name=registered_model.name,
                    version=registered_model.version,
                    description=f"Hyperparameter-optimized XGBoost model. Mean AUC-PR: {mean_auc_pr:.4f}, Best: {best_auc_pr:.4f}. Accumulating window training."
                )
                
                mlflow.log_param("registered_model_version", registered_model.version)
                mlflow.log_param("registered_model_name", registered_model.name)
                
            except Exception as e:
                print(f"Warning: Model registration failed: {e}")
    
    return {
        "mean_auc_pr": mean_auc_pr,
        "mean_auc_roc": mean_auc_roc,
        "mean_p@100": mean_p_at_100,
        "mean_lift@100": mean_lift,
        "n_windows": len(results),
        "results_df": results_df,
    }


def main(
    n_trials: int = 100,
    n_windows: int = 5,
    timeout_minutes: Optional[int] = None,
    validate: bool = True,
    initial_window_days: int = 180,
):
    """
    Main entry point for hyperparameter optimization.
    
    Args:
        n_trials: Number of Optuna trials
        n_windows: Number of CV windows per trial
        timeout_minutes: Optional timeout
        validate: Whether to run full validation after optimization
        initial_window_days: Initial training window size for accumulating windows
    """
    # Run optimization
    result = run_hyperparameter_optimization(
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes,
        initial_window_days=initial_window_days,
    )
    
    # Validate best params
    if validate:
        validation_result = validate_best_params(
            result["best_params"],
            initial_window_days=initial_window_days
        )
        
        # Final comparison
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        
        baseline = 0.6713
        optimized = validation_result["mean_auc_pr"]
        improvement = ((optimized - baseline) / baseline) * 100
        
        print(f"\n{'Model':<30} {'AUC-PR':>12} {'P@100':>12}")
        print("-" * 54)
        print(f"{'Baseline (time-weighted)':<30} {baseline:>12.4f} {'0.7710':>12}")
        print(f"{'Optimized XGBoost':<30} {optimized:>12.4f} {validation_result['mean_p@100']:>12.4f}")
        print("-" * 54)
        print(f"{'Improvement':<30} {improvement:>+11.2f}%")
        
        if optimized >= 0.70:
            print("\n🎯 SUCCESS: Achieved AUC-PR >= 0.70!")
        elif improvement >= 1.0:
            print(f"\n✅ SUCCESS: Achieved >{1.0}% improvement")
        else:
            print(f"\n⚠️ Marginal improvement ({improvement:.2f}%)")
    
    return result


if __name__ == "__main__":
    main()


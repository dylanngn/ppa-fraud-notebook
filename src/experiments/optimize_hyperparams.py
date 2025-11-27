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


def objective(trial: optuna.Trial, df: pl.DataFrame, features: list, n_windows: int = 5) -> float:
    """
    Optuna objective function. Evaluates XGBoost with suggested hyperparameters
    using sliding window cross-validation.
    
    Args:
        trial: Optuna trial object
        df: Preprocessed DataFrame with features
        features: List of feature column names
        n_windows: Number of sliding windows for CV
    
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
    
    # Define window parameters
    window_days = 90
    step_days = 28  # Larger steps for faster optimization (less overlap)
    test_days = 14
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=test_days)
    
    # Start from a point that gives us n_windows evaluations
    total_days = (end_date - start_date).days
    max_windows = (total_days - window_days - test_days) // step_days
    actual_windows = min(n_windows, max_windows)
    
    # Start from the end to use the most recent (and likely most relevant) data
    # Skip first window to ensure we have enough training data
    skip_windows = max(0, max_windows - actual_windows)
    current_date = start_date + window_size + (skip_windows * step_size)
    
    auc_pr_scores = []
    p_at_100_scores = []
    
    target = "is_fraud"
    
    windows_evaluated = 0
    while current_date + test_size <= end_date and windows_evaluated < actual_windows:
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
        
        if len(test_data) < 50 or len(train_data) < 100:
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
        mlflow.log_params(params)
        mlflow.log_metrics({
            "composite_score": composite_score,
            "mean_auc_pr": mean_auc_pr,
            "mean_p_at_100": mean_p_at_100,
            "n_windows": float(windows_evaluated)
        })
    
    return composite_score


def run_hyperparameter_optimization(
    n_trials: int = 100,
    n_windows: int = 5,
    timeout_minutes: Optional[int] = None,
    study_name: str = "xgboost_fraud_detection",
    results_dir: str = "artifacts/results",
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
    mlflow.set_experiment("ppa-fraud-detection")
    with mlflow.start_run(run_name=f"hyperopt_{study_name}", tags={"type": "hyperopt"}):
        mlflow.log_params({
            "n_trials": n_trials,
            "n_windows": n_windows,
            "study_name": study_name
        })
        
        study.optimize(
            lambda trial: objective(trial, df, features, n_windows),
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
    
    # Extract results
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
            "baseline_auc_pr": baseline_auc_pr,
            "improvement_pct": improvement,
            "n_trials": n_trials,
            "n_windows": n_windows,
        }, f)
    print(f"\n💾 Saved best params to: {params_path}")
    
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
    print(f"💾 Saved trial history to: {trials_path}")
    
    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_auc_pr": best_auc_pr,
        "best_p_at_100": best_p_at_100,
        "improvement_pct": improvement,
        "study": study,
    }


def validate_best_params(best_params: dict, full_evaluation: bool = True) -> dict:
    """
    Validate the best parameters using full sliding window evaluation
    (same as train_baseline.py for fair comparison).
    
    Args:
        best_params: Best hyperparameters from optimization
        full_evaluation: Whether to run full sliding window (vs quick validation)
    
    Returns:
        Dictionary with validation metrics
    """
    print("\n" + "=" * 70)
    print("VALIDATING BEST HYPERPARAMETERS")
    print("=" * 70)
    
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
    
    # Window parameters (same as train_baseline.py)
    window_days = 90
    step_days = 7 if full_evaluation else 28
    test_days = 14
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=test_days)
    
    current_date = start_date + window_size
    
    results = []
    target = "is_fraud"
    
    print(f"\nRunning sliding window validation (step={step_days} days)...")
    
    while current_date + test_size <= end_date:
        train_end = current_date
        test_end = current_date + test_size
        
        # Split
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
        y_train = train_data.select(target).to_numpy().flatten()
        X_test = test_data.select(features).to_numpy()
        y_test = test_data.select(target).to_numpy().flatten()
        
        if sum(y_test) == 0:
            current_date += step_size
            continue
        
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
        
        results.append({
            "window_start": train_end,
            **metrics
        })
        
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
    results_df.write_csv("artifacts/results/hyperopt_validation_results.csv")
    print(f"\n💾 Saved validation results to: artifacts/results/hyperopt_validation_results.csv")
    
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
):
    """
    Main entry point for hyperparameter optimization.
    
    Args:
        n_trials: Number of Optuna trials
        n_windows: Number of CV windows per trial
        timeout_minutes: Optional timeout
        validate: Whether to run full validation after optimization
    """
    # Run optimization
    result = run_hyperparameter_optimization(
        n_trials=n_trials,
        n_windows=n_windows,
        timeout_minutes=timeout_minutes,
    )
    
    # Validate best params
    if validate:
        validation_result = validate_best_params(result["best_params"])
        
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


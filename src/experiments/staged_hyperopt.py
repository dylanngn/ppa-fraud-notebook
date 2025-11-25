"""
Experiment 11b: Staged Hyperparameter Optimization

A structured approach to hyperparameter tuning:
1. Stage 1: Parameter sensitivity analysis (which params matter?)
2. Stage 2: Test promising combinations from domain knowledge
3. Stage 3: Compose final combination from winners

More interpretable and efficient than blind TPE search.
"""
import os
import pickle
from datetime import timedelta
from typing import Optional

import numpy as np
import polars as pl
import xgboost as xgb

from src.models.train_baseline import (
    ADVANCED_GRAPH_FEATURE_COLUMNS,
    GRAPH_FEATURE_COLUMNS,
    INTERACTION_FEATURE_COLUMNS,
    TIME_WEIGHTED_FEATURE_COLUMNS,
    feature_engineering,
    get_base_features,
    load_data,
)
from src.utils.metrics import calculate_metrics


# =============================================================================
# BASELINE PARAMETERS (current defaults)
# =============================================================================
BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# =============================================================================
# STAGE 1: PARAMETER SENSITIVITY CANDIDATES
# Test each parameter independently to find which ones matter
# =============================================================================
SENSITIVITY_TESTS = {
    # More trees + lower LR (classic ensemble improvement)
    "more_trees_low_lr": {"n_estimators": 300, "learning_rate": 0.05},
    "many_trees_very_low_lr": {"n_estimators": 500, "learning_rate": 0.02},
    
    # Depth variations (deeper for complex patterns, shallower for regularization)
    "deeper_trees": {"max_depth": 8},
    "shallower_trees": {"max_depth": 4},
    
    # Regularization (critical for imbalanced data)
    "high_gamma": {"gamma": 0.5},
    "high_reg_alpha": {"reg_alpha": 1.0},
    "high_reg_lambda": {"reg_lambda": 5.0},
    "combined_regularization": {"gamma": 0.3, "reg_alpha": 0.5, "reg_lambda": 3.0},
    
    # Subsampling (prevents overfitting, good for sparse data)
    "subsample_80": {"subsample": 0.8},
    "colsample_80": {"colsample_bytree": 0.8},
    "both_subsample_80": {"subsample": 0.8, "colsample_bytree": 0.8},
    
    # Min child weight (higher = more conservative, good for noise)
    "min_child_5": {"min_child_weight": 5},
    "min_child_10": {"min_child_weight": 10},
}

# =============================================================================
# STAGE 2: PROMISING COMBINATIONS
# Based on XGBoost best practices for fraud detection
# =============================================================================
PROMISING_COMBINATIONS = {
    # Combo A: Conservative (high regularization, prevent overfitting)
    "conservative": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.3,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
    },
    
    # Combo B: Aggressive (more capacity, less regularization)
    "aggressive": {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    
    # Combo C: Balanced (middle ground)
    "balanced": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.2,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
    },
    
    # Combo D: High capacity with strong regularization
    "high_capacity_regularized": {
        "n_estimators": 500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "gamma": 0.4,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
    },
    
    # Combo E: Fraud-specific (based on imbalanced learning literature)
    "fraud_optimized": {
        "n_estimators": 350,
        "max_depth": 6,
        "learning_rate": 0.04,
        "min_child_weight": 8,  # Higher - fraud is minority
        "subsample": 0.75,  # More randomness
        "colsample_bytree": 0.75,
        "gamma": 0.5,  # Prevent splits on noise
        "reg_alpha": 0.8,
        "reg_lambda": 4.0,
    },
    
    # Combo F: XGBoost competition winner style
    "competition_style": {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
    },
}


def evaluate_params(
    params: dict,
    df: pl.DataFrame,
    features: list,
    n_windows: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a parameter set using sliding window CV.
    
    Returns dict with auc_pr, p@100, and composite score.
    """
    # Merge with baseline
    full_params = {**BASELINE_PARAMS, **params}
    
    df = df.sort("submission_at")
    
    window_days = 90
    step_days = 28  # Larger steps for speed
    test_days = 14
    
    start_date = df["submission_at"].min()
    end_date = df["submission_at"].max()
    
    window_size = timedelta(days=window_days)
    step_size = timedelta(days=step_days)
    test_size = timedelta(days=test_days)
    
    # Start from end for most relevant data
    total_days = (end_date - start_date).days
    max_windows = (total_days - window_days - test_days) // step_days
    actual_windows = min(n_windows, max_windows)
    skip_windows = max(0, max_windows - actual_windows)
    current_date = start_date + window_size + (skip_windows * step_size)
    
    auc_pr_scores = []
    p_at_100_scores = []
    
    target = "is_fraud"
    windows_evaluated = 0
    
    while current_date + test_size <= end_date and windows_evaluated < actual_windows:
        train_end = current_date
        test_end = current_date + test_size
        
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
        
        if sum(y_test) == 0:
            current_date += step_size
            continue
        
        n_neg = len(y_train[y_train == 0])
        n_pos = len(y_train[y_train == 1])
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
            **full_params
        )
        
        model.fit(X_train, y_train, verbose=False)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, proba)
        
        auc_pr_scores.append(metrics["auc_pr"])
        p_at_100_scores.append(metrics["p@100"])
        
        current_date += step_size
        windows_evaluated += 1
    
    if not auc_pr_scores:
        return {"auc_pr": 0.0, "p@100": 0.0, "composite": 0.0, "n_windows": 0}
    
    mean_auc_pr = np.mean(auc_pr_scores)
    mean_p_at_100 = np.mean(p_at_100_scores)
    composite = 0.7 * mean_auc_pr + 0.3 * mean_p_at_100
    
    return {
        "auc_pr": mean_auc_pr,
        "p@100": mean_p_at_100,
        "composite": composite,
        "n_windows": windows_evaluated,
    }


def run_staged_optimization(n_windows: int = 5) -> dict:
    """
    Run the three-stage optimization process.
    """
    print("=" * 70)
    print("EXPERIMENT 11b: STAGED HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    print("\n[Setup] Loading data...")
    df = load_data()
    df = feature_engineering(df, include_graph_features=True)
    
    all_graph_cols = (
        GRAPH_FEATURE_COLUMNS + 
        ADVANCED_GRAPH_FEATURE_COLUMNS + 
        TIME_WEIGHTED_FEATURE_COLUMNS + 
        INTERACTION_FEATURE_COLUMNS
    )
    graph_columns = [col for col in all_graph_cols if col in df.columns]
    features = get_base_features() + graph_columns
    
    print(f"  Features: {len(features)}")
    print(f"  Samples: {len(df):,}")
    
    all_results = {}
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE (current defaults)")
    print("=" * 70)
    
    baseline_result = evaluate_params({}, df, features, n_windows)
    all_results["baseline"] = baseline_result
    
    print(f"  AUC-PR: {baseline_result['auc_pr']:.4f}")
    print(f"  P@100:  {baseline_result['p@100']:.4f}")
    print(f"  Composite: {baseline_result['composite']:.4f}")
    
    # =========================================================================
    # STAGE 1: SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Testing individual parameter variations...")
    
    stage1_results = {}
    
    for name, params in SENSITIVITY_TESTS.items():
        print(f"\n  Testing: {name}")
        print(f"    Params: {params}")
        
        result = evaluate_params(params, df, features, n_windows)
        stage1_results[name] = result
        all_results[f"s1_{name}"] = result
        
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        indicator = "✅" if delta > 0 else "❌" if delta < 0 else "➖"
        
        print(f"    AUC-PR: {result['auc_pr']:.4f} ({delta:+.4f}) {indicator}")
        print(f"    P@100:  {result['p@100']:.4f}")
    
    # Identify top performers from Stage 1
    sorted_s1 = sorted(stage1_results.items(), key=lambda x: x[1]["auc_pr"], reverse=True)
    
    print("\n" + "-" * 50)
    print("STAGE 1 WINNERS (Top 5):")
    for i, (name, result) in enumerate(sorted_s1[:5], 1):
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        print(f"  {i}. {name}: {result['auc_pr']:.4f} ({delta:+.4f})")
    
    # =========================================================================
    # STAGE 2: PROMISING COMBINATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: TESTING PROMISING COMBINATIONS")
    print("=" * 70)
    
    stage2_results = {}
    
    for name, params in PROMISING_COMBINATIONS.items():
        print(f"\n  Testing: {name}")
        
        result = evaluate_params(params, df, features, n_windows)
        stage2_results[name] = result
        all_results[f"s2_{name}"] = result
        
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        indicator = "✅" if delta > 0.005 else "➖" if delta > -0.005 else "❌"
        
        print(f"    AUC-PR: {result['auc_pr']:.4f} ({delta:+.4f}) {indicator}")
        print(f"    P@100:  {result['p@100']:.4f}")
        print(f"    Composite: {result['composite']:.4f}")
    
    # Identify best from Stage 2
    sorted_s2 = sorted(stage2_results.items(), key=lambda x: x[1]["auc_pr"], reverse=True)
    best_s2_name, best_s2_result = sorted_s2[0]
    
    print("\n" + "-" * 50)
    print("STAGE 2 RANKING:")
    for i, (name, result) in enumerate(sorted_s2, 1):
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        print(f"  {i}. {name}: {result['auc_pr']:.4f} ({delta:+.4f})")
    
    # =========================================================================
    # STAGE 3: COMPOSE FINAL COMBINATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: COMPOSING FINAL COMBINATION")
    print("=" * 70)
    
    # Start with best Stage 2 combination
    print(f"\nStarting from best S2 combo: {best_s2_name}")
    best_base_params = PROMISING_COMBINATIONS[best_s2_name].copy()
    
    # Identify which individual tweaks helped in Stage 1
    helpful_tweaks = [
        (name, result, SENSITIVITY_TESTS[name])
        for name, result in sorted_s1 
        if result["auc_pr"] > baseline_result["auc_pr"]
    ]
    
    print(f"\nHelpful individual tweaks from S1: {len(helpful_tweaks)}")
    for name, result, params in helpful_tweaks[:5]:
        print(f"  - {name}: {params}")
    
    # Create hybrid combinations
    stage3_candidates = {}
    
    # Candidate 1: Best S2 as-is
    stage3_candidates[f"best_s2_{best_s2_name}"] = best_base_params.copy()
    
    # Candidate 2: Best S2 + more trees
    if "more_trees_low_lr" in [h[0] for h in helpful_tweaks]:
        c2 = best_base_params.copy()
        c2["n_estimators"] = 400
        c2["learning_rate"] = 0.03
        stage3_candidates["best_s2_more_trees"] = c2
    
    # Candidate 3: Best S2 + stronger regularization
    c3 = best_base_params.copy()
    c3["gamma"] = max(c3.get("gamma", 0), 0.4)
    c3["reg_lambda"] = max(c3.get("reg_lambda", 1), 4.0)
    stage3_candidates["best_s2_strong_reg"] = c3
    
    # Candidate 4: Best S2 + higher min_child_weight
    c4 = best_base_params.copy()
    c4["min_child_weight"] = 8
    stage3_candidates["best_s2_high_mcw"] = c4
    
    # Candidate 5: Ensemble of best tweaks
    ensemble_params = BASELINE_PARAMS.copy()
    # Add winning elements
    ensemble_params["n_estimators"] = 350
    ensemble_params["learning_rate"] = 0.04
    ensemble_params["max_depth"] = 6
    ensemble_params["min_child_weight"] = 5
    ensemble_params["subsample"] = 0.8
    ensemble_params["colsample_bytree"] = 0.8
    ensemble_params["gamma"] = 0.3
    ensemble_params["reg_alpha"] = 0.5
    ensemble_params["reg_lambda"] = 3.0
    stage3_candidates["ensemble_best"] = ensemble_params
    
    # Candidate 6: Ultra-regularized for fraud
    ultra_reg = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 0.6,
        "reg_alpha": 1.5,
        "reg_lambda": 6.0,
    }
    stage3_candidates["ultra_regularized"] = ultra_reg
    
    print(f"\nTesting {len(stage3_candidates)} composed candidates...")
    
    stage3_results = {}
    
    for name, params in stage3_candidates.items():
        print(f"\n  Testing: {name}")
        
        result = evaluate_params(params, df, features, n_windows)
        stage3_results[name] = {"result": result, "params": params}
        all_results[f"s3_{name}"] = result
        
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        indicator = "🏆" if delta > 0.01 else "✅" if delta > 0 else "❌"
        
        print(f"    AUC-PR: {result['auc_pr']:.4f} ({delta:+.4f}) {indicator}")
        print(f"    P@100:  {result['p@100']:.4f}")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Find overall best
    all_sorted = sorted(
        [(k, v) for k, v in all_results.items()],
        key=lambda x: x[1]["auc_pr"],
        reverse=True
    )
    
    best_name, best_result = all_sorted[0]
    
    print(f"\n🏆 BEST CONFIGURATION: {best_name}")
    print(f"   AUC-PR: {best_result['auc_pr']:.4f}")
    print(f"   P@100:  {best_result['p@100']:.4f}")
    print(f"   Composite: {best_result['composite']:.4f}")
    
    improvement = ((best_result["auc_pr"] - baseline_result["auc_pr"]) / baseline_result["auc_pr"]) * 100
    print(f"\n   Improvement over baseline: {improvement:+.2f}%")
    
    # Get best params
    if best_name.startswith("s3_"):
        best_params = stage3_candidates[best_name[3:]].copy()
    elif best_name.startswith("s2_"):
        best_params = PROMISING_COMBINATIONS[best_name[3:]].copy()
    elif best_name.startswith("s1_"):
        best_params = {**BASELINE_PARAMS, **SENSITIVITY_TESTS[best_name[3:]]}
    else:
        best_params = BASELINE_PARAMS.copy()
    
    print(f"\n📊 BEST PARAMETERS:")
    for param, value in sorted(best_params.items()):
        if isinstance(value, float):
            print(f"   {param}: {value:.4f}")
        else:
            print(f"   {param}: {value}")
    
    # Summary table
    print("\n" + "-" * 70)
    print("TOP 10 CONFIGURATIONS:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Configuration':<35} {'AUC-PR':>10} {'Delta':>10}")
    print("-" * 70)
    
    for i, (name, result) in enumerate(all_sorted[:10], 1):
        delta = result["auc_pr"] - baseline_result["auc_pr"]
        print(f"{i:<5} {name:<35} {result['auc_pr']:>10.4f} {delta:>+10.4f}")
    
    # Save results
    os.makedirs("artifacts/results", exist_ok=True)
    
    with open("artifacts/results/staged_hyperopt_results.pkl", "wb") as f:
        pickle.dump({
            "best_params": best_params,
            "best_result": best_result,
            "all_results": all_results,
            "stage1_results": stage1_results,
            "stage2_results": stage2_results,
            "stage3_results": stage3_results,
            "baseline_result": baseline_result,
        }, f)
    
    # Save as CSV for easy viewing
    results_list = [
        {"config": name, **result}
        for name, result in all_results.items()
    ]
    pl.DataFrame(results_list).sort("auc_pr", descending=True).write_csv(
        "artifacts/results/staged_hyperopt_summary.csv"
    )
    
    print(f"\n💾 Results saved to artifacts/results/staged_hyperopt_*.pkl/csv")
    
    # Compare to target
    current_best = 0.6713
    target = 0.70
    
    print("\n" + "=" * 70)
    print("COMPARISON TO TARGETS")
    print("=" * 70)
    print(f"  Previous best (time-weighted): {current_best:.4f}")
    print(f"  New best:                      {best_result['auc_pr']:.4f}")
    print(f"  Target:                        {target:.4f}")
    
    if best_result["auc_pr"] >= target:
        print(f"\n  🎯 TARGET ACHIEVED!")
    else:
        gap = target - best_result["auc_pr"]
        print(f"\n  ⚠️ Gap to target: {gap:.4f}")
    
    return {
        "best_params": best_params,
        "best_result": best_result,
        "all_results": all_results,
        "improvement_pct": improvement,
    }


def main(n_windows: int = 5):
    """Entry point for staged optimization."""
    return run_staged_optimization(n_windows=n_windows)


if __name__ == "__main__":
    main()


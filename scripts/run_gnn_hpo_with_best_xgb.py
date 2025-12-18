#!/usr/bin/env python
"""
Run GNN HPO with best XGBoost params from previous HPO.

Usage:
    python scripts/run_gnn_hpo_with_best_xgb.py
    python scripts/run_gnn_hpo_with_best_xgb.py --trials 10  # Quick test
"""

import subprocess
import sqlite3
import argparse
import sys


def get_best_xgb_params(db_path: str = "ppa-fraud-detection-mlflow.db") -> dict:
    """Query MLflow for best XGBoost params."""
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT 
        MAX(CASE WHEN m.key = 'auc_pr' THEN m.value END) as auc_pr,
        MAX(CASE WHEN p.key = 'model.n_estimators' THEN p.value END) as n_estimators,
        MAX(CASE WHEN p.key = 'model.max_depth' THEN p.value END) as max_depth,
        MAX(CASE WHEN p.key = 'model.learning_rate' THEN p.value END) as learning_rate,
        MAX(CASE WHEN p.key = 'model.min_child_weight' THEN p.value END) as min_child_weight,
        MAX(CASE WHEN p.key = 'model.subsample' THEN p.value END) as subsample,
        MAX(CASE WHEN p.key = 'model.colsample_bytree' THEN p.value END) as colsample_bytree,
        MAX(CASE WHEN p.key = 'model.gamma' THEN CAST(p.value AS REAL) END) as gamma,
        MAX(CASE WHEN p.key = 'model.reg_alpha' THEN CAST(p.value AS REAL) END) as reg_alpha,
        MAX(CASE WHEN p.key = 'model.reg_lambda' THEN CAST(p.value AS REAL) END) as reg_lambda,
        MAX(CASE WHEN p.key = 'model.scale_pos_weight' THEN CAST(p.value AS REAL) END) as scale_pos_weight
    FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    LEFT JOIN metrics m ON r.run_uuid = m.run_uuid
    LEFT JOIN params p ON r.run_uuid = p.run_uuid
    WHERE e.name IN ('HPO_XGBoost', 'HPO_HighRecall')
    GROUP BY r.run_uuid
    HAVING auc_pr IS NOT NULL
    ORDER BY auc_pr DESC
    LIMIT 1
    '''
    
    result = conn.cursor().execute(query).fetchone()
    conn.close()
    
    if not result:
        return None
    
    return {
        "auc_pr": result[0],
        "n_estimators": int(float(result[1])) if result[1] else 100,
        "max_depth": int(float(result[2])) if result[2] else 6,
        "learning_rate": float(result[3]) if result[3] else 0.1,
        "min_child_weight": int(float(result[4])) if result[4] else 1,
        "subsample": float(result[5]) if result[5] else 0.8,
        "colsample_bytree": float(result[6]) if result[6] else 0.8,
        "gamma": float(result[7]) if result[7] else 0.0,
        "reg_alpha": float(result[8]) if result[8] else 0.0,
        "reg_lambda": float(result[9]) if result[9] else 1.0,
        "scale_pos_weight": float(result[10]) if result[10] else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run GNN HPO with best XGBoost params")
    parser.add_argument("--trials", type=int, default=20, help="Number of HPO trials")
    parser.add_argument("--train-start", default="2024-12-01")
    parser.add_argument("--train-end", default="2025-06-01")
    parser.add_argument("--test-end", default="2025-07-01")
    args = parser.parse_args()
    
    # Get best XGBoost params
    print("Fetching best XGBoost params from MLflow...")
    params = get_best_xgb_params()
    
    if not params:
        print("ERROR: No XGBoost HPO runs found. Run 'make hpo' first.")
        sys.exit(1)
    
    print(f"Best XGBoost run: AUC-PR = {params['auc_pr']:.4f}")
    print(f"  n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    print(f"  learning_rate={params['learning_rate']:.4f}, scale_pos_weight={params['scale_pos_weight']:.2f}")
    print()
    
    # Build command
    cmd = [
        "python", "-m", "src.training.trainer",
        "--config-name=hpo_gnn_only",
        f"hydra.sweeper.n_trials={args.trials}",
        f"data.train_start_date={args.train_start}",
        f"data.train_end_date={args.train_end}",
        f"data.test_end_date={args.test_end}",
        f"model.xgboost.n_estimators={params['n_estimators']}",
        f"model.xgboost.max_depth={params['max_depth']}",
        f"model.xgboost.learning_rate={params['learning_rate']}",
        f"model.xgboost.min_child_weight={params['min_child_weight']}",
        f"model.xgboost.subsample={params['subsample']}",
        f"model.xgboost.colsample_bytree={params['colsample_bytree']}",
        f"model.xgboost.gamma={params['gamma']}",
        f"model.xgboost.reg_alpha={params['reg_alpha']}",
        f"model.xgboost.reg_lambda={params['reg_lambda']}",
        f"model.xgboost.scale_pos_weight={params['scale_pos_weight']}",
    ]
    
    print("Running GNN HPO with fixed XGBoost params...")
    print(" ".join(cmd[:5]) + " ...")
    print()
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

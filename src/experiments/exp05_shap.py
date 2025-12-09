"""
Experiment 5: SHAP Explainability Analysis (RQ4)

Research Question: How can we derive actionable insights and interpretable
explanations from our fraud detection model using SHAP?

This experiment generates:
1. Global feature importance (SHAP summary plots)
2. Local explanations for individual predictions
3. Feature interaction analysis
4. Case studies for fraud analyst review

Run with: python -m src.experiments.exp5_shap experiment_name=shap-explainability-rq4
"""
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.data.loader import load_data
from src.features.definitions.base import compute_base_features
from src.features.processor import FeatureProcessor
from src.models.utils.common import setup_mlflow
from src.models.xgboost.utils import get_optimal_tree_method, validate_features, get_categorical_features
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def train_model_for_shap(
    df: pl.DataFrame,
    config: DictConfig,
    train_ratio: float = 0.8
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Train an XGBoost model for SHAP analysis.
    
    Args:
        df: DataFrame with features and target
        config: Hydra configuration
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test, feature_names)
    """
    xgb_params = dict(config.model.params)
    target = "is_fraud"
    
    # Sort by time
    df = df.sort("submission_at")
    
    # Time-based split
    split_idx = int(len(df) * train_ratio)
    train_data = df[:split_idx]
    test_data = df[split_idx:]
    
    # Get cutoff date for feature computation
    cutoff_date = train_data["submission_at"].max()
    
    # Compute features using processor
    processor = FeatureProcessor.from_config(config)
    
    logger.info(f"Computing features for training data ({len(train_data)} samples)...")
    train_features = processor.process(train_data, cutoff_date)
    
    logger.info(f"Computing features for test data ({len(test_data)} samples)...")
    test_features = processor.process(test_data, cutoff_date)
    
    # Convert to pandas
    train_pdf = train_features.to_pandas()
    test_pdf = test_features.to_pandas()
    
    # Validate features
    valid_features, _ = validate_features(train_pdf, strict=False)
    
    # Handle categorical features
    categorical_features = get_categorical_features(valid_features)
    for cat_feat in categorical_features:
        unique_vals = train_pdf[cat_feat].dropna().unique()
        train_pdf[cat_feat] = pd.Categorical(train_pdf[cat_feat], categories=unique_vals)
        test_pdf[cat_feat] = pd.Categorical(test_pdf[cat_feat], categories=unique_vals)
    
    X_train = train_pdf[valid_features]
    X_test = test_pdf[valid_features]
    y_train = train_features[target].to_pandas()
    y_test = test_features[target].to_pandas()
    
    # Train model
    tree_method = get_optimal_tree_method()
    xgb_params["tree_method"] = tree_method
    if categorical_features:
        xgb_params["enable_categorical"] = True
    
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, proba)
    logger.info(f"Model performance: AUC-PR={metrics['auc_pr']:.4f}, AUC-ROC={metrics['auc_roc']:.4f}")
    
    return model, X_train, X_test, y_train, y_test, valid_features


def generate_global_explanations(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: List[str],
    output_dir: Path,
    max_samples: int = 1000
) -> Dict[str, Any]:
    """
    Generate global SHAP explanations.
    
    Args:
        model: Trained XGBoost model
        X_test: Test data
        feature_names: List of feature names
        output_dir: Directory to save plots
        max_samples: Maximum samples for SHAP computation
        
    Returns:
        Dictionary with SHAP values and feature importance
    """
    logger.info(f"Computing SHAP values for {min(len(X_test), max_samples)} samples...")
    
    # Sample data if too large
    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_test
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate mean absolute SHAP values for feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 features by SHAP importance:")
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Generate plots
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary plot (bar)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'shap_summary_bar.png'}")
    
    # 2. Summary plot (dot)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_dot.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'shap_summary_dot.png'}")
    
    # 3. Feature importance comparison
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Features by SHAP Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_dir / 'shap_feature_importance.png'}")
    
    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "explainer": explainer,
        "X_sample": X_sample
    }


def generate_local_explanations(
    model: xgb.XGBClassifier,
    explainer: shap.TreeExplainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    output_dir: Path,
    n_cases: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate local SHAP explanations for individual fraud cases.
    
    Args:
        model: Trained XGBoost model
        explainer: SHAP explainer
        X_test: Test data
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory to save plots
        n_cases: Number of case studies to generate
        
    Returns:
        List of case study dictionaries
    """
    logger.info(f"Generating {n_cases} case studies...")
    
    # Get predictions
    proba = model.predict_proba(X_test)[:, 1]
    
    # Find high-confidence fraud predictions
    fraud_mask = y_test == 1
    fraud_indices = y_test[fraud_mask].index.tolist()
    fraud_proba = proba[fraud_mask]
    
    # Get top fraud cases by prediction confidence
    if len(fraud_indices) > 0:
        top_fraud_idx = np.argsort(fraud_proba)[-n_cases:][::-1]
        case_indices = [fraud_indices[i] for i in top_fraud_idx]
    else:
        # Fallback to highest predictions if no fraud in test
        case_indices = np.argsort(proba)[-n_cases:][::-1].tolist()
    
    case_studies = []
    
    for i, idx in enumerate(case_indices):
        case_data = X_test.loc[[idx]]
        case_shap = explainer.shap_values(case_data)[0]
        case_proba = proba[X_test.index.get_loc(idx)]
        case_label = y_test.iloc[X_test.index.get_loc(idx)]
        
        # Get top contributing features
        feature_contrib = pd.DataFrame({
            'feature': feature_names,
            'shap_value': case_shap,
            'feature_value': case_data.values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        case_study = {
            "case_id": i + 1,
            "index": idx,
            "prediction": float(case_proba),
            "actual_label": int(case_label),
            "top_features": feature_contrib.head(10).to_dict('records'),
            "shap_values": case_shap
        }
        case_studies.append(case_study)
        
        # Generate waterfall plot for this case
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=case_shap,
                base_values=explainer.expected_value,
                data=case_data.values[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=15
        )
        plt.title(f"Case {i+1}: Fraud Probability = {case_proba:.3f} (Actual: {'Fraud' if case_label else 'Normal'})")
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_case_{i+1}_waterfall.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Case {i+1}: prob={case_proba:.3f}, actual={'Fraud' if case_label else 'Normal'}")
        logger.info(f"  Top features: {[f['feature'] for f in case_study['top_features'][:5]]}")
    
    return case_studies


def run_shap_analysis(
    df: pl.DataFrame,
    config: DictConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run complete SHAP analysis.
    
    Args:
        df: DataFrame with features and target
        config: Hydra configuration
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all SHAP analysis results
    """
    start_time = time.time()
    
    # Train model
    model, X_train, X_test, y_train, y_test, feature_names = train_model_for_shap(
        df=df,
        config=config
    )
    
    # Generate global explanations
    global_results = generate_global_explanations(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        output_dir=output_dir,
        max_samples=1000
    )
    
    # Generate local explanations
    case_studies = generate_local_explanations(
        model=model,
        explainer=global_results["explainer"],
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
        n_cases=5
    )
    
    # Save results
    results = {
        "feature_importance": global_results["feature_importance"].to_dict('records'),
        "case_studies": case_studies,
        "n_features": len(feature_names),
        "n_samples_analyzed": len(global_results["X_sample"]),
        "analysis_time_seconds": time.time() - start_time
    }
    
    # Save feature importance to CSV
    global_results["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)
    logger.info(f"Saved: {output_dir / 'feature_importance.csv'}")
    
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 5: SHAP Explainability Analysis (RQ4)
    
    Run with: python -m src.experiments.exp5_shap experiment_name=shap-explainability-rq4
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 5: SHAP EXPLAINABILITY ANALYSIS (RQ4)")
    logger.info("="*60)
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"Feature categories: {cfg.features.categories}")
    logger.info("")
    logger.info("This will generate SHAP explanations for fraud detection model")
    logger.info("="*60)
    
    # Setup MLflow
    setup_mlflow(cfg.experiment_name)
    
    # Create output directory
    output_dir = Path("artifacts/shap_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    df = load_data()
    df = compute_base_features(df, cutoff_date=datetime.now(), config=None)
    
    # Run SHAP analysis with MLflow tracking
    with mlflow.start_run(
        run_name=f"shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "shap_explainability"}
    ):
        mlflow.log_params({
            "feature_categories": str(cfg.features.categories),
            "total_samples": len(df),
        })
        
        results = run_shap_analysis(
            df=df,
            config=cfg,
            output_dir=output_dir
        )
        
        # Log metrics
        mlflow.log_metric("n_features", results["n_features"])
        mlflow.log_metric("n_samples_analyzed", results["n_samples_analyzed"])
        mlflow.log_metric("analysis_time_seconds", results["analysis_time_seconds"])
        
        # Log artifacts
        mlflow.log_artifacts(str(output_dir), artifact_path="shap_outputs")
        
        logger.info(f"\n{'='*60}")
        logger.info("SHAP ANALYSIS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Features analyzed: {results['n_features']}")
        logger.info(f"Samples analyzed: {results['n_samples_analyzed']}")
        logger.info(f"Case studies generated: {len(results['case_studies'])}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Analysis time: {results['analysis_time_seconds']:.1f}s")
        
        # Print top 10 features
        logger.info("\nTop 10 Features by SHAP Importance:")
        for i, feat in enumerate(results["feature_importance"][:10]):
            logger.info(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        # Save to results registry
        from src.experiments.results import save_result, ExperimentID
        save_result(ExperimentID.EXP5_SHAP, {
            "top_features": results["feature_importance"][:20],
            "n_features": results["n_features"],
            "n_samples_analyzed": results["n_samples_analyzed"],
            "n_case_studies": len(results["case_studies"]),
        })
    
    return results


if __name__ == "__main__":
    main()


"""
Comprehensive comparison analysis between baseline and hybrid models.

This script helps answer: Why doesn't the hybrid model (GNN + XGBoost) 
outperform the baseline XGBoost model?

It analyzes:
1. Feature importance differences
2. GNN embedding contribution
3. Prediction agreement/disagreement
4. Performance by fraud type
5. Temporal stability
"""

import os
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.explainability import ModelExplainer, load_saved_model, compare_models


def load_model_windows(
    model_type: str,
    window_indices: List[int] = None
) -> Dict[int, Dict]:
    """
    Load multiple windows for a model type.
    
    Args:
        model_type: Model type (baseline, hybrid_hgt, etc.)
        window_indices: Specific windows to load, or None for all
        
    Returns:
        Dictionary mapping window_idx to model bundle
    """
    models_dir = f"artifacts/models/{model_type}"
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    model_files = sorted(glob.glob(os.path.join(models_dir, "model_window_*.pkl")))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    model_bundles = {}
    
    for model_file in model_files:
        bundle = load_saved_model(model_file)
        idx = bundle['window_info']['window_idx']
        
        if window_indices is None or idx in window_indices:
            model_bundles[idx] = bundle
    
    return model_bundles


def compare_performance_metrics(
    baseline_bundles: Dict[int, Dict],
    hybrid_bundles: Dict[int, Dict]
) -> pd.DataFrame:
    """
    Compare performance metrics across all windows.
    
    Args:
        baseline_bundles: Baseline model bundles by window
        hybrid_bundles: Hybrid model bundles by window
        
    Returns:
        DataFrame with comparative metrics
    """
    comparison_data = []
    
    for window_idx in sorted(baseline_bundles.keys()):
        if window_idx not in hybrid_bundles:
            continue
        
        baseline = baseline_bundles[window_idx]
        hybrid = hybrid_bundles[window_idx]
        
        row = {
            'window_idx': window_idx,
            'window_start': baseline['window_info']['window_start'],
            'window_end': baseline['window_info']['window_end'],
        }
        
        # Add baseline metrics
        for metric, value in baseline['metrics'].items():
            if metric != 'window_start':
                row[f'baseline_{metric}'] = value
        
        # Add hybrid metrics
        for metric, value in hybrid['metrics'].items():
            if metric != 'window_start':
                row[f'hybrid_{metric}'] = value
        
        # Calculate differences
        for metric in ['auc_pr', 'auc_roc', 'p@100', 'p@200', 'lift@100']:
            baseline_val = baseline['metrics'].get(metric, 0)
            hybrid_val = hybrid['metrics'].get(metric, 0)
            row[f'diff_{metric}'] = hybrid_val - baseline_val
            row[f'pct_diff_{metric}'] = ((hybrid_val - baseline_val) / baseline_val * 100 
                                          if baseline_val > 0 else 0)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def analyze_embedding_contribution(
    hybrid_bundle: Dict
) -> Dict:
    """
    Analyze how much GNN embeddings contribute to predictions.
    
    Args:
        hybrid_bundle: Hybrid model bundle
        
    Returns:
        Dictionary with embedding contribution statistics
    """
    explainer = ModelExplainer(
        model=hybrid_bundle['model'],
        feature_names=hybrid_bundle['features'],
        X_test=hybrid_bundle['X_test'],
        y_test=hybrid_bundle['y_test'],
        y_pred=hybrid_bundle['y_pred'],
        model_type='hybrid',
        window_info=hybrid_bundle['window_info']
    )
    
    embed_features = explainer._get_embedding_features()
    
    if not embed_features:
        return {'has_embeddings': False}
    
    embed_indices = [i for i, name in enumerate(explainer.feature_names) 
                    if name in embed_features]
    
    # SHAP-based analysis
    embed_shap = explainer.shap_values[:, embed_indices]
    total_shap = explainer.shap_values
    
    # Calculate contribution percentage
    embed_importance = np.abs(embed_shap).mean()
    total_importance = np.abs(total_shap).mean()
    contribution_pct = (embed_importance / total_importance * 100 
                       if total_importance > 0 else 0)
    
    # Calculate per-instance contribution
    per_instance_embed = np.abs(embed_shap).sum(axis=1)
    per_instance_total = np.abs(total_shap).sum(axis=1)
    per_instance_pct = per_instance_embed / per_instance_total * 100
    
    # Analyze by prediction type
    tp_mask = (explainer.df_test['y_true'] == 1) & (explainer.df_test['predicted_fraud'] == 1)
    fp_mask = (explainer.df_test['y_true'] == 0) & (explainer.df_test['predicted_fraud'] == 1)
    tn_mask = (explainer.df_test['y_true'] == 0) & (explainer.df_test['predicted_fraud'] == 0)
    fn_mask = (explainer.df_test['y_true'] == 1) & (explainer.df_test['predicted_fraud'] == 0)
    
    return {
        'has_embeddings': True,
        'n_embeddings': len(embed_features),
        'overall_contribution_pct': contribution_pct,
        'mean_per_instance_pct': per_instance_pct.mean(),
        'median_per_instance_pct': np.median(per_instance_pct),
        'tp_contribution': per_instance_pct[tp_mask].mean() if tp_mask.sum() > 0 else 0,
        'fp_contribution': per_instance_pct[fp_mask].mean() if fp_mask.sum() > 0 else 0,
        'tn_contribution': per_instance_pct[tn_mask].mean() if tn_mask.sum() > 0 else 0,
        'fn_contribution': per_instance_pct[fn_mask].mean() if fn_mask.sum() > 0 else 0,
        'top_5_features': explainer.get_feature_importance_df(aggregate_embeddings=False)
                                   .head(5)[['feature', 'mean_abs_shap']].to_dict('records')
    }


def analyze_prediction_agreement(
    baseline_bundle: Dict,
    hybrid_bundle: Dict
) -> Dict:
    """
    Analyze where baseline and hybrid models agree/disagree.
    
    Args:
        baseline_bundle: Baseline model bundle
        hybrid_bundle: Hybrid model bundle
        
    Returns:
        Dictionary with agreement statistics
    """
    baseline_pred = (baseline_bundle['y_pred'] >= 0.5).astype(int)
    hybrid_pred = (hybrid_bundle['y_pred'] >= 0.5).astype(int)
    y_true = baseline_bundle['y_test']
    
    agreement = (baseline_pred == hybrid_pred)
    agreement_rate = agreement.mean()
    
    # Analyze disagreement cases
    disagree_mask = ~agreement
    disagree_cases = {
        'total': disagree_mask.sum(),
        'baseline_right': ((baseline_pred == y_true) & disagree_mask).sum(),
        'hybrid_right': ((hybrid_pred == y_true) & disagree_mask).sum(),
        'both_wrong': ((baseline_pred != y_true) & (hybrid_pred != y_true) & disagree_mask).sum()
    }
    
    # Correlation between prediction scores
    score_correlation = np.corrcoef(baseline_bundle['y_pred'], hybrid_bundle['y_pred'])[0, 1]
    
    return {
        'agreement_rate': agreement_rate,
        'disagree_cases': disagree_cases,
        'score_correlation': score_correlation
    }


def generate_comparison_report(
    baseline_type: str = "baseline",
    hybrid_type: str = "hybrid_hgt",
    output_dir: str = "artifacts/shap/comparison"
) -> None:
    """
    Generate a comprehensive comparison report.
    
    Args:
        baseline_type: Baseline model type
        hybrid_type: Hybrid model type
        output_dir: Output directory for report and plots
    """
    print("=" * 80)
    print("BASELINE vs HYBRID MODEL COMPARISON")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print(f"\n1. Loading models...")
    baseline_bundles = load_model_windows(baseline_type)
    hybrid_bundles = load_model_windows(hybrid_type)
    
    print(f"   Baseline: {len(baseline_bundles)} windows")
    print(f"   Hybrid: {len(hybrid_bundles)} windows")
    
    # Compare performance metrics
    print(f"\n2. Comparing performance metrics...")
    perf_df = compare_performance_metrics(baseline_bundles, hybrid_bundles)
    
    print("\n   Mean Performance Differences:")
    for metric in ['auc_pr', 'auc_roc', 'p@100', 'lift@100']:
        diff_col = f'diff_{metric}'
        if diff_col in perf_df.columns:
            mean_diff = perf_df[diff_col].mean()
            pct_col = f'pct_diff_{metric}'
            mean_pct = perf_df[pct_col].mean() if pct_col in perf_df.columns else 0
            winner = "🏆 Hybrid" if mean_diff > 0 else "🏆 Baseline" if mean_diff < 0 else "🤝 Tie"
            print(f"   {metric}: {mean_diff:+.4f} ({mean_pct:+.2f}%) {winner}")
    
    # Save performance comparison
    perf_csv = os.path.join(output_dir, "performance_comparison.csv")
    perf_df.to_csv(perf_csv, index=False)
    print(f"\n   Saved: {perf_csv}")
    
    # Analyze embedding contribution
    print(f"\n3. Analyzing embedding contribution...")
    latest_hybrid = hybrid_bundles[max(hybrid_bundles.keys())]
    embed_stats = analyze_embedding_contribution(latest_hybrid)
    
    if embed_stats['has_embeddings']:
        print(f"   Number of embedding features: {embed_stats['n_embeddings']}")
        print(f"   Overall contribution: {embed_stats['overall_contribution_pct']:.2f}%")
        print(f"   Mean per-instance: {embed_stats['mean_per_instance_pct']:.2f}%")
        print(f"\n   Contribution by prediction type:")
        print(f"     True Positives: {embed_stats['tp_contribution']:.2f}%")
        print(f"     False Positives: {embed_stats['fp_contribution']:.2f}%")
        print(f"     True Negatives: {embed_stats['tn_contribution']:.2f}%")
        print(f"     False Negatives: {embed_stats['fn_contribution']:.2f}%")
        
        print(f"\n   Top 5 Features (including embeddings):")
        for i, feat in enumerate(embed_stats['top_5_features']):
            print(f"     {i+1}. {feat['feature']}: {feat['mean_abs_shap']:.6f}")
    else:
        print("   ⚠️ No embeddings found in hybrid model")
    
    # Analyze prediction agreement
    print(f"\n4. Analyzing prediction agreement...")
    latest_baseline = baseline_bundles[max(baseline_bundles.keys())]
    agreement_stats = analyze_prediction_agreement(latest_baseline, latest_hybrid)
    
    print(f"   Agreement rate: {agreement_stats['agreement_rate']:.2%}")
    print(f"   Score correlation: {agreement_stats['score_correlation']:.4f}")
    print(f"\n   Disagreement cases: {agreement_stats['disagree_cases']['total']}")
    print(f"     Baseline correct: {agreement_stats['disagree_cases']['baseline_right']}")
    print(f"     Hybrid correct: {agreement_stats['disagree_cases']['hybrid_right']}")
    print(f"     Both wrong: {agreement_stats['disagree_cases']['both_wrong']}")
    
    # Generate visualizations
    print(f"\n5. Generating visualizations...")
    
    # Plot 1: Performance over time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['auc_pr', 'auc_roc', 'p@100', 'lift@100']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        ax.plot(perf_df['window_idx'], perf_df[f'baseline_{metric}'], 
               marker='o', label='Baseline', linewidth=2)
        ax.plot(perf_df['window_idx'], perf_df[f'hybrid_{metric}'], 
               marker='s', label='Hybrid', linewidth=2)
        ax.set_xlabel('Window Index')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    perf_plot = os.path.join(output_dir, "performance_over_time.png")
    plt.savefig(perf_plot, dpi=300, bbox_inches='tight')
    print(f"   Saved: {perf_plot}")
    plt.close()
    
    # Plot 2: Feature importance comparison
    baseline_explainer = ModelExplainer(
        model=latest_baseline['model'],
        feature_names=latest_baseline['features'],
        X_test=latest_baseline['X_test'],
        y_test=latest_baseline['y_test'],
        y_pred=latest_baseline['y_pred'],
        model_type=baseline_type,
        window_info=latest_baseline['window_info']
    )
    
    hybrid_explainer = ModelExplainer(
        model=latest_hybrid['model'],
        feature_names=latest_hybrid['features'],
        X_test=latest_hybrid['X_test'],
        y_test=latest_hybrid['y_test'],
        y_pred=latest_hybrid['y_pred'],
        model_type=hybrid_type,
        window_info=latest_hybrid['window_info']
    )
    
    comparison_plot = os.path.join(output_dir, "feature_importance_comparison.png")
    compare_models(
        {baseline_type: baseline_explainer, hybrid_type: hybrid_explainer},
        top_n_features=15,
        aggregate_embeddings=True,
        output_path=comparison_plot
    )
    print(f"   Saved: {comparison_plot}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    mean_auc_pr_diff = perf_df['diff_auc_pr'].mean()
    
    if embed_stats['has_embeddings']:
        if mean_auc_pr_diff < -0.001:  # Hybrid worse
            print("\n❌ FINDING: Hybrid model performs WORSE than baseline")
            print("\n   Possible reasons:")
            print(f"   1. Low embedding contribution ({embed_stats['overall_contribution_pct']:.1f}%)")
            print("   2. GNN may be learning redundant information")
            print("   3. Graph structure may be too sparse or noisy")
            print("\n   Recommendations:")
            print("   - Investigate graph quality (connectivity, sparsity)")
            print("   - Try different GNN architectures (GAT, GraphSAGE)")
            print("   - Increase embedding dimensions (64 → 128 → 256)")
            print("   - Add graph-specific features instead of raw embeddings")
        elif mean_auc_pr_diff < 0.001:  # Roughly equal
            print("\n⚠️  FINDING: Hybrid model shows NO significant improvement")
            print("\n   Observations:")
            print(f"   - Embedding contribution: {embed_stats['overall_contribution_pct']:.1f}%")
            print(f"   - High prediction agreement: {agreement_stats['agreement_rate']:.1%}")
            print("\n   Recommendations:")
            print("   - Embeddings are not adding new information")
            print("   - Consider using graph features (PageRank, clustering) instead")
            print("   - Focus on improving tabular features")
        else:  # Hybrid better
            print("\n✅ FINDING: Hybrid model shows improvement")
            print(f"\n   AUC-PR improvement: +{mean_auc_pr_diff:.4f}")
            print(f"   Embedding contribution: {embed_stats['overall_contribution_pct']:.1f}%")
            print("\n   Recommendations:")
            print("   - Continue using hybrid approach")
            print("   - Experiment with ensemble methods")
            print("   - Fine-tune GNN architecture")
    else:
        print("\n⚠️  WARNING: No embeddings found in hybrid model")
    
    print("\n" + "=" * 80)
    print(f"\nFull report saved to: {output_dir}")


if __name__ == "__main__":
    import typer
    
    def main(
        baseline: str = "baseline",
        hybrid: str = "hybrid_hgt",
        output_dir: str = "artifacts/shap/comparison"
    ):
        """Compare baseline and hybrid models."""
        generate_comparison_report(baseline, hybrid, output_dir)
    
    typer.run(main)


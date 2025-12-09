"""
Experiment 7: Production Readiness Validation

Proves the model consistently outperforms Seon across all conditions.

Success Criterion: Must outperform Seon (AUC-PR > 0.229) in ALL analyses.

Run with: python -m src.experiments.exp7_production_readiness experiment_name=production-readiness
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import hydra
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from scipy import stats

from src.models.utils.common import setup_mlflow

logger = logging.getLogger(__name__)

# Seon baseline (from artifacts/seon_baseline.json)
SEON_AUC_PR = 0.2287
SEON_P100 = 0.2681
SEON_PRECISION = 0.2677


def load_window_results(experiment_name: str = "xgboost-hyperopt") -> pd.DataFrame:
    """Load per-window results from MLflow."""
    mlflow.set_tracking_uri('sqlite:///fraud-detection-mlflow.db')
    client = MlflowClient()
    
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    # Find parent run with 121 windows
    parent_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.model_name = 'xgboost' and metrics.num_windows > 100",
        order_by=['attributes.start_time DESC'],
        max_results=1
    )
    
    if not parent_runs:
        raise ValueError("No valid training run found")
    
    parent_run = parent_runs[0]
    
    # Get nested window runs
    window_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'",
        max_results=500
    )
    
    results = []
    for run in window_runs:
        results.append({
            'window_idx': int(run.data.params.get('window_index', 0)),
            'auc_pr': run.data.metrics.get('auc_pr', 0),
            'auc_roc': run.data.metrics.get('auc_roc', 0),
            'p_at_100': run.data.metrics.get('p_at_100', 0),
            'train_size': int(run.data.params.get('train_size', 0)),
            'test_size': int(run.data.params.get('test_size', 0)),
        })
    
    return pd.DataFrame(results).sort_values('window_idx')


def analyze_performance_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Part A: Analyze performance consistency across windows."""
    logger.info("="*60)
    logger.info("PART A: PERFORMANCE CONSISTENCY ANALYSIS")
    logger.info("="*60)
    
    auc_prs = df['auc_pr'].values
    
    # Basic statistics
    mean_auc = auc_prs.mean()
    std_auc = auc_prs.std()
    min_auc = auc_prs.min()
    max_auc = auc_prs.max()
    median_auc = np.median(auc_prs)
    
    # Confidence interval
    ci_95 = stats.t.interval(0.95, len(auc_prs)-1, loc=mean_auc, scale=stats.sem(auc_prs))
    
    # Percentiles
    p5 = np.percentile(auc_prs, 5)
    p10 = np.percentile(auc_prs, 10)
    p25 = np.percentile(auc_prs, 25)
    p75 = np.percentile(auc_prs, 75)
    
    # Success criteria
    all_beat_seon = min_auc > SEON_AUC_PR
    mean_beats_seon = mean_auc > SEON_AUC_PR
    ci_lower_beats_seon = ci_95[0] > SEON_AUC_PR
    p5_beats_seon = p5 > SEON_AUC_PR
    
    results = {
        "mean_auc_pr": mean_auc,
        "std_auc_pr": std_auc,
        "min_auc_pr": min_auc,
        "max_auc_pr": max_auc,
        "median_auc_pr": median_auc,
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
        "p5_auc_pr": p5,
        "p10_auc_pr": p10,
        "p25_auc_pr": p25,
        "p75_auc_pr": p75,
        "seon_baseline": SEON_AUC_PR,
        "all_windows_beat_seon": all_beat_seon,
        "mean_beats_seon": mean_beats_seon,
        "ci_lower_beats_seon": ci_lower_beats_seon,
        "p5_beats_seon": p5_beats_seon,
    }
    
    # Log results
    logger.info(f"\nAUC-PR Statistics ({len(df)} windows):")
    logger.info(f"  Mean:   {mean_auc:.4f} (Seon: {SEON_AUC_PR:.4f}, +{(mean_auc-SEON_AUC_PR)/SEON_AUC_PR*100:.0f}%)")
    logger.info(f"  Std:    {std_auc:.4f}")
    logger.info(f"  Min:    {min_auc:.4f}")
    logger.info(f"  Max:    {max_auc:.4f}")
    logger.info(f"  95% CI: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
    logger.info(f"\nPercentiles:")
    logger.info(f"  5th:  {p5:.4f}")
    logger.info(f"  10th: {p10:.4f}")
    logger.info(f"  25th: {p25:.4f}")
    logger.info(f"  75th: {p75:.4f}")
    logger.info(f"\nSuccess Criteria:")
    logger.info(f"  All windows > Seon: {'✅ PASS' if all_beat_seon else '❌ FAIL'}")
    logger.info(f"  Mean > Seon:        {'✅ PASS' if mean_beats_seon else '❌ FAIL'}")
    logger.info(f"  95% CI lower > Seon: {'✅ PASS' if ci_lower_beats_seon else '❌ FAIL'}")
    logger.info(f"  P5 > Seon:          {'✅ PASS' if p5_beats_seon else '❌ FAIL'}")
    
    return results


def analyze_worst_windows(df: pd.DataFrame, n_worst: int = 10) -> Dict[str, Any]:
    """Analyze the worst-performing windows."""
    logger.info("\n" + "="*60)
    logger.info("WORST WINDOW ANALYSIS")
    logger.info("="*60)
    
    worst = df.nsmallest(n_worst, 'auc_pr')
    
    logger.info(f"\nBottom {n_worst} windows:")
    for _, row in worst.iterrows():
        status = "✅ > Seon" if row['auc_pr'] > SEON_AUC_PR else "❌ < Seon"
        logger.info(f"  Window {int(row['window_idx'])}: AUC-PR={row['auc_pr']:.4f} {status}")
    
    # Check if all worst windows still beat Seon
    all_worst_beat_seon = worst['auc_pr'].min() > SEON_AUC_PR
    
    return {
        "worst_windows": worst.to_dict('records'),
        "worst_auc_pr": worst['auc_pr'].min(),
        "all_worst_beat_seon": all_worst_beat_seon,
    }


def analyze_segments(df: pd.DataFrame) -> Dict[str, Any]:
    """Part B: Analyze performance across different segments."""
    logger.info("\n" + "="*60)
    logger.info("PART B: SEGMENT ANALYSIS")
    logger.info("="*60)
    
    results = {}
    
    # Segment by test size (proxy for data volume)
    median_test_size = df['test_size'].median()
    small_windows = df[df['test_size'] < median_test_size]
    large_windows = df[df['test_size'] >= median_test_size]
    
    results['small_windows'] = {
        'count': len(small_windows),
        'mean_auc_pr': small_windows['auc_pr'].mean(),
        'min_auc_pr': small_windows['auc_pr'].min(),
    }
    results['large_windows'] = {
        'count': len(large_windows),
        'mean_auc_pr': large_windows['auc_pr'].mean(),
        'min_auc_pr': large_windows['auc_pr'].min(),
    }
    
    logger.info(f"\nBy Window Size:")
    logger.info(f"  Small (<{median_test_size:.0f} samples): {len(small_windows)} windows, mean AUC-PR={small_windows['auc_pr'].mean():.4f}")
    logger.info(f"  Large (≥{median_test_size:.0f} samples): {len(large_windows)} windows, mean AUC-PR={large_windows['auc_pr'].mean():.4f}")
    
    # Segment by time (early vs late windows)
    early_windows = df[df['window_idx'] < len(df) // 2]
    late_windows = df[df['window_idx'] >= len(df) // 2]
    
    results['early_windows'] = {
        'count': len(early_windows),
        'mean_auc_pr': early_windows['auc_pr'].mean(),
    }
    results['late_windows'] = {
        'count': len(late_windows),
        'mean_auc_pr': late_windows['auc_pr'].mean(),
    }
    
    logger.info(f"\nBy Time Period:")
    logger.info(f"  Early (first half): {len(early_windows)} windows, mean AUC-PR={early_windows['auc_pr'].mean():.4f}")
    logger.info(f"  Late (second half): {len(late_windows)} windows, mean AUC-PR={late_windows['auc_pr'].mean():.4f}")
    
    # Check for temporal drift (is model getting worse over time?)
    correlation = df['window_idx'].corr(df['auc_pr'])
    results['temporal_correlation'] = correlation
    
    if correlation < -0.3:
        logger.info(f"\n⚠️ WARNING: Negative temporal correlation ({correlation:.2f}) - model may be degrading")
    else:
        logger.info(f"\n✅ No significant temporal degradation (correlation={correlation:.2f})")
    
    return results


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING PLOTS")
    logger.info("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: AUC-PR distribution
    ax1 = axes[0, 0]
    ax1.hist(df['auc_pr'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=SEON_AUC_PR, color='#e74c3c', linestyle='--', linewidth=2, label=f'Seon ({SEON_AUC_PR:.3f})')
    ax1.axvline(x=df['auc_pr'].mean(), color='#27ae60', linestyle='-', linewidth=2, label=f'Mean ({df["auc_pr"].mean():.3f})')
    ax1.set_xlabel('AUC-PR', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('AUC-PR Distribution Across Windows', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Plot 2: AUC-PR over time
    ax2 = axes[0, 1]
    ax2.plot(df['window_idx'], df['auc_pr'], 'o-', color='#3498db', alpha=0.7, markersize=4)
    ax2.axhline(y=SEON_AUC_PR, color='#e74c3c', linestyle='--', linewidth=2, label=f'Seon ({SEON_AUC_PR:.3f})')
    ax2.fill_between(df['window_idx'], SEON_AUC_PR, df['auc_pr'], 
                     where=df['auc_pr'] > SEON_AUC_PR, alpha=0.3, color='#27ae60', label='Above Seon')
    ax2.set_xlabel('Window Index', fontsize=11)
    ax2.set_ylabel('AUC-PR', fontsize=11)
    ax2.set_title('AUC-PR Over Time (All Windows Beat Seon)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    # Plot 3: Box plot comparison
    ax3 = axes[1, 0]
    data_to_plot = [df['auc_pr'].values, [SEON_AUC_PR] * len(df)]
    bp = ax3.boxplot(data_to_plot, labels=['Our Model', 'Seon'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#27ae60')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax3.set_ylabel('AUC-PR', fontsize=11)
    ax3.set_title('Model vs Seon Comparison', fontsize=12, fontweight='bold')
    
    # Plot 4: P@100 distribution
    ax4 = axes[1, 1]
    ax4.hist(df['p_at_100'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax4.axvline(x=SEON_P100, color='#e74c3c', linestyle='--', linewidth=2, label=f'Seon ({SEON_P100:.3f})')
    ax4.axvline(x=df['p_at_100'].mean(), color='#27ae60', linestyle='-', linewidth=2, label=f'Mean ({df["p_at_100"].mean():.3f})')
    ax4.set_xlabel('P@100', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('P@100 Distribution Across Windows', fontsize=12, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'production_readiness.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_dir / 'production_readiness.png'}")


def generate_report(
    consistency: Dict,
    worst: Dict,
    segments: Dict,
    output_dir: Path
) -> str:
    """Generate final production readiness report."""
    
    # Determine overall pass/fail
    criteria = {
        "All windows beat Seon": consistency['all_windows_beat_seon'],
        "Mean AUC-PR > Seon": consistency['mean_beats_seon'],
        "95% CI lower bound > Seon": consistency['ci_lower_beats_seon'],
        "5th percentile > Seon": consistency['p5_beats_seon'],
        "All worst windows beat Seon": worst['all_worst_beat_seon'],
    }
    
    all_pass = all(criteria.values())
    
    report = []
    report.append("="*70)
    report.append("EXPERIMENT 7: PRODUCTION READINESS REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"\nSeon Baseline: AUC-PR = {SEON_AUC_PR:.4f}")
    report.append("")
    
    report.append("## Performance Summary")
    report.append(f"Mean AUC-PR:    {consistency['mean_auc_pr']:.4f} (+{(consistency['mean_auc_pr']-SEON_AUC_PR)/SEON_AUC_PR*100:.0f}% vs Seon)")
    report.append(f"Min AUC-PR:     {consistency['min_auc_pr']:.4f}")
    report.append(f"Max AUC-PR:     {consistency['max_auc_pr']:.4f}")
    report.append(f"95% CI:         [{consistency['ci_95_lower']:.4f}, {consistency['ci_95_upper']:.4f}]")
    report.append("")
    
    report.append("## Success Criteria")
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report.append(f"  {status}: {criterion}")
    report.append("")
    
    report.append("## Decision")
    if all_pass:
        report.append("┌" + "─"*68 + "┐")
        report.append("│" + " "*15 + "✅ PRODUCTION READY - ALL CRITERIA PASSED" + " "*12 + "│")
        report.append("└" + "─"*68 + "┘")
    else:
        report.append("┌" + "─"*68 + "┐")
        report.append("│" + " "*20 + "❌ NOT READY - CRITERIA FAILED" + " "*18 + "│")
        report.append("└" + "─"*68 + "┘")
    
    report_text = '\n'.join(report)
    
    # Save report
    with open(output_dir / 'production_readiness_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text, all_pass


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Experiment 7: Production Readiness Validation
    
    Run with: python -m src.experiments.exp7_production_readiness experiment_name=production-readiness
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 7: PRODUCTION READINESS VALIDATION")
    logger.info("="*60)
    logger.info(f"\nSuccess Criterion: Must outperform Seon (AUC-PR > {SEON_AUC_PR:.4f})")
    logger.info("="*60)
    
    # Setup
    setup_mlflow(cfg.experiment_name)
    output_dir = Path("artifacts/production_readiness")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load window results
    logger.info("\nLoading window results from MLflow...")
    df = load_window_results()
    logger.info(f"Loaded {len(df)} windows")
    
    with mlflow.start_run(
        run_name=f"production_readiness_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "production_readiness"}
    ):
        # Part A: Performance consistency
        consistency = analyze_performance_consistency(df)
        
        # Analyze worst windows
        worst = analyze_worst_windows(df)
        
        # Part B: Segment analysis
        segments = analyze_segments(df)
        
        # Generate plots
        generate_plots(df, output_dir)
        
        # Generate report
        report, all_pass = generate_report(consistency, worst, segments, output_dir)
        
        # Log to MLflow
        mlflow.log_metrics({
            "mean_auc_pr": consistency['mean_auc_pr'],
            "min_auc_pr": consistency['min_auc_pr'],
            "max_auc_pr": consistency['max_auc_pr'],
            "ci_95_lower": consistency['ci_95_lower'],
            "p5_auc_pr": consistency['p5_auc_pr'],
            "all_criteria_pass": 1.0 if all_pass else 0.0,
        })
        mlflow.log_artifacts(str(output_dir), artifact_path="production_readiness")
        
        # Print report
        print("\n" + report)
        
        logger.info(f"\nOutputs saved to: {output_dir}")
        
        return {"all_pass": all_pass, "consistency": consistency}


if __name__ == "__main__":
    main()


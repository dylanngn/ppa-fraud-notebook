"""
Experiment 9: Feature Selection Visualization

Generates:
- Method comparison bar chart
- Feature overlap analysis (if feature names available)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
ARTIFACTS_DIR = Path("artifacts/feature_selection")
OUTPUT_DIR = ARTIFACTS_DIR

plt.style.use('seaborn-v0_8-whitegrid')

def load_results():
    """Load all available results."""
    results = {}
    
    # Load CSV (9B results)
    csv_path = ARTIFACTS_DIR / "method_comparison.csv"
    if csv_path.exists():
        results['9b'] = pd.read_csv(csv_path)
    
    # Load JSON (9A/9C results)
    json_path = ARTIFACTS_DIR / "selection_results.json"
    if json_path.exists():
        with open(json_path) as f:
            results['json'] = json.load(f)
    
    return results


def create_method_comparison_chart(results):
    """Create comprehensive method comparison bar chart."""
    
    # Combine all results into one DataFrame
    rows = []
    
    # 9A results from JSON
    if 'json' in results and '9a_all_fields' in results['json']['results']:
        for config, data in results['json']['results']['9a_all_fields'].items():
            rows.append({
                'method': config.replace('_', ' ').title(),
                'n_features': data['n_features'],
                'mean_auc_pr': data['mean_auc_pr'],
                'std_auc_pr': data.get('std_auc_pr', 0),
                'category': '9A: All Fields'
            })
    
    # 9B results from CSV
    if '9b' in results:
        for _, row in results['9b'].iterrows():
            rows.append({
                'method': row['method'].replace('_', ' ').title(),
                'n_features': row['n_features'],
                'mean_auc_pr': row['mean_auc_pr'],
                'std_auc_pr': row.get('std_auc_pr', 0),
                'category': '9B: Selection Methods'
            })
    
    # 9C results from JSON (if available)
    if 'json' in results and '9c_graph_only' in results['json']['results']:
        for config, data in results['json']['results']['9c_graph_only'].items():
            rows.append({
                'method': config.replace('_', ' ').title(),
                'n_features': data['n_features'],
                'mean_auc_pr': data['mean_auc_pr'],
                'std_auc_pr': data.get('std_auc_pr', 0),
                'category': '9C: Ablation'
            })
    
    df = pd.DataFrame(rows)
    
    # Sort by AUC-PR
    df = df.sort_values('mean_auc_pr', ascending=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by category
    colors = {
        '9A: All Fields': '#3498db',
        '9B: Selection Methods': '#2ecc71',
        '9C: Ablation': '#9b59b6'
    }
    bar_colors = [colors.get(cat, '#95a5a6') for cat in df['category']]
    
    # Plot horizontal bars
    bars = ax.barh(df['method'], df['mean_auc_pr'], 
                   xerr=df['std_auc_pr'], capsize=3,
                   color=bar_colors, alpha=0.8)
    
    # Add feature count labels
    for bar, n, auc in zip(bars, df['n_features'], df['mean_auc_pr']):
        ax.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
                f'n={n}', va='center', fontsize=9)
    
    ax.set_xlabel('AUC-PR', fontsize=12)
    ax.set_title('Experiment 9: Feature Selection Method Comparison', fontsize=14)
    ax.set_xlim(0, 1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l, alpha=0.8) 
                       for l, c in colors.items() if l in df['category'].values]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add vertical line for reference
    ax.axvline(x=0.77, color='red', linestyle='--', alpha=0.5, label='~0.77 threshold')
    
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "exp9_complete_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return df


def create_summary_table(df):
    """Create summary table for thesis."""
    summary = df[['method', 'category', 'n_features', 'mean_auc_pr']].copy()
    summary['mean_auc_pr'] = summary['mean_auc_pr'].round(4)
    summary = summary.sort_values('mean_auc_pr', ascending=False)
    
    # Save as CSV
    output_path = OUTPUT_DIR / "exp9_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return summary


def main():
    print("=" * 60)
    print("EXPERIMENT 9: VISUALIZATION")
    print("=" * 60)
    
    results = load_results()
    
    # Create comparison chart
    df = create_method_comparison_chart(results)
    
    # Create summary table
    summary = create_summary_table(df)
    
    print("\n--- Summary Table ---")
    print(summary.to_string(index=False))
    
    print("\n✅ Visualizations complete!")


if __name__ == "__main__":
    main()

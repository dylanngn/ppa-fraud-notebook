#!/usr/bin/env python3
"""
Generate bar charts for Figure 5.3 and 5.4 with visible colors.
"""
import matplotlib.pyplot as plt
from pathlib import Path

def generate_model_comparison_chart():
    """Generate bar chart for Figure 5.3: Model Comparison."""
    models = ['Logistic\nRegression', 'Random\nForest', 'Vanilla\nXGBoost', 'GNN+XGBoost']
    auc_pr = [0.5099, 0.5646, 0.6639, 0.6990]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with distinct colors
    colors = ['#FF6B6B', '#FFA94D', '#4ECDC4', '#45B7D1']
    bars = ax.bar(models, auc_pr, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('AUC-PR', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: AUC-PR', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.4, 0.75)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Save figure
    output_path = Path('thesis/figures/fig_5_5_2_1.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()

def generate_ablation_chart():
    """Generate bar chart for Figure 5.4: Component Contribution."""
    components = ['Tabular\nOnly', '+ GNN\n(Self-Sup)', '+ GNN\n(Supervised)', 'Full\nHybrid']
    auc_pr = [0.6639, 0.6967, 0.6815, 0.6990]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with distinct colors
    colors = ['#95A5A6', '#3498DB', '#9B59B6', '#2ECC71']
    bars = ax.bar(components, auc_pr, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('AUC-PR', fontsize=12, fontweight='bold')
    ax.set_title('Component Contribution to AUC-PR', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.60, 0.72)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Save figure
    output_path = Path('thesis/figures/fig_5_5_4_1.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()

def generate_monthly_fraud_chart():
    """Generate bar chart for Figure 5.1: Monthly Fraud Rate."""
    months = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
    fraud_rate = [7.2, 8.5, 13.1, 9.8, 7.4, 6.2, 5.8, 5.1, 4.9, 5.2, 5.5, 5.3]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bars with color gradient (red for high, green for low)
    colors = ['#FFB74D' if rate < 8 else '#FF7043' if rate < 10 else '#E53935' for rate in fraud_rate]
    bars = ax.bar(months, fraud_rate, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on top of bars
    for bar, rate in zip(bars, fraud_rate):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Fraud Rate (Dec 2024 - Nov 2025)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Save figure
    output_path = Path('thesis/figures/fig_5_5_1_1.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_model_comparison_chart()
    generate_ablation_chart()
    generate_monthly_fraud_chart()


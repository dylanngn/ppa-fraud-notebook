#!/usr/bin/env python3
"""
Generate pie chart for Figure 5.2 with 4 distinct colors.
"""
import matplotlib.pyplot as plt
from pathlib import Path

def generate_fraud_region_pie_chart():
    """Generate pie chart for fraud by origin region."""
    # Data from chapter5_section5_1.md
    labels = [
        'West Africa (66%+ fraud rate) [847]',
        'Other High-Risk (30-65%) [1243]',
        'Medium Risk (10-30%) [2156]',
        'Low Risk (<10%) [2741]'
    ]
    sizes = [847, 1243, 2156, 2741]
    
    # 4 distinct colors - professional color palette
    colors = ['#FF6B6B', '#FFA94D', '#FFD93D', '#6BCB77']  # Red, Orange, Yellow, Green
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # We'll add custom legend
        autopct='%1.0f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    # Add title
    ax.set_title('Fraud by Origin Region', fontsize=14, weight='bold', pad=20)
    
    # Add legend
    ax.legend(
        wedges,
        labels,
        loc='center left',
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Save figure
    output_path = Path('thesis/figures/fig_5_5_1_2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Generated: {output_path}")
    plt.close()

if __name__ == '__main__':
    generate_fraud_region_pie_chart()


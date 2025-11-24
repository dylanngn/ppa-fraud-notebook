"""
Comprehensive Model Comparison: XGBoost vs Hybrid vs Seon

This module compares ALL fraud detection approaches:
1. Baseline XGBoost (our research model)
2. Hybrid GNN+XGBoost (our advanced research model)  
3. Seon (production baseline we want to beat)

Goal: Determine which approach performs best and whether we beat Seon.
"""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def load_model_results(results_path: str) -> pd.DataFrame:
    """
    Load results CSV for a model.
    
    Args:
        results_path: Path to results CSV file
        
    Returns:
        DataFrame with model results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    # Read with polars then convert to pandas for easier manipulation
    df = pl.read_csv(results_path).to_pandas()
    return df


def generate_comprehensive_comparison(
    window_days: int = 90,
    step_days: int = 14,
    output_dir: str = "artifacts/results/comparison"
):
    """
    Generate comprehensive comparison across all models.
    
    Args:
        window_days: Window size used for evaluation
        step_days: Step size used for evaluation
        output_dir: Output directory for results and plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("XGBoost Baseline vs Hybrid vs Seon")
    print("=" * 80)
    
    # Load results for each model
    results = {}
    model_names = {
        "seon": "Seon (Production)",
        "baseline": "XGBoost Baseline",
        "hybrid_hgt": "Hybrid (GNN+XGBoost)",
    }
    
    print("\n1. Loading model results...")
    for key, name in model_names.items():
        results_path = f"artifacts/results/{key}_baseline_results.csv" if key == "seon" else f"artifacts/results/{key}_results.csv"
        
        try:
            df = load_model_results(results_path)
            results[key] = df
            print(f"   ✓ {name}: {len(df)} windows")
        except FileNotFoundError:
            print(f"   ✗ {name}: Results not found at {results_path}")
            print(f"     Run: python -m src.cli {'evaluate-seon' if key == 'seon' else 'train-' + key}")
    
    if len(results) < 2:
        print("\n⚠️  Need at least 2 models to compare. Run training/evaluation first.")
        return
    
    # Merge results for comparison
    print("\n2. Comparing performance metrics...")
    
    # Prepare comparison DataFrame
    comparison_data = []
    
    for model_key, df in results.items():
        model_name = model_names[model_key]
        
        # Handle different column names
        if model_key == "seon":
            metrics = {
                "precision": df["precision"].mean() if "precision" in df.columns else 0,
                "recall": df["recall"].mean() if "recall" in df.columns else 0,
                "f1_score": df["f1_score"].mean() if "f1_score" in df.columns else 0,
                "fraud_count": df["fraud_count"].sum() if "fraud_count" in df.columns else 0,
            }
        else:
            # XGBoost models use different metric names
            metrics = {
                "precision": df["p@100"].mean() if "p@100" in df.columns else 0,
                "recall": df["r@100"].mean() if "r@100" in df.columns else 0,
                "f1_score": 0,  # Calculate if needed
                "auc_pr": df["auc_pr"].mean() if "auc_pr" in df.columns else 0,
                "fraud_count": df["fraud_count"].sum() if "fraud_count" in df.columns else 0,
            }
        
        comparison_data.append({
            "model": model_name,
            "model_key": model_key,
            **metrics,
            "windows": len(df)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Determine winner
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)
    
    # For Seon, use F1; for XGBoost, use AUC-PR or P@100
    seon_perf = comparison_df[comparison_df["model_key"] == "seon"]["f1_score"].values[0] if "seon" in results else 0
    baseline_perf = comparison_df[comparison_df["model_key"] == "baseline"]["precision"].values[0] if "baseline" in results else 0
    hybrid_perf = comparison_df[comparison_df["model_key"] == "hybrid_hgt"]["precision"].values[0] if "hybrid_hgt" in results else 0
    
    print(f"\nSeon F1 Score: {seon_perf:.4f}")
    print(f"Baseline P@100: {baseline_perf:.4f}")
    if "hybrid_hgt" in results:
        print(f"Hybrid P@100: {hybrid_perf:.4f}")
    
    # Determine best research model
    if "hybrid_hgt" in results and "baseline" in results:
        best_research = "Hybrid" if hybrid_perf > baseline_perf else "Baseline"
        best_research_perf = max(hybrid_perf, baseline_perf)
        print(f"\n🏆 Best Research Model: {best_research} (P@100 = {best_research_perf:.4f})")
        
        if "seon" in results:
            # Note: Direct comparison is tricky due to different metrics
            print(f"\n📊 Comparison with Seon:")
            print(f"   Seon operates with binary decisions (F1 = {seon_perf:.4f})")
            print(f"   Research models provide ranked predictions (P@100 = {best_research_perf:.4f})")
            print(f"   → P@100 shows precision in top 100 riskiest listings")
            print(f"   → Direct comparison requires threshold calibration")
    
    # Save comparison table
    comparison_csv = os.path.join(output_dir, "all_models_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n✓ Comparison table saved to: {comparison_csv}")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    
    # Plot 1: Performance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')
    
    # Precision
    ax = axes[0, 0]
    ax.bar(comparison_df["model"], comparison_df["precision"], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(comparison_df)])
    ax.set_ylabel("Precision")
    ax.set_title("Precision Comparison")
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Recall
    ax = axes[0, 1]
    ax.bar(comparison_df["model"], comparison_df["recall"], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(comparison_df)])
    ax.set_ylabel("Recall")
    ax.set_title("Recall Comparison")
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # F1 Score (for Seon)
    ax = axes[1, 0]
    f1_data = comparison_df[comparison_df["f1_score"] > 0]
    if not f1_data.empty:
        ax.bar(f1_data["model"], f1_data["f1_score"], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(f1_data)])
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score (Binary Decisions)")
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "F1 Score not available\nfor ranking models", 
                ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # AUC-PR (for XGBoost)
    ax = axes[1, 1]
    auc_data = comparison_df[comparison_df.get("auc_pr", 0) > 0]
    if not auc_data.empty:
        ax.bar(auc_data["model"], auc_data["auc_pr"], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(auc_data)])
        ax.set_ylabel("AUC-PR")
        ax.set_title("AUC-PR (Ranking Models)")
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "AUC-PR not available\nfor binary models", 
                ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    comparison_plot = os.path.join(output_dir, "all_models_performance.png")
    plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {comparison_plot}")
    plt.close()
    
    # Plot 2: Performance over time (if multiple windows)
    if all(len(df) > 1 for df in results.values()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Performance Over Time", fontsize=16, fontweight='bold')
        
        for model_key, df in results.items():
            model_name = model_names[model_key]
            
            if "window_idx" in df.columns or "window_start" in df.columns:
                x = df.index if "window_idx" not in df.columns else df["window_idx"]
                
                # Plot precision
                if "precision" in df.columns:
                    axes[0, 0].plot(x, df["precision"], marker='o', label=model_name, linewidth=2)
                elif "p@100" in df.columns:
                    axes[0, 0].plot(x, df["p@100"], marker='o', label=model_name, linewidth=2)
                
                # Plot recall
                if "recall" in df.columns:
                    axes[0, 1].plot(x, df["recall"], marker='o', label=model_name, linewidth=2)
                elif "r@100" in df.columns:
                    axes[0, 1].plot(x, df["r@100"], marker='o', label=model_name, linewidth=2)
                
                # Plot F1
                if "f1_score" in df.columns:
                    axes[1, 0].plot(x, df["f1_score"], marker='o', label=model_name, linewidth=2)
                
                # Plot AUC-PR
                if "auc_pr" in df.columns:
                    axes[1, 1].plot(x, df["auc_pr"], marker='o', label=model_name, linewidth=2)
        
        axes[0, 0].set_title("Precision Over Time")
        axes[0, 0].set_xlabel("Window")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_title("Recall Over Time")
        axes[0, 1].set_xlabel("Window")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_title("F1 Score Over Time")
        axes[1, 0].set_xlabel("Window")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].set_title("AUC-PR Over Time")
        axes[1, 1].set_xlabel("Window")
        axes[1, 1].set_ylabel("AUC-PR")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        trends_plot = os.path.join(output_dir, "all_models_trends.png")
        plt.savefig(trends_plot, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {trends_plot}")
        plt.close()
    
    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if "seon" in results and "baseline" in results:
        print("\n📌 Key Findings:")
        print("   1. Seon is the production baseline (binary approve/reject)")
        print("   2. Research models provide risk rankings (useful for top-K review)")
        print("   3. Different use cases favor different approaches:")
        print("      - Automated decisions → Seon-style binary classification")
        print("      - Manual review prioritization → Research models (P@100)")
        
        if "hybrid_hgt" in results:
            if hybrid_perf > baseline_perf:
                print("\n✅ Hybrid model shows improvement over baseline")
                print("   → GNN embeddings are adding value")
            else:
                print("\n⚠️  Hybrid model does NOT outperform baseline")
                print("   → GNN embeddings not providing additional signal")
                print("   → Recommend SHAP analysis to understand why")
    
    print("\n" + "=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    import typer
    typer.run(generate_comprehensive_comparison)


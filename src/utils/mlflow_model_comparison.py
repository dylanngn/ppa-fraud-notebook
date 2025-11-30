"""
MLflow-Native Model Comparison and Drift Detection

This module provides MLflow-native utilities for:
1. Comparing models (production vs new)
2. Detecting performance drift
3. Making deployment recommendations using Model Registry

Follows MLflow best practices for model lifecycle management.
"""

from typing import Dict, List, Optional, Tuple
import mlflow
from mlflow.tracking import MlflowClient
import typer


def compare_models(
    production_run_id: str,
    candidate_run_id: str,
    primary_metric: str = "auc_pr",
    improvement_threshold: float = 0.01,
) -> Dict:
    """
    Compare a candidate model against production using MLflow runs.
    
    Args:
        production_run_id: MLflow run ID of production model
        candidate_run_id: MLflow run ID of candidate model
        primary_metric: Primary metric to compare (default: auc_pr)
        improvement_threshold: Minimum improvement to recommend deployment
        
    Returns:
        Dict with comparison results and recommendation
    """
    client = MlflowClient()
    
    try:
        prod_run = client.get_run(production_run_id)
        cand_run = client.get_run(candidate_run_id)
        
        # Get metrics
        prod_metrics = prod_run.data.metrics
        cand_metrics = cand_run.data.metrics
        
        # Compare primary metric
        prod_value = prod_metrics.get(primary_metric, 0)
        cand_value = cand_metrics.get(primary_metric, 0)
        
        improvement = cand_value - prod_value
        improvement_pct = (improvement / prod_value * 100) if prod_value > 0 else 0
        
        # Compare all metrics
        metric_comparison = {}
        all_metrics = set(prod_metrics.keys()) | set(cand_metrics.keys())
        
        for metric in all_metrics:
            prod_val = prod_metrics.get(metric, 0)
            cand_val = cand_metrics.get(metric, 0)
            metric_comparison[metric] = {
                "production": prod_val,
                "candidate": cand_val,
                "delta": cand_val - prod_val,
                "delta_pct": ((cand_val - prod_val) / prod_val * 100) if prod_val > 0 else 0,
            }
        
        # Make recommendation
        should_deploy = improvement >= improvement_threshold
        
        recommendation = {
            "should_deploy": should_deploy,
            "reason": (
                f"Candidate improves {primary_metric} by {improvement:.4f} ({improvement_pct:+.2f}%)"
                if should_deploy
                else f"Candidate does not meet improvement threshold ({improvement_threshold:.4f})"
            ),
            "primary_metric": primary_metric,
            "production_value": prod_value,
            "candidate_value": cand_value,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "metric_comparison": metric_comparison,
        }
        
        return recommendation
        
    except Exception as e:
        return {
            "error": str(e),
            "should_deploy": False,
            "reason": f"Comparison failed: {e}",
        }


def compare_with_production(
    model_name: str,
    candidate_run_id: str,
    primary_metric: str = "auc_pr",
    improvement_threshold: float = 0.01,
) -> Dict:
    """
    Compare a candidate model against the current production model in Model Registry.
    
    Args:
        model_name: Registered model name
        candidate_run_id: MLflow run ID of candidate model
        primary_metric: Primary metric to compare
        improvement_threshold: Minimum improvement to recommend deployment
        
    Returns:
        Dict with comparison results and recommendation
    """
    client = MlflowClient()
    
    try:
        # Get production model version
        prod_versions = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )
        
        if not prod_versions:
            return {
                "should_deploy": True,
                "reason": "No production model found - first deployment",
                "primary_metric": primary_metric,
                "production_value": None,
                "candidate_value": None,
            }
        
        prod_version = prod_versions[0]
        prod_run_id = prod_version.run_id
        
        # Compare
        result = compare_models(
            production_run_id=prod_run_id,
            candidate_run_id=candidate_run_id,
            primary_metric=primary_metric,
            improvement_threshold=improvement_threshold,
        )
        
        result["production_version"] = prod_version.version
        result["production_stage"] = "Production"
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "should_deploy": False,
            "reason": f"Comparison failed: {e}",
        }


def get_model_drift_summary(
    model_name: str,
    n_recent_versions: int = 5,
) -> Dict:
    """
    Get drift summary by comparing recent model versions.
    
    Args:
        model_name: Registered model name
        n_recent_versions: Number of recent versions to analyze
        
    Returns:
        Dict with drift summary
    """
    client = MlflowClient()
    
    try:
        # Get all versions
        all_versions = client.search_model_versions(f"name='{model_name}'")
        all_versions.sort(key=lambda v: v.version, reverse=True)
        
        recent_versions = all_versions[:n_recent_versions]
        
        if len(recent_versions) < 2:
            return {
                "drift_detected": False,
                "reason": "Not enough versions for comparison",
                "versions_analyzed": len(recent_versions),
            }
        
        # Get production version
        prod_versions = [v for v in recent_versions if "Production" in v.current_stage]
        if not prod_versions:
            prod_versions = [recent_versions[0]]  # Use latest if no production
        
        prod_version = prod_versions[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_metrics = prod_run.data.metrics
        
        # Compare with other recent versions
        comparisons = []
        for version in recent_versions:
            if version.version == prod_version.version:
                continue
            
            run = client.get_run(version.run_id)
            metrics = run.data.metrics
            
            # Compare key metrics
            comparison = {
                "version": version.version,
                "stage": version.current_stage,
                "run_id": version.run_id,
                "metrics": {},
            }
            
            for metric in ["auc_pr", "mean_auc_pr", "p@100", "mean_p_at_100"]:
                if metric in prod_metrics and metric in metrics:
                    delta = metrics[metric] - prod_metrics[metric]
                    comparison["metrics"][metric] = {
                        "production": prod_metrics[metric],
                        "version": metrics[metric],
                        "delta": delta,
                        "delta_pct": (delta / prod_metrics[metric] * 100) if prod_metrics[metric] > 0 else 0,
                    }
            
            comparisons.append(comparison)
        
        # Detect drift (significant performance degradation)
        drift_detected = False
        drift_reasons = []
        
        for comp in comparisons:
            for metric, values in comp["metrics"].items():
                if values["delta_pct"] < -5.0:  # 5% degradation
                    drift_detected = True
                    drift_reasons.append(
                        f"Version {comp['version']}: {metric} degraded by {abs(values['delta_pct']):.1f}%"
                    )
        
        return {
            "drift_detected": drift_detected,
            "production_version": prod_version.version,
            "production_metrics": prod_metrics,
            "comparisons": comparisons,
            "drift_reasons": drift_reasons if drift_detected else ["No significant drift detected"],
            "versions_analyzed": len(recent_versions),
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "drift_detected": False,
            "reason": f"Drift analysis failed: {e}",
        }


def recommend_deployment(
    model_name: str,
    candidate_run_id: str,
    primary_metric: str = "auc_pr",
    improvement_threshold: float = 0.01,
    min_improvement_pct: float = 1.0,
) -> Dict:
    """
    Make deployment recommendation using MLflow Model Registry best practices.
    
    This function:
    1. Compares candidate with production
    2. Checks if improvement meets threshold
    3. Recommends staging or production deployment
    
    Args:
        model_name: Registered model name
        candidate_run_id: MLflow run ID of candidate model
        primary_metric: Primary metric to evaluate
        improvement_threshold: Absolute improvement threshold
        min_improvement_pct: Minimum percentage improvement
        
    Returns:
        Dict with deployment recommendation
    """
    comparison = compare_with_production(
        model_name=model_name,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
    )
    
    if "error" in comparison:
        return {
            "recommendation": "REJECT",
            "stage": None,
            "reason": comparison["reason"],
            "comparison": comparison,
        }
    
    improvement = comparison.get("improvement", 0)
    improvement_pct = comparison.get("improvement_pct", 0)
    
    # Decision logic
    if improvement >= improvement_threshold and improvement_pct >= min_improvement_pct:
        # Significant improvement - recommend production
        recommendation = "PRODUCTION"
        stage = "Production"
        reason = (
            f"Meets deployment criteria: {improvement:.4f} absolute improvement "
            f"({improvement_pct:+.2f}% relative) on {primary_metric}"
        )
    elif improvement > 0:
        # Small improvement - recommend staging for validation
        recommendation = "STAGING"
        stage = "Staging"
        reason = (
            f"Small improvement ({improvement:.4f}, {improvement_pct:+.2f}%). "
            f"Deploy to Staging for validation before Production."
        )
    else:
        # No improvement or degradation
        recommendation = "REJECT"
        stage = None
        reason = (
            f"No improvement or degradation ({improvement:.4f}, {improvement_pct:+.2f}%). "
            f"Does not meet threshold ({improvement_threshold:.4f})."
        )
    
    return {
        "recommendation": recommendation,
        "stage": stage,
        "reason": reason,
        "comparison": comparison,
    }


app = typer.Typer(help="MLflow Model Comparison and Drift Detection CLI")


@app.command()
def compare(
    model_name: str = typer.Option(..., help="Registered model name"),
    candidate_run_id: str = typer.Option(..., help="Candidate run ID"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to compare"),
    improvement_threshold: float = typer.Option(0.01, help="Minimum improvement threshold"),
):
    """Compare a candidate model with production."""
    result = compare_with_production(
        model_name=model_name,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold
    )
    
    import json
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def drift(
    model_name: str = typer.Option(..., help="Registered model name"),
    n_versions: int = typer.Option(5, help="Number of recent versions to analyze"),
):
    """Analyze model drift across recent versions."""
    result = get_model_drift_summary(
        model_name=model_name,
        n_recent_versions=n_versions
    )
    
    import json
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def recommend(
    model_name: str = typer.Option(..., help="Registered model name"),
    candidate_run_id: str = typer.Option(..., help="Candidate run ID"),
    primary_metric: str = typer.Option("auc_pr", help="Primary metric to compare"),
    improvement_threshold: float = typer.Option(0.01, help="Minimum improvement threshold"),
    min_improvement_pct: float = typer.Option(1.0, help="Minimum percentage improvement"),
):
    """Get deployment recommendation."""
    result = recommend_deployment(
        model_name=model_name,
        candidate_run_id=candidate_run_id,
        primary_metric=primary_metric,
        improvement_threshold=improvement_threshold,
        min_improvement_pct=min_improvement_pct
    )
    
    import json
    typer.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    app()


"""
Baseline XGBoost model training with graph features.
"""
from src.models.training_window import train_accumulating_window
from src.models.feature_engineering import load_data, add_base_tabular_features


def run_baseline():
    """
    Train baseline XGBoost with accumulating window and MLflow tracking.
    
    Always enabled:
    - MLflow tracking
    - Model registration
    - Graph features with temporal filtering (computed on-the-fly per window)
    """
    df = load_data()
    df = add_base_tabular_features(df)
    
    model_name = "baseline_graph"
    result = train_accumulating_window(df, model_name=model_name)
    
    return result


def main():
    """CLI entry point for baseline training."""
    run_baseline()


if __name__ == "__main__":
    main()

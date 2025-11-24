from src.models.train_baseline import run_baseline


def main(window_days: int = 90, step_days: int = 14):
    """
    Graph-feature-enhanced XGBoost baseline.
    """
    return run_baseline(
        window_days=window_days,
        step_days=step_days,
        include_graph_features=True,
        results_filename="artifacts/results/baseline_graph_results.csv",
        save_models=True,  # Save models by default
        models_dir="artifacts/models/baseline_graph"
    )


if __name__ == "__main__":
    main()


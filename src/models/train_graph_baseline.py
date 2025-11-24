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
    )


if __name__ == "__main__":
    main()


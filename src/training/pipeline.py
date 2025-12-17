"""
Training Pipelines for Fraud Detection.

Two pipeline modes:
1. SingleTrainingPipeline: Train once on fixed train/test split
2. ExpandingWindowPipeline: Train across multiple expanding windows (concept drift)

Key Features:
- Point-in-Time labels to prevent label leakage
- Temporal split with gap to prevent feature leakage
- Heterogeneous graph with GNN embeddings
- Support for VANILLA and GNN_XGBOOST variants
"""

import logging
import mlflow
import polars as pl
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import os

from src.features.schema import ModelVariant, FEATURE_SCHEMA
from src.features.store import FeatureStore
from src.features.graph_builder import HeterogeneousGraphBuilder, HeteroGraphConfig
from src.features.temporal_split import (
    TemporalSplitter,
    TemporalSplitConfig,
    ExpandingWindowSplitter,
    LabelPropagation,
    validate_temporal_integrity,
)
from src.models.hybrid import HybridPipeline
from src.models.xgboost_classifier import XGBoostClassifier
from src.models.graphsage import GraphSAGEEmbedder
from src.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def _log(msg: str, level: str = "INFO"):
    """Log to both logger and stdout for visibility in Hydra."""
    getattr(logger, level.lower())(msg)
    print(f"[{level}] {msg}")


class SingleTrainingPipeline:
    """
    Trains a single model on a fixed temporal split.
    
    Design Philosophy:
    - Simple: One model, one train/test split
    - Controlled: Fixed dates for reproducible A/B comparisons
    - Clean: No accumulation loops, no batch drift detection
    
    Usage:
        pipeline = SingleTrainingPipeline(cfg, feature_store)
        metrics = pipeline.run()
        # Compare multiple variants by running with different configs
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        feature_store: FeatureStore,
    ):
        self.cfg = cfg
        self.feature_store = feature_store
        self.variant = ModelVariant(cfg["model"]["variant"])
        
        # Parse date boundaries
        train_start_str = cfg["data"].get("train_start_date")
        self.train_start = (
            datetime.fromisoformat(train_start_str).replace(tzinfo=timezone.utc)
            if train_start_str else None
        )
        self.train_end = datetime.fromisoformat(cfg["data"]["train_end_date"]).replace(tzinfo=timezone.utc)
        self.test_end = datetime.fromisoformat(cfg["data"]["test_end_date"]).replace(tzinfo=timezone.utc)
        self.gap_days = cfg["training"].get("gap_days", 7)
        
    def run(self) -> Dict[str, float]:
        """
        Execute training and evaluation.
        
        Returns:
            Dict[str, float]: Model metrics including 'auc_pr'.
        """
        _log(f"Starting Single Training for {self.variant.value}")
        _log(f"Train: {self.train_start} → {self.train_end}, Test end: {self.test_end}")
        
        # 1. Load Data
        df = self.feature_store.load_data().collect()
        _log(f"Loaded {len(df)} total records")
        
        # Filter by train_start if specified (exclude incomplete early data)
        time_col = FEATURE_SCHEMA.time_column
        if self.train_start is not None:
            df = df.filter(pl.col(time_col) >= self.train_start)
            _log(f"Filtered to {len(df)} records (after {self.train_start.date()})")
        
        # 2. Split Data using temporal splitter with Point-in-Time labels
        split_config = TemporalSplitConfig(
            time_column=time_col,
            insertion_column=FEATURE_SCHEMA.temporal_config.insertion_id_column,
            label_column=FEATURE_SCHEMA.target,
            fraud_flag_column=FEATURE_SCHEMA.raw_target_source,  # FLAGGEDFORFRAUD
            gap_days=self.gap_days,
            label_propagation=LabelPropagation.POINT_IN_TIME,  # Prevents label leakage
        )
        splitter = TemporalSplitter(split_config)
        train_df, test_df = splitter.split(df, self.train_end, self.test_end)
        
        # Validate temporal integrity
        validate_temporal_integrity(train_df, test_df, time_col)
        
        _log(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        if len(train_df) == 0 or len(test_df) == 0:
            _log("Empty train or test set!")
            return {}
        
        # 3. Build Pipeline
        pipeline = self._build_pipeline()
        
        # 4. Build Graphs (if needed)
        train_graph = None
        inference_graph = None
        
        if self.variant == ModelVariant.GNN_XGBOOST:
            _log("Building Train Graph...")
            train_graph = pipeline.graph_builder.build_graph(train_df, cutoff_date=self.train_end)
            _log(f"Train Graph: {train_graph['listing'].num_nodes} listing nodes")
            
            # Inference graph includes train + test (safe: GNN frozen, edges temporal)
            inference_df = df.filter(pl.col(time_col) <= self.test_end)
            _log("Building Inference Graph...")
            inference_graph = pipeline.graph_builder.build_graph(inference_df, cutoff_date=self.test_end)
            _log(f"Inference Graph: {inference_graph['listing'].num_nodes} listing nodes")
        
        # 5. Fit Pipeline
        _log("Fitting model...")
        pipeline.fit(train_df, train_graph)
        
        # 6. Predict
        _log("Generating predictions...")
        
        if self.variant == ModelVariant.GNN_XGBOOST:
            inference_df = df.filter(pl.col(time_col) <= self.test_end)
            all_probs = pipeline.predict(inference_df, inference_graph)
            test_start = self.train_end + timedelta(days=self.gap_days)
            test_mask = (inference_df[time_col] > test_start) & (inference_df[time_col] <= self.test_end)
            probs = all_probs[test_mask.to_numpy()]
        else:
            probs = pipeline.predict(test_df, None)
        
        labels = test_df[FEATURE_SCHEMA.target].to_numpy()
        
        # 7. Calculate Metrics
        metrics = calculate_metrics(labels, probs, include_confusion_matrix=True)
        _log(f"Model Metrics: {metrics}")
        
        # Log to MLflow
        mlflow.log_metrics(metrics)
        
        if "benchmark_seon_approved" in test_df.columns:
            approved = test_df["benchmark_seon_approved"].to_numpy()
            seon_preds = 1 - approved
            
            seon_full = calculate_metrics(labels, seon_preds, include_confusion_matrix=True)
            seon_metrics = {
                "seon_precision": seon_full["precision"],
                "seon_recall": seon_full["recall"],
                "seon_f1": seon_full["f1_score"],
            }
            _log(f"Seon Baseline: {seon_metrics}")
            mlflow.log_metrics(seon_metrics)
        
        self._log_model(pipeline)
        
        return metrics
    
    def _build_pipeline(self) -> HybridPipeline:
        """Instantiate pipeline components based on variant."""
        xgb_params = self.cfg["model"]["xgboost"]
        classifier = XGBoostClassifier(**xgb_params)
        
        graph_builder = None
        embedder = None
        
        if self.variant == ModelVariant.GNN_XGBOOST:
            # Build heterogeneous graph with multiple node/edge types
            graph_config = HeteroGraphConfig(
                primary_id_column=FEATURE_SCHEMA.temporal_config.insertion_id_column,
                time_column=FEATURE_SCHEMA.time_column,
                temporal_identity_columns=list(FEATURE_SCHEMA.graph_config.high_signal_columns),
            )
            graph_builder = HeterogeneousGraphBuilder(graph_config)
            
            # GNN embedder for heterogeneous graph
            gnn_params = self.cfg["model"].get("gnn", {})
            embedder = GraphSAGEEmbedder(
                in_channels=len(FEATURE_SCHEMA.get_gnn_input_features()),
                hidden_channels=gnn_params.get("hidden_dim", FEATURE_SCHEMA.gnn_hidden_dim),
                out_channels=gnn_params.get("output_dim", FEATURE_SCHEMA.gnn_embedding_dim),
                num_layers=gnn_params.get("num_layers", FEATURE_SCHEMA.gnn_num_layers),
            )
            
        return HybridPipeline(
            variant=self.variant,
            graph_builder=graph_builder,
            embedder=embedder,
            classifier=classifier,
        )
    
    def _log_model(self, pipeline: HybridPipeline):
        """Log trained model to MLflow."""
        try:
            from mlflow.models import infer_signature
            
            xgb_native = pipeline.classifier.model
            input_example = pipeline.input_example_
            
            signature = None
            if input_example is not None:
                signature = infer_signature(
                    input_example, 
                    xgb_native.predict_proba(input_example)[:, 1]
                )
            
            req_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "requirements.txt"
            )
            
            mlflow.xgboost.log_model(
                xgb_native,
                name="model",
                signature=signature,
                pip_requirements=req_path
            )
            
            # Log GNN state if exists
            if pipeline.embedder:
                os.makedirs("tmp", exist_ok=True)
                path = "tmp/gnn_state.pt"
                pipeline.embedder.save(path)
                mlflow.log_artifact(path, artifact_path="gnn_model")
                os.remove(path)
                
            _log("Model artifacts logged to MLflow")
            
        except Exception as e:
            _log(f"Failed to log model artifacts: {e}")


class ExpandingWindowPipeline:
    """
    Trains and evaluates across multiple expanding time windows.
    
    Use case: Validate model stability and detect concept drift over time (RQ3).
    
    How it works:
    - Window 1: Train [start → T1], Test [T1+gap → T2]
    - Window 2: Train [start → T2], Test [T2+gap → T3]
    - ...each window expands training data
    
    Returns per-window metrics + aggregate statistics.
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        feature_store: FeatureStore,
    ):
        self.cfg = cfg
        self.feature_store = feature_store
        self.variant = ModelVariant(cfg["model"]["variant"])
        
        # Window configuration
        expanding_cfg = cfg.get("expanding_window", {})
        self.window_days = expanding_cfg.get("window_days", 30)
        self.min_train_windows = expanding_cfg.get("min_train_windows", 2)
        
        # Date boundaries
        self.start_date = datetime.fromisoformat(
            cfg["data"].get("train_start_date", "2025-01-01")
        ).replace(tzinfo=timezone.utc)
        self.end_date = datetime.fromisoformat(
            cfg["data"]["test_end_date"]
        ).replace(tzinfo=timezone.utc)
        self.gap_days = cfg["training"].get("gap_days", 7)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute expanding window training and evaluation.
        
        Returns:
            Dict with per-window metrics and aggregate statistics.
        """
        from src.features.temporal_split import ExpandingWindowSplitter
        
        _log(f"Starting Expanding Window Training for {self.variant.value}")
        _log(f"Window size: {self.window_days} days, Gap: {self.gap_days} days")
        _log(f"Date range: {self.start_date.date()} → {self.end_date.date()}")
        
        # Load data once
        df = self.feature_store.load_data().collect()
        time_col = FEATURE_SCHEMA.time_column
        
        # Filter by start date
        df = df.filter(pl.col(time_col) >= self.start_date)
        _log(f"Loaded {len(df)} records")
        
        # Generate expanding window splits
        split_config = TemporalSplitConfig(
            time_column=time_col,
            insertion_column=FEATURE_SCHEMA.temporal_config.insertion_id_column,
            label_column=FEATURE_SCHEMA.target,
            fraud_flag_column=FEATURE_SCHEMA.raw_target_source,
            gap_days=self.gap_days,
            label_propagation=LabelPropagation.POINT_IN_TIME,
        )
        
        splitter = ExpandingWindowSplitter(split_config, self.window_days)
        splits = splitter.generate_splits(
            df, self.start_date, self.end_date, self.min_train_windows
        )
        
        if not splits:
            _log("No valid splits generated!")
            return {"error": "No valid splits"}
        
        _log(f"Generated {len(splits)} expanding window splits")
        
        # Train and evaluate each window
        all_metrics = []
        
        for i, (train_df, test_df, train_end, test_end) in enumerate(splits):
            window_num = i + 1
            _log(f"\n=== Window {window_num}/{len(splits)} ===")
            _log(f"Train end: {train_end.date()}, Test end: {test_end.date()}")
            _log(f"Train: {len(train_df)}, Test: {len(test_df)}")
            
            # Validate temporal integrity
            validate_temporal_integrity(train_df, test_df, time_col)
            
            # Build fresh pipeline for this window
            pipeline = self._build_pipeline()
            
            # Build graphs if needed
            train_graph = None
            inference_graph = None
            
            if self.variant == ModelVariant.GNN_XGBOOST:
                train_graph = pipeline.graph_builder.build_graph(train_df, cutoff_date=train_end)
                
                # Inference includes test period
                inference_df = df.filter(pl.col(time_col) <= test_end)
                inference_graph = pipeline.graph_builder.build_graph(inference_df, cutoff_date=test_end)
            
            # Fit
            pipeline.fit(train_df, train_graph)
            
            # Predict
            if self.variant == ModelVariant.GNN_XGBOOST:
                inference_df = df.filter(pl.col(time_col) <= test_end)
                all_probs = pipeline.predict(inference_df, inference_graph)
                test_start = train_end + timedelta(days=self.gap_days)
                test_mask = (inference_df[time_col] > test_start) & (inference_df[time_col] <= test_end)
                probs = all_probs[test_mask.to_numpy()]
            else:
                probs = pipeline.predict(test_df, None)
            
            labels = test_df[FEATURE_SCHEMA.target].to_numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(labels, probs, include_confusion_matrix=False)
            metrics["window"] = window_num
            metrics["train_end"] = train_end.isoformat()
            metrics["test_end"] = test_end.isoformat()
            metrics["train_size"] = len(train_df)
            metrics["test_size"] = len(test_df)
            
            all_metrics.append(metrics)
            _log(f"Window {window_num} AUC-PR: {metrics['auc_pr']:.4f}")
            
            # Log to MLflow with window prefix
            mlflow.log_metrics({f"w{window_num}_{k}": v for k, v in metrics.items() 
                               if isinstance(v, (int, float))})
        
        # Aggregate statistics
        auc_prs = [m["auc_pr"] for m in all_metrics]
        auc_rocs = [m["auc_roc"] for m in all_metrics]
        
        aggregate = {
            "mean_auc_pr": np.mean(auc_prs),
            "std_auc_pr": np.std(auc_prs),
            "min_auc_pr": np.min(auc_prs),
            "max_auc_pr": np.max(auc_prs),
            "mean_auc_roc": np.mean(auc_rocs),
            "std_auc_roc": np.std(auc_rocs),
            "num_windows": len(splits),
            "drift_range": np.max(auc_prs) - np.min(auc_prs),
        }
        
        mlflow.log_metrics(aggregate)
        
        _log(f"\n=== Aggregate Results ===")
        _log(f"AUC-PR: {aggregate['mean_auc_pr']:.4f} ± {aggregate['std_auc_pr']:.4f}")
        _log(f"AUC-ROC: {aggregate['mean_auc_roc']:.4f} ± {aggregate['std_auc_roc']:.4f}")
        _log(f"Drift Range: {aggregate['drift_range']:.4f}")
        
        return {
            "windows": all_metrics,
            "aggregate": aggregate,
        }
    
    def _build_pipeline(self) -> HybridPipeline:
        """Instantiate pipeline components based on variant."""
        xgb_params = self.cfg["model"]["xgboost"]
        classifier = XGBoostClassifier(**xgb_params)
        
        graph_builder = None
        embedder = None
        
        if self.variant == ModelVariant.GNN_XGBOOST:
            graph_config = HeteroGraphConfig(
                primary_id_column=FEATURE_SCHEMA.temporal_config.insertion_id_column,
                time_column=FEATURE_SCHEMA.time_column,
                temporal_identity_columns=list(FEATURE_SCHEMA.graph_config.high_signal_columns),
            )
            graph_builder = HeterogeneousGraphBuilder(graph_config)
            
            gnn_params = self.cfg["model"].get("gnn", {})
            embedder = GraphSAGEEmbedder(
                in_channels=len(FEATURE_SCHEMA.get_gnn_input_features()),
                hidden_channels=gnn_params.get("hidden_dim", FEATURE_SCHEMA.gnn_hidden_dim),
                out_channels=gnn_params.get("output_dim", FEATURE_SCHEMA.gnn_embedding_dim),
                num_layers=gnn_params.get("num_layers", FEATURE_SCHEMA.gnn_num_layers),
            )
            
        return HybridPipeline(
            variant=self.variant,
            graph_builder=graph_builder,
            embedder=embedder,
            classifier=classifier,
        )


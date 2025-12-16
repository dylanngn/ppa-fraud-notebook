"""
Single Training Pipeline.

Trains ONE model on a fixed train/test split.
Simple, controlled, and suitable for A/B comparisons.
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
from src.features.graph_builder import TemporalGraphBuilder, GraphConfig
from src.features.graph_features import HandcraftedGraphFeatures
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
        _log(f"Train end: {self.train_end}, Test end: {self.test_end}")
        
        # 1. Load Data
        df = self.feature_store.load_data().collect()
        _log(f"Loaded {len(df)} total records")
        
        # 2. Split Data
        # Train: All data up to train_end
        train_df = df.filter(pl.col("submission_at") <= self.train_end)
        
        # Test: Data after gap, up to test_end
        test_start = self.train_end + timedelta(days=self.gap_days)
        test_df = df.filter(
            (pl.col("submission_at") > test_start) & 
            (pl.col("submission_at") <= self.test_end)
        )
        
        _log(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        if len(train_df) == 0 or len(test_df) == 0:
            _log("Empty train or test set!")
            return {}
        
        # 3. Build Pipeline
        pipeline = self._build_pipeline()
        
        # 4. Build Graphs (if needed)
        train_graph = None
        inference_graph = None
        
        if self.variant != ModelVariant.VANILLA_XGBOOST:
            _log("Building Train Graph...")
            train_graph = pipeline.graph_builder.build_graph(train_df, cutoff_date=self.train_end)
            
            train_timestamps = train_df["submission_at"].to_list()
            pipeline.graph_builder.validate_no_leakage(train_graph, train_timestamps)
            _log(f"Train Graph: {train_graph['listing'].num_nodes} nodes, validated")
            
            # Inference graph includes train + test (safe: GNN frozen, edges temporal)
            inference_df = df.filter(pl.col("submission_at") <= self.test_end)
            _log("Building Inference Graph...")
            inference_graph = pipeline.graph_builder.build_graph(inference_df, cutoff_date=self.test_end)
            
            inference_timestamps = inference_df["submission_at"].to_list()
            pipeline.graph_builder.validate_no_leakage(inference_graph, inference_timestamps)
            _log(f"Inference Graph: {inference_graph['listing'].num_nodes} nodes, validated")
        
        # 5. Fit Pipeline
        _log("Fitting model...")
        pipeline.fit(train_df, train_graph)
        
        # 6. Predict
        _log("Generating predictions...")
        
        if self.variant != ModelVariant.VANILLA_XGBOOST:
            inference_df = df.filter(pl.col("submission_at") <= self.test_end)
            all_probs = pipeline.predict(inference_df, inference_graph)
            test_mask = (inference_df["submission_at"] > test_start) & (inference_df["submission_at"] <= self.test_end)
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
        handcrafted = None
        
        if self.variant != ModelVariant.VANILLA_XGBOOST:
            graph_config = GraphConfig(
                identity_columns=FEATURE_SCHEMA.graph_identity_columns,
                directed=True
            )
            graph_builder = TemporalGraphBuilder(graph_config)
        
        if self.variant == ModelVariant.GRAPHSAGE_XGBOOST:
            gnn_params = self.cfg["model"]["gnn"]
            embedder = GraphSAGEEmbedder(**gnn_params)
            
        if self.variant == ModelVariant.HANDCRAFTED_XGBOOST:
            handcrafted = HandcraftedGraphFeatures()
            
        return HybridPipeline(
            variant=self.variant,
            graph_builder=graph_builder,
            embedder=embedder,
            classifier=classifier,
            handcrafted_features=handcrafted
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


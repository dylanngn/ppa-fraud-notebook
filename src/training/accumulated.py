"""
Accumulated Training Pipeline.
Iterates over temporal splits, trains models on expanding history, and evaluates on future data.
"""

import logging
import mlflow
import polars as pl
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta
import os
import shutil
import numpy as np

from src.data.schema import ModelVariant, DataSplit, FEATURE_SCHEMA
from src.data.temporal_split import TemporalSplitter, AccumulatedTrainingConfig
from src.data.feature_store import FeatureStore
from src.graph.builder import TemporalGraphBuilder, GraphConfig
from src.graph.features import HandcraftedGraphFeatures
from src.models.hybrid import HybridPipeline
from src.models.xgboost_classifier import XGBoostClassifier
from src.models.graphsage import GraphSAGEEmbedder
from src.utils.metrics import calculate_metrics
from src.drift.detector import DriftDetector

logger = logging.getLogger(__name__)


class AccumulatedTrainingPipeline:
    """
    Orchestrates the accumulated training process.
    """
    
    def __init__(
        self,
        cfg: Dict[str, Any],
        feature_store: FeatureStore,
    ):
        self.cfg = cfg
        self.feature_store = feature_store
        
        # Parse configs
        self.training_config = AccumulatedTrainingConfig(
            data_start_date=cfg["data"]["start_date"],
            data_end_date=cfg["data"]["end_date"],
            initial_train_months=cfg["training"]["initial_train_months"],
            accumulation_frequency=cfg["training"]["accumulation_frequency"],
            prediction_window_days=cfg["training"]["prediction_window_days"],
        )
        
        self.variant = ModelVariant(cfg["model"]["variant"])
        self.drift_detector = DriftDetector() # Defaults to ADWIN
        
    def run(self):
        """Execute the training loop."""
        logger.info(f"Starting Accumulated Training for {self.variant.value}")
        
        # 1. Load Data
        # We collect() here to work with Eager Polars DataFrame in memory.
        # This allows conversion to Pandas (needed for XGBoost) and simplifies slicing.
        df = self.feature_store.load_data().collect()
        
        # 2. Setup Splitter
        splitter = TemporalSplitter(self.training_config)
        
        # 3. Iterate Splits
        accum_results: List[Dict] = []
        
        # Track best model for parent run logging
        best_model = None
        best_input_example = None
        best_auc_pr = -1.0
        best_split_idx = -1
        
        for i, split in enumerate(splitter.generate_accumulated_splits()):
            logger.info(f"Processing Split {i}: Train End {split.train.cutoff_date}")
            
            with mlflow.start_run(run_name=f"Split_{i}_{split.train.accumulation_id}", nested=True):
                # Log Split params
                log_params = {
                    "split_idx": i,
                    "train_cutoff": split.train.cutoff_date,
                    "val_cutoff": split.val.cutoff_date,
                    "test_cutoff": split.test.cutoff_date,
                    "variant": self.variant.value,
                }
                mlflow.log_params(log_params)
                
                # 4. Prepare Components for this iteration
                pipeline = self._build_pipeline()
                
                # 5. Build Graphs
                # STRICT ADHERENCE TO GNN TIMING GUIDE:
                # 1. Train Graph: Strictly Training Data (No Future Leakage)
                # 2. Inference Graph: Training + Test Data (Inductive Inference)
                
                # A. Prepare Dataframes
                # Ensure UTC to match Polars Schema (datetime[μs, UTC])
                train_cutoff = datetime.fromisoformat(split.train.cutoff_date).replace(tzinfo=timezone.utc)
                test_cutoff = datetime.fromisoformat(split.test.cutoff_date).replace(tzinfo=timezone.utc)
                
                # Train DF (Strictly <= Train Cutoff)
                train_df = df.filter(pl.col("submission_at") <= train_cutoff)
                
                # Inference Window DF (Data up to Test Cutoff)
                # We need full history to connect Test nodes back to Train nodes
                inference_df = df.filter(pl.col("submission_at") <= test_cutoff)
                
                # B. Build Train Graph (for GNN Training)
                train_graph = None
                if self.variant != ModelVariant.VANILLA_XGBOOST:
                    logger.info("Building Train Graph (Strict)...")
                    train_graph = pipeline.graph_builder.build_graph(
                        train_df, 
                        cutoff_date=train_cutoff
                    )

                # 6. Fit Pipeline (on Train)
                logger.info("Fitting Pipeline (GNN acts on Train Graph)...")
                pipeline.fit(train_df, train_graph, split)
                
                # 7. Evaluate (on Test)
                # We need to generate predictions for Test nodes using the Inductive assumption.
                # Pipeline.predict expects a DF and a Graph aligned with it.
                
                inference_graph = None
                if self.variant != ModelVariant.VANILLA_XGBOOST:
                    logger.info("Building Inference Graph (Train + Test)...")
                    inference_graph = pipeline.graph_builder.build_graph(
                        inference_df,
                        cutoff_date=test_cutoff
                    )
                
                logger.info("Generating Predictions (Inductive Inference)...")
                # Predict on the FULL inference window to ensure embeddings are generated correctly
                # (Test nodes need Train neighbors)
                all_probs = pipeline.predict(inference_df, inference_graph)
                
                # Filter predictions to just the Test Set
                # We can join or filter by timestamp
                # Calculate Test Start
                val_end = datetime.fromisoformat(split.val.cutoff_date).replace(tzinfo=timezone.utc)
                test_start = val_end + timedelta(days=split.gap_days)
                
                # Create mask for Test rows in inference_df
                # Polars masking on the dataframe implies we need to align 'all_probs' (numpy) with 'inference_df' rows.
                # Assuming 1:1 row correspondence maintained.
                
                test_mask = (inference_df["submission_at"] > test_start) & (inference_df["submission_at"] <= test_cutoff)
                test_mask_np = test_mask.to_numpy()
                
                probs = all_probs[test_mask_np]
                labels = inference_df.filter(test_mask)[FEATURE_SCHEMA.target].to_numpy()
                
                # Start Validation logic
                test_df = inference_df.filter(test_mask)
                
                metrics = calculate_metrics(labels, probs)
                logger.info(f"Split {i} Model Metrics: {metrics}")
                mlflow.log_metrics(metrics)
                
                # Baseline Evaluation (Seon)
                if "benchmark_seon_approved" in test_df.columns:
                    # Seon Approved: 1 = Legit, 0 = Rejected (Fraud)
                    # We need Fraud Probabilities (1 = Fraud)
                    # So Pred = 1 - Approved
                    approved = test_df["benchmark_seon_approved"].to_numpy()
                    baseline_preds = 1 - approved
                    
                    baseline_metrics = calculate_metrics(labels, baseline_preds)
                    # Prefix keys
                    baseline_metrics = {f"baseline_{k}": v for k, v in baseline_metrics.items()}
                    logger.info(f"Split {i} Baseline Metrics (Binary): {baseline_metrics}")
                    mlflow.log_metrics(baseline_metrics)
                
                # Drift Detection
                # Monitor AUC-PR stability
                # ADWIN expects a scalar (usually bounded 0-1)
                current_perf = metrics["auc_pr"]
                if self.drift_detector.update(current_perf):
                    logger.warning(f"DRIFT DETECTED at Split {i}! Performance changed significantly.")
                    mlflow.log_event(f"Drift Detected at Split {i}")
                
                accum_results.append(metrics)
                
                # Track best model for parent run (based on AUC-PR)
                current_auc_pr = metrics.get("auc_pr", 0)
                if current_auc_pr > best_auc_pr:
                    best_auc_pr = current_auc_pr
                    best_split_idx = i
                    best_model = pipeline.classifier.model  # XGBoost native model
                    best_input_example = pipeline.input_example_
                
                # Save Model (Log Artifacts to MLflow - per split for traceability)
                # We log the XGBoost model as the primary inference artifact
                # For GNN, we should log the weights as an extra artifact
                
                try:
                    import pandas as pd
                    from mlflow.models import infer_signature
                    
                    # Log the XGBoost Classifier with proper signature
                    xgb_native = pipeline.classifier.model 
                    
                    # Use input example stored during fit (works for all variants)
                    input_example = pipeline.input_example_
                    signature = None
                    if input_example is not None:
                        # Infer signature from training sample
                        signature = infer_signature(input_example, xgb_native.predict_proba(input_example)[:, 1])

                    # Requirements path
                    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "requirements.txt")

                    # Note: input_example is omitted because MLflow JSON serialization
                    # loses pandas categorical dtype, causing XGBoost validation errors.
                    # The signature alone is sufficient for schema documentation.
                    mlflow.xgboost.log_model(
                        xgb_native, 
                        name="xgboost_model",
                        signature=signature,
                        pip_requirements=req_path
                    )
                    
                    # Log GNN Embedder state if exists
                    if pipeline.embedder:
                        path = f"tmp/gnn_state_{i}.pt"
                        pipeline.embedder.save(path)
                        mlflow.log_artifact(path, artifact_path="gnn_model")
                        os.remove(path)
                        
                except Exception as e:
                    logger.warning(f"Failed to log model artifacts: {e}")

                
        # 8. Log Final Model to Parent Run
        # The FINAL model is what gets deployed - log its metrics to parent
        if accum_results:
            # Use the best model (highest AUC-PR)
            final_metrics = accum_results[best_split_idx]
            final_metrics_labeled = {f"final_{k}": v for k, v in final_metrics.items()}
            mlflow.log_metrics(final_metrics_labeled)
            
            # Log metadata about the final model
            mlflow.log_params({
                "final_model_split_idx": best_split_idx,
                "total_splits": len(accum_results),
            })
            
            logger.info(f"Final Model Metrics: {final_metrics}")
            
            # Log the FINAL model to parent run for easy deployment access
            # (Child runs have per-split models for traceability)
            if best_model is not None:
                try:
                    import pandas as pd
                    from mlflow.models import infer_signature
                    
                    signature = None
                    if best_input_example is not None:
                        signature = infer_signature(
                            best_input_example, 
                            best_model.predict_proba(best_input_example)[:, 1]
                        )
                    
                    req_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                        "requirements.txt"
                    )
                    
                    mlflow.xgboost.log_model(
                        best_model,
                        name="final_model",
                        signature=signature,
                        pip_requirements=req_path
                    )
                    logger.info(f"Logged final model to parent run (from split {best_split_idx}, AUC-PR: {best_auc_pr:.4f})")
                except Exception as e:
                    logger.warning(f"Failed to log final model to parent run: {e}")
            
    def _build_pipeline(self) -> HybridPipeline:
        """Instantiate pipeline for the window."""
        # Config params
        xgb_params = self.cfg["model"]["xgboost"]
        
        # Fresh classifier for each split
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

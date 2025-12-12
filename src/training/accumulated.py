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

logger = logging.getLogger(__name__)

from src.drift.detector import DriftDetector

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
        
        for i, split in enumerate(splitter.generate_accumulated_splits()):
            logger.info(f"Processing Split {i}: Train End {split.train.cutoff_date}")
            
            with mlflow.start_run(run_name=f"Split_{i}_{split.train.accumulation_id}", nested=True):
                # Log Split params
                mlflow.log_params({
                    "split_idx": i,
                    "train_cutoff": split.train.cutoff_date,
                    "val_cutoff": split.val.cutoff_date,
                    "test_cutoff": split.test.cutoff_date,
                    "variant": self.variant.value
                })
                
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
                
                # Save Model (Log Artifacts to MLflow)
                # We log the XGBoost model as the primary inference artifact
                # For GNN, we should log the weights as an extra artifact
                
                try:
                    import pandas as pd
                    from mlflow.models import infer_signature
                    
                    # To be safe, we log the XGBoost Classifier object directly
                    xgb_native = pipeline.classifier.model 
                    
                    # We need an input example that matches X_test structure (Base + Embeddings)
                    # We don't have X_test explicitly here (it's inside predict).
                    # Refactor: We can't easily get X_test from outside without modifying predict to return it.
                    # Workaround: Log without signature OR modify pipeline.predict to return data?
                    # Better: Log the *Pipeline* as a pyfunc?
                    # For now: Log XGBoost model without signature if input is complex, 
                    # OR attempt to inspect pipeline.classifier (it might have feature_names_in_)
                    
                    # RE-LOG with signature if possible (cleaner code structure):
                    signature = None
                    input_example = None

                    # Requirements path
                    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "requirements.txt")

                    mlflow.xgboost.log_model(
                        xgb_native, 
                        name="xgboost_model",
                        signature=signature,
                        input_example=input_example,
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

                
        # 8. Aggregated Results
        if accum_results:
            avg_metrics = {k: float(np.mean([r[k] for r in accum_results])) for k in accum_results[0]}
            logger.info(f"Average Metrics: {avg_metrics}")
            mlflow.log_metrics(avg_metrics) # In parent run?
            
    def _build_pipeline(self) -> HybridPipeline:
        """Instantiate a fresh pipeline for the window."""
        
        # Config params
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

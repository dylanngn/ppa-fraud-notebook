"""
SHAP Analysis for Fraud Detection Models.

Provides:
1. Global explanations: Feature importance across all predictions
2. Local explanations: Why a specific listing was flagged
3. Temporal analysis: Feature importance changes over time windows

Usage:
    # After training
    analyzer = SHAPAnalyzer(model, X_train, feature_names)
    global_importance = analyzer.compute_global_importance(X_test)
    
    # For specific predictions
    local_explanation = analyzer.explain_prediction(X_test.iloc[0])
    
    # For temporal drift analysis
    drift_report = analyzer.compare_windows(X_window1, X_window2)
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SHAPConfig:
    """Configuration for SHAP analysis."""
    max_samples: int = 1000  # Max samples for SHAP computation (for speed)
    top_k_features: int = 20  # Top features to display
    output_dir: str = "artifacts/shap"
    figure_dpi: int = 150


class SHAPAnalyzer:
    """
    SHAP-based explainability for fraud detection models.
    
    Supports XGBoost models (including those from HybridPipeline).
    """
    
    def __init__(
        self,
        model: xgb.XGBClassifier,
        background_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[SHAPConfig] = None,
    ):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained XGBoost classifier
            background_data: Training data for SHAP baseline (optional for TreeExplainer)
            feature_names: Feature names for labeling
            config: Analysis configuration
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.config = config or SHAPConfig()
        
        # Create explainer
        self.explainer = shap.TreeExplainer(model)
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("SHAP analyzer initialized")
    
    def _prepare_for_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for SHAP computation.
        
        Converts object columns to appropriate types for XGBoost/SHAP.
        """
        X = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric first
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception:
                    pass
                
                # If still object, convert to categorical
                if X[col].dtype == 'object':
                    X[col] = X[col].astype('category')
            
            # Handle nullable integers
            elif pd.api.types.is_integer_dtype(X[col]):
                X[col] = X[col].astype('float64')
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        
        return X
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Feature DataFrame
            max_samples: Maximum samples to use (for speed)
            
        Returns:
            Tuple of (shap_values array, sampled DataFrame)
        """
        max_samples = max_samples or self.config.max_samples
        
        # Sample if needed
        if len(X) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(X)} samples for SHAP")
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X.copy()
        
        # Prepare for SHAP (handle object types)
        X_sample = self._prepare_for_shap(X_sample)
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification (returns list of 2 arrays)
        if isinstance(shap_values, list):
            # Use positive class (fraud) SHAP values
            shap_values = shap_values[1]
        
        return shap_values, X_sample
    
    def compute_global_importance(
        self,
        X: pd.DataFrame,
        save_plot: bool = True,
        plot_name: str = "global_importance",
    ) -> pd.DataFrame:
        """
        Compute global feature importance using mean absolute SHAP values.
        
        Args:
            X: Feature DataFrame (test data recommended)
            save_plot: Whether to save visualization
            plot_name: Base name for saved plots
            
        Returns:
            DataFrame with feature importance rankings
        """
        shap_values, X_sample = self.compute_shap_values(X)
        
        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": mean_abs_shap,
            "mean_shap": shap_values.mean(axis=0),  # Direction (positive = fraud)
            "std_shap": shap_values.std(axis=0),
        })
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df = importance_df.reset_index(drop=True)
        
        logger.info(f"Top 5 features: {importance_df['feature'].head().tolist()}")
        
        if save_plot:
            self._save_global_plots(shap_values, X_sample, plot_name)
        
        return importance_df
    
    def _save_global_plots(
        self,
        shap_values: np.ndarray,
        X_sample: pd.DataFrame,
        plot_name: str,
    ):
        """Save global importance visualizations."""
        output_dir = Path(self.config.output_dir)
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            max_display=self.config.top_k_features,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name}_summary.png", dpi=self.config.figure_dpi)
        plt.close()
        logger.info(f"Saved: {output_dir / f'{plot_name}_summary.png'}")
        
        # Bar plot (mean absolute SHAP)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_sample,
            plot_type="bar",
            max_display=self.config.top_k_features,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{plot_name}_bar.png", dpi=self.config.figure_dpi)
        plt.close()
        logger.info(f"Saved: {output_dir / f'{plot_name}_bar.png'}")
    
    def explain_prediction(
        self,
        instance: pd.Series,
        save_plot: bool = True,
        instance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            instance: Single row of features
            save_plot: Whether to save waterfall plot
            instance_id: Identifier for naming saved plot
            
        Returns:
            Dictionary with local explanation details
        """
        # Convert to DataFrame if needed
        if isinstance(instance, pd.Series):
            X_instance = instance.to_frame().T
        else:
            X_instance = instance.copy()
        
        # Prepare for SHAP
        X_instance = self._prepare_for_shap(X_instance)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_vals = shap_values[0]
        
        # Get expected value (base prediction)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Create explanation details
        feature_contributions = pd.DataFrame({
            "feature": X_instance.columns,
            "value": X_instance.iloc[0].values,
            "shap_value": shap_vals,
        })
        feature_contributions["abs_shap"] = feature_contributions["shap_value"].abs()
        feature_contributions = feature_contributions.sort_values("abs_shap", ascending=False)
        
        # Top contributors
        top_positive = feature_contributions[
            feature_contributions["shap_value"] > 0
        ].head(5)
        top_negative = feature_contributions[
            feature_contributions["shap_value"] < 0
        ].head(5)
        
        explanation = {
            "base_value": float(expected_value),
            "prediction_value": float(expected_value + shap_vals.sum()),
            "top_fraud_indicators": top_positive[["feature", "value", "shap_value"]].to_dict("records"),
            "top_legitimate_indicators": top_negative[["feature", "value", "shap_value"]].to_dict("records"),
            "all_contributions": feature_contributions.to_dict("records"),
        }
        
        if save_plot:
            self._save_local_plot(X_instance, shap_vals, expected_value, instance_id)
        
        return explanation
    
    def _save_local_plot(
        self,
        X_instance: pd.DataFrame,
        shap_vals: np.ndarray,
        expected_value: float,
        instance_id: Optional[str],
    ):
        """Save waterfall plot for local explanation."""
        output_dir = Path(self.config.output_dir)
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=expected_value,
            data=X_instance.iloc[0].values,
            feature_names=list(X_instance.columns),
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.tight_layout()
        
        plot_name = f"local_{instance_id or 'prediction'}.png"
        plt.savefig(output_dir / plot_name, dpi=self.config.figure_dpi)
        plt.close()
        logger.info(f"Saved: {output_dir / plot_name}")
    
    def explain_top_predictions(
        self,
        X: pd.DataFrame,
        y_pred_proba: np.ndarray,
        n_top: int = 5,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for top fraud predictions.
        
        Args:
            X: Feature DataFrame
            y_pred_proba: Predicted probabilities
            n_top: Number of top predictions to explain
            threshold: Minimum probability to consider
            
        Returns:
            List of local explanations
        """
        # Get indices of top predictions above threshold
        high_risk = np.where(y_pred_proba >= threshold)[0]
        sorted_indices = high_risk[np.argsort(y_pred_proba[high_risk])[::-1]]
        top_indices = sorted_indices[:n_top]
        
        explanations = []
        for i, idx in enumerate(top_indices):
            logger.info(f"Explaining prediction {i+1}/{n_top} (prob={y_pred_proba[idx]:.3f})")
            exp = self.explain_prediction(
                X.iloc[idx],
                save_plot=True,
                instance_id=f"top_{i+1}_prob{y_pred_proba[idx]:.2f}",
            )
            exp["probability"] = float(y_pred_proba[idx])
            exp["index"] = int(idx)
            explanations.append(exp)
        
        return explanations
    
    def compare_feature_importance(
        self,
        X_window1: pd.DataFrame,
        X_window2: pd.DataFrame,
        window1_name: str = "window_1",
        window2_name: str = "window_2",
    ) -> pd.DataFrame:
        """
        Compare feature importance between two time windows.
        
        Useful for detecting concept drift through changing feature importance.
        
        Args:
            X_window1: Features from first time window
            X_window2: Features from second time window
            window1_name: Label for first window
            window2_name: Label for second window
            
        Returns:
            DataFrame with importance comparison and drift indicators
        """
        logger.info(f"Computing SHAP for {window1_name}...")
        imp1 = self.compute_global_importance(
            X_window1, save_plot=False
        ).set_index("feature")
        
        logger.info(f"Computing SHAP for {window2_name}...")
        imp2 = self.compute_global_importance(
            X_window2, save_plot=False
        ).set_index("feature")
        
        # Merge importances
        comparison = pd.DataFrame({
            f"importance_{window1_name}": imp1["importance"],
            f"importance_{window2_name}": imp2["importance"],
            f"rank_{window1_name}": imp1["rank"],
            f"rank_{window2_name}": imp2["rank"],
        })
        
        # Compute changes
        comparison["importance_change"] = (
            comparison[f"importance_{window2_name}"] - 
            comparison[f"importance_{window1_name}"]
        )
        comparison["importance_change_pct"] = (
            comparison["importance_change"] / 
            (comparison[f"importance_{window1_name}"] + 1e-8) * 100
        )
        comparison["rank_change"] = (
            comparison[f"rank_{window1_name}"] - 
            comparison[f"rank_{window2_name}"]  # Positive = improved rank
        )
        
        # Flag significant changes
        comparison["significant_drift"] = (
            (comparison["importance_change_pct"].abs() > 50) |
            (comparison["rank_change"].abs() > 5)
        )
        
        comparison = comparison.sort_values(
            "importance_change_pct", 
            ascending=False,
            key=abs,
        )
        
        # Save comparison plot
        self._save_comparison_plot(comparison, window1_name, window2_name)
        
        return comparison.reset_index()
    
    def _save_comparison_plot(
        self,
        comparison: pd.DataFrame,
        window1_name: str,
        window2_name: str,
    ):
        """Save feature importance comparison plot."""
        output_dir = Path(self.config.output_dir)
        
        # Top 15 features by average importance
        top_features = comparison.head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(top_features))
        width = 0.35
        
        ax.barh(
            x - width/2, 
            top_features[f"importance_{window1_name}"],
            width,
            label=window1_name,
            color='steelblue',
        )
        ax.barh(
            x + width/2,
            top_features[f"importance_{window2_name}"],
            width,
            label=window2_name,
            color='coral',
        )
        
        ax.set_yticks(x)
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance Comparison Across Windows")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(
            output_dir / f"comparison_{window1_name}_vs_{window2_name}.png",
            dpi=self.config.figure_dpi,
        )
        plt.close()
        logger.info(f"Saved comparison plot")
    
    def generate_report(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive SHAP analysis report.
        
        Args:
            X_test: Test features
            y_test: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Complete analysis report
        """
        logger.info("Generating comprehensive SHAP report...")
        
        # Global importance
        global_importance = self.compute_global_importance(X_test)
        
        # Explain top fraud predictions
        top_fraud_explanations = self.explain_top_predictions(
            X_test, y_pred_proba, n_top=5, threshold=0.7
        )
        
        # Explain some true positives (correctly caught fraud)
        fraud_indices = np.where((y_test == 1) & (y_pred_proba >= 0.5))[0]
        tp_explanations = []
        for i, idx in enumerate(fraud_indices[:3]):
            exp = self.explain_prediction(
                X_test.iloc[idx],
                save_plot=True,
                instance_id=f"true_positive_{i+1}",
            )
            exp["actual_fraud"] = True
            tp_explanations.append(exp)
        
        # Explain some false negatives (missed fraud)
        fn_indices = np.where((y_test == 1) & (y_pred_proba < 0.5))[0]
        fn_explanations = []
        for i, idx in enumerate(fn_indices[:3]):
            exp = self.explain_prediction(
                X_test.iloc[idx],
                save_plot=True,
                instance_id=f"false_negative_{i+1}",
            )
            exp["actual_fraud"] = True
            exp["probability"] = float(y_pred_proba[idx])
            fn_explanations.append(exp)
        
        # GNN vs Tabular feature analysis
        gnn_features = [f for f in X_test.columns if f.startswith("gnn_emb_")]
        tabular_features = [f for f in X_test.columns if not f.startswith("gnn_emb_")]
        
        gnn_importance = global_importance[
            global_importance["feature"].isin(gnn_features)
        ]["importance"].sum()
        tabular_importance = global_importance[
            global_importance["feature"].isin(tabular_features)
        ]["importance"].sum()
        
        report = {
            "global_importance": global_importance.to_dict("records"),
            "top_10_features": global_importance.head(10)["feature"].tolist(),
            "gnn_contribution": {
                "total_importance": float(gnn_importance),
                "percentage": float(gnn_importance / (gnn_importance + tabular_importance) * 100),
                "feature_count": len(gnn_features),
            },
            "tabular_contribution": {
                "total_importance": float(tabular_importance),
                "percentage": float(tabular_importance / (gnn_importance + tabular_importance) * 100),
                "feature_count": len(tabular_features),
            },
            "top_fraud_explanations": top_fraud_explanations,
            "true_positive_explanations": tp_explanations,
            "false_negative_explanations": fn_explanations,
        }
        
        # Save report
        output_dir = Path(self.config.output_dir)
        import json
        with open(output_dir / "shap_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"SHAP report saved to {output_dir / 'shap_report.json'}")
        
        return report


def run_shap_analysis(
    model: xgb.XGBClassifier, 
    X_sample: pd.DataFrame, 
    save_path: str = "shap_summary.png"
) -> pd.DataFrame:
    """
    Compute and plot SHAP summary (legacy interface).
    
    For more comprehensive analysis, use SHAPAnalyzer class.
    """
    analyzer = SHAPAnalyzer(model)
    importance = analyzer.compute_global_importance(
        X_sample, 
        save_plot=True,
        plot_name=Path(save_path).stem,
    )
    return importance

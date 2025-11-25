"""
SHAP-based explainability module for XGBoost fraud detection models.

This module provides utilities to:
1. Generate global feature importance explanations
2. Explain individual predictions (local explanations)
3. Compare feature importance across model variants
4. Analyze temporal changes in feature importance
5. Investigate false positives and false negatives
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
import xgboost as xgb
from matplotlib.figure import Figure

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
shap.initjs()


def _fix_xgboost_model(model: xgb.XGBClassifier) -> xgb.XGBClassifier:
    """
    Fix XGBoost model that may have corrupted internal parameters after pickling.
    
    This function handles cases where model parameters (like base_score) are stored
    as strings instead of floats, which can cause SHAP to fail.
    
    Args:
        model: XGBoost model that may need fixing
        
    Returns:
        Fixed XGBoost model
    """
    # Always try JSON save/reload first - this is the most reliable fix
    # for corrupted pickled models
    try:
        print("Checking model integrity and fixing if needed...")
        # Save model to JSON and reload - this normalizes all parameters
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        # Save to JSON (this will normalize parameters)
        model.save_model(temp_path)
        
        # Create a new model and load from JSON
        fixed_model = xgb.XGBClassifier()
        fixed_model.load_model(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Test if fixed model works with TreeExplainer
        try:
            _ = shap.TreeExplainer(fixed_model)
            print("✓ Model parameters normalized successfully")
            return fixed_model
        except Exception as test_error:
            # If it still fails, return the fixed model anyway
            # The caller will handle it with fallback explainers
            print(f"Note: Model normalized but TreeExplainer test failed: {test_error}")
            print("Will use fallback SHAP explainer methods...")
            return fixed_model
            
    except Exception as fix_error:
        print(f"Warning: Could not normalize model via JSON: {fix_error}")
        print("Will use original model with fallback SHAP explainer methods...")
        # Return original model - we'll handle it with fallback explainers
        return model


class ModelExplainer:
    """
    Wrapper class for SHAP explanations of XGBoost models.
    """
    
    def __init__(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        model_type: str = "baseline",
        window_info: Optional[Dict] = None
    ):
        """
        Initialize explainer with model and test data.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            X_test: Test features
            y_test: True labels
            y_pred: Predicted probabilities
            model_type: Type of model (baseline, graph_baseline, hybrid)
            window_info: Dictionary with window metadata
        """
        self.model = model
        self.feature_names = feature_names
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.model_type = model_type
        self.window_info = window_info or {}
        
        # Fix model if needed (handles pickled models with corrupted parameters)
        fixed_model = _fix_xgboost_model(model)
        
        # Create SHAP explainer (TreeExplainer is optimized for XGBoost)
        print("Initializing SHAP TreeExplainer...")
        self.use_kernel_explainer = False
        try:
            self.explainer = shap.TreeExplainer(fixed_model)
        except (ValueError, TypeError, AttributeError) as e:
            # If TreeExplainer still fails, try with model_output='probability'
            print(f"Standard TreeExplainer failed: {e}")
            print("Trying TreeExplainer with model_output='probability'...")
            try:
                self.explainer = shap.TreeExplainer(fixed_model, model_output='probability')
            except Exception as e2:
                # Last resort: use KernelExplainer (slower but more robust)
                print(f"TreeExplainer with probability output also failed: {e2}")
                print("Falling back to KernelExplainer (this will be slower)...")
                # Use a sample of data for background
                background_size = min(100, len(X_test))
                background = X_test[:background_size]
                self.explainer = shap.KernelExplainer(fixed_model.predict_proba, background)
                self.use_kernel_explainer = True
        
        # Compute SHAP values
        print("Computing SHAP values...")
        if self.use_kernel_explainer:
            # KernelExplainer returns values for both classes, we want the positive class (fraud)
            shap_values_all = self.explainer.shap_values(X_test)
            # shap_values_all is a list [values_class_0, values_class_1] for binary classification
            if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
                self.shap_values = shap_values_all[1]  # Use class 1 (fraud) SHAP values
            else:
                self.shap_values = shap_values_all
            self.expected_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value
        else:
            shap_values_raw = self.explainer.shap_values(X_test)
            # Handle different return formats from TreeExplainer
            if isinstance(shap_values_raw, list):
                # List of arrays - use the positive class (index 1 for binary classification)
                if len(shap_values_raw) == 2:
                    self.shap_values = shap_values_raw[1]
                else:
                    self.shap_values = shap_values_raw[0]
            else:
                # Single array - check if it has extra dimensions
                self.shap_values = shap_values_raw
            
            # Ensure SHAP values are 2D: (n_samples, n_features)
            if self.shap_values.ndim > 2:
                # Flatten extra dimensions - take the last meaningful dimension
                # Shape might be (n_samples, n_features, n_classes, ...) - we want (n_samples, n_features)
                original_shape = self.shap_values.shape
                if len(original_shape) == 3 and original_shape[2] == 2:
                    # Shape is (n_samples, n_features, 2) - take the positive class
                    self.shap_values = self.shap_values[:, :, 1]
                elif len(original_shape) == 4:
                    # Shape is (n_samples, n_features, 2, 2) or similar - flatten
                    # Take the mean across extra dimensions or select appropriate slice
                    self.shap_values = self.shap_values[:, :, 0, 0]  # Or use appropriate indexing
                else:
                    # Flatten to 2D by taking mean across extra dimensions
                    self.shap_values = self.shap_values.mean(axis=tuple(range(2, self.shap_values.ndim)))
            
            # Final validation: ensure 2D shape
            if self.shap_values.ndim != 2:
                raise ValueError(f"SHAP values should be 2D (n_samples, n_features), got shape {self.shap_values.shape}")
            
            # Ensure shape matches X_test
            if self.shap_values.shape[0] != X_test.shape[0]:
                raise ValueError(f"SHAP values first dimension ({self.shap_values.shape[0]}) doesn't match X_test ({X_test.shape[0]})")
            if self.shap_values.shape[1] != X_test.shape[1]:
                # This might be okay if we're using a different feature set, but log a warning
                print(f"Warning: SHAP values features ({self.shap_values.shape[1]}) doesn't match X_test ({X_test.shape[1]})")
            
            self.expected_value = self.explainer.expected_value
            # Handle expected_value if it's an array
            if isinstance(self.expected_value, (list, np.ndarray)):
                if len(self.expected_value) == 2:
                    self.expected_value = self.expected_value[1]  # Use positive class
                else:
                    self.expected_value = float(self.expected_value[0])
            else:
                self.expected_value = float(self.expected_value)
        
        # Create DataFrame for easier analysis
        self.df_test = pd.DataFrame(X_test, columns=feature_names)
        self.df_test['y_true'] = y_test
        self.df_test['y_pred'] = y_pred
        self.df_test['predicted_fraud'] = (y_pred >= 0.5).astype(int)
        
        # Identify prediction types
        self.df_test['prediction_type'] = 'TN'
        self.df_test.loc[(self.df_test['y_true'] == 1) & (self.df_test['predicted_fraud'] == 1), 'prediction_type'] = 'TP'
        self.df_test.loc[(self.df_test['y_true'] == 0) & (self.df_test['predicted_fraud'] == 1), 'prediction_type'] = 'FP'
        self.df_test.loc[(self.df_test['y_true'] == 1) & (self.df_test['predicted_fraud'] == 0), 'prediction_type'] = 'FN'
        
    def _normalize_shap_values(self, shap_vals: np.ndarray, n_features: int) -> np.ndarray:
        """
        Normalize SHAP values to 2D shape (n_samples, n_features).
        
        Args:
            shap_vals: SHAP values array (may have extra dimensions)
            n_features: Expected number of features
            
        Returns:
            Normalized 2D SHAP values array
        """
        if shap_vals.ndim == 2:
            # Already correct shape
            return shap_vals
        elif shap_vals.ndim == 3:
            # Shape is (n_samples, n_features, n_classes) - take positive class or mean
            if shap_vals.shape[2] == 2:
                return shap_vals[:, :, 1]  # Take positive class (fraud)
            else:
                return shap_vals.mean(axis=2)
        elif shap_vals.ndim == 4:
            # Shape is (n_samples, n_features, 2, 2) or similar
            # Take the first class, first output dimension
            return shap_vals[:, :, 0, 0]
        else:
            # Flatten extra dimensions by taking mean across them
            return shap_vals.mean(axis=tuple(range(2, shap_vals.ndim)))
    
    def _get_embedding_features(self) -> List[str]:
        """Get list of embedding feature names."""
        return [f for f in self.feature_names if f.startswith('embed_')]
    
    def _aggregate_embedding_shap(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Aggregate SHAP values for embedding features into a single 'GNN_embeddings' feature.
        
        Returns:
            Tuple of (aggregated_shap_values, new_feature_names, aggregated_X_test)
        """
        embed_features = self._get_embedding_features()
        if not embed_features:
            # Normalize before returning
            shap_vals = self._normalize_shap_values(self.shap_values, len(self.feature_names))
            return shap_vals, self.feature_names, self.X_test
        
        # Normalize SHAP values first
        shap_vals_normalized = self._normalize_shap_values(self.shap_values, len(self.feature_names))
        
        embed_indices = [i for i, name in enumerate(self.feature_names) if name in embed_features]
        non_embed_indices = [i for i, name in enumerate(self.feature_names) if name not in embed_features]
        
        # Aggregate SHAP values (sum absolute values for importance, or sum for direction)
        embed_shap_aggregated = np.sum(shap_vals_normalized[:, embed_indices], axis=1, keepdims=True)
        non_embed_shap = shap_vals_normalized[:, non_embed_indices]
        
        # Aggregate feature values (mean)
        embed_X_aggregated = np.mean(self.X_test[:, embed_indices], axis=1, keepdims=True)
        non_embed_X = self.X_test[:, non_embed_indices]
        
        # Combine
        aggregated_shap = np.hstack([non_embed_shap, embed_shap_aggregated])
        aggregated_X = np.hstack([non_embed_X, embed_X_aggregated])
        aggregated_features = [self.feature_names[i] for i in non_embed_indices] + ['GNN_embeddings']
        
        return aggregated_shap, aggregated_features, aggregated_X
    
    def plot_summary(
        self,
        max_display: int = 20,
        aggregate_embeddings: bool = True,
        output_path: Optional[str] = None
    ) -> Figure:
        """
        Generate SHAP summary plot showing global feature importance.
        
        Args:
            max_display: Maximum number of features to display
            aggregate_embeddings: Whether to aggregate embedding features
            output_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        print("Generating SHAP summary plot...")
        
        if aggregate_embeddings and self._get_embedding_features():
            shap_vals, features, X = self._aggregate_embedding_shap()
        else:
            shap_vals, features, X = self.shap_values, self.feature_names, self.X_test
        
        # Normalize SHAP values to 2D
        shap_vals = self._normalize_shap_values(shap_vals, len(features))
        
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_vals,
            X,
            feature_names=features,
            max_display=max_display,
            show=False
        )
        plt.title(f"SHAP Feature Importance - {self.model_type.upper()}\n{self._get_window_str()}")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved summary plot to {output_path}")
        
        return fig
    
    def plot_feature_importance_bar(
        self,
        max_display: int = 20,
        aggregate_embeddings: bool = True,
        output_path: Optional[str] = None
    ) -> Figure:
        """
        Generate bar chart of mean absolute SHAP values.
        
        Args:
            max_display: Maximum number of features to display
            aggregate_embeddings: Whether to aggregate embedding features
            output_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        print("Generating feature importance bar chart...")
        
        if aggregate_embeddings and self._get_embedding_features():
            shap_vals, features, X = self._aggregate_embedding_shap()
        else:
            shap_vals, features, X = self.shap_values, self.feature_names, self.X_test
        
        # Normalize SHAP values to 2D
        shap_vals = self._normalize_shap_values(shap_vals, len(features))
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        # Ensure mean_abs_shap is 1D and matches number of features
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        if len(mean_abs_shap) != len(features):
            # Take first len(features) elements if shape mismatch
            mean_abs_shap = mean_abs_shap[:len(features)]
        
        # Sort and select top features
        sorted_idx = np.argsort(mean_abs_shap)[-max_display:]
        
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
        ax.barh(
            range(len(sorted_idx)),
            mean_abs_shap[sorted_idx],
            color='steelblue',
            alpha=0.7
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'Global Feature Importance - {self.model_type.upper()}\n{self._get_window_str()}')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance bar chart to {output_path}")
        
        return fig
    
    def plot_waterfall(
        self,
        instance_idx: int,
        aggregate_embeddings: bool = True,
        output_path: Optional[str] = None
    ) -> Figure:
        """
        Generate waterfall plot for a specific prediction.
        
        Args:
            instance_idx: Index of the instance to explain
            aggregate_embeddings: Whether to aggregate embedding features
            output_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        print(f"Generating waterfall plot for instance {instance_idx}...")
        
        if aggregate_embeddings and self._get_embedding_features():
            shap_vals, features, X = self._aggregate_embedding_shap()
        else:
            shap_vals, features, X = self.shap_values, self.feature_names, self.X_test
        
        # Normalize SHAP values to 2D
        shap_vals = self._normalize_shap_values(shap_vals, len(features))
        
        fig = plt.figure(figsize=(12, 8))
        
        # Create an Explanation object for the waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_vals[instance_idx],
            base_values=self.expected_value,
            data=X[instance_idx],
            feature_names=features
        )
        
        shap.waterfall_plot(shap_explanation, show=False)
        
        # Add additional information
        pred_type = self.df_test.iloc[instance_idx]['prediction_type']
        y_true = self.df_test.iloc[instance_idx]['y_true']
        y_pred = self.df_test.iloc[instance_idx]['y_pred']
        
        plt.title(
            f"SHAP Waterfall - Instance {instance_idx} ({pred_type})\n"
            f"True: {y_true}, Predicted: {y_pred:.3f}\n"
            f"{self.model_type.upper()} - {self._get_window_str()}"
        )
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved waterfall plot to {output_path}")
        
        return fig
    
    def plot_dependence(
        self,
        feature: str,
        interaction_feature: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> Figure:
        """
        Generate SHAP dependence plot for a specific feature.
        
        Args:
            feature: Feature to plot
            interaction_feature: Feature to use for coloring (auto-selected if None)
            output_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in model features")
        
        print(f"Generating dependence plot for {feature}...")
        
        # Normalize SHAP values to 2D
        shap_vals_normalized = self._normalize_shap_values(self.shap_values, len(self.feature_names))
        
        fig = plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_vals_normalized,
            self.X_test,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature}\n{self.model_type.upper()} - {self._get_window_str()}")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved dependence plot to {output_path}")
        
        return fig
    
    def explain_top_predictions(
        self,
        top_k: int = 20,
        prediction_type: str = 'FP',
        aggregate_embeddings: bool = True,
        output_dir: Optional[str] = None
    ) -> List[Figure]:
        """
        Generate waterfall plots for top-K predictions of a specific type.
        
        Args:
            top_k: Number of instances to explain
            prediction_type: Type of predictions to explain (TP, FP, TN, FN)
            aggregate_embeddings: Whether to aggregate embedding features
            output_dir: Directory to save figures
            
        Returns:
            List of Figure objects
        """
        print(f"Explaining top {top_k} {prediction_type} predictions...")
        
        # Filter by prediction type
        subset = self.df_test[self.df_test['prediction_type'] == prediction_type]
        
        if len(subset) == 0:
            print(f"No {prediction_type} predictions found.")
            return []
        
        # Sort by prediction confidence
        subset = subset.sort_values('y_pred', ascending=False)
        top_instances = subset.head(top_k)
        
        figures = []
        for i, (idx, row) in enumerate(top_instances.iterrows()):
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{prediction_type}_instance_{i}_idx{idx}.png")
            
            fig = self.plot_waterfall(idx, aggregate_embeddings, output_path)
            figures.append(fig)
            plt.close(fig)  # Close to save memory
        
        return figures
    
    def analyze_cohort_differences(
        self,
        cohort1_type: str = 'TP',
        cohort2_type: str = 'FP',
        top_n_features: int = 15,
        output_path: Optional[str] = None
    ) -> Figure:
        """
        Compare SHAP value distributions between two prediction cohorts.
        
        Args:
            cohort1_type: First cohort type (TP, FP, TN, FN)
            cohort2_type: Second cohort type
            top_n_features: Number of top features to compare
            output_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        print(f"Comparing {cohort1_type} vs {cohort2_type}...")
        
        cohort1_mask = self.df_test['prediction_type'] == cohort1_type
        cohort2_mask = self.df_test['prediction_type'] == cohort2_type
        
        if cohort1_mask.sum() == 0 or cohort2_mask.sum() == 0:
            print(f"Insufficient data for comparison. {cohort1_type}: {cohort1_mask.sum()}, {cohort2_type}: {cohort2_mask.sum()}")
            return None
        
        # Normalize SHAP values to 2D
        shap_vals_normalized = self._normalize_shap_values(self.shap_values, len(self.feature_names))
        
        # Calculate mean absolute SHAP values for each cohort
        shap_cohort1 = np.abs(shap_vals_normalized[cohort1_mask]).mean(axis=0)
        shap_cohort2 = np.abs(shap_vals_normalized[cohort2_mask]).mean(axis=0)
        
        # Get top features based on total importance
        total_importance = shap_cohort1 + shap_cohort2
        top_indices = np.argsort(total_importance)[-top_n_features:]
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(12, max(6, top_n_features * 0.4)))
        
        x = np.arange(len(top_indices))
        width = 0.35
        
        ax.barh(x - width/2, shap_cohort1[top_indices], width, label=cohort1_type, alpha=0.7)
        ax.barh(x + width/2, shap_cohort2[top_indices], width, label=cohort2_type, alpha=0.7)
        
        ax.set_yticks(x)
        ax.set_yticklabels([self.feature_names[i] for i in top_indices])
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(
            f'Feature Importance Comparison: {cohort1_type} vs {cohort2_type}\n'
            f'{self.model_type.upper()} - {self._get_window_str()}'
        )
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved cohort comparison to {output_path}")
        
        return fig
    
    def get_feature_importance_df(
        self,
        aggregate_embeddings: bool = True
    ) -> pd.DataFrame:
        """
        Get a DataFrame with feature importance metrics.
        
        Args:
            aggregate_embeddings: Whether to aggregate embedding features
            
        Returns:
            DataFrame with feature names and importance metrics
        """
        if aggregate_embeddings and self._get_embedding_features():
            shap_vals, features, _ = self._aggregate_embedding_shap()
        else:
            shap_vals, features, _ = self.shap_values, self.feature_names, self.X_test
        
        # Normalize SHAP values to 2D
        shap_vals = self._normalize_shap_values(shap_vals, len(features))
        
        # Calculate various importance metrics
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        mean_shap = shap_vals.mean(axis=0)
        std_shap = shap_vals.std(axis=0)
        
        # Ensure all metrics are 1D and match number of features
        if mean_abs_shap.ndim > 1:
            mean_abs_shap = mean_abs_shap.flatten()[:len(features)]
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()[:len(features)]
        if std_shap.ndim > 1:
            std_shap = std_shap.flatten()[:len(features)]
        
        df = pd.DataFrame({
            'feature': features,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap,
            'std_shap': std_shap,
            'model_type': self.model_type
        })
        
        # Add window info if available
        if self.window_info:
            df['window_start'] = str(self.window_info.get('window_start', ''))
            df['window_end'] = str(self.window_info.get('window_end', ''))
        
        return df.sort_values('mean_abs_shap', ascending=False)
    
    def _get_window_str(self) -> str:
        """Get formatted window string for plot titles."""
        if self.window_info:
            start = self.window_info.get('window_start', '')
            end = self.window_info.get('window_end', '')
            if start and end:
                return f"Window: {start} to {end}"
        return ""


def compare_models(
    explainers: Dict[str, ModelExplainer],
    top_n_features: int = 15,
    aggregate_embeddings: bool = True,
    output_path: Optional[str] = None
) -> Figure:
    """
    Compare feature importance across multiple models.
    
    Args:
        explainers: Dictionary mapping model names to ModelExplainer objects
        top_n_features: Number of top features to display
        aggregate_embeddings: Whether to aggregate embedding features
        output_path: Path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    print(f"Comparing {len(explainers)} models...")
    
    # Collect feature importance from all models
    all_importances = {}
    all_features = set()
    
    for model_name, explainer in explainers.items():
        df = explainer.get_feature_importance_df(aggregate_embeddings)
        importance_dict = dict(zip(df['feature'], df['mean_abs_shap']))
        all_importances[model_name] = importance_dict
        all_features.update(df['feature'].tolist())
    
    # Get top features across all models
    total_importance = {feat: sum(model_imp.get(feat, 0) for model_imp in all_importances.values())
                       for feat in all_features}
    top_features = sorted(total_importance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
    top_feature_names = [f[0] for f in top_features]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, max(8, top_n_features * 0.4)))
    
    x = np.arange(len(top_feature_names))
    width = 0.8 / len(explainers)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(explainers)))
    
    for i, (model_name, importance_dict) in enumerate(all_importances.items()):
        values = [importance_dict.get(feat, 0) for feat in top_feature_names]
        offset = (i - len(explainers)/2 + 0.5) * width
        ax.barh(x + offset, values, width, label=model_name, alpha=0.8, color=colors[i])
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_feature_names)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Model Comparison: Feature Importance Across {len(explainers)} Models')
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {output_path}")
    
    return fig


def save_explainer(
    explainer: ModelExplainer,
    output_dir: str,
    prefix: str = ""
) -> None:
    """
    Save all standard SHAP visualizations for an explainer.
    
    Args:
        explainer: ModelExplainer object
        output_dir: Directory to save outputs
        prefix: Prefix for output filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    
    print(f"\nGenerating all SHAP visualizations in {output_dir}...")
    
    # 1. Summary plot
    fig = explainer.plot_summary(output_path=os.path.join(output_dir, f"{prefix}summary_plot.png"))
    plt.close(fig)
    
    # 2. Feature importance bar chart
    fig = explainer.plot_feature_importance_bar(output_path=os.path.join(output_dir, f"{prefix}feature_importance.png"))
    plt.close(fig)
    
    # 3. Top false positives
    explainer.explain_top_predictions(
        top_k=10,
        prediction_type='FP',
        output_dir=os.path.join(output_dir, f"{prefix}fp_explanations")
    )
    
    # 4. Top false negatives (if any)
    explainer.explain_top_predictions(
        top_k=5,
        prediction_type='FN',
        output_dir=os.path.join(output_dir, f"{prefix}fn_explanations")
    )
    
    # 5. Cohort comparison (TP vs FP)
    fig = explainer.analyze_cohort_differences(
        'TP', 'FP',
        output_path=os.path.join(output_dir, f"{prefix}tp_vs_fp_comparison.png")
    )
    if fig:
        plt.close(fig)
    
    # 6. Save feature importance DataFrame
    df = explainer.get_feature_importance_df()
    df.to_csv(os.path.join(output_dir, f"{prefix}feature_importance.csv"), index=False)
    print(f"Saved feature importance CSV to {output_dir}/{prefix}feature_importance.csv")
    
    # 7. Key dependence plots for top 3 features
    top_features = df.head(3)['feature'].tolist()
    for feat in top_features:
        if feat != 'GNN_embeddings':  # Skip aggregated embedding feature
            try:
                fig = explainer.plot_dependence(
                    feat,
                    output_path=os.path.join(output_dir, f"{prefix}dependence_{feat}.png")
                )
                plt.close(fig)
            except Exception as e:
                print(f"Could not create dependence plot for {feat}: {e}")
    
    print(f"✓ All SHAP visualizations saved to {output_dir}")


def load_saved_model(model_path: str) -> Dict:
    """
    Load a saved model bundle.
    
    Args:
        model_path: Path to the saved model pickle
        
    Returns:
        Dictionary containing model, features, and metadata
    """
    import pickle
    
    with open(model_path, 'rb') as f:
        model_bundle = pickle.load(f)
    
    return model_bundle


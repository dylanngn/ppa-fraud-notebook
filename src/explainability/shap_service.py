"""
SHAP Explainability Service

Provides on-demand explanations for predictions and discovers new feature opportunities.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64


@dataclass
class Explanation:
    """SHAP explanation for a single prediction."""
    listing_id: int
    fraud_score: float
    base_value: float
    shap_values: Dict[str, float]
    top_features: List[Tuple[str, float]]
    waterfall_plot_base64: Optional[str] = None


class SHAPService:
    """
    Generates SHAP explanations for model predictions.
    
    Also analyzes feature interactions to suggest new composite features.
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        # Initialize SHAP explainer (TreeExplainer for XGBoost)
        self.explainer = shap.TreeExplainer(model)
        
    def explain(
        self,
        features: np.ndarray,
        listing_id: int,
        fraud_score: float,
        generate_plot: bool = True
    ) -> Explanation:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            features: Feature vector (1D array)
            listing_id: Listing ID
            fraud_score: Predicted fraud score
            generate_plot: Whether to generate waterfall plot
            
        Returns:
            Explanation object with SHAP values and visualization
        """
        # Get SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))
        
        # Handle multi-output (XGBoost sometimes returns both classes)
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]  # Take positive class
            
        shap_values = shap_values[0]  # Single prediction
        base_value = self.explainer.expected_value
        
        # Handle multi-class base value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
            
        # Create feature-to-SHAP dict
        shap_dict = {
            name: float(val)
            for name, val in zip(self.feature_names, shap_values)
        }
        
        # Get top contributing features (by absolute SHAP value)
        top_features = sorted(
            shap_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        # Generate waterfall plot if requested
        waterfall_base64 = None
        if generate_plot:
            waterfall_base64 = self._generate_waterfall_plot(
                shap_values, base_value, features[0]
            )
            
        return Explanation(
            listing_id=listing_id,
            fraud_score=fraud_score,
            base_value=float(base_value),
            shap_values=shap_dict,
            top_features=top_features,
            waterfall_plot_base64=waterfall_base64
        )
    
    def _generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        base_value: float,
        features: np.ndarray
    ) -> str:
        """Generate SHAP waterfall plot and return as base64 string."""
        plt.figure(figsize=(10, 6))
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=features,
            feature_names=self.feature_names
        )
        
        # Generate waterfall plot
        shap.waterfall_plot(explanation, show=False)
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def analyze_feature_interactions(
        self,
        X: np.ndarray,
        top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Analyze feature interactions using SHAP interaction Values.
        
        Args:
            X: Feature matrix (sample of recent predictions)
            top_n: Number of top interactions to return
            
        Returns:
            List of (feature1, feature2, interaction_strength) tuples
        """
        # Compute SHAP interaction values
        shap_interaction_values = self.explainer.shap_interaction_values(X)
        
        # Handle multi-output
        if len(shap_interaction_values.shape) > 3:
            shap_interaction_values = shap_interaction_values[:, :, :, 1]
            
        # Get mean absolute interaction for each pair
        mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
        
        # Find top interactions (excluding diagonal)
        interactions = []
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                strength = mean_abs_interactions[i, j]
                interactions.append((
                    self.feature_names[i],
                    self.feature_names[j],
                    float(strength)
                ))
                
        # Sort by interaction strength
        interactions.sort(key=lambda x: x[2], reverse=True)
        
        return interactions[:top_n]
    
    def suggest_new_features(
        self,
        X: np.ndarray,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Suggest new composite features based on SHAP interactions.
        
        Args:
            X: Feature matrix (sample of recent predictions)
            threshold: Minimum interaction strength to suggest
            
        Returns:
            List of feature suggestions (as strings)
        """
        interactions = self.analyze_feature_interactions(X, top_n=10)
        
        suggestions = []
        for feat1, feat2, strength in interactions:
            if strength > threshold:
                # Suggest composite features
                suggestions.append(f"{feat1} * {feat2}  # Interaction strength: {strength:.4f}")
                suggestions.append(f"{feat1} / ({feat2} + 1)  # Ratio feature")
                
        return suggestions

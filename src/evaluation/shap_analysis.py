"""
SHAP Analysis for XGBoost models.
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

def run_shap_analysis(
    model: xgb.XGBClassifier, 
    X_sample: pd.DataFrame, 
    save_path: str = "shap_summary.png"
):
    """
    Compute and plot SHAP summary.
    """
    logger.info("Computing SHAP values...")
    
    # TreeExplainer is strict on model type. 
    # Ensure model is the underlying Booster or XGBClassifier
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"SHAP summary saved to {save_path}")
    
    # Also return importance df
    if isinstance(shap_values, list):
        # Multiclass?
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    importance = pd.DataFrame(
        list(zip(X_sample.columns, vals)),
        columns=["feature", "importance"]
    )
    importance.sort_values(by="importance", ascending=False, inplace=True)
    
    return importance

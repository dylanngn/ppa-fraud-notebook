"""
XGBoost Classifier wrapper.
"""

import xgboost as xgb
import numpy as np
import pickle
from typing import Optional, Dict, Any

from src.models.base import BaseClassifier

class XGBoostClassifier(BaseClassifier):
    """
    Wrapper around XGBoost classifier.
    """
    
    def __init__(self, **params):
        """
        Initialize with XGBoost parameters.
        
        Args:
            **params: XGBoost parameters (n_estimators, max_depth, learning_rate, etc.)
        """
        self.params = params
        self.model = None
        
    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> "XGBoostClassifier":
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False 
        )
        
        return self
    
    def predict_proba(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fit")
        # Return probability of positive class
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not fit")
        self.model.save_model(path)
        
    @classmethod
    def load(cls, path: str) -> "XGBoostClassifier":
        # Load logic depends on how we want to re-instantiate.
        # XGBoost save_model saves internal state.
        # We need to create instance then load.
        instance = cls()
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(path)
        return instance

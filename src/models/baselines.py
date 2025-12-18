"""
Baseline Classifiers for Comparison.

Provides simple sklearn-based baselines:
1. Logistic Regression - Linear baseline
2. Random Forest - Non-boosted tree ensemble

These serve as comparison points to demonstrate 
the value of XGBoost and GNN+XGBoost approaches.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class BaselineClassifier:
    """Base class for baseline classifiers with preprocessing."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.label_encoders = {}
        self.feature_names = None
    
    def _preprocess(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for sklearn models.
        
        - Encode categorical columns
        - Impute missing values
        - Scale numeric features
        """
        X = X.copy()
        self.feature_names = list(X.columns)
        
        # Handle categorical columns
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                # Handle unseen categories by adding 'unknown'
                X[col] = X[col].astype(str).fillna('unknown')
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            else:
                X[col] = X[col].astype(str).fillna('unknown')
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X[col] = X[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    X[col] = 0
        
        # Convert to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Impute and scale
        if fit:
            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the classifier."""
        logger.info(f"Fitting {self.__class__.__name__} on {len(X)} samples")
        X_processed = self._preprocess(X, fit=True)
        self.model.fit(X_processed, y)
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict_proba(X_processed)[:, 1]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes."""
        X_processed = self._preprocess(X, fit=False)
        return self.model.predict(X_processed)


class LogisticRegressionClassifier(BaselineClassifier):
    """
    Logistic Regression baseline.
    
    A simple linear model that serves as the most basic baseline.
    Expected to underperform on complex fraud patterns.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        **kwargs,
    ):
        super().__init__()
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            n_jobs=-1,
            **kwargs,
        )
        logger.info(f"LogisticRegression initialized (C={C})")


class RandomForestBaseline(BaselineClassifier):
    """
    Random Forest baseline.
    
    A non-boosted tree ensemble. Should perform better than LR
    but worse than XGBoost due to lack of boosting.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = 'balanced',
        **kwargs,
    ):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
            **kwargs,
        )
        logger.info(
            f"RandomForest initialized (n_estimators={n_estimators}, "
            f"max_depth={max_depth})"
        )

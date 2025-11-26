"""
Adaptation Engine: SHAP-Driven Automatic Adaptation Suggestions

This module analyzes SHAP values across training windows to automatically suggest:
1. New rules to add (high-importance features with clear thresholds)
2. Features to prune (zero importance across windows)
3. Interaction features to engineer (SHAP interaction values)
4. When to retrain (drift detected)

This is the core innovation for the data mining research:
transforming SHAP insights into actionable adaptations.

Usage:
    from src.explainability.adaptation_engine import AdaptationEngine
    
    engine = AdaptationEngine(feature_names=features, history_window=5)
    report = engine.analyze_window(model, X_test, y_test, y_pred, window_info)
    report.save_markdown("artifacts/reports/adaptation_report.md")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle

import numpy as np
import shap


@dataclass
class AdaptationSuggestion:
    """Single adaptation recommendation."""
    type: str  # "rule", "feature", "retrain", "prune"
    priority: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    implementation: str  # Code/SQL snippet
    expected_impact: str
    shap_evidence: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Feature importance drift alert."""
    feature: str
    previous_rank: int
    current_rank: int
    rank_change: int
    importance_delta: float
    severity: str  # "critical", "high", "medium"
    direction: str  # "rising", "falling"


@dataclass 
class AdaptationReport:
    """Complete adaptation report for a window."""
    window_idx: int
    timestamp: datetime
    model_performance: Dict[str, float]
    
    # Feature importance tracking
    current_importance: Dict[str, float]
    drift_alerts: List[DriftAlert]
    rising_features: List[DriftAlert]
    falling_features: List[DriftAlert]
    
    # Suggestions
    pruning_candidates: List[str]
    rule_suggestions: List[AdaptationSuggestion]
    feature_suggestions: List[AdaptationSuggestion]
    
    # Recommendations
    retrain_recommendation: bool
    retrain_reason: str
    
    # Optional narrative (filled by LLM)
    narrative: str = ""
    
    def save_markdown(self, output_path: str) -> None:
        """Save report as formatted markdown."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            f.write(self._to_markdown())
    
    def save_json(self, output_path: str) -> None:
        """Save report as JSON for programmatic access."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "window_idx": self.window_idx,
            "timestamp": self.timestamp.isoformat(),
            "model_performance": self.model_performance,
            "current_importance": self.current_importance,
            "drift_alerts": [vars(d) for d in self.drift_alerts],
            "rising_features": [vars(d) for d in self.rising_features],
            "falling_features": [vars(d) for d in self.falling_features],
            "pruning_candidates": self.pruning_candidates,
            "rule_suggestions": [vars(s) for s in self.rule_suggestions],
            "feature_suggestions": [vars(s) for s in self.feature_suggestions],
            "retrain_recommendation": self.retrain_recommendation,
            "retrain_reason": self.retrain_reason,
            "narrative": self.narrative,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _to_markdown(self) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# Adaptation Report - Window {self.window_idx}",
            f"\n**Generated**: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        # Model Performance
        lines.append("## Model Performance")
        for metric, value in self.model_performance.items():
            if isinstance(value, float):
                lines.append(f"- **{metric}**: {value:.4f}")
            else:
                lines.append(f"- **{metric}**: {value}")
        lines.append("")
        
        # Drift Alerts
        if self.drift_alerts:
            lines.append("## ⚠️ Drift Alerts")
            for alert in self.drift_alerts[:10]:
                direction_emoji = "📈" if alert.direction == "rising" else "📉"
                lines.append(
                    f"- {direction_emoji} **{alert.feature}**: "
                    f"Rank {alert.previous_rank} → {alert.current_rank} "
                    f"({alert.rank_change:+d}) [{alert.severity}]"
                )
            lines.append("")
        
        # Rising Features
        if self.rising_features:
            lines.append("## 📈 Rising Features (Gaining Importance)")
            lines.append("These features are becoming more important - may indicate new fraud patterns.")
            lines.append("")
            for alert in self.rising_features[:5]:
                lines.append(
                    f"- **{alert.feature}**: "
                    f"+{alert.importance_delta:.4f} importance "
                    f"(now rank #{alert.current_rank})"
                )
            lines.append("")
        
        # Falling Features
        if self.falling_features:
            lines.append("## 📉 Falling Features (Losing Importance)")
            lines.append("These features are becoming less important - fraudsters may be adapting.")
            lines.append("")
            for alert in self.falling_features[:5]:
                lines.append(
                    f"- **{alert.feature}**: "
                    f"{alert.importance_delta:.4f} importance "
                    f"(now rank #{alert.current_rank})"
                )
            lines.append("")
        
        # Top Features
        lines.append("## Top 10 Features (Current Window)")
        sorted_importance = sorted(
            self.current_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for feature, importance in sorted_importance:
            lines.append(f"- **{feature}**: {importance:.4f}")
        lines.append("")
        
        # Pruning Candidates
        if self.pruning_candidates:
            lines.append(f"## 🗑️ Pruning Candidates ({len(self.pruning_candidates)} features)")
            lines.append("These features have zero or near-zero importance and can be removed.")
            lines.append("")
            examples = self.pruning_candidates[:10]
            lines.append(f"Examples: {', '.join(examples)}")
            if len(self.pruning_candidates) > 10:
                lines.append(f"... and {len(self.pruning_candidates) - 10} more")
            lines.append("")
            lines.append("> **Tip**: Removing these features can improve training speed without affecting performance.")
            lines.append("")
        
        # Rule Suggestions
        if self.rule_suggestions:
            lines.append("## 🎯 Rule Suggestions")
            lines.append("Based on SHAP analysis, consider these business rules for immediate flagging.")
            lines.append("")
            for i, suggestion in enumerate(self.rule_suggestions, 1):
                priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                    suggestion.priority, "⚪"
                )
                lines.append(f"### {i}. {priority_emoji} {suggestion.title}")
                lines.append(f"**Priority**: {suggestion.priority.upper()}")
                lines.append(f"**Expected Impact**: {suggestion.expected_impact}")
                lines.append("")
                lines.append(suggestion.description)
                lines.append("")
                lines.append("**Implementation**:")
                lines.append("```python")
                lines.append(suggestion.implementation)
                lines.append("```")
                lines.append("")
        
        # Feature Suggestions
        if self.feature_suggestions:
            lines.append("## 🔧 Feature Engineering Suggestions")
            lines.append("Consider creating these new features based on SHAP interaction analysis.")
            lines.append("")
            for suggestion in self.feature_suggestions:
                lines.append(f"- **{suggestion.title}**: {suggestion.description}")
            lines.append("")
        
        # Retrain Recommendation
        lines.append("## 🔄 Retrain Recommendation")
        if self.retrain_recommendation:
            lines.append(f"**⚠️ RETRAIN RECOMMENDED**: {self.retrain_reason}")
        else:
            lines.append("✅ No immediate retraining needed. Model is stable.")
        lines.append("")
        
        # Narrative (if present)
        if self.narrative:
            lines.append("## 📝 Executive Summary")
            lines.append(self.narrative)
            lines.append("")
        
        return "\n".join(lines)


class AdaptationEngine:
    """
    Analyzes SHAP values across windows to automatically suggest:
    1. New rules to add (high-importance features)
    2. Features to prune (zero importance)
    3. Interaction features to engineer
    4. When to retrain (drift detected)
    """
    
    def __init__(
        self,
        feature_names: List[str],
        history_window: int = 5,
        drift_rank_threshold: int = 10,
        prune_importance_threshold: float = 0.001,
        rule_importance_threshold: float = 0.03,
        rule_lift_threshold: float = 3.0,
    ):
        """
        Initialize the adaptation engine.
        
        Args:
            feature_names: List of feature names in order
            history_window: Number of past windows to track for drift detection
            drift_rank_threshold: Rank change threshold to trigger drift alert
            prune_importance_threshold: Features below this importance are prune candidates
            rule_importance_threshold: Features above this importance get rule suggestions
            rule_lift_threshold: Minimum lift for a rule to be suggested
        """
        self.feature_names = feature_names
        self.history_window = history_window
        self.drift_rank_threshold = drift_rank_threshold
        self.prune_importance_threshold = prune_importance_threshold
        self.rule_importance_threshold = rule_importance_threshold
        self.rule_lift_threshold = rule_lift_threshold
        
        # History tracking
        self.importance_history: List[Dict[str, float]] = []
        self.performance_history: List[Dict[str, float]] = []
    
    def analyze_window(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        window_info: Dict[str, Any],
        compute_interactions: bool = False,
    ) -> AdaptationReport:
        """
        Generate comprehensive adaptation report for a training window.
        
        Args:
            model: Trained XGBoost model
            X_test: Test features (n_samples, n_features)
            y_test: True labels
            y_pred: Predicted probabilities
            window_info: Dict with window metadata (window_idx, start, end, etc.)
            compute_interactions: Whether to compute SHAP interaction values (slow)
            
        Returns:
            AdaptationReport with all analysis results
        """
        window_idx = window_info.get("window_idx", len(self.importance_history))
        
        print(f"[Adaptation] Analyzing window {window_idx}...")
        
        # 1. Compute SHAP values and importance
        print("[Adaptation] Computing SHAP values...")
        shap_values, current_importance = self._compute_importance(model, X_test)
        
        # 2. Compute performance metrics
        metrics = self._compute_metrics(y_test, y_pred)
        
        # 3. Detect drift from history
        drift_alerts = self._detect_drift(current_importance)
        rising = [d for d in drift_alerts if d.direction == "rising"]
        falling = [d for d in drift_alerts if d.direction == "falling"]
        
        # 4. Identify pruning candidates
        prune_candidates = self._find_pruning_candidates(current_importance)
        
        # 5. Generate rule suggestions from top features
        print("[Adaptation] Generating rule suggestions...")
        rule_suggestions = self._generate_rule_suggestions(
            current_importance, X_test, y_test, y_pred
        )
        
        # 6. Analyze feature interactions (optional, slow)
        feature_suggestions = []
        if compute_interactions:
            print("[Adaptation] Computing SHAP interactions (this may take a while)...")
            feature_suggestions = self._analyze_interactions(model, X_test)
        
        # 7. Decide if retrain is needed
        retrain, reason = self._should_retrain(
            drift_alerts, rising, metrics
        )
        
        # 8. Update history
        self.importance_history.append(current_importance)
        self.performance_history.append(metrics)
        if len(self.importance_history) > self.history_window:
            self.importance_history.pop(0)
            self.performance_history.pop(0)
        
        print(f"[Adaptation] Analysis complete. {len(rule_suggestions)} rules suggested.")
        
        return AdaptationReport(
            window_idx=window_idx,
            timestamp=datetime.now(),
            model_performance=metrics,
            current_importance=current_importance,
            drift_alerts=drift_alerts,
            rising_features=rising,
            falling_features=falling,
            pruning_candidates=prune_candidates,
            rule_suggestions=rule_suggestions,
            feature_suggestions=feature_suggestions,
            retrain_recommendation=retrain,
            retrain_reason=reason,
        )
    
    def _compute_importance(
        self,
        model,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Compute SHAP values and mean absolute importance."""
        try:
            # Try TreeExplainer first (fast)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
                
        except Exception as e:
            print(f"[Adaptation] TreeExplainer failed: {e}")
            print("[Adaptation] Falling back to model feature importance...")
            
            # Try to get feature importance from XGBoost
            try:
                # First try: Use feature_importances_ attribute (sklearn API)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance = {
                        name: float(val)
                        for name, val in zip(self.feature_names, importances)
                    }
                else:
                    # Second try: Use booster's get_score with feature names mapping
                    booster = model.get_booster()
                    importance_dict = booster.get_score(importance_type='gain')
                    
                    # XGBoost may use f0, f1, etc. - map by index
                    importance = {}
                    for i, feat in enumerate(self.feature_names):
                        # Try both formats: f0, f1 or actual feature names
                        val = importance_dict.get(feat, 0)
                        if val == 0:
                            val = importance_dict.get(f'f{i}', 0)
                        importance[feat] = val
                
                # Normalize to sum to 1
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v / total for k, v in importance.items()}
                else:
                    # All zeros - distribute equally
                    importance = {k: 1.0 / len(self.feature_names) for k in self.feature_names}
                    
            except Exception as e2:
                print(f"[Adaptation] Feature importance fallback also failed: {e2}")
                # Last resort: equal weights
                importance = {k: 1.0 / len(self.feature_names) for k in self.feature_names}
            
            return np.array([]), importance
        
        # Compute mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance = {
            name: float(val)
            for name, val in zip(self.feature_names, mean_abs_shap)
        }
        
        # Normalize to sum to 1
        total = sum(importance.values()) or 1
        importance = {k: v / total for k, v in importance.items()}
        
        return shap_values, importance
    
    def _compute_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
        )
        
        # Binary predictions at 0.5 threshold
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        metrics = {
            "auc_pr": float(average_precision_score(y_test, y_pred)),
            "auc_roc": float(roc_auc_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_binary, zero_division=0)),
            "fraud_count": int(y_test.sum()),
            "test_size": int(len(y_test)),
        }
        
        # P@K metrics
        sorted_indices = np.argsort(y_pred)[::-1]
        for k in [50, 100, 200]:
            if len(y_test) >= k:
                top_k_labels = y_test[sorted_indices[:k]]
                metrics[f"p@{k}"] = float(top_k_labels.mean())
        
        return metrics
    
    def _detect_drift(
        self,
        current: Dict[str, float]
    ) -> List[DriftAlert]:
        """Detect feature importance drift from history."""
        if len(self.importance_history) < 1:
            # First window - all features are "new"
            return self._create_initial_alerts(current)
        
        previous = self.importance_history[-1]
        
        # Calculate ranks
        current_ranks = self._compute_ranks(current)
        prev_ranks = self._compute_ranks(previous)
        
        alerts = []
        for feature in current.keys():
            curr_rank = current_ranks.get(feature, len(current))
            prev_rank = prev_ranks.get(feature, len(previous))
            rank_change = prev_rank - curr_rank  # Positive = climbed (improved)
            importance_delta = current.get(feature, 0) - previous.get(feature, 0)
            
            if abs(rank_change) >= self.drift_rank_threshold:
                severity = self._classify_severity(abs(rank_change))
                direction = "rising" if rank_change > 0 else "falling"
                
                alerts.append(DriftAlert(
                    feature=feature,
                    previous_rank=prev_rank + 1,
                    current_rank=curr_rank + 1,
                    rank_change=rank_change,
                    importance_delta=importance_delta,
                    severity=severity,
                    direction=direction,
                ))
        
        # Sort by absolute rank change
        alerts.sort(key=lambda x: abs(x.rank_change), reverse=True)
        
        return alerts
    
    def _create_initial_alerts(
        self,
        current: Dict[str, float]
    ) -> List[DriftAlert]:
        """Create drift alerts for the first window (all features are 'new')."""
        current_ranks = self._compute_ranks(current)
        
        alerts = []
        for feature, importance in current.items():
            if importance > self.rule_importance_threshold:
                alerts.append(DriftAlert(
                    feature=feature,
                    previous_rank=len(current),  # Assume last
                    current_rank=current_ranks[feature] + 1,
                    rank_change=len(current) - current_ranks[feature],
                    importance_delta=importance,
                    severity="high" if importance > 0.05 else "medium",
                    direction="rising",
                ))
        
        return alerts
    
    def _compute_ranks(self, importance: Dict[str, float]) -> Dict[str, int]:
        """Compute ranks from importance dict (0 = most important)."""
        sorted_features = sorted(
            importance.keys(),
            key=lambda x: importance[x],
            reverse=True
        )
        return {f: r for r, f in enumerate(sorted_features)}
    
    def _classify_severity(self, rank_change: int) -> str:
        """Classify drift severity based on rank change magnitude."""
        if rank_change >= 30:
            return "critical"
        elif rank_change >= 20:
            return "high"
        elif rank_change >= 10:
            return "medium"
        return "low"
    
    def _find_pruning_candidates(
        self,
        current: Dict[str, float]
    ) -> List[str]:
        """Find features with near-zero importance."""
        candidates = [
            feature
            for feature, importance in current.items()
            if importance < self.prune_importance_threshold
        ]
        
        # Sort by importance (lowest first)
        candidates.sort(key=lambda x: current[x])
        
        return candidates
    
    def _generate_rule_suggestions(
        self,
        importance: Dict[str, float],
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[AdaptationSuggestion]:
        """Generate rule suggestions from high-importance features."""
        suggestions = []
        
        # Get top features above threshold
        top_features = [
            (feat, imp)
            for feat, imp in importance.items()
            if imp >= self.rule_importance_threshold
        ]
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        for feature, imp in top_features[:5]:  # Top 5 features
            suggestion = self._create_rule_for_feature(
                feature, imp, X_test, y_test, y_pred
            )
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _create_rule_for_feature(
        self,
        feature: str,
        importance: float,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> Optional[AdaptationSuggestion]:
        """Create a rule suggestion for a high-importance feature."""
        try:
            feature_idx = self.feature_names.index(feature)
        except ValueError:
            return None
        
        feature_values = X_test[:, feature_idx]
        base_fraud_rate = y_test.mean()
        
        if base_fraud_rate == 0:
            return None
        
        # Test different thresholds to find best rule
        best_rule = None
        best_lift = 1.0
        
        # For boolean features
        unique_values = np.unique(feature_values)
        if len(unique_values) <= 3:
            for val in unique_values:
                mask = feature_values == val
                if mask.sum() < 10:
                    continue
                
                fraud_rate = y_test[mask].mean()
                lift = fraud_rate / base_fraud_rate
                
                if lift > best_lift:
                    best_lift = lift
                    best_rule = {
                        "type": "equality",
                        "value": val,
                        "fraud_rate": fraud_rate,
                        "count": int(mask.sum()),
                    }
        else:
            # For continuous features, test percentile thresholds
            for percentile in [5, 10, 25, 50, 75, 90, 95]:
                threshold = np.percentile(feature_values, percentile)
                
                # Test "below threshold"
                mask_below = feature_values < threshold
                if mask_below.sum() >= 10:
                    fraud_rate = y_test[mask_below].mean()
                    lift = fraud_rate / base_fraud_rate
                    
                    if lift > best_lift:
                        best_lift = lift
                        best_rule = {
                            "type": "less_than",
                            "value": threshold,
                            "fraud_rate": fraud_rate,
                            "count": int(mask_below.sum()),
                        }
                
                # Test "above threshold"
                mask_above = feature_values > threshold
                if mask_above.sum() >= 10:
                    fraud_rate = y_test[mask_above].mean()
                    lift = fraud_rate / base_fraud_rate
                    
                    if lift > best_lift:
                        best_lift = lift
                        best_rule = {
                            "type": "greater_than",
                            "value": threshold,
                            "fraud_rate": fraud_rate,
                            "count": int(mask_above.sum()),
                        }
        
        if best_rule is None or best_lift < self.rule_lift_threshold:
            return None
        
        # Generate implementation code
        if best_rule["type"] == "equality":
            condition = f"listing['{feature}'] == {best_rule['value']}"
            sql_condition = f"{feature} = {best_rule['value']}"
        elif best_rule["type"] == "less_than":
            condition = f"listing['{feature}'] < {best_rule['value']:.4f}"
            sql_condition = f"{feature} < {best_rule['value']:.4f}"
        else:  # greater_than
            condition = f"listing['{feature}'] > {best_rule['value']:.4f}"
            sql_condition = f"{feature} > {best_rule['value']:.4f}"
        
        implementation = f"""# Python Rule
if {condition}:
    flag_priority = "HIGH"  # {best_lift:.1f}x lift, {best_rule['fraud_rate']:.1%} fraud rate

# SQL Query
SELECT * FROM listings 
WHERE {sql_condition}
AND status = 'PENDING_APPROVAL';"""
        
        priority = "critical" if best_lift > 10 else "high" if best_lift > 5 else "medium"
        
        return AdaptationSuggestion(
            type="rule",
            priority=priority,
            title=f"High-risk pattern: {feature}",
            description=(
                f"Listings matching this condition have {best_lift:.1f}x higher fraud rate "
                f"({best_rule['fraud_rate']:.1%} vs {base_fraud_rate:.1%} baseline). "
                f"This pattern covers {best_rule['count']:,} listings in the test set."
            ),
            implementation=implementation,
            expected_impact=f"{best_lift:.1f}x lift over baseline ({best_rule['fraud_rate']:.1%} fraud rate)",
            shap_evidence={feature: importance},
            metadata=best_rule,
        )
    
    def _analyze_interactions(
        self,
        model,
        X_test: np.ndarray,
        sample_size: int = 500,
    ) -> List[AdaptationSuggestion]:
        """Analyze SHAP interaction values to suggest new features."""
        suggestions = []
        
        try:
            explainer = shap.TreeExplainer(model)
            
            # Sample for speed
            if len(X_test) > sample_size:
                indices = np.random.choice(len(X_test), sample_size, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test
            
            # Compute interaction values
            interaction_values = explainer.shap_interaction_values(X_sample)
            
            # Handle multi-class
            if isinstance(interaction_values, list):
                interaction_values = interaction_values[1]
            elif len(interaction_values.shape) == 4:
                interaction_values = interaction_values[:, :, :, 1]
            
            # Get mean absolute interaction for each pair
            mean_interactions = np.abs(interaction_values).mean(axis=0)
            
            # Find top interactions (excluding diagonal)
            n_features = len(self.feature_names)
            interactions = []
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    strength = mean_interactions[i, j]
                    if strength > 0.001:  # Threshold for significance
                        interactions.append({
                            "feature1": self.feature_names[i],
                            "feature2": self.feature_names[j],
                            "strength": float(strength),
                        })
            
            # Sort by strength
            interactions.sort(key=lambda x: x["strength"], reverse=True)
            
            # Create suggestions for top 3
            for inter in interactions[:3]:
                f1, f2 = inter["feature1"], inter["feature2"]
                suggestions.append(AdaptationSuggestion(
                    type="feature",
                    priority="medium",
                    title=f"Interaction: {f1} × {f2}",
                    description=(
                        f"Strong interaction detected between {f1} and {f2}. "
                        f"Consider creating a composite feature."
                    ),
                    implementation=f"""# Create interaction feature
df['{f1}_{f2}_interaction'] = df['{f1}'] * df['{f2}']
df['{f1}_{f2}_ratio'] = df['{f1}'] / (df['{f2}'] + 1)""",
                    expected_impact=f"Interaction strength: {inter['strength']:.4f}",
                    shap_evidence={f1: inter["strength"], f2: inter["strength"]},
                ))
            
        except Exception as e:
            print(f"[Adaptation] Interaction analysis failed: {e}")
        
        return suggestions
    
    def _should_retrain(
        self,
        drift_alerts: List[DriftAlert],
        rising: List[DriftAlert],
        metrics: Dict[str, float],
    ) -> Tuple[bool, str]:
        """Decide if model should be retrained."""
        reasons = []
        
        # Check for critical drift
        critical_drift = [a for a in drift_alerts if a.severity == "critical"]
        if len(critical_drift) >= 3:
            reasons.append(
                f"{len(critical_drift)} features with critical importance drift"
            )
        
        # Check for many rising features (new fraud patterns)
        if len(rising) >= 5:
            reasons.append(
                f"{len(rising)} rising features - possible new fraud tactics"
            )
        
        # Check for performance drop (if we have history)
        if len(self.performance_history) >= 2:
            prev_auc = self.performance_history[-1].get("auc_pr", 0)
            curr_auc = metrics.get("auc_pr", 0)
            
            if prev_auc > 0 and curr_auc < prev_auc * 0.95:  # 5% drop
                reasons.append(
                    f"AUC-PR dropped {((prev_auc - curr_auc) / prev_auc * 100):.1f}%"
                )
        
        if reasons:
            return True, "; ".join(reasons)
        
        return False, "Model is stable"
    
    def save_state(self, output_path: str) -> None:
        """Save engine state for persistence across sessions."""
        state = {
            "feature_names": self.feature_names,
            "importance_history": self.importance_history,
            "performance_history": self.performance_history,
            "config": {
                "history_window": self.history_window,
                "drift_rank_threshold": self.drift_rank_threshold,
                "prune_importance_threshold": self.prune_importance_threshold,
                "rule_importance_threshold": self.rule_importance_threshold,
                "rule_lift_threshold": self.rule_lift_threshold,
            }
        }
        
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, input_path: str) -> "AdaptationEngine":
        """Load engine from saved state."""
        with open(input_path, "rb") as f:
            state = pickle.load(f)
        
        engine = cls(
            feature_names=state["feature_names"],
            **state["config"]
        )
        engine.importance_history = state["importance_history"]
        engine.performance_history = state["performance_history"]
        
        return engine


def run_adaptation_analysis(
    model_path: str,
    output_dir: str = "artifacts/reports",
    compute_interactions: bool = False,
) -> AdaptationReport:
    """
    Convenience function to run adaptation analysis on a saved model.
    
    Args:
        model_path: Path to saved model pickle
        output_dir: Directory for output reports
        compute_interactions: Whether to compute SHAP interactions (slow)
        
    Returns:
        AdaptationReport
    """
    from src.utils.explainability import load_saved_model
    
    print(f"Loading model from {model_path}...")
    bundle = load_saved_model(model_path)
    
    # Initialize engine
    engine = AdaptationEngine(
        feature_names=bundle["features"],
        history_window=5,
    )
    
    # Check for existing state
    state_path = Path(output_dir) / "adaptation_engine_state.pkl"
    if state_path.exists():
        print(f"Loading engine state from {state_path}...")
        engine = AdaptationEngine.load_state(str(state_path))
    
    # Run analysis
    report = engine.analyze_window(
        model=bundle["model"],
        X_test=bundle["X_test"],
        y_test=bundle["y_test"],
        y_pred=bundle["y_pred"],
        window_info=bundle.get("window_info", {}),
        compute_interactions=compute_interactions,
    )
    
    # Save outputs
    window_idx = report.window_idx
    report.save_markdown(f"{output_dir}/adaptation_report_window_{window_idx}.md")
    report.save_json(f"{output_dir}/adaptation_report_window_{window_idx}.json")
    engine.save_state(str(state_path))
    
    print(f"Report saved to {output_dir}/adaptation_report_window_{window_idx}.md")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run adaptation analysis on saved model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/models/baseline/model_window_99.pkl",
        help="Path to saved model pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--interactions",
        action="store_true",
        help="Compute SHAP interaction values (slow)",
    )
    
    args = parser.parse_args()
    
    report = run_adaptation_analysis(
        model_path=args.model_path,
        output_dir=args.output_dir,
        compute_interactions=args.interactions,
    )
    
    print("\n" + "=" * 60)
    print("ADAPTATION SUMMARY")
    print("=" * 60)
    print(f"Window: {report.window_idx}")
    print(f"AUC-PR: {report.model_performance.get('auc_pr', 0):.4f}")
    print(f"Drift Alerts: {len(report.drift_alerts)}")
    print(f"Rising Features: {len(report.rising_features)}")
    print(f"Pruning Candidates: {len(report.pruning_candidates)}")
    print(f"Rule Suggestions: {len(report.rule_suggestions)}")
    print(f"Retrain Recommended: {report.retrain_recommendation}")
    if report.retrain_reason:
        print(f"Reason: {report.retrain_reason}")


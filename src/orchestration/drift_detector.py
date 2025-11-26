"""
Drift Detection for Continuous Fraud Detection Pipeline.

Detects feature distribution drift between training and production data.

Methods:
- PSI (Population Stability Index): Recommended for continuous features
- KS Test: Kolmogorov-Smirnov test for distribution comparison
- Mean Shift: Simple normalized mean difference

Usage:
    detector = DriftDetector(training_stats)
    report = detector.detect_drift(current_data)
    if report.has_critical_drift:
        alert("Critical drift detected!")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import numpy as np
import polars as pl


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""
    feature: str
    method: str
    score: float
    threshold: float
    is_drifted: bool
    severity: str  # "critical", "warning", "normal"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Complete drift report for all features."""
    timestamp: datetime
    total_features: int
    drifted_features: int
    critical_count: int
    warning_count: int
    feature_drifts: List[FeatureDrift]
    has_critical_drift: bool
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_features": int(self.total_features),
            "drifted_features": int(self.drifted_features),
            "critical_count": int(self.critical_count),
            "warning_count": int(self.warning_count),
            "has_critical_drift": bool(self.has_critical_drift),
            "summary": str(self.summary),
            "feature_drifts": [
                {
                    "feature": str(fd.feature),
                    "method": str(fd.method),
                    "score": float(fd.score),
                    "threshold": float(fd.threshold),
                    "is_drifted": bool(fd.is_drifted),
                    "severity": str(fd.severity),
                }
                for fd in self.feature_drifts
            ],
        }
    
    def save_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DriftDetector:
    """
    Detects feature distribution drift.
    
    Supports three methods:
    - PSI: Population Stability Index (recommended)
    - KS: Kolmogorov-Smirnov test
    - mean_shift: Simple normalized mean difference
    """
    
    def __init__(
        self,
        training_stats: Optional[Dict[str, Dict]] = None,
        method: str = "psi",
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,  # p-value threshold
        mean_shift_threshold: float = 2.0,  # standard deviations
        critical_threshold_multiplier: float = 1.5,
    ):
        """
        Initialize drift detector.
        
        Args:
            training_stats: Pre-computed training statistics per feature
            method: Detection method ("psi", "ks", "mean_shift")
            psi_threshold: PSI threshold for drift (>0.2 = significant)
            ks_threshold: KS test p-value threshold (<0.05 = drift)
            mean_shift_threshold: Mean shift in std deviations
            critical_threshold_multiplier: Multiplier for critical severity
        """
        self.training_stats = training_stats or {}
        self.method = method
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.mean_shift_threshold = mean_shift_threshold
        self.critical_multiplier = critical_threshold_multiplier
    
    def compute_training_stats(
        self,
        df: pl.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, Dict]:
        """
        Compute training statistics for drift detection.
        
        Args:
            df: Training data
            feature_columns: Feature columns to track
            
        Returns:
            Dict of feature -> statistics
        """
        stats = {}
        
        for col in feature_columns:
            if col not in df.columns:
                continue
            
            # Get column type
            dtype = df[col].dtype
            
            # Skip boolean columns - they don't work well with histogram
            if dtype == pl.Boolean:
                continue
            
            values = df[col].drop_nulls().to_numpy()
            
            if len(values) == 0:
                continue
            
            # Convert to float for numerical operations
            try:
                values = values.astype(np.float64)
            except (ValueError, TypeError):
                # Skip non-numeric columns
                continue
            
            # Check for valid numeric data
            if not np.issubdtype(values.dtype, np.number):
                continue
            
            stats[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "n_samples": len(values),
            }
            
            # Compute histogram for PSI
            if self.method == "psi":
                try:
                    hist, bin_edges = np.histogram(values, bins=10)
                    stats[col]["hist"] = hist.tolist()
                    stats[col]["bin_edges"] = bin_edges.tolist()
                except Exception:
                    # Skip if histogram fails
                    del stats[col]
                    continue
        
        self.training_stats = stats
        return stats
    
    def detect_drift(
        self,
        current_data: pl.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Detect drift between training and current data.
        
        Args:
            current_data: Current/production data
            feature_columns: Features to check (default: all in training_stats)
            
        Returns:
            DriftReport with detailed results
        """
        if not self.training_stats:
            raise ValueError("Training stats not initialized. Call compute_training_stats first.")
        
        if feature_columns is None:
            feature_columns = list(self.training_stats.keys())
        
        feature_drifts = []
        
        for col in feature_columns:
            if col not in self.training_stats:
                continue
            
            if col not in current_data.columns:
                continue
            
            # Skip boolean columns
            if current_data[col].dtype == pl.Boolean:
                continue
            
            current_values = current_data[col].drop_nulls().to_numpy()
            
            # Convert to float
            try:
                current_values = current_values.astype(np.float64)
            except (ValueError, TypeError):
                continue
            
            if len(current_values) == 0:
                continue
            
            # Compute drift score based on method
            if self.method == "psi":
                drift = self._compute_psi_drift(col, current_values)
            elif self.method == "ks":
                drift = self._compute_ks_drift(col, current_values)
            else:  # mean_shift
                drift = self._compute_mean_shift_drift(col, current_values)
            
            feature_drifts.append(drift)
        
        # Compute summary
        drifted = [fd for fd in feature_drifts if fd.is_drifted]
        critical = [fd for fd in feature_drifts if fd.severity == "critical"]
        warning = [fd for fd in feature_drifts if fd.severity == "warning"]
        
        has_critical = len(critical) > 0
        
        summary = self._generate_summary(feature_drifts, critical, warning)
        
        return DriftReport(
            timestamp=datetime.now(),
            total_features=len(feature_drifts),
            drifted_features=len(drifted),
            critical_count=len(critical),
            warning_count=len(warning),
            feature_drifts=sorted(
                feature_drifts,
                key=lambda x: x.score,
                reverse=True
            ),
            has_critical_drift=has_critical,
            summary=summary,
        )
    
    def _compute_psi_drift(
        self,
        feature: str,
        current_values: np.ndarray,
    ) -> FeatureDrift:
        """Compute PSI (Population Stability Index) drift."""
        train_stats = self.training_stats[feature]
        
        # Use same bins as training
        bin_edges = np.array(train_stats["bin_edges"])
        expected = np.array(train_stats["hist"])
        
        # Compute current histogram with same bins
        actual, _ = np.histogram(current_values, bins=bin_edges)
        
        # Normalize to proportions
        expected_prop = (expected + 1) / (expected.sum() + len(expected))  # Add 1 for smoothing
        actual_prop = (actual + 1) / (actual.sum() + len(actual))
        
        # PSI formula
        psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
        
        # Determine severity
        is_drifted = psi > self.psi_threshold
        if psi > self.psi_threshold * self.critical_multiplier:
            severity = "critical"
        elif is_drifted:
            severity = "warning"
        else:
            severity = "normal"
        
        return FeatureDrift(
            feature=feature,
            method="psi",
            score=float(psi),
            threshold=self.psi_threshold,
            is_drifted=is_drifted,
            severity=severity,
            details={
                "expected_mean": train_stats["mean"],
                "actual_mean": float(np.mean(current_values)),
            },
        )
    
    def _compute_ks_drift(
        self,
        feature: str,
        current_values: np.ndarray,
    ) -> FeatureDrift:
        """Compute Kolmogorov-Smirnov test drift."""
        from scipy import stats
        
        train_stats = self.training_stats[feature]
        
        # Generate samples from training distribution (approximate)
        train_mean = train_stats["mean"]
        train_std = train_stats["std"]
        n_train = train_stats["n_samples"]
        
        # For KS test, we need the original data or simulate from distribution
        # Here we use a normal approximation
        train_samples = np.random.normal(train_mean, train_std, min(n_train, 1000))
        
        # KS test
        ks_stat, p_value = stats.ks_2samp(train_samples, current_values)
        
        is_drifted = p_value < self.ks_threshold
        if p_value < self.ks_threshold / 10:  # Very significant
            severity = "critical"
        elif is_drifted:
            severity = "warning"
        else:
            severity = "normal"
        
        return FeatureDrift(
            feature=feature,
            method="ks",
            score=float(ks_stat),
            threshold=self.ks_threshold,
            is_drifted=is_drifted,
            severity=severity,
            details={
                "p_value": float(p_value),
                "expected_mean": train_mean,
                "actual_mean": float(np.mean(current_values)),
            },
        )
    
    def _compute_mean_shift_drift(
        self,
        feature: str,
        current_values: np.ndarray,
    ) -> FeatureDrift:
        """Compute simple mean shift drift."""
        train_stats = self.training_stats[feature]
        
        train_mean = train_stats["mean"]
        train_std = train_stats["std"]
        current_mean = float(np.mean(current_values))
        
        # Normalized shift in standard deviations
        if train_std > 0:
            shift = abs(current_mean - train_mean) / train_std
        else:
            shift = 0 if current_mean == train_mean else float('inf')
        
        is_drifted = shift > self.mean_shift_threshold
        if shift > self.mean_shift_threshold * self.critical_multiplier:
            severity = "critical"
        elif is_drifted:
            severity = "warning"
        else:
            severity = "normal"
        
        return FeatureDrift(
            feature=feature,
            method="mean_shift",
            score=float(shift),
            threshold=self.mean_shift_threshold,
            is_drifted=is_drifted,
            severity=severity,
            details={
                "expected_mean": train_mean,
                "actual_mean": current_mean,
                "std_deviations": float(shift),
            },
        )
    
    def _generate_summary(
        self,
        all_drifts: List[FeatureDrift],
        critical: List[FeatureDrift],
        warning: List[FeatureDrift],
    ) -> str:
        """Generate human-readable summary."""
        if not all_drifts:
            return "No features analyzed."
        
        if not critical and not warning:
            return f"No drift detected across {len(all_drifts)} features. Model is stable."
        
        lines = []
        
        if critical:
            lines.append(f"⚠️ CRITICAL DRIFT in {len(critical)} features:")
            for fd in critical[:5]:
                lines.append(f"  - {fd.feature}: {fd.method}={fd.score:.4f} (threshold: {fd.threshold})")
        
        if warning:
            lines.append(f"Warning: Drift detected in {len(warning)} additional features.")
        
        return "\n".join(lines)
    
    def save_stats(self, path: str) -> None:
        """Save training statistics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.training_stats, f, indent=2)
    
    def load_stats(self, path: str) -> None:
        """Load training statistics from JSON."""
        with open(path) as f:
            self.training_stats = json.load(f)


if __name__ == "__main__":
    # Example usage
    import polars as pl
    
    # Create dummy training data
    np.random.seed(42)
    train_df = pl.DataFrame({
        "feature_a": np.random.normal(10, 2, 1000),
        "feature_b": np.random.normal(0, 1, 1000),
    })
    
    # Create current data with drift
    current_df = pl.DataFrame({
        "feature_a": np.random.normal(12, 2, 200),  # Mean shifted!
        "feature_b": np.random.normal(0.2, 1.5, 200),  # Slight shift
    })
    
    # Detect drift
    detector = DriftDetector(method="psi")
    detector.compute_training_stats(train_df, ["feature_a", "feature_b"])
    
    report = detector.detect_drift(current_df)
    
    print("=== Drift Report ===")
    print(f"Total features: {report.total_features}")
    print(f"Drifted: {report.drifted_features}")
    print(f"Critical: {report.critical_count}")
    print(f"\nSummary:\n{report.summary}")


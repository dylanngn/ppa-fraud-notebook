"""
Drift Detection for ML Production Monitoring.

Streaming Drift: MultiMethodDriftDetector (ADWIN, DDM, EDDM, Page-Hinkley)
Data Drift: BatchDriftMonitor (PSI, Kolmogorov-Smirnov, Jensen-Shannon)
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from river import drift
from scipy import stats


class MultiMethodDriftDetector:
    """
    Ensemble drift detector using ADWIN, DDM, EDDM, and Page-Hinkley.
    Drift signaled when ≥consensus_threshold methods agree.
    """
    
    def __init__(
        self,
        consensus_threshold: int = 2,
        adwin_delta: float = 0.002,
        ddm_min_instances: int = 30,
        ph_threshold: float = 50.0,
        ph_alpha: float = 0.9999,
    ):
        self.consensus_threshold = consensus_threshold
        self.adwin_delta = adwin_delta
        self.ddm_min_instances = ddm_min_instances
        self.ph_threshold = ph_threshold
        self.ph_alpha = ph_alpha
        
        self._init_detectors()
        
        self.n_updates = 0
        self.n_drifts = 0
        self.drift_counts = {name: 0 for name in self.detectors}
        self.last_triggered: List[str] = []
    
    def _init_detectors(self):
        """Initialize all detector instances."""
        self.detectors = {
            "adwin": drift.ADWIN(delta=self.adwin_delta),
            "ddm": drift.binary.DDM(min_num_instances=self.ddm_min_instances),
            "eddm": drift.binary.EDDM(min_num_instances=self.ddm_min_instances),
            "page_hinkley": drift.PageHinkley(
                threshold=self.ph_threshold, 
                alpha=self.ph_alpha
            ),
        }
    
    def update(self, value: float) -> bool:
        """Update all detectors. Returns True if consensus drift detected."""
        self.n_updates += 1
        triggered = []
        
        for name, detector in self.detectors.items():
            detector.update(value)
            if detector.drift_detected:
                triggered.append(name)
                self.drift_counts[name] += 1
        
        self.last_triggered = triggered
        
        if len(triggered) >= self.consensus_threshold:
            self.n_drifts += 1
            return True
        return False
    
    def update_batch(self, values: np.ndarray) -> List[Tuple[int, List[str]]]:
        """Process batch, return list of (index, triggered_methods) for drift points."""
        drift_points = []
        for i, val in enumerate(values):
            if self.update(float(val)):
                drift_points.append((i, self.last_triggered.copy()))
        return drift_points
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "n_updates": self.n_updates,
            "n_drifts": self.n_drifts,
            "drift_counts_by_method": self.drift_counts.copy(),
            "consensus_threshold": self.consensus_threshold,
        }
    
    def reset(self):
        """Reset all detector states."""
        self._init_detectors()
        self.n_updates = 0
        self.n_drifts = 0
        self.drift_counts = {name: 0 for name in self.detectors}
        self.last_triggered = []


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index. PSI < 0.1: stable, 0.1-0.25: moderate, ≥0.25: significant.
    """
    if len(expected) < bins or len(actual) < bins:
        return 0.0
    
    # Create bins from expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)
    
    # Ensure actual values fall within bin range
    bin_edges[0] = min(bin_edges[0], np.min(actual))
    bin_edges[-1] = max(bin_edges[-1], np.max(actual))
    
    expected_hist, _ = np.histogram(expected, bins=bin_edges)
    actual_hist, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to percentages with smoothing
    expected_pct = (expected_hist + 1e-6) / (len(expected) + bins * 1e-6)
    actual_pct = (actual_hist + 1e-6) / (len(actual) + bins * 1e-6)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


def ks_test_drift(
    reference: np.ndarray, 
    current: np.ndarray, 
    alpha: float = 0.05
) -> Tuple[bool, float, float]:
    """Kolmogorov-Smirnov test. Returns (drift_detected, p_value, ks_statistic)."""
    if len(reference) < 2 or len(current) < 2:
        return False, 1.0, 0.0
    
    stat, p_value = stats.ks_2samp(reference, current)
    return p_value < alpha, float(p_value), float(stat)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, bins: int = 10) -> float:
    """Jensen-Shannon Divergence (0 = identical, 1 = maximally different)."""
    if len(p) < bins or len(q) < bins:
        return 0.0
    
    combined = np.concatenate([p, q])
    _, bin_edges = np.histogram(combined, bins=bins)
    
    p_hist, _ = np.histogram(p, bins=bin_edges, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    
    p_hist = (p_hist + 1e-10) / (np.sum(p_hist) + bins * 1e-10)
    q_hist = (q_hist + 1e-10) / (np.sum(q_hist) + bins * 1e-10)
    
    m = 0.5 * (p_hist + q_hist)
    
    kl_pm = np.sum(p_hist * np.log(p_hist / m))
    kl_qm = np.sum(q_hist * np.log(q_hist / m))
    
    return float(0.5 * kl_pm + 0.5 * kl_qm)


class BatchDriftMonitor:
    """Monitor feature and prediction distribution drift using PSI, KS, and JSD."""
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        jsd_threshold: float = 0.1,
        feature_psi_threshold: float = 0.25,
    ):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.jsd_threshold = jsd_threshold
        self.feature_psi_threshold = feature_psi_threshold
        
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_features: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
    
    def set_reference(
        self,
        predictions: np.ndarray,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """Set reference/baseline distributions."""
        self.reference_predictions = np.asarray(predictions).flatten()
        if features is not None:
            self.reference_features = np.asarray(features)
        self.feature_names = feature_names
    
    def check_drift(
        self,
        current_predictions: np.ndarray,
        current_features: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Check for drift against reference distributions."""
        if self.reference_predictions is None:
            raise ValueError("Reference not set. Call set_reference() first.")
        
        current_preds = np.asarray(current_predictions).flatten()
        
        results: Dict[str, Any] = {
            "n_reference": len(self.reference_predictions),
            "n_current": len(current_preds),
        }
        
        pred_psi = calculate_psi(self.reference_predictions, current_preds)
        ks_drift, ks_p, ks_stat = ks_test_drift(
            self.reference_predictions, current_preds, self.ks_alpha
        )
        pred_jsd = jensen_shannon_divergence(self.reference_predictions, current_preds)
        
        results["prediction_psi"] = pred_psi
        results["prediction_psi_drift"] = pred_psi > self.psi_threshold
        results["prediction_ks_pvalue"] = ks_p
        results["prediction_ks_stat"] = ks_stat
        results["prediction_ks_drift"] = ks_drift
        results["prediction_jsd"] = pred_jsd
        results["prediction_jsd_drift"] = pred_jsd > self.jsd_threshold
        
        pred_drift_methods = sum([
            results["prediction_psi_drift"],
            results["prediction_ks_drift"],
            results["prediction_jsd_drift"],
        ])
        results["prediction_drift"] = pred_drift_methods >= 2
        
        if current_features is not None and self.reference_features is not None:
            current_feats = np.asarray(current_features)
            
            if current_feats.ndim == 1:
                current_feats = current_feats.reshape(-1, 1)
            if self.reference_features.ndim == 1:
                ref_feats = self.reference_features.reshape(-1, 1)
            else:
                ref_feats = self.reference_features
            
            n_features = min(current_feats.shape[1], ref_feats.shape[1])
            
            feature_psi = {}
            feature_ks = {}
            drifted_features = []
            
            for i in range(n_features):
                name = self.feature_names[i] if self.feature_names and i < len(self.feature_names) else f"feature_{i}"
                
                psi = calculate_psi(ref_feats[:, i], current_feats[:, i])
                ks_d, ks_pval, _ = ks_test_drift(ref_feats[:, i], current_feats[:, i])
                
                feature_psi[name] = psi
                feature_ks[name] = ks_pval
                
                if psi > self.feature_psi_threshold:
                    drifted_features.append(name)
            
            results["feature_psi"] = feature_psi
            results["feature_ks_pvalues"] = feature_ks
            results["max_feature_psi"] = max(feature_psi.values()) if feature_psi else 0.0
            results["drifted_features"] = drifted_features
            results["n_drifted_features"] = len(drifted_features)
            results["feature_drift"] = len(drifted_features) > 0
        else:
            results["feature_drift"] = False
            results["n_drifted_features"] = 0
        
        results["overall_drift"] = results["prediction_drift"] or results.get("feature_drift", False)
        
        return results
    
    def reset(self):
        """Reset reference distributions."""
        self.reference_predictions = None
        self.reference_features = None
        self.feature_names = None


DriftDetector = MultiMethodDriftDetector

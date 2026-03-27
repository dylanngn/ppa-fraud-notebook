"""
Stateful Drift Monitor Singleton.

Wraps BatchDriftMonitor from src/evaluation/drift.py so it can accumulate
predictions between API requests.  The monitor maintains a rolling reference
window and a rolling current window; once both are full enough it runs a
full drift check automatically.

Designed to be initialised once in the FastAPI lifespan and shared across
all request handlers.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftState:
    """
    Accumulates fraud-probability predictions and periodically checks for
    distribution shift using PSI and KS tests.

    Args:
        reference_window: Number of predictions to use as the stable reference
                          distribution.  The reference is set once this many
                          predictions have been collected.
        current_window:   Size of the rolling window compared against the
                          reference on each status request.
        check_interval:   How many new predictions to wait between full drift
                          checks (avoids running scipy every single request).
    """

    def __init__(
        self,
        reference_window: int = 100,
        current_window: int = 50,
        check_interval: int = 10,
    ) -> None:
        self.reference_window = reference_window
        self.current_window = current_window
        self.check_interval = check_interval

        self._all_predictions: Deque[float] = deque(maxlen=reference_window + current_window)
        self._reference_predictions: List[float] = []
        self._reference_set: bool = False
        self._predictions_since_check: int = 0

        self._last_check: Optional[datetime] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_prediction(self, fraud_probability: float) -> None:
        """Record a new model prediction for drift tracking."""
        with self._lock:
            self._all_predictions.append(fraud_probability)
            self._predictions_since_check += 1

            # Set reference once we have enough data
            if not self._reference_set and len(self._all_predictions) >= self.reference_window:
                self._reference_predictions = list(self._all_predictions)[: self.reference_window]
                self._reference_set = True
                logger.info(
                    f"Drift reference set from {self.reference_window} predictions "
                    f"(mean={np.mean(self._reference_predictions):.3f})"
                )

            # Periodically run a full drift check
            if self._reference_set and self._predictions_since_check >= self.check_interval:
                self._run_check()

    def check_drift(self) -> Optional[Dict[str, Any]]:
        """Force a drift check and return the result (or None if not ready)."""
        with self._lock:
            if not self._reference_set:
                return None
            self._run_check()
            return self._last_result

    def get_status(self) -> Dict[str, Any]:
        """Return the current drift status for the dashboard."""
        with self._lock:
            current = list(self._all_predictions)[-self.current_window:]

            base: Dict[str, Any] = {
                "reference_size": len(self._reference_predictions),
                "current_size": len(current),
                "reference_set": self._reference_set,
                "last_checked": self._last_check.isoformat() if self._last_check else None,
                "score_distribution": self._bucket_distribution(current),
            }

            if self._last_result:
                # Per-feature PSI values (if available from BatchDriftMonitor)
                raw_fpsi = self._last_result.get("feature_psi", {})
                feature_psi = {str(k): round(float(v), 4) for k, v in raw_fpsi.items()} if raw_fpsi else {}

                base.update({
                    "prediction_psi": round(float(self._last_result.get("prediction_psi", 0.0)), 4),
                    "prediction_ks_drift": bool(self._last_result.get("prediction_ks_drift", False)),
                    "overall_drift": bool(self._last_result.get("overall_drift", False)),
                    "drifted_features": [str(f) for f in self._last_result.get("drifted_features", [])],
                    "drift_label": self._drift_label(float(self._last_result.get("prediction_psi", 0.0))),
                    "feature_psi": feature_psi,
                })
            else:
                base.update({
                    "prediction_psi": 0.0,
                    "prediction_ks_drift": False,
                    "overall_drift": False,
                    "drifted_features": [],
                    "drift_label": "COLLECTING DATA",
                    "feature_psi": {},
                })

            return base

    def reset(self) -> None:
        """Reset the reference distribution (useful for demo resets)."""
        with self._lock:
            self._all_predictions.clear()
            self._reference_predictions = []
            self._reference_set = False
            self._last_check = None
            self._last_result = None
            self._predictions_since_check = 0
        logger.info("Drift state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_check(self) -> None:
        """Run BatchDriftMonitor.check_drift() (must be called with lock held)."""
        try:
            from src.evaluation.drift import BatchDriftMonitor

            current = list(self._all_predictions)[-self.current_window:]
            if len(current) < 10:
                return

            monitor = BatchDriftMonitor()
            monitor.set_reference(
                np.array(self._reference_predictions),
                features=None,
                feature_names=[],
            )
            result = monitor.check_drift(
                np.array(current),
                current_features=None,
            )
            self._last_result = result
            self._last_check = datetime.now(timezone.utc)
            self._predictions_since_check = 0

        except Exception as exc:
            logger.warning(f"Drift check failed: {exc}")

    @staticmethod
    def _bucket_distribution(predictions: List[float]) -> Dict[str, int]:
        """Count predictions per probability band for the histogram."""
        buckets = {"0.0-0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0}
        for p in predictions:
            if p < 0.1:
                buckets["0.0-0.1"] += 1
            elif p < 0.3:
                buckets["0.1-0.3"] += 1
            elif p < 0.5:
                buckets["0.3-0.5"] += 1
            elif p < 0.7:
                buckets["0.5-0.7"] += 1
            else:
                buckets["0.7-1.0"] += 1
        return buckets

    @staticmethod
    def _drift_label(psi: float) -> str:
        if psi < 0.1:
            return "STABLE"
        if psi < 0.25:
            return "MODERATE SHIFT"
        return "SIGNIFICANT DRIFT"

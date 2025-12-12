"""
Drift Detection logic.
"""

from typing import List, Optional
import numpy as np

# Try importing river, else fallback
try:
    from river import drift
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False

class DriftDetector:
    """
    Wrapper for Concept Drift Detection (ADWIN).
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.detector = None
        if HAS_RIVER:
            self.detector = drift.ADWIN(delta=delta)
        else:
            print("Warning: river not installed. Drift detection disabled.")
            
    def update(self, val: float) -> bool:
        """
        Update detector with a single data point (e.g. error rate or mean).
        Returns True if drift detected.
        """
        if self.detector:
            self.detector.update(val)
            return self.detector.drift_detected
        return False
    
    def reset(self):
        if self.detector:
            self.detector.reset()

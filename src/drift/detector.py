"""
Drift Detection logic.
"""

from river import drift

class DriftDetector:
    """
    Wrapper for Concept Drift Detection (ADWIN).
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.detector = drift.ADWIN(delta=delta)
            
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

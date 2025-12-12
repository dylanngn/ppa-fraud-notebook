"""
Experience Replay Buffer for Incremental GNN Learning.
Prioritizes fraud samples to prevent catastrophic forgetting of rare patterns.
"""

import numpy as np
from typing import List, Dict, Optional

class ExperienceReplayBuffer:
    """
    Maintain a buffer of representative samples for incremental GNN training.
    Prioritizes fraud samples to prevent forgetting rare patterns.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        fraud_priority_weight: float = 3.0,
    ):
        self.max_size = max_size
        self.fraud_priority_weight = fraud_priority_weight
        self.buffer: List[Dict] = []
        self.fraud_indices = set()
    
    def add_batch(
        self,
        node_indices: List[str], # Changed to str for consistency with listing_id
        labels: np.ndarray,
        embeddings: np.ndarray,
    ):
        """Add new samples to buffer with priority sampling."""
        
        for idx, label, emb in zip(node_indices, labels, embeddings):
            priority = self.fraud_priority_weight if label == 1 else 1.0
            self.buffer.append({
                "idx": idx,
                "label": int(label),
                "embedding": emb,
                "priority": priority,
            })
        
        # Trim buffer if needed (keep high-priority samples)
        if len(self.buffer) > self.max_size:
            self._trim_buffer()
    
    def _trim_buffer(self):
        """Remove low-priority samples to maintain buffer size."""
        # Sort by priority (descending), keep top max_size
        # To break ties randomly, we could shuffle first, but stable sort is fine.
        sorted_buffer = sorted(self.buffer, key=lambda x: x["priority"], reverse=True)
        self.buffer = sorted_buffer[:self.max_size]
    
    def sample(self, batch_size: int) -> Optional[Dict]:
        """Sample a batch for replay during training."""
        if len(self.buffer) == 0:
            return None
        
        # Weighted sampling by priority
        priorities = np.array([item["priority"] for item in self.buffer])
        probs = priorities / priorities.sum()
        
        # Sample indices
        sample_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=sample_size, replace=False, p=probs)
        
        batch_items = [self.buffer[i] for i in indices]
        
        return {
            "indices": [item["idx"] for item in batch_items],
            "labels": np.array([item["label"] for item in batch_items]),
            "embeddings": np.stack([item["embedding"] for item in batch_items]),
        }

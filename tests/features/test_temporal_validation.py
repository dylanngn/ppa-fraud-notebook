"""
Tests for Feature Store - Temporal Validation

Ensures no time-travel violations occur.
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.store import FeatureStore


class TestFeatureStoreTemporal:
    """Test temporal consistency guarantees."""
    
    def setup_method(self):
        """Set up feature store for testing."""
        # This assumes you have artifacts/ directory with test data
        self.feature_store = FeatureStore()
        
    def test_no_future_features(self):
        """
        Test that we cannot get features from BEFORE a listing was created.
        
        This would be a time-travel violation.
        """
        # Get a test listing
        listing_id = self.feature_store.listing_nodes["insertion_id"][0]
        submission_time = self.feature_store.listing_nodes.filter(
            self.feature_store.listing_nodes["insertion_id"] == listing_id
        )["submission_at"][0]
        
        # Try to get features from 1 day BEFORE submission
        with pytest.raises(ValueError, match="Time travel violation"):
            past_time = submission_time - timedelta(days=1)
            self.feature_store.get_features(listing_id, as_of_time=past_time)
            
    def test_features_at_submission_time(self):
        """Test that we CAN get features AT the submission time."""
        listing_id = self.feature_store.listing_nodes["insertion_id"][0]
        submission_time = self.feature_store.listing_nodes.filter(
            self.feature_store.listing_nodes["insertion_id"] == listing_id
        )["submission_at"][0]
        
        # This should work
        snapshot = self.feature_store.get_features(listing_id, as_of_time=submission_time)
        assert snapshot.listing_id == listing_id
        assert snapshot.as_of_time == submission_time
        
    def test_features_after_submission_time(self):
        """Test that we CAN get features AFTER submission time."""
        listing_id = self.feature_store.listing_nodes["insertion_id"][0]
        submission_time = self.feature_store.listing_nodes.filter(
            self.feature_store.listing_nodes["insertion_id"] == listing_id
        )["submission_at"][0]
        
        # Get features 1 day AFTER submission
        future_time = submission_time + timedelta(days=1)
        snapshot = self.feature_store.get_features(listing_id, as_of_time=future_time)
        assert snapshot.listing_id == listing_id
        
    def test_cold_start_flag(self):
        """Test that listings without graph features are flagged as cold start."""
        # This test requires a listing that exists in listing_nodes but NOT in graph_features
        # For now, we'll test the general logic
        listing_id = self.feature_store.listing_nodes["insertion_id"][0]
        snapshot = self.feature_store.get_features(listing_id)
        
        # Confidence should be lower for cold start
        if snapshot.is_cold_start:
            assert snapshot.confidence < 1.0
        else:
            assert snapshot.confidence == 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

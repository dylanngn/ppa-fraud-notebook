"""
Internal Event Store for the Fraud Detection Microservice.

Stores events in-memory (with optional SQLite persistence) for:
1. Graph building - linking entities across events
2. Feature engineering - historical patterns
3. Real-time prediction - instant feature computation

The store is self-contained - no external database dependencies.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading
import json
from pathlib import Path

import polars as pl

from src.api.models import EventPayload

logger = logging.getLogger(__name__)


class EventStore:
    """
    In-memory event store with optional persistence.
    
    Designed for microservice deployment where the service
    owns its own data and doesn't depend on external databases.
    """
    
    def __init__(
        self,
        max_events: int = 1_000_000,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize event store.
        
        Args:
            max_events: Maximum events to keep (LRU eviction)
            persist_path: Optional path for SQLite persistence
        """
        self.max_events = max_events
        self.persist_path = persist_path
        
        # Primary storage: insertion_id -> list of events
        self._events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Indexes for graph building
        self._by_email: Dict[str, set] = defaultdict(set)  # email_hash -> {insertion_ids}
        self._by_phone: Dict[str, set] = defaultdict(set)  # phone_hash -> {insertion_ids}
        self._by_ip: Dict[str, set] = defaultdict(set)     # ip_hash -> {insertion_ids}
        self._by_device: Dict[str, set] = defaultdict(set) # device_hash -> {insertion_ids}
        self._by_user: Dict[str, set] = defaultdict(set)   # user_id -> {insertion_ids}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metadata
        self._total_events = 0
        self._last_updated = None
        
        # Load persisted data if exists
        if persist_path and Path(persist_path).exists():
            self._load_from_disk()
    
    def store_event(self, event: EventPayload) -> str:
        """
        Store an event and update indexes.
        
        Args:
            event: Event payload to store
            
        Returns:
            Status: 'stored' or 'updated'
        """
        with self._lock:
            event_dict = event.model_dump()
            event_dict["event_timestamp"] = event.event_timestamp.isoformat()
            
            insertion_id = event.insertion_id
            
            # Check if we already have events for this insertion
            is_new = insertion_id not in self._events
            
            # Store event
            self._events[insertion_id].append(event_dict)
            self._total_events += 1
            
            # Update indexes
            self._by_email[event.email_hash].add(insertion_id)
            self._by_phone[event.phone_hash].add(insertion_id)
            self._by_ip[event.ip_hash].add(insertion_id)
            self._by_device[event.device_hash].add(insertion_id)
            self._by_user[event.user_id].add(insertion_id)
            
            self._last_updated = datetime.now(timezone.utc)
            
            # Eviction if needed
            if self._total_events > self.max_events:
                self._evict_oldest()
            
            return "stored" if is_new else "updated"
    
    def get_events(self, insertion_id: str) -> List[Dict[str, Any]]:
        """Get all events for an insertion."""
        with self._lock:
            return self._events.get(insertion_id, [])
    
    def get_latest_event(self, insertion_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent event for an insertion."""
        events = self.get_events(insertion_id)
        return events[-1] if events else None
    
    def get_related_insertions(
        self,
        event: EventPayload,
        max_related: int = 100,
    ) -> Dict[str, set]:
        """
        Find insertions related to this event via shared identities.
        
        Used for graph feature computation.
        
        Args:
            event: Event to find relations for
            max_related: Max related insertions per identity type
            
        Returns:
            Dict with related insertion IDs by relationship type
        """
        with self._lock:
            related = {
                "shares_email": set(list(self._by_email.get(event.email_hash, set()))[:max_related]),
                "shares_phone": set(list(self._by_phone.get(event.phone_hash, set()))[:max_related]),
                "shares_ip": set(list(self._by_ip.get(event.ip_hash, set()))[:max_related]),
                "shares_device": set(list(self._by_device.get(event.device_hash, set()))[:max_related]),
                "same_user": set(list(self._by_user.get(event.user_id, set()))[:max_related]),
            }
            
            # Remove self
            for rel_type in related:
                related[rel_type].discard(event.insertion_id)
            
            return related
    
    def count_related(self, event: EventPayload) -> Dict[str, int]:
        """
        Count related insertions by identity type.
        
        Fast graph-like features without full graph construction.
        """
        with self._lock:
            return {
                "email_link_count": len(self._by_email.get(event.email_hash, set())) - 1,
                "phone_link_count": len(self._by_phone.get(event.phone_hash, set())) - 1,
                "ip_link_count": len(self._by_ip.get(event.ip_hash, set())) - 1,
                "device_link_count": len(self._by_device.get(event.device_hash, set())) - 1,
                "user_listing_count": len(self._by_user.get(event.user_id, set())),
            }
    
    def to_dataframe(self) -> pl.DataFrame:
        """
        Export all events as a Polars DataFrame.
        
        Used for batch graph building or model retraining.
        """
        with self._lock:
            all_events = []
            for insertion_id, events in self._events.items():
                for event in events:
                    all_events.append(event)
            
            if not all_events:
                return pl.DataFrame()
            
            return pl.DataFrame(all_events)
    
    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            return {
                "total_insertions": len(self._events),
                "total_events": self._total_events,
                "unique_emails": len(self._by_email),
                "unique_phones": len(self._by_phone),
                "unique_ips": len(self._by_ip),
                "unique_devices": len(self._by_device),
                "unique_users": len(self._by_user),
                "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            }
    
    def _evict_oldest(self):
        """Evict oldest events when store is full."""
        # Simple strategy: remove oldest insertions until under limit
        # In production, use proper LRU with timestamps
        if len(self._events) < 100:
            return  # Don't evict if very few insertions
        
        # Remove 10% of oldest
        to_remove = len(self._events) // 10
        keys_to_remove = list(self._events.keys())[:to_remove]
        
        for key in keys_to_remove:
            events = self._events.pop(key)
            self._total_events -= len(events)
            
            # Clean up indexes (simplified - may leave orphans)
            # In production, maintain reverse indexes
        
        logger.info(f"Evicted {to_remove} insertions from store")
    
    def persist(self):
        """Save store to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            data = {
                "events": dict(self._events),
                "metadata": {
                    "total_events": self._total_events,
                    "last_updated": self._last_updated.isoformat() if self._last_updated else None,
                }
            }
            
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
            
            logger.info(f"Persisted {self._total_events} events to {self.persist_path}")
    
    def _load_from_disk(self):
        """Load store from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            # Rebuild events
            for insertion_id, events in data.get("events", {}).items():
                self._events[insertion_id] = events
                self._total_events += len(events)
                
                # Rebuild indexes from latest event
                if events:
                    latest = events[-1]
                    self._by_email[latest.get("email_hash", "")].add(insertion_id)
                    self._by_phone[latest.get("phone_hash", "")].add(insertion_id)
                    self._by_ip[latest.get("ip_hash", "")].add(insertion_id)
                    self._by_device[latest.get("device_hash", "")].add(insertion_id)
                    self._by_user[latest.get("user_id", "")].add(insertion_id)
            
            logger.info(f"Loaded {self._total_events} events from {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to load from disk: {e}")
    
    def clear(self):
        """Clear all data (for testing)."""
        with self._lock:
            self._events.clear()
            self._by_email.clear()
            self._by_phone.clear()
            self._by_ip.clear()
            self._by_device.clear()
            self._by_user.clear()
            self._total_events = 0
            self._last_updated = None


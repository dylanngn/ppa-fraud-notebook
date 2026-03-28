"""
Admin Decision Queue and Parquet Persistence.

Manages the human-review queue for MEDIUM-risk predictions and persists all
decisions (auto and human) to date-partitioned parquet files.

Storage layout under artifacts/decisions/:
  approved/YYYY-MM-DD.parquet  — LOW auto-approvals + admin-approved MEDIUM
  declined/YYYY-MM-DD.parquet  — HIGH auto-declines + admin-rejected MEDIUM
  retrain/pending.parquet       — union of approved + labelled-flagged rows,
                                  ready for the retraining trigger
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """A prediction queued for human review."""
    insertion_id: str
    fraud_probability: float
    risk_tier: str          # HIGH / MEDIUM / LOW
    event_summary: Dict[str, Any]  # subset of event fields for quick display
    queued_at: str          # ISO-8601


@dataclass
class DecisionRecord:
    """A final decision on a listing (auto or human)."""
    insertion_id: str
    fraud_probability: float
    risk_tier: str
    event_summary: Dict[str, Any]
    queued_at: str
    admin_decision: str     # APPROVE / DECLINE
    admin_id: str           # "auto" for system decisions
    notes: str
    decided_at: str
    bucket: str             # approved / declined


# ---------------------------------------------------------------------------
# AdminStore
# ---------------------------------------------------------------------------

class AdminStore:
    """
    In-memory queue + on-disk parquet persistence for listing decisions.

    Thread-safe.  All writes are flushed synchronously to parquet files
    on every decide() call (low-volume demo use case).
    """

    def __init__(self, storage_dir: str = "artifacts/decisions") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        for bucket in ("approved", "declined", "retrain"):
            (self.storage_dir / bucket).mkdir(exist_ok=True)

        # insertion_id → ReviewItem (ordered for FIFO display)
        self._queue: OrderedDict[str, ReviewItem] = OrderedDict()
        self._decided: List[DecisionRecord] = []
        self._lock = threading.RLock()

        # Running counters for fast stats
        self._counts: Dict[str, int] = {"approved": 0, "declined": 0}

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------

    def enqueue(
        self,
        insertion_id: str,
        prediction: Dict[str, Any],
        event: Dict[str, Any],
    ) -> None:
        """Add a MEDIUM-risk prediction to the admin review queue."""
        with self._lock:
            if insertion_id in self._queue:
                return  # Already queued; ignore duplicate

            summary = {
                "listing_category": event.get("listing_category"),
                "listing_platform": event.get("listing_platform"),
                "listing_price": event.get("listing_price"),
                "listing_country": event.get("listing_country"),
                "seon_tor": event.get("seon_tor"),
                "seon_vpn": event.get("seon_vpn"),
                "seon_datacenter": event.get("seon_datacenter"),
                "seon_ip_country": event.get("seon_ip_country"),
                "user_id": event.get("user_id"),
            }
            item = ReviewItem(
                insertion_id=insertion_id,
                fraud_probability=prediction.get("fraud_probability", 0.0),
                risk_tier=prediction.get("risk_tier", "MEDIUM"),
                event_summary=summary,
                queued_at=datetime.now(timezone.utc).isoformat(),
            )
            self._queue[insertion_id] = item
            logger.info(f"Queued for review: {insertion_id} (prob={item.fraud_probability:.3f})")

    def get_queue(self) -> List[ReviewItem]:
        with self._lock:
            return list(self._queue.values())

    def clear_queue(self) -> None:
        """Remove all items from the review queue (used on simulation reset)."""
        with self._lock:
            self._queue.clear()
        logger.info("Review queue cleared")

    def get_queue_item(self, insertion_id: str) -> Optional[ReviewItem]:
        with self._lock:
            return self._queue.get(insertion_id)

    # ------------------------------------------------------------------
    # Decision recording
    # ------------------------------------------------------------------

    def decide(
        self,
        insertion_id: str,
        decision: str,
        admin_id: str = "admin-1",
        notes: str = "",
    ) -> DecisionRecord:
        """
        Record a human decision for a queued item.

        Args:
            insertion_id: The listing being decided.
            decision: "APPROVE" or "DECLINE".
            admin_id: Who made the decision.
            notes: Optional free-text reason.

        Returns:
            The completed DecisionRecord.

        Raises:
            KeyError: If insertion_id is not in the queue.
        """
        with self._lock:
            item = self._queue.pop(insertion_id, None)
            if item is None:
                raise KeyError(f"insertion_id not in review queue: {insertion_id}")

            bucket = "approved" if decision == "APPROVE" else "declined"
            record = DecisionRecord(
                **{k: v for k, v in asdict(item).items()},
                admin_decision=decision,
                admin_id=admin_id,
                notes=notes,
                decided_at=datetime.now(timezone.utc).isoformat(),
                bucket=bucket,
            )
            self._decided.append(record)
            self._counts[bucket] = self._counts.get(bucket, 0) + 1

            self._write_parquet(record)
            self._append_retrain(record)
            return record

    def auto_decide(
        self,
        insertion_id: str,
        decision: str,
        event: Optional[Dict] = None,
        fraud_probability: float = 0.0,
    ) -> DecisionRecord:
        """
        Record an automatic decision (LOW → APPROVE, HIGH → DECLINE).

        If the insertion is already in the queue it is removed.
        """
        with self._lock:
            item = self._queue.pop(insertion_id, None)

            # Build a minimal ReviewItem if not previously queued
            if item is None:
                summary: Dict[str, Any] = {}
                if event:
                    summary = {
                        "listing_category": event.get("listing_category"),
                        "listing_platform": event.get("listing_platform"),
                        "listing_price": event.get("listing_price"),
                        "listing_country": event.get("listing_country"),
                        "seon_tor": event.get("seon_tor"),
                        "seon_vpn": event.get("seon_vpn"),
                        "seon_datacenter": event.get("seon_datacenter"),
                        "seon_ip_country": event.get("seon_ip_country"),
                        "user_id": event.get("user_id"),
                    }
                item = ReviewItem(
                    insertion_id=insertion_id,
                    fraud_probability=fraud_probability,
                    risk_tier="HIGH" if decision == "DECLINE" else "LOW",
                    event_summary=summary,
                    queued_at=datetime.now(timezone.utc).isoformat(),
                )

            bucket = "approved" if decision == "APPROVE" else "declined"
            record = DecisionRecord(
                **{k: v for k, v in asdict(item).items()},
                admin_decision=decision,
                admin_id="auto",
                notes="",
                decided_at=datetime.now(timezone.utc).isoformat(),
                bucket=bucket,
            )
            self._decided.append(record)
            self._counts[bucket] = self._counts.get(bucket, 0) + 1

            self._write_parquet(record)
            self._append_retrain(record)
            return record

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _write_parquet(self, record: DecisionRecord) -> None:
        """Append a single record to the appropriate date-partitioned parquet file."""
        try:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = self.storage_dir / record.bucket / f"{date_str}.parquet"

            row = {
                "insertion_id": record.insertion_id,
                "fraud_probability": record.fraud_probability,
                "risk_tier": record.risk_tier,
                "admin_decision": record.admin_decision,
                "admin_id": record.admin_id,
                "notes": record.notes,
                "decided_at": record.decided_at,
                "bucket": record.bucket,
                **{f"evt_{k}": str(v) for k, v in (record.event_summary or {}).items()},
            }
            df = pd.DataFrame([row])
            table = pa.Table.from_pandas(df)

            if path.exists():
                existing = pq.read_table(str(path))
                combined = pa.concat_tables([existing, table], promote_options="default")
                pq.write_table(combined, str(path))
            else:
                pq.write_table(table, str(path))
        except Exception as exc:
            logger.warning(f"Parquet write failed ({record.bucket}): {exc}")

    def _append_retrain(self, record: DecisionRecord) -> None:
        """
        Append approved records (and admin-decided MEDIUM records labelled as fraud)
        to the retrain pending pool.

        Label convention:
          APPROVE → is_fraud = 0
          DECLINE → is_fraud = 1
        """
        try:
            path = self.storage_dir / "retrain" / "pending.parquet"
            row = {
                "insertion_id": record.insertion_id,
                "fraud_probability": record.fraud_probability,
                "is_fraud": 0 if record.admin_decision == "APPROVE" else 1,
                "admin_decision": record.admin_decision,
                "decided_at": record.decided_at,
                **{f"evt_{k}": str(v) for k, v in (record.event_summary or {}).items()},
            }
            df = pd.DataFrame([row])
            table = pa.Table.from_pandas(df)

            if path.exists():
                existing = pq.read_table(str(path))
                combined = pa.concat_tables([existing, table], promote_options="default")
                pq.write_table(combined, str(path))
            else:
                pq.write_table(table, str(path))
        except Exception as exc:
            logger.warning(f"Retrain parquet write failed: {exc}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_storage_stats(self) -> Dict[str, Any]:
        """Return row counts and file paths for each storage bucket."""
        stats: Dict[str, Any] = {
            "approved_count": 0,
            "declined_count": 0,
            "pending_retrain_rows": 0,
            "queue_size": 0,
            "storage_paths": {
                "approved": str(self.storage_dir / "approved"),
                "declined": str(self.storage_dir / "declined"),
                "retrain":  str(self.storage_dir / "retrain"),
            },
            "last_decision": None,
        }

        with self._lock:
            stats["queue_size"] = len(self._queue)

            for bucket in ("approved", "declined"):
                bucket_dir = self.storage_dir / bucket
                count = 0
                for f in bucket_dir.glob("*.parquet"):
                    try:
                        count += pq.read_metadata(str(f)).num_rows
                    except Exception:
                        pass
                stats[f"{bucket}_count"] = count

            retrain_path = self.storage_dir / "retrain" / "pending.parquet"
            if retrain_path.exists():
                try:
                    stats["pending_retrain_rows"] = pq.read_metadata(str(retrain_path)).num_rows
                except Exception:
                    pass

            if self._decided:
                stats["last_decision"] = self._decided[-1].decided_at

        return stats

    def get_pending_retrain_count(self) -> int:
        retrain_path = self.storage_dir / "retrain" / "pending.parquet"
        if not retrain_path.exists():
            return 0
        try:
            return pq.read_metadata(str(retrain_path)).num_rows
        except Exception:
            return 0

    def get_recent_decisions(self, n: int = 20) -> List[DecisionRecord]:
        with self._lock:
            return list(self._decided[-n:])

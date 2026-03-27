"""
Simulation State Machine.

Ports the Node-RED virtual clock (fn-virtual-clock) into a Python asyncio
background task.  Drives the demo by generating synthetic events at
configurable speed and tracking counters, drift windows, and retrain triggers.

Usage:
    from src.api.simulator import simulator
    await simulator.start()   # in FastAPI lifespan
    await simulator.stop()    # on shutdown
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window definitions — verbatim from Node-RED fn-virtual-clock
# ---------------------------------------------------------------------------

_WINDOWS = [
    {"start": "2025-03-21", "end": "2025-05-20", "low": 70, "med": 20, "high": 10, "auc": None,  "label": "W0 Training"},
    {"start": "2025-05-21", "end": "2025-06-20", "low": 70, "med": 20, "high": 10, "auc": 0.64, "label": "W1 Stable"},
    {"start": "2025-06-21", "end": "2025-07-20", "low": 65, "med": 22, "high": 13, "auc": 0.59, "label": "W2 Declining"},
    {"start": "2025-07-21", "end": "2025-08-20", "low": 70, "med": 20, "high": 10, "auc": 0.73, "label": "W3 Peak"},
    {"start": "2025-08-21", "end": "2025-09-20", "low": 65, "med": 22, "high": 13, "auc": 0.64, "label": "W4 Stable"},
    {"start": "2025-09-21", "end": "2025-10-20", "low": 60, "med": 25, "high": 15, "auc": 0.58, "label": "W5 Drift!"},
    {"start": "2025-10-21", "end": "2025-11-20", "low": 50, "med": 28, "high": 22, "auc": 0.49, "label": "W6 Severe!"},
    {"start": "2025-11-21", "end": "2025-12-20", "low": 65, "med": 22, "high": 13, "auc": 0.63, "label": "W7 Recovery"},
    {"start": "2025-12-21", "end": "2026-02-01", "low": 67, "med": 20, "high": 13, "auc": 0.54, "label": "W8 Stabilizing"},
]

_RETRAIN_AT: Set[int] = {6, 7}


def _date_to_ms(date_str: str) -> int:
    """Parse YYYY-MM-DD to epoch milliseconds (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# Pre-compute epoch-ms for each window boundary
WINDOWS = []
for w in _WINDOWS:
    WINDOWS.append({
        **w,
        "start_ms": _date_to_ms(w["start"]),
        "end_ms": _date_to_ms(w["end"]),
    })

SIM_START = _date_to_ms("2025-03-21")
SIM_END = _date_to_ms("2026-02-01")
SIM_RANGE = SIM_END - SIM_START


def _find_window(ms: int) -> int:
    """Return the index of the window containing the given timestamp."""
    for i in range(len(WINDOWS) - 1, -1, -1):
        if ms >= WINDOWS[i]["start_ms"]:
            return i
    return 0


class Simulator:
    """Background simulation loop that drives the fraud detection demo."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self.reset_state()

    def reset_state(self) -> None:
        """Reset all simulation state to initial values."""
        self.virtual_ms: int = SIM_START
        self.paused: bool = False
        self.retraining: bool = False
        self.speed: float = 1.0
        self.approved_count: int = 0
        self.rejected_count: int = 0
        self.pending_count: int = 0
        self.current_window: int = 0
        self.last_retrain_ms: int = 0
        self._retrain_poll_active: bool = False
        # Last event info — consumed by the dashboard for pipeline animation
        self._event_seq: int = 0           # monotonic counter
        self._last_event_risk: str = ""    # LOW / MEDIUM / HIGH
        self._last_event_prob: float = 0.0
        self._last_event_id: str = ""

    async def start(self) -> None:
        """Launch the background tick loop."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._tick_loop())
        logger.info("Simulator started")

    async def stop(self) -> None:
        """Cancel the background tick loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Simulator stopped")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def toggle_pause(self) -> bool:
        """Toggle pause state and return the new value."""
        self.paused = not self.paused
        return self.paused

    def set_speed(self, speed: float) -> float:
        """Set simulation speed (clamped to 0.25–8.0)."""
        self.speed = max(0.25, min(8.0, speed))
        return self.speed

    async def reset(self) -> None:
        """Reset all state and the drift reference distribution."""
        self.reset_state()
        # Reset drift state in the API
        try:
            import src.api.main as m
            if m.drift_state:
                m.drift_state.reset()
        except Exception as exc:
            logger.warning(f"Could not reset drift state: {exc}")

    async def force_future_event(self) -> Optional[Dict[str, Any]]:
        """Pull an event from 1-2 windows ahead (port of fn-future-event)."""
        cur = self.current_window
        ahead = min(cur + 1 + random.randint(0, 1), len(WINDOWS) - 1)
        w = WINDOWS[ahead]
        lo, me, hi = w["low"], w["med"], w["high"]
        # +/- 15% jitter then renormalise
        lo = max(1, lo + (random.random() - 0.5) * 30)
        me = max(1, me + (random.random() - 0.5) * 30)
        hi = max(1, hi + (random.random() - 0.5) * 30)
        total = lo + me + hi
        lo /= total
        me /= total
        r = random.random()
        scenario = "LOW_RISK" if r < lo else ("MEDIUM_RISK" if r < lo + me else "HIGH_RISK")
        return await self._fire_event(scenario)

    # ------------------------------------------------------------------
    # State for dashboard
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return the full simulation state dict for the dashboard."""
        cur_win = _find_window(self.virtual_ms)
        win = WINDOWS[cur_win]
        sim_done = self.virtual_ms >= SIM_END

        # Train bar: shows how far the training data extends
        if cur_win > 0:
            t_end = WINDOWS[cur_win - 1]["end_ms"]
        else:
            t_end = min(self.virtual_ms, WINDOWS[0]["end_ms"])
        train_pct = min((t_end - SIM_START) / SIM_RANGE * 100, 100)
        cursor_pct = min((self.virtual_ms - SIM_START) / SIM_RANGE * 100, 100)

        windows_detail = []
        for i, w in enumerate(WINDOWS):
            w_end = min(w["end_ms"], SIM_END)
            windows_detail.append({
                "label": w["label"],
                "auc": w["auc"],
                "start_pct": (w["start_ms"] - SIM_START) / SIM_RANGE * 100,
                "width_pct": (w_end - w["start_ms"]) / SIM_RANGE * 100,
                "passed": self.virtual_ms > w["end_ms"],
                "current": i == cur_win,
                "retrain_entry": i in _RETRAIN_AT,
            })

        return {
            "virtual_date": datetime.utcfromtimestamp(self.virtual_ms / 1000).strftime("%Y-%m-%d"),
            "window_index": cur_win,
            "window_label": win["label"],
            "window_auc": win["auc"],
            "speed": self.speed,
            "paused": self.paused,
            "retraining": self.retraining,
            "sim_done": sim_done,
            "approved": self.approved_count,
            "rejected": self.rejected_count,
            "pending": self.pending_count,
            "windows_detail": windows_detail,
            "train_pct": round(train_pct, 2),
            "cursor_pct": round(cursor_pct, 2),
            # Last event — drives pipeline animation in the dashboard
            "event_seq": self._event_seq,
            "last_event": {
                "risk": self._last_event_risk,
                "prob": self._last_event_prob,
                "id": self._last_event_id,
            } if self._last_event_risk else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Background loop — fires every 1 second."""
        while True:
            try:
                await asyncio.sleep(1.0)
                if self.paused or self.retraining or self.virtual_ms >= SIM_END:
                    continue

                prev_win = self.current_window
                self.virtual_ms += int(self.speed * 86_400_000)
                new_win = _find_window(self.virtual_ms)
                self.current_window = new_win

                # Pick scenario based on current window distribution
                w = WINDOWS[new_win]
                r = random.random() * 100
                if r < w["low"]:
                    scenario = "LOW_RISK"
                elif r < w["low"] + w["med"]:
                    scenario = "MEDIUM_RISK"
                else:
                    scenario = "HIGH_RISK"

                await self._fire_event(scenario)

                # Auto-trigger retrain at window boundaries
                if new_win != prev_win and new_win in _RETRAIN_AT:
                    elapsed_since_retrain = self.virtual_ms - self.last_retrain_ms
                    if elapsed_since_retrain > 25 * 86_400_000:
                        self.last_retrain_ms = self.virtual_ms
                        self.retraining = True
                        self.paused = True
                        asyncio.create_task(self._trigger_retrain())

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"Simulator tick error: {exc}")

    async def _fire_event(self, scenario: str) -> Optional[Dict[str, Any]]:
        """Call the simulate endpoint and update counters."""
        try:
            from src.api.routers.admin import simulate, SimulateRequest
            result = await simulate(SimulateRequest(scenario=scenario))
            tier = result.get("risk_tier", "LOW")
            if tier == "HIGH":
                self.rejected_count += 1
            elif tier == "MEDIUM":
                self.pending_count += 1
            else:
                self.approved_count += 1
            # Record for dashboard animation
            self._event_seq += 1
            self._last_event_risk = tier
            self._last_event_prob = result.get("fraud_probability", 0.0)
            self._last_event_id = result.get("insertion_id", "")
            return result
        except Exception as exc:
            logger.warning(f"Simulate failed: {exc}")
            return None

    async def _trigger_retrain(self) -> None:
        """Trigger retraining and poll until complete, then resume simulation."""
        try:
            from src.api.routers.admin import _retrain_state, _run_retrain, _MIN_RETRAIN_ROWS
            import src.api.main as m

            if m.admin_store is None:
                logger.warning("Admin store not available — skipping retrain")
                self.retraining = False
                self.paused = False
                return

            row_count = m.admin_store.get_pending_retrain_count()
            if row_count < _MIN_RETRAIN_ROWS:
                logger.info(f"Insufficient retrain data ({row_count}/{_MIN_RETRAIN_ROWS}) — resuming")
                self.retraining = False
                self.paused = False
                return

            # Run retraining in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_retrain, row_count)
        except Exception as exc:
            logger.warning(f"Retrain trigger failed: {exc}")

        self.retraining = False
        self.paused = False
        logger.info("Retrain cycle complete — simulation resumed")


# Module-level singleton
simulator = Simulator()

"""
Simulator API Router.

Provides endpoints for controlling the demo simulation loop and
retrieving its current state (polled by the dashboard).

All endpoints are mounted at /admin/sim via:
    app.include_router(sim_router, prefix="/admin/sim")
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.api.simulator import simulator

router = APIRouter(tags=["Simulator"])


class SpeedRequest(BaseModel):
    speed: float = 1.0


@router.get("/state", summary="Current simulation state (polled by dashboard)")
async def sim_state():
    return simulator.get_state()


@router.post("/start", summary="Start the simulation loop")
async def sim_start():
    await simulator.start()
    return {"status": "started"}


@router.post("/pause", summary="Toggle pause")
async def sim_pause():
    paused = simulator.toggle_pause()
    return {"paused": paused}


@router.post("/speed", summary="Set simulation speed")
async def sim_speed(req: SpeedRequest):
    speed = simulator.set_speed(req.speed)
    return {"speed": speed}


@router.post("/reset", summary="Reset simulation state")
async def sim_reset():
    await simulator.reset()
    return {"status": "reset"}


@router.post("/force-event", summary="Pull one future event")
async def sim_force_event():
    result = await simulator.force_future_event()
    return {"status": "fired", "result": result}

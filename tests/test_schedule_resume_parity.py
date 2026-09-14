"""A crash must not change how many turns a schedule request buys.

A schedule request is one more turn -- for a fanned-out process, one more actor
turn. A straight run crosses one request off per turn as it takes it. A fresh
controller rebuilding from persistence replayed every request and crossed off
one per *completed occurrence*, so a batch interrupted part-way kept its
requests open and the resumed run took actor turns the uninterrupted run never
took. Probed at every possible pause point, serially and concurrently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from genesis.persistence import PersistenceCoordinator
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    StateStore,
)

WRITERS = ["w1", "w2", "w3"]


class _Emit:
    def __init__(self, effects: list[dict[str, Any]]) -> None:
        self.effects = effects

    def execute(self, invocation: Any) -> Any:
        return ProcessResult(scheduling_effects=list(self.effects))


class _Noop:
    def execute(self, invocation: Any) -> Any:
        return ProcessResult()


def _controller(db: Any, processes: list[dict], effects: list[dict], limit: int, status=None):
    return RunController(
        Scheduler(processes),
        ExecutorRegistry({"sched": _Emit(effects), "write": _Noop(), "settle": _Noop()}),
        ContextEngine({"none": {"allow": []}}),
        persistence=db,
        state_store=StateStore({"x": int}, {"x": 0}),
        status_provider=status,
        max_concurrency={"write": limit},
    )


def _sequence(db: Any) -> list[tuple[Any, ...]]:
    return [
        (event["process_id"], tuple(event.get("actors") or ()), event.get("phase"))
        for event in db.list_events("r")
        if event.get("kind") != "process_failed"
    ]


def _straight(tmp: Path, processes, effects, limit: int, phases: int) -> list[tuple[Any, ...]]:
    db = PersistenceCoordinator(tmp / "straight.db", tmp / "straight-objects")
    try:
        _controller(db, processes, effects, limit).run("r", phase_limit=phases)
        return _sequence(db)
    finally:
        db.close()


def _resumed(tmp: Path, processes, effects, limit: int, phases: int, pause_at: int):
    db = PersistenceCoordinator(tmp / f"resumed-{pause_at}.db", tmp / f"resumed-{pause_at}-objects")
    try:
        _controller(
            db,
            processes,
            effects,
            limit,
            status=lambda: "paused" if len(db.list_events("r")) >= pause_at else "running",
        ).run("r", phase_limit=phases)
        # A fresh controller, as after a crash: it knows only what was persisted.
        _controller(db, processes, effects, limit).run("r", phase_limit=phases)
        return _sequence(db)
    finally:
        db.close()


SCENARIOS = {
    "one request, fanned-out": (
        [
            {"id": "sched", "context_policy": "none", "trigger": {"phase": 0}},
            {
                "id": "write",
                "actors": WRITERS,
                "executor": {"mode": "generative"},
                "context_policy": "none",
                "trigger": {"type": "phase", "phase": 2},
            },
        ],
        [{"type": "schedule", "process_id": "write", "phase": 1}],
    ),
    "two requests, fanned-out": (
        [
            {"id": "sched", "context_policy": "none", "trigger": {"phase": 0}},
            {
                "id": "write",
                "actors": WRITERS,
                "executor": {"mode": "generative"},
                "context_policy": "none",
                "trigger": {"type": "phase", "phase": 5},
            },
        ],
        [
            {"type": "schedule", "process_id": "write", "phase": 1},
            {"type": "schedule", "process_id": "write", "phase": 1},
        ],
    ),
    "two requests, run-once": (
        [
            {"id": "sched", "context_policy": "none", "trigger": {"phase": 0}},
            {"id": "settle", "context_policy": "none", "trigger": {"type": "event", "event": "x"}},
        ],
        [
            {"type": "schedule", "process_id": "settle", "phase": 0},
            {"type": "schedule", "process_id": "settle", "phase": 1},
        ],
    ),
}


@pytest.mark.parametrize("limit", [1, 8], ids=["serial", "concurrent"])
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_resumed_run_takes_the_same_turns_at_every_pause_point(
    tmp_path: Path, name: str, limit: int
) -> None:
    processes, effects = SCENARIOS[name]
    straight = _straight(tmp_path, processes, effects, limit, phases=3)
    assert len(straight) > 1, "the scenario must do something to interrupt"
    for pause_at in range(1, len(straight)):
        resumed = _resumed(tmp_path, processes, effects, limit, phases=3, pause_at=pause_at)
        assert resumed == straight, f"paused after {pause_at}: {resumed} != {straight}"

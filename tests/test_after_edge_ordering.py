"""`after: [X]` holds within a round, without giving up mid-round starts.

A repeating, condition-triggered process that is not triggered when a dependent
is considered does not hold that dependent back, which is what lets a round
proceed. But if its trigger turned true later in the same round, it ran *after*
the process declared to run after it. Deciding every trigger at the start of the
round would have fixed that by removing mid-round starts, which the concurrency
guard and several studies rely on; so instead, once something waiting on X has
run in a round, X waits for the next round.
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

FLAG = {"type": "condition", "predicate": {"path": "flag", "op": "eq", "value": 1}, "repeat": True}


class _SetFlag:
    def execute(self, invocation: Any) -> Any:
        return ProcessResult(state_effects={"flag": 1})


class _Noop:
    def execute(self, invocation: Any) -> Any:
        return ProcessResult()


def _controller(processes: list[dict], executors: dict, **kwargs: Any) -> RunController:
    return RunController(
        Scheduler(processes),
        ExecutorRegistry(executors),
        ContextEngine({"none": {"allow": []}}),
        state_store=StateStore({"flag": int}, {"flag": 0}),
        **kwargs,
    )


INVERSION = [
    {"id": "c", "context_policy": "none", "trigger": {"phase": 0, "repeat": True}, "after": ["p"]},
    {
        "id": "s",
        "context_policy": "none",
        "trigger": {"phase": 0, "repeat": True},
        "after": ["c"],
        "state_effects": ["flag"],
    },
    {"id": "p", "context_policy": "none", "trigger": FLAG},
]
INVERSION_EXECUTORS = {"c": _Noop(), "s": _SetFlag(), "p": _Noop()}


def test_a_process_never_runs_after_its_dependent_in_the_same_round() -> None:
    order = _controller(INVERSION, INVERSION_EXECUTORS).run("r", phase_limit=2)
    # Round 0: p is untriggered as c starts, becomes triggered later -- and waits.
    # Round 1: triggered from the start, so c waits for it.
    assert order == ["c", "s", "p", "c", "s"]


def test_a_mid_round_start_still_happens_when_nothing_waits_on_it() -> None:
    processes = [{**INVERSION[0], "after": []}, INVERSION[1], INVERSION[2]]
    order = _controller(processes, INVERSION_EXECUTORS).run("r", phase_limit=1)
    assert order == ["c", "s", "p"]


def test_a_dependent_interrupted_mid_batch_holds_the_process_back_too() -> None:
    """c's first actor writes the flag; p must not slip in before c's second actor."""
    processes = [
        {
            "id": "c",
            "actors": ["a1", "a2"],
            "context_policy": "none",
            "trigger": {"phase": 0, "repeat": True},
            "after": ["p"],
            "state_effects": ["flag"],
        },
        {"id": "p", "context_policy": "none", "trigger": FLAG},
    ]
    order = _controller(processes, {"c": _SetFlag(), "p": _Noop()}).run("r", phase_limit=2)
    # Before: c(a1), p, c(a2) -- p slipped between c's actors. A first fix held
    # p back but stranded c(a2) for the round, silently dropping an actor turn.
    assert order == ["c", "c", "p", "c", "c"], order


def test_a_delayed_edge_is_not_an_ordering_within_the_round() -> None:
    """A positive delay reads an earlier round, so there is no same-round order
    to keep and the producer is not held back. Checked at the scheduler, with a
    zero-delay control: a run-level version cannot isolate this, because a
    delayed edge on a producer that has never run blocks the consumer for
    reasons unrelated to ordering."""

    def scheduler(dependencies: dict[str, Any]) -> Scheduler:
        consumer = {
            "id": "c",
            "trigger": {"phase": 0, "repeat": True},
            "dependencies": dependencies,
        }
        return Scheduler([consumer, {"id": "p", "trigger": FLAG}])

    state = {"flag": 1}
    delayed = scheduler({"after": ["p"], "delay": 1})
    delayed.mark_started("c", 1)
    assert "p" in [item.process_id for item in delayed.ready(1, state=state)]

    immediate = scheduler({"after": ["p"]})
    immediate.mark_started("c", 1)
    assert "p" not in [item.process_id for item in immediate.ready(1, state=state)]


@pytest.mark.parametrize(
    ("name", "processes", "executors"),
    [
        ("inversion", INVERSION, INVERSION_EXECUTORS),
        (
            "mid-batch",
            [
                {
                    "id": "c",
                    "actors": ["a1", "a2"],
                    "context_policy": "none",
                    "trigger": {"phase": 0, "repeat": True},
                    "after": ["p"],
                    "state_effects": ["flag"],
                },
                {"id": "p", "context_policy": "none", "trigger": FLAG},
            ],
            {"c": _SetFlag(), "p": _Noop()},
        ),
    ],
)
def test_a_resumed_run_orders_the_round_the_same_way(
    tmp_path: Path, name: str, processes: list[dict], executors: dict
) -> None:
    """What has started is rebuilt from committed events, not remembered."""

    def sequence(db: Any) -> list[tuple[Any, ...]]:
        return [
            (e["process_id"], tuple(e.get("actors") or ()), e["phase"]) for e in db.list_events("r")
        ]

    straight_db = PersistenceCoordinator(tmp_path / "s.db", tmp_path / "s-obj")
    try:
        _controller(processes, executors, persistence=straight_db).run("r", phase_limit=3)
        straight = sequence(straight_db)
    finally:
        straight_db.close()
    for pause_at in range(1, len(straight)):
        db = PersistenceCoordinator(tmp_path / f"{pause_at}.db", tmp_path / f"{pause_at}-obj")
        try:
            _controller(
                processes,
                executors,
                persistence=db,
                status_provider=lambda db=db, pause_at=pause_at: (
                    "paused" if len(db.list_events("r")) >= pause_at else "running"
                ),
            ).run("r", phase_limit=3)
            _controller(processes, executors, persistence=db).run("r", phase_limit=3)
            assert sequence(db) == straight, f"{name}, paused after {pause_at}"
        finally:
            db.close()

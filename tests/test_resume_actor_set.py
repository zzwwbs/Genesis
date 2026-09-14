"""A batch's actors are fixed when it opens, not re-derived after a pause.

A process may draw its actors from a state field. When that same field is
written by the process, a resumed run re-expanded the batch from the mutated
field and invented an actor out of a value the run itself had appended: the
uninterrupted run visits a, b, c, d and the resumed run visits a, b, c, d, za.

That breaks the equality pause/resume rests on, so the probe is the comparison
itself -- run it straight through, run it again with a pause, and require the
same actors in the same order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genesis.service import GenesisService
from tests.test_engine_gaps import _write_package

START = ["a", "b", "c", "d"]


def _package(workspace: Path) -> Path:
    return _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    {
                        "id": "note",
                        "actors": {"source": "notes"},
                        "information_timing": {"mode": "sequential"},
                        "openness_rationale": "who speaks is drawn from the roster",
                        "closure_rationale": "the note itself is computed",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "sees-notes",
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                        "state_effects": [{"field": "notes", "op": "append"}],
                    }
                ]
            },
            "domain": {
                "visibility": [{"id": "sees-notes", "allow": ["notes"]}],
                "states": [{"id": "notes", "value_type": "array", "initial": list(START)}],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 0}},
        },
        "roster-study",
    )


def _visited(tmp_path: Path, name: str, pause_after: int | None) -> list[str]:
    """Actors the run actually invoked, optionally pausing and resuming."""
    service = GenesisService(tmp_path / name)
    try:
        source = _package(tmp_path / name)
        build = service.compile_study(source, "builds/roster")["path"]
        service.create_run({"id": "r", "study_id": "roster-study", "build": build})
        seen: list[str] = []

        def note(invocation: Any) -> dict[str, Any]:
            actor = invocation.actor_ids[0]
            seen.append(actor)
            if pause_after is not None and len(seen) == pause_after:
                current = service.get_run("r")
                service.transition_run("r", "paused", current["version"])
            return {"notes": f"z{actor}"}

        service.execute_run("r", executor_overrides={"note": note})
        if pause_after is not None:
            service.execute_run("r", executor_overrides={"note": note})
        return seen
    finally:
        service.close()


def test_a_resumed_run_visits_the_actors_the_batch_opened_with(tmp_path: Path) -> None:
    straight = _visited(tmp_path, "straight", pause_after=None)
    assert straight == START, straight
    resumed = _visited(tmp_path, "resumed", pause_after=1)
    assert resumed == START, resumed

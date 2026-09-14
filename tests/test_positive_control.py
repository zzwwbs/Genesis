"""A positive control: a study whose correct answer is known before it runs.

Every other test checks a mechanism. None checked the claim the system actually
makes -- that a declared treatment arrives and produces its effect -- and so none
failed when every treatment in the clickbait study was inert: its gates named a
path the scheduler never resolves, the run completed, the outcomes existed, and
the result read as a null effect.

This study runs the real pipeline -- compile, protocol expansion across cells,
dispatch -- with a treatment that reaches its actor through each channel a real
design uses, each with a distinct magnitude, so a broken channel shows up as a
recognisably wrong number rather than a plausible one:

    gated process      `boost` writes signal = 10, only when treated
    gated context      `hint` = 1000 is written in *both* cells; only the
                       context gate keeps it from the control actor
    gated measurement  `reading` = 100 is produced in both cells; only the
                       measurement use keeps it from the control actor

Treated must total 1110 and control 0. Each executor is trivial arithmetic over
what it was handed; everything under test is what it was handed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genesis.service import GenesisService
from tests.test_engine_gaps import _write_package

TREATED = {"path": "condition.factors.treatment", "op": "eq", "value": "treated"}
SIGNAL, HINT, READING = 10, 1000, 100


def _process(pid: str, phase: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "executor": {"mode": "deterministic"},
        "context_policy": "none",
        "trigger": {"type": "phase", "phase": phase},
        **extra,
    }


def _package(workspace: Path) -> Path:
    source = _write_package(
        workspace,
        {
            "openness": {
                "processes": [
                    _process("seed", 0, state_effects=[{"field": "hint", "op": "set"}]),
                    _process(
                        "boost",
                        0,
                        trigger={"type": "condition", "predicate": TREATED},
                        state_effects=[{"field": "signal", "op": "set"}],
                    ),
                    _process(
                        "measure",
                        0,
                        measurement=True,
                        outputs=[{"artifact_type": "reading", "schema_ref": "reading"}],
                    ),
                    _process(
                        "respond",
                        1,
                        actors=["respondent"],
                        context_policy="actor-view",
                        inputs=["reading"],
                        measurement_use=[
                            {"source": "measure", "rationale": "the treatment", "when": TREATED}
                        ],
                        state_effects=[{"field": "outcome", "op": "set"}],
                    ),
                ]
            },
            "domain": {
                "states": [
                    {"id": "signal", "value_type": "integer", "initial": 0},
                    {"id": "hint", "value_type": "integer", "initial": 0},
                    {"id": "outcome", "value_type": "integer", "initial": 0},
                ],
                "visibility": [
                    {"id": "none", "allow": []},
                    {
                        "id": "actor-view",
                        "allow": ["signal", "hint"],
                        "available_when": {"hint": TREATED},
                    },
                ],
                "artifacts": [
                    {"id": "reading", "artifact_type": "reading", "schema_ref": "reading"}
                ],
            },
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 1},
                "factors": [{"id": "treatment", "levels": ["control", "treated"]}],
                "matching": {"enabled": True, "shared_streams": ["conventional"]},
            },
        },
        "control-study",
    )
    (source / "schemas").mkdir(exist_ok=True)
    (source / "schemas" / "reading.json").write_text(
        json.dumps({"type": "object", "properties": {"value": {"type": "integer"}}})
    )
    return source


def _respond(invocation: Any) -> dict[str, Any]:
    context = dict(getattr(invocation.context, "data", {}) or {})
    reading = sum(
        int((record.get("value") or {}).get("value", 0))
        for record in (invocation.inputs or {}).values()
    )
    return {"outcome": int(context.get("signal", 0)) + int(context.get("hint", 0)) + reading}


OVERRIDES = {
    "seed": lambda _inv: {"hint": HINT},
    "boost": lambda _inv: {"signal": SIGNAL},
    "measure": lambda _inv: {"reading": {"value": READING}},
    "respond": _respond,
}


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    workspace = tmp_path_factory.mktemp("positive-control") / "ws"
    service = GenesisService(workspace)
    try:
        build = service.compile_study(_package(workspace), "builds/control")["path"]
        service.create_run({"id": "e", "study_id": "control-study", "build": build})
        runs = service.execute_protocol("e", replications=2, executor_overrides=OVERRIDES)["runs"]
        by_cell: dict[str, Any] = {}
        for run_id in runs:
            run = service.get_run(run_id)
            final = dict(list(service.persistence.list_state_history(run_id))[-1][1])
            key = f"{run['manifest']['condition_id']}/{run['manifest']['replication']}"
            by_cell[key] = {"state": final, "seeds": run["manifest"]["seeds"], "run": run}
        return by_cell
    finally:
        service.close()


def test_every_cell_ran_to_completion(results: dict[str, Any]) -> None:
    assert sorted(results) == [
        "treatment-control/1",
        "treatment-control/2",
        "treatment-treated/1",
        "treatment-treated/2",
    ]
    assert all(cell["run"]["status"] == "completed" for cell in results.values())


def test_the_treated_cell_receives_every_channel(results: dict[str, Any]) -> None:
    """A dead gate on any channel removes its digit: 1010, 110 and 1100 each
    name the channel that failed to deliver."""
    for replication in (1, 2):
        assert results[f"treatment-treated/{replication}"]["state"]["outcome"] == (
            SIGNAL + HINT + READING
        )


def test_the_control_cell_receives_no_channel(results: dict[str, Any]) -> None:
    """A leaking gate adds its digit: 1000 is the context gate, 100 the
    measurement gate, 10 the process gate."""
    for replication in (1, 2):
        assert results[f"treatment-control/{replication}"]["state"]["outcome"] == 0


def test_the_ungated_writers_really_did_write_in_the_control_cell(results: dict[str, Any]) -> None:
    """Without this, a control outcome of 0 could mean the hint was never
    written rather than that the gate withheld it."""
    control = results["treatment-control/1"]["state"]
    assert control["hint"] == HINT
    assert control["signal"] == 0


def test_matched_cells_share_their_dice(results: dict[str, Any]) -> None:
    for replication in (1, 2):
        control = results[f"treatment-control/{replication}"]["seeds"]
        treated = results[f"treatment-treated/{replication}"]["seeds"]
        assert control == treated
    assert results["treatment-control/1"]["seeds"] != results["treatment-control/2"]["seeds"]

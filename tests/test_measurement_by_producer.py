"""A gated measurement is hidden by who produced it, not by what kind it is.

The gate hid a record *type*. Two consequences, both from that one choice:

* It leaked. When any other process also produced the type, the type could not
  be hidden without starving that process's contribution, so nothing was hidden
  -- and the consumer received the measurement's own records with the gate off.
* It over-hid. A consumer that itself produced the type lost its own earlier
  records whenever the gate was off.

The rule a researcher states is about a producer: "sanction must not see the
detector's scores." Every stored record names the process that produced it, so
the gate now drops exactly those records and keeps everyone else's. These tests
drive a real run, because the leak lived in resolution, not in any list a unit
test could inspect.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genesis.service import GenesisService
from tests.test_engine_gaps import _write_package

GATE = {"path": "condition.factors.governance", "op": "eq", "value": "strict"}


def _process(pid: str, phase: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "executor": {"mode": "deterministic"},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": phase},
        **extra,  # an explicit trigger overrides the default
    }


def _run(tmp_path: Path, *, other_producer: bool, consumer_produces: bool) -> dict[str, list[str]]:
    """Map each condition to the producers of the records `settle` received."""
    processes = [
        _process(
            "detect",
            0,
            measurement=True,
            outputs=[{"artifact_type": "score", "schema_ref": "score"}],
        ),
        _process(
            "settle",
            1,
            # Repeats, so from its second run it has earlier records of its own.
            trigger={"type": "phase", "phase": 1, "repeat": True},
            inputs=["score"],
            measurement_use=[{"source": "detect", "rationale": "sanction", "when": GATE}],
            **(
                {"outputs": [{"artifact_type": "score", "schema_ref": "score"}]}
                if consumer_produces
                else {}
            ),
        ),
    ]
    if other_producer:
        processes.append(
            _process("tally", 0, outputs=[{"artifact_type": "score", "schema_ref": "score"}])
        )
    workspace = tmp_path / f"ws-{other_producer}-{consumer_produces}"
    source = _write_package(
        workspace,
        {
            "openness": {"processes": processes},
            "domain": {
                "visibility": [{"id": "p", "allow": []}],
                "artifacts": [{"id": "score", "artifact_type": "score", "schema_ref": "score"}],
            },
            "protocol": {
                "time_model": {"type": "rounds", "start": 0, "end": 2},
                "factors": [{"id": "governance", "levels": ["none", "strict"]}],
            },
        },
        "gate-study",
    )
    (source / "schemas").mkdir(exist_ok=True)
    (source / "schemas" / "score.json").write_text(
        json.dumps({"type": "object", "properties": {"by": {"type": "string"}}})
    )
    service = GenesisService(workspace)
    received: dict[str, list[str]] = {}
    try:
        build = service.compile_study(source, "builds/gate")["path"]
        service.create_run({"id": "e", "study_id": "gate-study", "build": build})

        def settle(invocation: Any) -> dict[str, Any]:
            seen = sorted(
                str(record.get("producer_process")) for record in (invocation.inputs or {}).values()
            )
            received.setdefault(str(invocation.condition.get("id")), []).append(",".join(seen))
            return {"score": {"by": "settle"}} if consumer_produces else {}

        service.execute_protocol(
            "e",
            executor_overrides={
                "detect": lambda inv: {"score": {"by": "detect"}},
                "tally": lambda inv: {"score": {"by": "tally"}},
                "settle": settle,
            },
        )
    finally:
        service.close()
    return received


def _last(received: dict[str, list[str]], condition: str) -> set[str]:
    return set(filter(None, received[condition][-1].split(",")))


def test_the_measurement_is_hidden_when_the_gate_is_off(tmp_path: Path) -> None:
    received = _run(tmp_path, other_producer=False, consumer_produces=False)
    assert "detect" not in _last(received, "governance-none")
    assert "detect" in _last(received, "governance-strict")


def test_another_producer_of_the_same_type_no_longer_opens_the_gate(tmp_path: Path) -> None:
    """The leak: tally producing `score` too meant nothing was hidden."""
    received = _run(tmp_path, other_producer=True, consumer_produces=False)
    off = _last(received, "governance-none")
    assert "detect" not in off, off
    assert "tally" in off, "the other producer's records must still arrive"


def test_a_consumer_keeps_its_own_records_when_the_gate_is_off(tmp_path: Path) -> None:
    """The over-hiding: settle lost its own earlier `score` records."""
    received = _run(tmp_path, other_producer=False, consumer_produces=True)
    # Control: with the gate on it must see itself, or this test proves nothing.
    assert "settle" in _last(received, "governance-strict"), received
    off = _last(received, "governance-none")
    assert "detect" not in off, off
    assert "settle" in off, off


@pytest.mark.parametrize("other_producer", [False, True])
def test_the_gate_opens_fully_when_its_condition_holds(
    tmp_path: Path, other_producer: bool
) -> None:
    received = _run(tmp_path, other_producer=other_producer, consumer_produces=False)
    assert "detect" in _last(received, "governance-strict")

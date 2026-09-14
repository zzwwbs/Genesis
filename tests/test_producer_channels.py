"""An input is unproduced only if nothing in the package produces it.

The producer check read `process.outputs` and nothing else, but the runtime also
produces an artifact from a recorded_artifact executor's declared outputs and
from a retry policy's fallback outputs. A package whose only producer used one
of those was refused INPUT_ARTIFACT_UNPRODUCED -- and with the check removed it
compiled, ran, and delivered the artifact.

The refusal blocks a package outright, so the cost of missing a channel is much
higher than the cost of accepting one it cannot see.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from genesis.compiler import StudyCompiler, ValidationIssue
from tests.test_engine_gaps import _write_package

CONSUMER: dict[str, Any] = {
    "id": "reader",
    "executor": {"mode": "deterministic"},
    "context_policy": "p",
    "trigger": {"type": "phase", "phase": 1},
    "inputs": ["report"],
}


def _package(workspace: Path, producer: dict[str, Any], name: str) -> Path:
    return _write_package(
        workspace,
        {
            "openness": {"processes": [producer, CONSUMER]},
            "domain": {
                "visibility": [{"id": "p", "allow": []}],
                "artifacts": [{"id": "report", "artifact_type": "report", "schema_ref": "report"}],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 1}},
        },
        name,
    )


def _codes(tmp_path: Path, producer: dict[str, Any], name: str) -> list[str]:
    source = _package(tmp_path / name, producer, name)
    try:
        StudyCompiler(source).compile(tmp_path / name / "build")
    except ValidationIssue as issue:
        return [item.code for item in issue.issues]
    return []


def test_the_ordinary_outputs_channel_still_counts(tmp_path: Path) -> None:
    producer = {
        "id": "write",
        "executor": {"mode": "deterministic"},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0},
        "outputs": [{"artifact_type": "report", "schema_ref": "report"}],
    }
    assert "INPUT_ARTIFACT_UNPRODUCED" not in _codes(tmp_path, producer, "ordinary")


def test_a_recorded_artifact_executor_counts_as_a_producer(tmp_path: Path) -> None:
    producer = {
        "id": "write",
        "executor": {"mode": "recorded_artifact", "parameters": {"outputs": {"report": {}}}},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0},
    }
    assert "INPUT_ARTIFACT_UNPRODUCED" not in _codes(tmp_path, producer, "recorded")


def test_a_retry_fallback_counts_as_a_producer(tmp_path: Path) -> None:
    producer = {
        "id": "write",
        "executor": {"mode": "deterministic"},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0},
        "retry_policy": {"max_attempts": 2, "fallback_outputs": {"report": {}}},
    }
    assert "INPUT_ARTIFACT_UNPRODUCED" not in _codes(tmp_path, producer, "fallback")


def test_an_input_nothing_produces_is_still_refused(tmp_path: Path) -> None:
    """The check must keep catching what it was built for."""
    producer = {
        "id": "write",
        "executor": {"mode": "deterministic"},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0},
    }
    assert "INPUT_ARTIFACT_UNPRODUCED" in _codes(tmp_path, producer, "none")


def test_an_input_matched_only_by_type_is_flagged(tmp_path: Path) -> None:
    """It compiles -- some channel this check cannot read might key by id -- but
    the ordinary path keys inputs by the artifact's own id, so the consumer gets
    nothing. Probed: the matching-id case delivers one input, this delivers none."""
    import json as _json

    source = _write_package(
        tmp_path / "typeonly",
        {
            "openness": {
                "processes": [
                    {
                        "id": "write",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "p",
                        "trigger": {"type": "phase", "phase": 0},
                        "outputs": [{"artifact_type": "document", "schema_ref": "document"}],
                    },
                    {**CONSUMER, "trigger": {"type": "phase", "phase": 1}},
                ]
            },
            "domain": {
                "visibility": [{"id": "p", "allow": []}],
                "artifacts": [
                    {"id": "report", "artifact_type": "document", "schema_ref": "document"}
                ],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 1}},
        },
        "typeonly-study",
    )
    (source / "schemas").mkdir(exist_ok=True)
    (source / "schemas" / "document.json").write_text(
        _json.dumps({"type": "object", "properties": {"body": {"type": "string"}}})
    )
    build = StudyCompiler(source).compile(tmp_path / "typeonly" / "build")
    report = _json.loads((build.path / "validation_report.json").read_text())
    codes = {warning["code"] for warning in report.get("warnings", [])}
    assert "INPUT_ARTIFACT_TYPE_ONLY" in codes

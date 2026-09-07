"""AW-16: state and artifact execution wiring through the application service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis.service import GenesisService

PACKAGE_OPENNESS = {
    "schema_version": "1.0",
    "study_id": "state-study",
    "processes": [
        {
            "id": "tick",
            "executor": {},
            "context_policy": "observe-context",
            "state_effects": [{"field": "counter", "op": "set"}],
        },
        {
            "id": "observe",
            "executor": {},
            "context_policy": "observe-context",
            "dependencies": {"after": ["tick"]},
            "state_effects": [{"field": "counter", "op": "set"}],
        },
    ],
}

PACKAGE = {
    "id": "state-study",
    "title": "State wiring study",
    "processes": PACKAGE_OPENNESS["processes"],
    "theory": {"theory_family": "exploratory"},
    "domain": {
        "states": [
            {"id": "counter", "value_type": "integer", "initial": 0},
        ],
        "artifacts": [{"id": "note", "artifact_type": "text"}],
        "visibility": [{"id": "observe-context", "allow": ["counter"]}],
    },
    "protocol": {
        "time_model": {"type": "rounds", "end": 5},
        "conditions": [],
        "replications": 1,
    },
    "outcomes": [],
    "models": [],
}


def _setup(tmp_path: Path) -> GenesisService:
    service = GenesisService(tmp_path / "workspace")
    draft = service.create_specification(PACKAGE)
    approved = service.approve_specification("state-study", draft["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(None, "builds/state-study", specification_id="state-study")
    service.create_run({"id": "run-1", "study_id": "state-study", "build": compiled["path"]})
    return service


def test_service_applies_declared_state_effects_and_persists_state(tmp_path: Path) -> None:
    service = _setup(tmp_path)
    try:
        overrides = {"tick": lambda _invocation: {"counter": 5}}
        result = service.execute_run("run-1", executor_overrides=overrides)
        assert result["status"] == "completed"
        version, state = service.persistence.latest_json_state("run-1")
        assert version >= 1
        assert state["counter"] == 5
    finally:
        service.close()


def test_later_process_receives_committed_state_in_its_context(tmp_path: Path) -> None:
    service = _setup(tmp_path)
    try:
        seen: dict[str, object] = {}

        def observe(invocation) -> dict[str, object]:
            seen["context"] = invocation.context.data
            return {}

        overrides = {"tick": lambda _invocation: {"counter": 5}, "observe": observe}
        result = service.execute_run("run-1", executor_overrides=overrides)
        assert result["status"] == "completed"
        # The observer ran after tick committed, so its authorised context carries the new state.
        assert seen["context"] == {"counter": 5}
    finally:
        service.close()


def test_declared_artifact_output_is_persisted(tmp_path: Path) -> None:
    service = _setup(tmp_path)
    try:
        overrides = {"tick": lambda _invocation: {"counter": 5, "note": "hello-artifact"}}
        result = service.execute_run("run-1", executor_overrides=overrides)
        assert result["status"] == "completed"
        artifacts = service.persistence.list_artifacts("run-1")
        payloads = [json.loads(artifact["payload"]) for artifact in artifacts]
        assert any(
            payload.get("outputs", {}).get("note") == "hello-artifact" for payload in payloads
        )
        # AW-16/artifact contract: declared identity, metadata, and lineage survive.
        declared = [
            (artifact["artifact_id"], json.loads(artifact["payload"]))
            for artifact in artifacts
            if artifact["artifact_id"].startswith("note-")
        ]
        assert declared, "no declared artifact row persisted"
        artifact_id, payload = declared[0]
        assert payload["value"] == "hello-artifact"
        assert payload["content_hash"]
        assert payload["producer_event"]
        assert isinstance(payload["lineage"], list)
        assert payload.get("visibility") is None  # catalog declared no visibility
        trace = service.trace_run("run-1")
        assert any(event.get("kind") == "process_completed" for event in trace)
    finally:
        service.close()


def test_invalid_state_effect_rejects_the_commit(tmp_path: Path) -> None:
    service = _setup(tmp_path)
    try:
        with pytest.raises((TypeError, ValueError)):
            service.execute_run(
                "run-1", executor_overrides={"tick": lambda _invocation: {"counter": "bad"}}
            )
        run = service.get_run("run-1")
        assert run["status"] == "failed"
        version, state = service.persistence.latest_json_state("run-1")
        assert state["counter"] in (0, None) or version == 1
    finally:
        service.close()


def test_state_snapshot_is_persisted_even_when_no_executor_emits_effects(
    tmp_path: Path, tmp_path_factory
) -> None:
    service = _setup(tmp_path)
    try:
        result = service.execute_run("run-1")
        assert result["status"] == "completed"
        version, state = service.persistence.latest_json_state("run-1")
        assert state["counter"] == 0
        assert version >= 1
    finally:
        service.close()

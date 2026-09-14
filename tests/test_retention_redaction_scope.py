"""Retention purges provider bodies, never a study's declared outputs.

Redaction matched `response`, `raw_response` and `parsed_response` at any
depth, so a declared output whose field was named `response` -- the obvious name
for a model's answer -- was overwritten in exports and by the durable purge. And
retention was matched as a substring, so "never-purge-this" purged.
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis.compiler import _validate_retention
from genesis.persistence import PersistenceCoordinator, _redact_raw_responses, retention_purges
from genesis.service import GenesisService
from genesis.specification.models import OpennessSpec

PURGED = "<purged-by-retention>"


def _artifact() -> dict:
    return {
        "process_id": "rate",
        "outputs": {"rating": {"response": "HIGH", "reason": "strong"}, "response": "HIGH"},
        "value": {"response": "HIGH"},
        "raw_response": '{"response": "HIGH"}',
        "parsed_response": {"response": "HIGH"},
        "provider_attempts": [{"raw_response": "body", "parsed_response": {"a": 1}, "status": 200}],
    }


def test_only_provider_bodies_are_redacted() -> None:
    redacted, changed = _redact_raw_responses(_artifact())
    assert changed
    assert redacted["outputs"] == _artifact()["outputs"]
    assert redacted["value"] == {"response": "HIGH"}
    assert redacted["raw_response"] == PURGED and redacted["parsed_response"] == PURGED
    assert redacted["provider_attempts"] == [
        {"raw_response": PURGED, "parsed_response": PURGED, "status": 200}
    ]


def test_export_rows_redact_inside_event_metadata_and_artifact_payload() -> None:
    event = {"event_id": "e", "metadata": {"raw_response": "body", "response": "kept"}}
    row = {"artifact_id": "a", "payload": _artifact()}
    events = GenesisService._redact_raw_responses([event])
    rows = GenesisService._redact_raw_responses([row])
    assert events[0]["metadata"] == {"raw_response": PURGED, "response": "kept"}
    assert rows[0]["payload"]["outputs"]["rating"]["response"] == "HIGH"
    assert rows[0]["payload"]["raw_response"] == PURGED


def test_the_durable_purge_keeps_declared_outputs(tmp_path: Path) -> None:
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        store.commit_process_result(
            {"event_id": "e1", "run_id": "r", "kind": "process_completed", "state_version": 1},
            {"run_id": "r", "state_version": 1, "payload": b"{}"},
            [{"artifact_id": "a", "run_id": "r", "payload": json.dumps(_artifact()).encode()}],
        )
        assert store.retention_purge("r", {"rate"}) == 1
        [row] = list(store.iter_artifacts("r"))
        payload = json.loads(row["payload"])
        assert payload["outputs"] == _artifact()["outputs"]
        assert payload["raw_response"] == PURGED
    finally:
        store.close()


def test_retention_is_an_exact_value() -> None:
    assert retention_purges("purge-raw-after-run")
    assert not retention_purges("never-purge-this")
    assert not retention_purges(None)

    def refused(value: str) -> list[str]:
        spec = OpennessSpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "s",
                "processes": [
                    {
                        "id": "p",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "none",
                        "trace_policy": {"retention": value},
                    }
                ],
            }
        )
        return [error["code"] for error in _validate_retention(spec)]

    assert refused("never-purge-this") == ["RETENTION_UNKNOWN"]
    assert refused("full") == [] and refused("purge") == []


def test_the_purge_keeps_the_record_of_an_object_something_else_still_names(
    tmp_path: Path,
) -> None:
    """Its transaction checked four tables, its file pass every reference (L5)."""
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        payload = json.dumps(_artifact()).encode()
        store.commit_process_result(
            {"event_id": "e1", "run_id": "r", "kind": "process_completed", "state_version": 1},
            {"run_id": "r", "state_version": 1, "payload": b"{}"},
            [{"artifact_id": "a", "run_id": "r", "payload": payload}],
        )
        [old_ref] = [
            row[0]
            for row in store.connection.execute(
                "SELECT payload_ref FROM artifacts WHERE run_id='r'"
            )
        ]
        store.record_package_version("s", 1, "h", None, "draft", {"snapshot_digest": old_ref})
        assert store.retention_purge("r", {"rate"}) == 1
        kept = store.connection.execute(
            "SELECT COUNT(*) FROM objects WHERE digest = ?", (old_ref,)
        ).fetchone()[0]
        assert kept == 1
        assert (store.object_store.root / old_ref[:2] / old_ref[2:]).is_file()
    finally:
        store.close()

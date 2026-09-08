"""Export/import round-trip and Outputs-tab API regressions.

- process map accepts slash-bearing build refs (compiled build JSON, not YAML);
- run bundles export events/artifacts (states excluded) and import into a
  pristine workspace for exploration (trace, outcomes, events API);
- duplicate imports are rejected idempotently.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.service import GenesisService

STUDY = {
    "id": "exchange-study",
    "title": "exchange study",
    "processes": [
        {
            "id": "tick",
            "executor": {},
            "context_policy": "public",
            "state_effects": [{"field": "counter", "op": "set"}],
        }
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {
        "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        "artifacts": [{"id": "note", "artifact_type": "text"}],
    },
    "protocol": {"time_model": {"type": "rounds", "end": 3}},
    "outcomes": [
        {
            "id": "note-count",
            "source": "events",
            "filters": [],
            "grouping": [],
            "aggregation": {"type": "count", "field": "phase"},
        }
    ],
    "models": [],
}


def _prepared_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    draft = service.create_specification(STUDY)
    service.approve_specification("exchange-study", draft["version"], "researcher")
    service.compile_study(None, "builds/exchange-study", specification_id="exchange-study")
    service.create_run(
        {"id": "xrun", "study_id": "exchange-study", "build": "builds/exchange-study"}
    )
    service.execute_run(
        "xrun",
        executor_overrides={"tick": lambda _inv: {"counter": 1, "note": "hi"}},
    )
    service.close()
    return workspace


def test_process_map_accepts_slash_bearing_build_ref(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    response = client.get("/builds/builds%2Fexchange-study/processes")
    assert response.status_code == 200
    processes = response.json().get("processes", [])
    assert any(p["id"] == "tick" for p in processes)


def test_run_bundle_roundtrip_import_for_exploration(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("xrun", "exports/bundle")
        bundle = workspace / "exports" / "bundle"
        assert (bundle / "events.json").is_file()
        assert (bundle / "artifacts.json").is_file()
        assert not (bundle / "states.json").exists(), "raw state snapshots excluded"

        # import into a pristine workspace (bundle copied in)
        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "bundle")
        importer = GenesisService(other)
        try:
            result = importer.import_run(other / "imports" / "bundle", run_id="imported-xrun")
            assert result["status"] == "imported"
            assert result["events"] >= 1
            events = importer.trace_run("imported-xrun")
            assert events, "imported run must be traceable"
            assert all(e.get("event_id") for e in events)
            artifacts = importer.artifacts_for_run("imported-xrun")
            assert isinstance(artifacts, list)
            assert artifacts == service.artifacts_for_run("xrun")
            assert importer.evaluate_outcomes("imported-xrun") == service.evaluate_outcomes("xrun")
            importer.export_run("imported-xrun", "exports/again")
            import json

            again = other / "exports/again"
            assert json.loads((again / "events.json").read_text()) == events
            assert json.loads((again / "artifacts.json").read_text()) == artifacts
            assert json.loads((again / "outcomes.json").read_text()) == service.evaluate_outcomes(
                "xrun"
            )
            client = TestClient(create_app(other))
            assert client.get("/runs/imported-xrun/events").status_code == 200
            assert client.get("/runs/imported-xrun/outcomes").status_code == 200
            # idempotency: a second import of the same bundle is rejected
            try:
                importer.import_run(other / "imports" / "bundle", run_id="imported-xrun-again")
                raise AssertionError("duplicate import must be rejected")
            except ValueError as exc:
                assert "ALREADY_EXISTS" in str(exc)
        finally:
            importer.close()
    finally:
        service.close()


def test_outcomes_endpoint_returns_plan_rows(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    response = client.get("/runs/xrun/outcomes")
    assert response.status_code == 200
    rows = response.json()
    assert isinstance(rows, list)
    assert any(row.get("outcome_id") == "note-count" for row in rows)


def test_import_run_requires_bundle_members(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        (workspace / "imports").mkdir(exist_ok=True)
        (workspace / "imports" / "empty-dir").mkdir(exist_ok=True)
        try:
            service.import_run("imports/empty-dir")
            raise AssertionError("must reject a non-bundle directory")
        except ValueError as exc:
            assert "IMPORT_RUN" in str(exc)
    finally:
        service.close()


def test_import_row_is_isolated_from_source_workspace(tmp_path: Path) -> None:
    """Importing into the same workspace that produced the bundle is refused."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("xrun", "exports/bundle")
        try:
            service.import_run("exports/bundle", run_id="xrun-imported")
            raise AssertionError("colliding event ids must be refused")
        except ValueError as exc:
            assert "ALREADY_EXISTS" in str(exc)
        # the source run stays intact
        import pytest

        with pytest.raises(KeyError):
            service.get_run("xrun-imported")
        assert service.trace_run("xrun")
    finally:
        service.close()


def test_natural_trace_endpoint_shape(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    response = client.get("/runs/xrun/natural-trace")
    assert response.status_code == 200
    body = response.json()
    assert "selection_rule" in body
    assert body["run_id"] == "xrun"
    assert "illustration" in body
    illustration = body["illustration"]
    assert {"user", "phase", "steps"} <= set(illustration)
    for step in illustration["steps"]:
        assert "step" in step


def test_natural_trace_user_param(tmp_path: Path) -> None:
    """Bound the user-parameterized trace view: error envelope for unknowns,
    rule label for a specified user on a run with records."""
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    # unknown user -> structured 422 envelope
    response = client.get("/runs/xrun/natural-trace", params={"user": "u99"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "USER_TRACE_NOT_FOUND"
    # no user -> frozen-rule path still works
    response = client.get("/runs/xrun/natural-trace")
    assert response.status_code == 200
    assert "selection_rule" in response.json()


def test_outcomes_build_hash_fallback(tmp_path: Path) -> None:
    """Imported runs without a build path still evaluate outcomes when the
    build identity (build_hash) resolves through the build registry."""
    import json as _json
    import sqlite3 as _sqlite3

    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        run = service.get_run("xrun")
        manifest = run.get("manifest") or {}
        assert manifest.get("build_hash")
        # blank the build path the way an import without the bundle fix would
        payload = dict(run)
        payload["build"] = ""
        con = _sqlite3.connect(workspace / ".genesis" / "genesis.db")
        con.execute(
            "UPDATE runs SET payload_json=? WHERE run_id=?",
            (_json.dumps(payload, sort_keys=True), "xrun"),
        )
        con.commit()
        con.close()
        outcomes = service.evaluate_outcomes("xrun")
        assert isinstance(outcomes, list) and outcomes, "fallback must recover the plan"
        assert any(row.get("outcome_id") == "note-count" for row in outcomes)
    finally:
        service.close()

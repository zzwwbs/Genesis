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


def test_natural_trace_requires_a_starting_point_when_none_is_declared(tmp_path: Path) -> None:
    """A package declaring no trace must be told what to pass, not guessed at."""
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    assert client.get("/runs/xrun/traces").json()["traces"] == []
    response = client.get("/runs/xrun/natural-trace")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "TRACE_SELECTION_REQUIRED"


def test_natural_trace_walks_from_any_actor_without_a_declaration(tmp_path: Path) -> None:
    """Traversal is generic: it works on a study that declares no trace at all."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        actors = [
            actor for event in service.trace_run("xrun") for actor in (event.get("actors") or ())
        ]
        events = service.trace_run("xrun")
    finally:
        service.close()
    client = TestClient(create_app(workspace))

    # An unknown actor is a structured error, not an empty illustration.
    response = client.get("/runs/xrun/natural-trace", params={"actor": "nobody"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ACTOR_TRACE_NOT_FOUND"

    # Any recorded event is a valid starting point.
    seed = str(events[0]["event_id"])
    response = client.get("/runs/xrun/natural-trace", params={"event": seed})
    assert response.status_code == 200
    body = response.json()
    assert body["seed_event"] == seed
    assert body["illustration"]["steps"], "the seed itself is always a step"
    assert any(step["relation"] == "seed" for step in body["illustration"]["steps"])
    if actors:
        response = client.get("/runs/xrun/natural-trace", params={"actor": actors[0]})
        assert response.status_code == 200


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


def test_import_preserves_condition_and_replication(tmp_path, monkeypatch) -> None:
    """F1: importing a reproducibility bundle of a conditioned run must keep
    the condition (factors), condition_id and replication on the imported run
    and on its replay children, with the original seed preserved."""
    import importlib.util

    from genesis.evidence import ExportMode
    from genesis.replay import ReplayMode

    spec = importlib.util.spec_from_file_location(
        "replay_configuration", str(Path(__file__).parent / "test_replay_configuration.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    source_service = mod._source_run(tmp_path, monkeypatch)
    try:
        source_service.export_run(
            "source-strict-3", "exports/cond-bundle", mode=ExportMode.REPRODUCIBILITY
        )
        bundle = tmp_path / "workspace" / "exports" / "cond-bundle"

        other = tmp_path / "other-workspace"
        (other / "imports").mkdir(parents=True)
        shutil.copytree(bundle, other / "imports" / "cond-bundle")
        importer = GenesisService(other)
        try:
            imported = importer.import_run(other / "imports" / "cond-bundle", run_id="imp-strict")
            assert imported["status"] == "imported"
            record = importer.get_run("imp-strict")
            assert record.get("condition_id") == "strict"
            assert record.get("condition", {}).get("factors") == {
                "policy": "strict",
                "peer": "low",
            }
            assert record.get("replication") == 3
            replay = importer.replay_run("imp-strict", mode=ReplayMode.FULL)
            child = importer.get_run(replay["run_id"])
            assert child.get("condition_id") == "strict"
            assert child.get("replication") == 3
            assert child.get("condition", {}).get("factors") == {
                "policy": "strict",
                "peer": "low",
            }
        finally:
            importer.close()
    finally:
        source_service.close()


def test_import_rejects_bundle_without_any_manifest(tmp_path: Path) -> None:
    """F3: a bundle directory that contains the required member files but
    NEITHER bundle_manifest.json NOR integrity.json must be rejected, instead
    of importing with a silently unverified / unsized manifest."""
    import json as _json

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        (workspace / "imports").mkdir(exist_ok=True)
        bare = workspace / "imports" / "bare-bundle"
        bare.mkdir(exist_ok=True)
        (bare / "run_manifest.json").write_text(_json.dumps({"run_id": "bare"}))
        (bare / "events.json").write_text("[]")
        (bare / "artifacts.json").write_text("[]")
        try:
            service.import_run("imports/bare-bundle", size_limit_bytes=1)
            raise AssertionError("a bare bundle must be rejected")
        except ValueError as exc:
            assert "IMPORT_MANIFEST_REQUIRED" in str(exc)
    finally:
        service.close()

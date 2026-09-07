"""AW-17/AW-19: API surface completion, package import/export, correlation IDs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.analysis import AnalysisExporter
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
    approved = service.approve_specification("exchange-study", draft["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(
        None, "builds/exchange-study", specification_id="exchange-study"
    )
    service.create_run({"id": "xrun", "study_id": "exchange-study", "build": compiled["path"]})
    result = service.execute_run(
        "xrun", executor_overrides={"tick": lambda _inv: {"counter": 1, "note": "hi"}}
    )
    assert result["status"] == "completed"
    service.close()
    return workspace


def test_required_routes_are_available(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    assert client.get("/specifications/exchange-study/checklist").status_code == 200
    assert client.get("/builds").status_code == 200
    builds = client.get("/builds").json()
    assert any(build["study_id"] == "exchange-study" for build in builds)
    exported = client.post("/exports", json={"run_id": "xrun", "output": "exports/xrun"})
    assert exported.status_code == 200
    assert exported.json()["status"] == "exported"


def test_experiments_api_group_is_available(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    client = TestClient(create_app(workspace))
    assert client.get("/experiments").status_code == 200
    listed = client.get("/experiments").json()
    assert isinstance(listed, list)


def test_error_envelope_carries_correlation_id(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "empty"))
    response = client.post("/runs/missing/execute", headers={"X-Correlation-ID": "corr-123"})
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["correlation_id"] == "corr-123"
    generated = client.get("/runs/missing")
    assert generated.status_code == 404
    assert generated.json()["error"]["correlation_id"]


def test_export_bundle_contains_complete_evidence(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        service.export_run("xrun", "exports/full")
        exported = workspace / "exports/full"
        assert AnalysisExporter.verify_bundle(exported)
        for name in (
            "run_manifest.json",
            "build_manifest.json",
            "validation_report.json",
            "protocol.json",
            "events.json",
            "artifacts.json",
            "package/study.yaml",
            "outcomes.json",
            "outcomes.parquet",
            "data_dictionary.json",
            "methods.json",
            "replay_lineage.json",
        ):
            assert (exported / name).is_file(), f"missing {name}"
        manifest = json.loads((exported / "run_manifest.json").read_text())
        assert manifest["run_id"] == "xrun"
        events = json.loads((exported / "events.json").read_text())
        assert events  # run executed and emitted events
    finally:
        service.close()


def test_import_package_validates_and_registers_draft(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        source = workspace / ".genesis/specifications/exchange-study"
        imported = service.import_package(source, specification_id="imported-study")
        assert imported["status"] == "draft"
        assert imported["id"] == "imported-study"
        assert service.get_specification("imported-study")["version"] == 1
        # The imported draft can be approved and compiled like any other.
        approved = service.approve_specification("imported-study", 1, "researcher")
        assert approved["status"] == "approved"
    finally:
        service.close()


def test_import_reconstructs_full_editable_form(tmp_path: Path) -> None:
    """Finding 10: imported detail survives guided-edit reconstruct/erase cycles."""
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        source = workspace / ".genesis/specifications/exchange-study"
        imported = service.import_package(source, specification_id="form-study")
        form = service.get_specification("form-study")["form"]
        assert form["processes"]  # processes survive
        assert "theory" in form and "protocol" in form and "domain" in form
        # Editing from the reconstructed form round-trips: canonical equality holds.
        rebuilt = service.update_specification(
            "form-study", {"title": "renamed form study"}, imported["version"]
        )
        assert rebuilt["form"]["processes"] == form["processes"]
    finally:
        service.close()


def test_import_enforces_size_limit(tmp_path: Path) -> None:
    """Finding 10: oversized packages are rejected."""
    import pytest

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        source = workspace / "imports" / "big"
        shutil.copytree(
            workspace / "studies" / "demo"
            if False
            else _prepared_workspace(tmp_path) / ".genesis/specifications/exchange-study",
            source,
        )
        (source / "data").mkdir(exist_ok=True)
        (source / "data" / "blob.bin").write_bytes(b"x" * 2048)
        with pytest.raises(ValueError, match="IMPORT_LIMIT"):
            service.import_package(source, specification_id="big-study", size_limit_bytes=1024)
    finally:
        service.close()


def test_import_rejects_tampered_package(tmp_path: Path) -> None:
    workspace = _prepared_workspace(tmp_path)
    service = GenesisService(workspace)
    try:
        source = workspace / ".genesis/specifications/exchange-study"
        tampered = workspace / "tampered"
        import shutil

        shutil.copytree(source, tampered)
        (tampered / "openness.yaml").write_text("schema_version: '1.0'\nstudy_id: nope\n")
        with pytest.raises(ValueError, match="SCHEMA_INVALID|STUDY_ID_MISMATCH"):
            service.import_package(tampered, specification_id="tampered-study")
        with pytest.raises(KeyError):
            service.get_specification("tampered-study")
    finally:
        service.close()

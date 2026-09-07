"""AW-20: security, retention, backup, and migration safeguards."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from genesis.cli import warn_non_local_binding
from genesis.service import GenesisService


def _workspace_with_run(tmp_path: Path) -> GenesisService:
    service = GenesisService(tmp_path / "workspace")
    draft = service.create_specification(
        {
            "id": "ops-study",
            "title": "ops study",
            "processes": [
                {
                    "id": "tick",
                    "executor": {},
                    "context_policy": "public",
                    "state_effects": [{"field": "counter", "op": "set"}],
                }
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]},
            "protocol": {"time_model": {"type": "rounds", "end": 2}},
            "outcomes": [],
            "models": [],
        }
    )
    service.approve_specification("ops-study", draft["version"], "researcher")
    compiled = service.compile_study(None, "builds/ops-study", specification_id="ops-study")
    service.create_run({"id": "ops-run", "study_id": "ops-study", "build": compiled["path"]})
    service.execute_run("ops-run", executor_overrides={"tick": lambda _inv: {"counter": 7}})
    return service


def test_non_local_binding_warns(tmp_path: Path, capsys) -> None:
    assert not warn_non_local_binding("127.0.0.1")
    assert warn_non_local_binding("0.0.0.0") is True
    captured = capsys.readouterr()
    assert "non-local interface" in captured.err


def test_backup_produces_restorable_sqlite_copy(tmp_path: Path) -> None:
    service = _workspace_with_run(tmp_path)
    try:
        target = service.workspace / "exports" / "backup.db"
        result = service.backup_run("ops-run", target)
        assert result["status"] == "backed-up"
        assert target.is_file()
        connection = sqlite3.connect(target)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert {"events", "states", "runs", "artifacts"} <= tables
            run = connection.execute(
                "SELECT payload_json FROM runs WHERE run_id = ?", ("ops-run",)
            ).fetchone()
            assert json.loads(run[0])["status"] in {"completed", "running"}
        finally:
            connection.close()
    finally:
        service.close()


def test_doctor_reports_healthy_workspace(tmp_path: Path) -> None:
    service = _workspace_with_run(tmp_path)
    try:
        report = service.doctor()
        assert report["status"] == "ok"
        assert report["checks"]["schema_current"] is True
        assert report["checks"]["wal_mode"] is True
        assert report["checks"]["object_integrity_failures"] == 0
        assert report["checks"]["disk_ok"] is True
        assert report["checks"]["workspace_writable"] is True
        assert isinstance(report["checks"]["credential_presence"], dict)
    finally:
        service.close()


def test_doctor_detects_tampered_object(tmp_path: Path) -> None:
    service = _workspace_with_run(tmp_path)
    try:
        objects = list(service.persistence.object_store.root.glob("??/*"))
        assert objects
        victim = objects[0]
        victim.write_bytes(b"tampered")
        report = service.doctor()
        assert report["status"] == "degraded"
        assert report["checks"]["object_integrity_failures"] >= 1
    finally:
        service.close()


def test_exports_contain_no_credentials(tmp_path: Path) -> None:
    service = _workspace_with_run(tmp_path)
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_OPS_KEY",
            }
        )
        service.export_run("ops-run", "exports/bundle")
        bundle = service.workspace / "exports/bundle"
        for path in bundle.rglob("*"):
            if not path.is_file():
                continue
            content = path.read_bytes() + b""
            assert b"GENESIS_OPS_KEY" not in content
            assert b"sk-secret-value" not in content
    finally:
        service.close()


def test_retained_objects_survive_split_into_two_services(tmp_path: Path) -> None:
    """Reopening a workspace preserves schema version and committed events."""
    service = _workspace_with_run(tmp_path)
    service.close()
    reopened = GenesisService(tmp_path / "workspace")
    try:
        assert reopened.persistence.schema_version() == 7
        assert reopened.trace_run("ops-run")
    finally:
        reopened.close()


def test_retention_policy_purges_raw_responses_from_exports(tmp_path: Path) -> None:
    """AW-20: retention 'purge' removes raw provider responses from export bundles."""
    import json as _json

    from genesis.providers import ProviderResponse

    class RetentionProvider:
        provider = "openai-compatible"

        def __init__(self, **_kwargs) -> None:
            pass

        def generate(self, request) -> ProviderResponse:
            body = _json.dumps({"text": "top-secret-raw-response"})
            return ProviderResponse(
                body, self.provider, request.model, "req-1", parsed=_json.loads(body)
            )

    import genesis.service as svc

    original_provider = svc.OpenAICompatibleProvider
    svc.OpenAICompatibleProvider = RetentionProvider
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_RETENTION_KEY",
            }
        )
        draft = service.create_specification(
            {
                "id": "retention-study",
                "title": "retention",
                "models": [{"id": "mp", "provider": "openai-compatible", "model": "m1"}],
                "processes": [
                    {
                        "id": "compose",
                        "openness_rationale": "open",
                        "closure_rationale": "closed",
                        "executor": {"mode": "generative", "model_profile": "mp"},
                        "context_policy": "public",
                        "prompt_ref": "compose",
                        "outputs": [{"artifact_type": "response", "schema_ref": "out-art"}],
                        "trace_policy": {"retention": "purge-raw-after-run"},
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"artifacts": [{"id": "out-art", "artifact_type": "text"}]},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [],
                "prompts": {"compose": "Say something"},
            }
        )
        schema_dir = workspace / ".genesis" / "specifications" / "retention-study" / "schemas"
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "out-art.yaml").write_text(
            "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
        )
        revised = service.update_specification(
            "retention-study", {"description": "with schema"}, draft["version"]
        )
        service.approve_specification("retention-study", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/retention-study", specification_id="retention-study"
        )
        service.create_run(
            {"id": "ret-run", "study_id": "retention-study", "build": compiled["path"]}
        )
        result = service.execute_run("ret-run")
        assert result["status"] == "completed"
        preview = service.export_preview("ret-run")
        assert preview["sensitive"]["raw_responses"] >= 1
        # Durable purge removes raw-response artifact payloads, keeps the rest.
        enforced = service.retention_enforce("ret-run")
        assert enforced["policy"] == "purge"
        assert enforced["purged_rows"] >= 1
        remaining = service.persistence.list_artifacts("ret-run")
        assert all(b'"response"' not in artifact["payload"] for artifact in remaining)
        # The sensitive bytes are gone from durable object storage too.
        sensitive = b"top-secret-raw-response"
        objects_root = service.persistence.object_store.root
        for path in objects_root.rglob("*"):
            if path.is_file():
                assert sensitive not in path.read_bytes()
        # Events (the immutable ledger) are untouched.
        assert service.trace_run("ret-run")
        service.export_run("ret-run", "exports/ret")
        artifacts_json = json.loads((service.workspace / "exports/ret/artifacts.json").read_text())
        assert "top-secret-raw-response" not in _json.dumps(artifacts_json)
        events_json = json.loads((service.workspace / "exports/ret/events.json").read_text())
        assert "top-secret-raw-response" not in _json.dumps(events_json)
    finally:
        svc.OpenAICompatibleProvider = original_provider
        service.close()


def test_preflight_or_setup_failure_marks_run_failed(tmp_path: Path) -> None:
    """Finding 7: setup failures after transition never leave the run 'running'."""
    import pytest

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "setup-fail-study",
                "title": "setup fail",
                "processes": [
                    {
                        "id": "ghost",
                        "executor": {"mode": "extension", "extension_ref": "ghost-extension"},
                        "context_policy": "public",
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [],
                "models": [],
            }
        )
        service.approve_specification("setup-fail-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/setup-fail-study", specification_id="setup-fail-study"
        )
        service.create_run(
            {"id": "setup-fail-run", "study_id": "setup-fail-study", "build": compiled["path"]}
        )
        with pytest.raises(ValueError, match="EXECUTOR_UNREGISTERED"):
            service.execute_run("setup-fail-run")
        assert service.get_run("setup-fail-run")["status"] == "failed"
    finally:
        service.close()

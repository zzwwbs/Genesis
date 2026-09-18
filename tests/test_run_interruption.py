"""A run a provider cannot serve pauses, resumes anywhere, and runs in one process.

Running out of credit, a rejected key or an outage longer than the retries failed
the run, and a failed run cannot be resumed. Resume set the status to running and
executed nothing. Nothing stopped two processes executing one run at once.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.compiler import _validate_retry_policies
from genesis.persistence import PersistenceCoordinator
from genesis.provider_errors import provider_pause_reason
from genesis.service import GenesisService
from genesis.specification.models import OpennessSpec
from tests.test_concurrent_model_calls import WRITERS, _pooled_controller, _PooledCall, _record
from tests.test_dataset_export import STUDY

CREDIT = "PROVIDER_HTTP: provider returned HTTP 402: Insufficient Balance"


@pytest.mark.parametrize(
    ("message", "kind"),
    [
        (CREDIT, "provider_credit"),
        ("PROVIDER_HTTP: provider returned HTTP 401 after 0 retries: bad key", "provider_auth"),
        (
            "PROVIDER_HTTP: provider returned HTTP 429 after 3 retries: slow down",
            "provider_rate_limit",
        ),
        ("PROVIDER_HTTP: provider returned HTTP 503 after 3 retries: down", "provider_unavailable"),
        (
            "PROVIDER_UNAVAILABLE: provider request failed after 3 retries: refused",
            "provider_unavailable",
        ),
        ("PROVIDER_HTTP: provider returned HTTP 400: bad parameter", None),
        ("PROVIDER_HTTP: provider returned HTTP 404: no such model", None),
        ("PROVIDER_RESPONSE: provider returned invalid JSON", None),
        ("KeyError in study code", None),
    ],
)
def test_only_a_provider_that_cannot_serve_the_call_pauses(message: str, kind: str | None) -> None:
    reason = provider_pause_reason(ValueError(message))
    assert (reason or {}).get("kind") == kind


def _service(tmp_path: Path, end: int = 4) -> GenesisService:
    service = GenesisService(tmp_path)
    study = {
        **STUDY,
        "protocol": {**STUDY["protocol"], "time_model": {"type": "rounds", "start": 1, "end": end}},
    }
    draft = service.create_specification(study)
    service.approve_specification("tables", draft["version"], "researcher")
    build = service.compile_study(None, "builds/t", specification_id="tables")["path"]
    service.create_run({"id": "run", "study_id": "tables", "build": build})
    return service


def _raising_at(phase: int, message: str) -> Any:
    def tick(invocation: Any) -> dict[str, Any]:
        if invocation.phase == phase:
            raise ValueError(message)
        return {}

    return tick


def test_running_out_of_credit_pauses_the_run_and_resume_completes_it(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        paused = service.execute_run("run", executor_overrides={"tick": _raising_at(2, CREDIT)})
        assert paused["status"] == "paused"
        reason = service.get_run("run")["executions"][-1]["paused_by"]
        assert reason["kind"] == "provider_credit" and reason["status"] == 402
        assert reason["process_id"] == "tick" and reason["phase"] == 2
        # The call that could not be served is not a failure of the process.
        kinds = [(e["phase"], e["kind"]) for e in service.trace_run("run", evidence=False)]
        assert kinds == [(1, "process_completed")]
        done = service.resume_run("run")
        assert done["status"] == "completed"
        phases = [e["phase"] for e in service.trace_run("run", evidence=False)]
        assert phases == [1, 2, 3, 4]
    finally:
        service.close()


def test_a_fault_in_the_study_still_fails_the_run(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="bad parameter"):
            service.execute_run(
                "run",
                executor_overrides={
                    "tick": _raising_at(
                        2, "PROVIDER_HTTP: provider returned HTTP 400: bad parameter"
                    )
                },
            )
        assert service.get_run("run")["status"] == "failed"
    finally:
        service.close()


class _Outage(_PooledCall):
    """Serves every writer except w5, whose provider is down until it recovers."""

    def __init__(self) -> None:
        super().__init__()
        self.down = True

    def execute(self, invocation: Any) -> Any:
        if self.down and invocation.actor_ids[0] == "w5":
            raise ValueError("PROVIDER_HTTP: provider returned HTTP 503 after 3 retries: down")
        return super().execute(invocation)


def test_an_outage_in_a_concurrent_batch_pauses_and_resumes_to_the_uninterrupted_run(
    tmp_path: Path,
) -> None:
    clean, clean_db = _pooled_controller(tmp_path, _PooledCall(), limit=4, name="clean")
    try:
        clean.run("r", phase_limit=1)
        expected = _record(clean, clean_db)
    finally:
        clean_db.close()

    outage = _Outage()
    first, db = _pooled_controller(tmp_path, outage, limit=4, name="outage")
    try:
        first.run("r", phase_limit=1)
        assert first.status == "paused" and first.pause_reason["status"] == 503
        committed = [e["actors"][0] for e in db.list_events("r")]
        assert committed == WRITERS[: WRITERS.index("w5")]
    finally:
        db.close()

    outage.down = False
    resumed, db = _pooled_controller(tmp_path, outage, limit=4, name="outage")
    try:
        resumed.run("r", phase_limit=1)
        assert resumed.status == "completed"
        record = _record(resumed, db)
        assert record["events"] == expected["events"]
        assert record["state"] == expected["state"]
    finally:
        db.close()


def test_a_protocol_reports_a_paused_cell_apart_from_failures(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        result = service.execute_protocol(
            "run", executor_overrides={"tick": _raising_at(2, CREDIT)}
        )
        assert result["status"] == "partial"
        assert result["failed_runs"] == []
        assert len(result["paused_runs"]) == 2
        assert all("402" in detail for detail in result["paused_runs"].values())
    finally:
        service.close()


def test_a_run_lease_is_held_by_one_process_at_a_time(tmp_path: Path) -> None:
    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    other = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.acquire_run_lease("r", "host:1:a", 60)
        with pytest.raises(ValueError, match=r"RUN_ALREADY_EXECUTING.*host:1:a"):
            other.acquire_run_lease("r", "host:2:b", 60)
        assert store.renew_run_lease("r", "host:1:a", 60)
        store.connection.execute("UPDATE run_leases SET expires_at = '2000-01-01T00:00:00+00:00'")
        other.acquire_run_lease("r", "host:2:b", 60)  # the holder stopped renewing
        assert not store.renew_run_lease("r", "host:1:a", 60)
        store.release_run_lease("r", "host:1:a")  # not its lease: a no-op
        with pytest.raises(ValueError, match="RUN_ALREADY_EXECUTING"):
            store.acquire_run_lease("r", "host:1:a", 60)
        other.release_run_lease("r", "host:2:b")
        store.acquire_run_lease("r", "host:1:a", 60)
    finally:
        store.close()
        other.close()


def test_a_run_another_process_is_executing_is_refused_and_left_alone(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        service.persistence.acquire_run_lease("run", "elsewhere:99:zz", 60)
        with pytest.raises(ValueError, match="RUN_ALREADY_EXECUTING.*another process"):
            service.execute_run("run", executor_overrides={"tick": lambda _inv: {}})
        assert service.get_run("run")["status"] == "created"
        assert service.trace_run("run", evidence=False) == []
    finally:
        service.close()


def test_losing_the_lease_mid_run_stops_without_claiming_the_run(tmp_path: Path) -> None:
    service = _service(tmp_path, end=6)
    service.LEASE_TTL_SECONDS = 0.4  # renewed every 0.1 s
    try:

        def tick(invocation: Any) -> dict[str, Any]:
            if invocation.phase == 3:
                service.persistence.connection.execute(
                    "UPDATE run_leases SET expires_at = '2000-01-01T00:00:00+00:00'"
                )
                service.persistence.acquire_run_lease("run", "elsewhere:99:zz", 60)
                time.sleep(0.4)
            return {}

        with pytest.raises(ValueError, match="RUN_LEASE_LOST"):
            service.execute_run("run", executor_overrides={"tick": tick})
        run = service.get_run("run")
        assert run["status"] == "running"  # the other process's to decide
        assert max(e["phase"] for e in service.trace_run("run", evidence=False)) < 6
    finally:
        service.close()


def test_resume_executes_the_run_from_the_api_and_refuses_an_ended_one(tmp_path: Path) -> None:
    client = TestClient(create_app(service=_service(tmp_path)))
    service = client.app.state.service
    try:
        service.execute_run("run", executor_overrides={"tick": _raising_at(2, CREDIT)})
        assert service.get_run("run")["status"] == "paused"
        version = service.get_run("run")["version"]
        stale = client.post("/runs/run/resume", headers={"If-Match": f'"{version - 1}"'})
        assert stale.status_code == 409
        resumed = client.post("/runs/run/resume", headers={"If-Match": f'"{version}"'})
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "completed"
        again = client.post("/runs/run/resume")
        assert again.status_code == 409 and again.json()["error"]["code"] == "RUN_TRANSITION"
    finally:
        service.close()


def test_the_cli_resume_command_executes_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from genesis.cli import main

    service = _service(tmp_path)
    try:
        service.execute_run("run", executor_overrides={"tick": _raising_at(2, CREDIT)})
    finally:
        service.close()
    main(["resume", str(tmp_path), "--run-id", "run"])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def _retry_codes(policy: dict[str, Any]) -> list[str]:
    spec = OpennessSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "s",
            "processes": [
                {
                    "id": "p",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "none",
                    "retry_policy": policy,
                }
            ],
        }
    )
    return [error["message"] for error in _validate_retry_policies(spec)]


def test_a_retry_policy_the_runtime_does_not_read_is_refused() -> None:
    assert _retry_codes({"max_attempts": 2}) == []
    assert (
        _retry_codes({"failure_policy": "use_declared_fallback", "fallback_outputs": {"x": {}}})
        == []
    )
    assert "on_exhausted" in _retry_codes({"max_attempts": 2, "on_exhausted": "record-missing"})[0]
    assert "at least 1" in _retry_codes({"max_attempts": 0})[0]
    assert "not applied" in _retry_codes({"failure_policy": "record-missing"})[0]
    assert "needs fallback_outputs" in _retry_codes({"failure_policy": "use_declared_fallback"})[0]

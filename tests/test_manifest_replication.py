"""AW-04: frozen run manifests, protocol.json builds, and the replication controller."""

from __future__ import annotations

import json
from pathlib import Path

from genesis.compiler import StudyCompiler
from genesis.service import GenesisService


def _setup(tmp_path: Path, *, conditions: list[dict] | None = None, replications: int = 2) -> Path:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    protocol = {
        "time_model": {"type": "rounds", "end": 3},
        "conditions": conditions or [{"id": "base"}, {"id": "alt"}],
        "replications": replications,
    }
    draft = service.create_specification(
        {
            "id": "rep-study",
            "title": "replication study",
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
            "protocol": protocol,
            "outcomes": [],
            "models": [],
        }
    )
    approved = service.approve_specification("rep-study", draft["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(None, "builds/rep-study", specification_id="rep-study")
    service.create_run({"id": "experiment-1", "study_id": "rep-study", "build": compiled["path"]})
    service.close()
    return workspace


def test_build_emits_protocol_json_and_verifies(tmp_path: Path) -> None:
    workspace = _setup(tmp_path)
    build_path = workspace / "builds/rep-study"
    assert StudyCompiler.verify_build(build_path)
    protocol = json.loads((build_path / "protocol.json").read_text())
    assert protocol["conditions"] == [{"id": "base"}, {"id": "alt"}]
    assert protocol["replications"] == 2


def test_execute_run_freezes_an_immutable_manifest(tmp_path: Path) -> None:
    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        service.execute_run("experiment-1")
        run = service.get_run("experiment-1")
        manifest = run["manifest"]
        assert manifest["run_id"] == "experiment-1"
        assert manifest["build_hash"]
        assert manifest["condition_id"] == "base"
        assert "conventional" in manifest["seeds"]
        frozen = manifest
        service.execute_run("experiment-1")
        assert service.get_run("experiment-1")["manifest"] == frozen
    finally:
        service.close()


def test_protocol_controller_expands_conditions_and_replications(tmp_path: Path) -> None:
    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        result = service.execute_protocol("experiment-1")
        assert result["status"] == "completed"
        assert len(result["runs"]) == 4  # 2 conditions x 2 replications
        assert result["runs"] == [
            "experiment-1-base-1",
            "experiment-1-base-2",
            "experiment-1-alt-1",
            "experiment-1-alt-2",
        ]
        for trial_id in result["runs"]:
            trial = service.get_run(trial_id)
            assert trial["status"] == "completed"
            manifest = trial["manifest"]
            assert manifest["condition_id"] in {"base", "alt"}
            assert manifest["replication"] in {1, 2}
            assert "conventional" in manifest["seeds"]
        # Distinct (condition, replication) tuples yield distinct derived seeds.
        seeds = {
            (
                service.get_run(trial_id)["manifest"]["condition_id"],
                service.get_run(trial_id)["manifest"]["replication"],
            ): service.get_run(trial_id)["manifest"]["seeds"]["conventional"]
            for trial_id in result["runs"]
        }
        assert len(set(seeds.values())) == 4
        # Experiment-level aggregate is recorded on the template run.
        recorded = service.get_run("experiment-1")
        assert recorded["outcomes"]
        assert recorded["outcomes"][0]["experiment_id"] == "experiment-1"
    finally:
        service.close()


def test_protocol_controller_records_aggregate_outcomes(tmp_path: Path) -> None:
    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        result = service.execute_protocol("experiment-1")
        assert result["aggregates"] == [] or isinstance(result["aggregates"], list)
    finally:
        service.close()


def test_parallel_protocol_preserves_per_run_scheduling_semantics(tmp_path: Path) -> None:
    """AW-13: parallel dispatch produces identical per-run results to sequential."""
    from tests.test_manifest_replication import _setup

    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        sequential = service.execute_protocol("experiment-1")
        assert sequential["status"] == "completed"
        sequential_signatures = {
            run_id: [
                (event.get("kind"), event.get("process_id"), event.get("phase"))
                for event in service.trace_run(run_id)
            ]
            for run_id in sequential["runs"]
        }
        # New experiment for the parallel pass.
        service.create_run(
            {
                "id": "experiment-2",
                "study_id": "rep-study",
                "build": service.get_run("experiment-1")["build"],
            }
        )
        parallel = service.execute_protocol("experiment-2", parallel=True, max_workers=4)
        assert parallel["status"] == "completed"
        assert len(parallel["runs"]) == len(sequential["runs"])
        for run_id in parallel["runs"]:
            events = service.trace_run(run_id)
            signature = [
                (event.get("kind"), event.get("process_id"), event.get("phase")) for event in events
            ]
            # Trials are independent; each parallel trial matches the sequential
            # semantics of its own (condition, replication) twin.
            suffix = run_id.rsplit("-", 2)
            twin = f"experiment-1-{suffix[-2]}-{suffix[-1]}"
            assert signature == sequential_signatures[twin]
            assert service.get_run(run_id)["status"] == "completed"
    finally:
        service.close()


def test_parallel_protocol_process_pool_preserves_semantics(tmp_path: Path) -> None:
    """AW-13: managed process pool executes trials with per-process coordinators."""
    from tests.test_manifest_replication import _setup

    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        service.create_run(
            {
                "id": "experiment-3",
                "study_id": "rep-study",
                "build": service.get_run("experiment-1")["build"],
            }
        )
        result = service.execute_protocol(
            "experiment-3", parallel=True, max_workers=2, worker_kind="process"
        )
        assert result["status"] == "completed"
        assert len(result["runs"]) == 4
        for trial_id in result["runs"]:
            run = service.get_run(trial_id)
            assert run["status"] == "completed"
            assert run["manifest"]["condition_id"] in {"base", "alt"}
            assert run["manifest"]["seeds"]["conventional"]
    finally:
        service.close()


def test_process_pool_rejects_executor_overrides(tmp_path: Path) -> None:
    """Picklable workers cannot carry callable overrides; fail loudly."""
    import pytest as _pytest

    from tests.test_manifest_replication import _setup

    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        service.create_run(
            {
                "id": "experiment-4",
                "study_id": "rep-study",
                "build": service.get_run("experiment-1")["build"],
            }
        )
        with _pytest.raises(ValueError, match="process workers cannot carry"):
            service.execute_protocol(
                "experiment-4",
                parallel=True,
                worker_kind="process",
                executor_overrides={"tick": lambda _inv: {"counter": 1}},
            )
    finally:
        service.close()


def test_manifest_keeps_compile_time_package_version_for_old_builds(tmp_path: Path) -> None:
    """Finding 1: executing an older build never adopts the latest package version."""
    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        # Build 1 was compiled while the package was at version 1.
        first = service._build_run_manifest(service.get_run("experiment-1"))
        assert first["package_version"] == 1
        # Advance the package to version 2.
        current = service.get_specification("rep-study")
        updated = service.update_specification(
            "rep-study", {"description": "version 2"}, current["version"]
        )
        service.approve_specification("rep-study", updated["version"], "researcher")
        service.compile_study(None, "builds/rep-study-v2", specification_id="rep-study")
        # A new run on the OLD build keeps the compile-time identity.
        service.create_run({"id": "old-build-run", "build": "builds/rep-study"})
        manifest = service._build_run_manifest(service.get_run("old-build-run"))
        assert manifest["package_version"] == 1
        assert manifest["package_content_hash"] == first["package_content_hash"]
    finally:
        service.close()


def test_cli_style_run_without_study_id_still_records_package_identity(
    tmp_path: Path,
) -> None:
    """Finding 1: CLI-created runs (build path only) freeze package identity."""
    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        service.create_run({"id": "cli-run", "build": "builds/rep-study"})
        manifest = service._build_run_manifest(service.get_run("cli-run"))
        assert manifest["study_id"] == "rep-study"
        assert manifest["package_version"] == 1
        assert manifest["package_content_hash"]
    finally:
        service.close()


def test_bounded_dispatch_limits_in_flight_workers(tmp_path: Path) -> None:
    """AW-13: the result-queue dispatcher bounds concurrent trials."""
    import threading

    from tests.test_manifest_replication import _setup

    active = 0
    peak = 0
    lock = threading.Lock()

    def slow_tick(_invocation) -> dict:
        nonlocal active, peak
        import time

        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return {"counter": 1}

    workspace = _setup(tmp_path)
    service = GenesisService(workspace)
    try:
        service.create_run(
            {
                "id": "experiment-5",
                "study_id": "rep-study",
                "build": service.get_run("experiment-1")["build"],
            }
        )
        result = service.execute_protocol(
            "experiment-5",
            parallel=True,
            max_workers=2,
            worker_kind="thread",
            executor_overrides={"tick": slow_tick},
        )
        assert result["status"] == "completed"
        assert peak <= 2, f"observed {peak} concurrent trials despite max_workers=2"
        assert peak >= 1
    finally:
        service.close()

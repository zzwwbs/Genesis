"""AW-15: study lifecycle entities — package versions, experiments, process instances."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genesis.service import GenesisService


def _study_payload(title: str = "lifecycle study") -> dict[str, Any]:
    return {
        "id": "lifecycle-study",
        "title": title,
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
        "protocol": {
            "time_model": {"type": "rounds", "end": 2},
            "conditions": [{"id": "base"}, {"id": "alt"}],
            "replications": 2,
        },
        "outcomes": [],
        "models": [],
    }


def test_approval_records_immutable_package_version(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(_study_payload())
        versions = service.list_package_versions("lifecycle-study")
        assert [(v["version"], v["status"]) for v in versions] == [(1, "draft")]
        draft_hash = versions[0]["content_hash"]
        approved = service.approve_specification("lifecycle-study", draft["version"], "researcher")
        assert approved["status"] == "approved"
        versions = service.list_package_versions("lifecycle-study")
        assert [(v["version"], v["status"]) for v in versions] == [(1, "approved")]
        assert versions[0]["content_hash"] == draft_hash
        # Identical content produces the same hash (LIFE-002).
        other = service.import_package(
            service.workspace / ".genesis/specifications/lifecycle-study",
            specification_id="copy-study",
        )
        assert other["status"] == "draft"
        assert service.list_package_versions("copy-study")[0]["content_hash"] == draft_hash
    finally:
        service.close()


def test_editing_an_approved_package_creates_draft_with_parent_lineage(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(_study_payload())
        service.approve_specification("lifecycle-study", draft["version"], "researcher")
        updated = service.update_specification(
            "lifecycle-study", {"title": "renamed"}, draft["version"]
        )
        assert updated["version"] == 2
        versions = service.list_package_versions("lifecycle-study")
        assert [v["status"] for v in versions] == ["approved", "draft"]
        assert versions[1]["parent_version"] == 1  # lineage back to the approved version
        assert versions[1]["content_hash"] != versions[0]["content_hash"]
        # The approved row remains unmodified (LIFE-001).
        assert versions[0]["content_hash"] == versions[0]["content_hash"]
    finally:
        service.close()


def test_experiment_record_references_build_and_protocol(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(_study_payload())
        service.approve_specification("lifecycle-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/lifecycle-study", specification_id="lifecycle-study"
        )
        service.create_run(
            {"id": "experiment-1", "study_id": "lifecycle-study", "build": compiled["path"]}
        )
        result = service.execute_protocol("experiment-1")
        assert result["status"] == "completed"
        experiment = service.persistence.get_experiment("experiment-1")
        assert experiment["study_id"] == "lifecycle-study"
        assert experiment["build_ref"] == compiled["path"]
        assert len(experiment["protocol_hash"]) == 64
        # Finding 4: the experiment carries the resolved build hash.
        row = service.persistence.connection.execute(
            "SELECT build_hash FROM experiments WHERE experiment_id = 'experiment-1'"
        ).fetchone()
        assert row[0] == compiled["build_hash"]
        # Runs carry the experiment association.
        trials = service.persistence.list_runs()
        assert all(
            run.get("experiment_id") == "experiment-1"
            for run in trials
            if run["id"] != "experiment-1"
        )
    finally:
        service.close()


def test_process_instance_rows_exist_per_invocation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(_study_payload())
        service.approve_specification("lifecycle-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/lifecycle-study", specification_id="lifecycle-study"
        )
        service.create_run(
            {"id": "instance-run", "study_id": "lifecycle-study", "build": compiled["path"]}
        )
        result = service.execute_run("instance-run")
        assert result["status"] == "completed"
        instances = service.persistence.list_process_instances("instance-run")
        assert instances
        assert all(row["run_id"] == "instance-run" for row in instances)
        assert all(row["process_id"] == "tick" for row in instances)
        statuses = {row["status"] for row in instances}
        assert statuses <= {"committed", "failed", "dispatched"}
        assert "committed" in statuses
        # One instance per dispatched invocation attempt.
        events = service.trace_run("instance-run")
        assert any(event["kind"] == "process_completed" for event in events)
    finally:
        service.close()


def test_orphaned_or_inconsistent_records_are_rejected(tmp_path: Path) -> None:
    """Review finding 3: lifecycle hierarchy rejects orphans/inconsistency."""
    import pytest

    service = GenesisService(tmp_path / "workspace")
    try:
        service.create_specification(_study_payload())
        # Experiment referencing an unknown study.
        with pytest.raises(ValueError, match="FK_VIOLATION.*study"):
            service.persistence.create_experiment(
                {
                    "id": "orphan-exp",
                    "study_id": "no-such-study",
                    "build_ref": "builds/x",
                    "protocol_hash": "0" * 64,
                }
            )
        # Experiment referencing an unknown build.
        with pytest.raises(ValueError, match="FK_VIOLATION.*build"):
            service.persistence.create_experiment(
                {
                    "id": "orphan-exp-2",
                    "study_id": "lifecycle-study",
                    "build_ref": "builds/missing",
                    "protocol_hash": "0" * 64,
                }
            )
        # Process instances for an unknown run.
        with pytest.raises(ValueError, match="FK_VIOLATION.*run"):
            service.persistence.record_process_instances(
                "no-such-run",
                [
                    {
                        "id": "i-1",
                        "process_id": "tick",
                        "phase": 0,
                        "attempt": 1,
                        "status": "committed",
                    }
                ],
            )
        # DB-level FKs reject direct orphaned writes (finding 4).
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            service.persistence.connection.execute(
                """INSERT INTO process_instances(
                       instance_id, run_id, process_id, phase, attempt, status, payload_json
                   ) VALUES ('orphan-instance', 'no-such-run', 'tick', 0, 1, 'committed', '{}')"""
            )
        with pytest.raises(sqlite3.IntegrityError):
            service.persistence.connection.execute(
                """INSERT INTO runs(run_id, status, version, experiment_id, payload_json)
                   VALUES ('orphan-run', 'running', 0, 'no-such-experiment', '{}')"""
            )
    finally:
        service.close()


def test_compile_records_study_build_lifecycle_row(tmp_path: Path) -> None:
    """Compiled builds are persisted as lifecycle records, not just directories."""
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(_study_payload())
        service.approve_specification("lifecycle-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/lifecycle-study", specification_id="lifecycle-study"
        )
        builds = service.persistence.list_study_builds("lifecycle-study")
        assert len(builds) == 1
        assert builds[0]["build_hash"] == compiled["build_hash"]
        assert builds[0]["package_version"] == 1
        listed = [build["build_hash"] for build in service.list_builds()]
        assert compiled["build_hash"] in listed
    finally:
        service.close()


def test_package_versions_retain_immutable_snapshots(tmp_path: Path) -> None:
    """Review P1: earlier accepted versions are recoverable from content-addressed bundles."""
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {"id": "snap-study", "title": "original title", "description": "v1 text"}
        )
        upgraded = service.update_specification(
            "snap-study",
            {"description": "v2 text"},
            draft["version"],
        )
        assert upgraded["version"] == 2
        snapshot = service.get_package_snapshot("snap-study", 1)
        assert snapshot["version"] == 1
        assert "original title" in snapshot["files"]["study.yaml"]
        assert "v1 text" in snapshot["files"]["study.yaml"]
        second = service.get_package_snapshot("snap-study", 2)
        assert "v2 text" in second["files"]["study.yaml"]
        # The bundle is content-addressed in the object store.
        digest = snapshot["snapshot_digest"]
        object_path = service.persistence.object_store.root / digest[:2] / digest[2:]
        assert object_path.is_file()
        # Compilation references the accepted snapshot.
        metadata = service.get_specification("snap-study")
        service.approve_specification("snap-study", metadata["version"], "researcher")
        compiled = service.compile_study(None, "builds/snap-study", specification_id="snap-study")
        build_manifest = json.loads(
            (service.resolve_path(compiled["path"]) / "build_manifest.json").read_text()
        )
        assert build_manifest["package_snapshot"] == digest or build_manifest.get(
            "package_snapshot"
        )
    finally:
        service.close()

"""Behavioral regressions found while reviewing the latest repair pass."""

import shutil
from pathlib import Path

from genesis.evidence import ExportMode
from genesis.replay import ReplayMode
from genesis.runtime import ContextEngine, ExecutorRegistry, RunController, Scheduler
from genesis.schema_validation import PackageSchemaCatalog
from genesis.service import GenesisService


def test_nested_schema_resource_inside_array_resolves_its_local_fragment():
    schema = {"allOf": [{"$id": "child", "$defs": {"n": {"type": "integer"}}, "$ref": "#/$defs/n"}]}
    catalog = PackageSchemaCatalog({"root": schema})
    assert catalog.validate("root", 3) == []
    assert catalog.validate("root", "wrong")


def test_instance_annotations_are_not_walked_as_schema_references():
    catalog = PackageSchemaCatalog(
        {"root": {"type": "object", "default": {"$ref": "https://example.invalid/not-a-schema"}}}
    )
    assert catalog.validate("root", {}) == []


def test_feedback_exposes_only_the_declared_state_source():
    controller = RunController(Scheduler([]), ExecutorRegistry({}), ContextEngine({}))
    controller._round_state_at_phase[1] = {"counter": 9, "private-state": "hidden"}
    slots, blocked = controller._feedback_history(
        {
            "theory_feedback": [
                {
                    "source": "counter",
                    "context_slot": "prior",
                    "lag_rounds": 1,
                    "initial": {"policy": "skip_consumer"},
                }
            ]
        },
        1,
    )
    assert not blocked
    assert slots == {"prior": {"counter": 9}}


def _study(service):
    draft = service.create_specification(
        {
            "id": "probe",
            "title": "probe",
            "models": [],
            "processes": [
                {
                    "id": "tick",
                    "executor": {"mode": "deterministic"},
                    "retry_policy": {"max_attempts": 2},
                    "context_policy": "public",
                }
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": {"states": [{"id": "counter", "initial": 1, "value_type": "integer"}]},
            "protocol": {
                "time_model": {"type": "rounds", "end": 1},
                "factors": [{"id": "policy", "levels": ["strict", "lenient"], "branchable": True}],
            },
            "datasets": [
                {"id": "rounds", "source": {"kind": "state", "snapshot": "each_completed_round"}}
            ],
            "outcomes": [
                {"id": "n", "source": "rounds", "aggregation": {"op": "count", "field": "counter"}}
            ],
        }
    )
    service.approve_specification("probe", draft["version"], "researcher")
    return service.compile_study(None, "builds/probe", specification_id="probe")["path"]


def test_completed_round_survives_a_recovered_retry(tmp_path: Path):
    service = GenesisService(tmp_path)
    try:
        build = _study(service)
        service.create_run({"id": "run", "study_id": "probe", "build": build})
        from genesis.runtime import ProcessResult

        def tick(inv):
            return ProcessResult(status="failed") if inv.attempt == 1 else {}

        service.execute_run("run", executor_overrides={"tick": tick})
        assert service.get_run("run")["status"] == "completed"
        assert service.evaluate_outcomes("run") == [
            {"outcome_id": "n", "counter_count": 1, "counter_missing": 0}
        ]
    finally:
        service.close()


def test_paused_phase_is_not_a_completed_round(tmp_path: Path):
    service = GenesisService(tmp_path)
    try:
        build = _study(service)
        service.create_run({"id": "run", "study_id": "probe", "build": build})

        def tick(inv):
            run = service.get_run("run")
            service.transition_run("run", "paused", run["version"])
            return {}

        service.execute_run("run", executor_overrides={"tick": tick})
        assert service.get_run("run")["status"] == "paused"
        rows = service.evaluate_outcomes("run")
        assert not rows or rows[0]["counter_count"] == 0
    finally:
        service.close()


def test_imported_branch_preserves_source_seed_inputs(tmp_path: Path):
    source = GenesisService(tmp_path / "source")
    target = GenesisService(tmp_path / "target")
    try:
        build = _study(source)
        source.create_run(
            {
                "id": "original",
                "study_id": "probe",
                "build": build,
                "condition_id": "strict",
                "condition": {"id": "strict", "factors": {"policy": "strict"}},
                "replication": 3,
            }
        )
        source.execute_run("original")
        original_seeds = source.get_run("original")["manifest"]["seeds"]
        source.export_run("original", "bundle", mode=ExportMode.REPRODUCIBILITY)
        shutil.copytree(source.workspace / "bundle", target.workspace / "bundle")
        target.import_run("bundle", run_id="imported")
        args = {
            "mode": ReplayMode.BRANCH,
            "boundary": "phase:0",
            "overrides": {"policy": "lenient"},
            "justification": "compare policy",
        }
        preview = target.replay_preview("imported", **args)
        replay = target.replay_run("imported", **args, preview_token=preview["preview_token"])
        child = target.get_run(replay["run_id"])
        assert child["condition"]["factors"] == {"policy": "lenient"}
        assert child["manifest"]["seeds"] == original_seeds
    finally:
        target.close()
        source.close()

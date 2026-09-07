from __future__ import annotations

from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    derive_seed,
    expand_protocol_conditions,
)
from genesis.service import GenesisService


def test_factorial_protocol_expands_to_stable_three_by_two_cells() -> None:
    conditions = expand_protocol_conditions(
        {
            "factors": [
                {"id": "governance", "levels": ["none", "opaque", "disclosed"]},
                {"id": "peer-visibility", "levels": ["low", "high"]},
            ]
        }
    )

    assert len(conditions) == 6
    assert conditions[0] == {
        "id": "governance-none-peer-visibility-low",
        "factors": {"governance": "none", "peer-visibility": "low"},
    }
    assert conditions[-1]["id"] == "governance-disclosed-peer-visibility-high"


def test_explicit_conditions_remain_backward_compatible() -> None:
    conditions = [{"id": "control", "governance": "none"}]
    assert expand_protocol_conditions({"conditions": conditions}) == conditions


def test_matched_seed_excludes_trial_and_condition_identity() -> None:
    left = derive_seed(
        7,
        "experiment-a-control-1",
        "recommend",
        "user-1",
        experiment_id="experiment-a",
        condition_id="control",
        replication=1,
        matching_key="conventional",
    )
    right = derive_seed(
        7,
        "experiment-a-treatment-1",
        "recommend",
        "user-1",
        experiment_id="experiment-a",
        condition_id="treatment",
        replication=1,
        matching_key="conventional",
    )

    assert left == right
    assert left != derive_seed(
        7,
        "experiment-a-treatment-2",
        "recommend",
        "user-1",
        experiment_id="experiment-a",
        condition_id="treatment",
        replication=2,
        matching_key="conventional",
    )


class PhaseRecorder:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, invocation):
        self.calls.append(invocation)
        return ProcessResult()


def test_protocol_phase_bounds_and_condition_reach_every_invocation() -> None:
    recorder = PhaseRecorder()
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "creator-decision",
                    "trigger": {"phase": 1, "repeat": True},
                    "context_policy": "private",
                }
            ]
        ),
        ExecutorRegistry({"creator-decision": recorder}),
        ContextEngine({"private": {"allow": []}}),
    )

    controller.run(
        "study-run",
        phase_start=1,
        phase_end=3,
        condition={"id": "disclosed-high", "factors": {"governance": "disclosed"}},
    )

    assert [call.phase for call in recorder.calls] == [1, 2, 3]
    assert all(call.condition["factors"]["governance"] == "disclosed" for call in recorder.calls)


def test_service_realizes_factor_cells_and_matched_manifest_seeds(tmp_path) -> None:
    service = GenesisService(tmp_path / "workspace")
    draft = service.create_specification(
        {
            "id": "factor-study",
            "title": "factor study",
            "processes": [
                {
                    "id": "tick",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "public",
                }
            ],
            "theory": {"theory_family": "exploratory"},
            "domain": {},
            "protocol": {
                "time_model": {"type": "rounds", "start": 1, "end": 2},
                "factors": [
                    {"id": "governance", "levels": ["none", "opaque", "disclosed"]},
                    {"id": "peer-visibility", "levels": ["low", "high"]},
                ],
                "matching": {"enabled": True, "shared_streams": ["conventional"]},
            },
            "outcomes": [],
            "models": [],
        }
    )
    service.approve_specification("factor-study", draft["version"], "researcher")
    build = service.compile_study(None, "builds/factor-study", specification_id="factor-study")
    service.create_run(
        {"id": "factor-experiment", "study_id": "factor-study", "build": build["path"]}
    )

    result = service.execute_protocol("factor-experiment")

    assert len(result["runs"]) == 6
    trials = [service.get_run(run_id) for run_id in result["runs"]]
    assert all(trial["condition"]["factors"] for trial in trials)
    assert len({trial["manifest"]["seeds"]["conventional"] for trial in trials}) == 1
    assert all(
        [event["phase"] for event in service.trace_run(trial["id"])] == [1] for trial in trials
    )
    service.close()

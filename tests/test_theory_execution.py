"""THY — bounded theory-to-execution operationalization (G2).

Theory prose is never interpreted as executable mathematics: explicit,
researcher-approved execution bindings of a small supported kind are compiled
into the schedule/context/mechanism plan, everything else is reported as
annotation-only in the coverage report. Covers THY-001..006.
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis.theory_execution import compile_theory_execution

PROCESSES = {"form-strategy", "write-article", "update-relations"}


def _codes(plan) -> set[str]:
    return {issue.code for issue in plan.issues}


def test_unbound_declarations_are_annotations_not_operational() -> None:
    """Spec §6.3: old prose without an execution binding must not become executable."""
    plan = compile_theory_execution(
        {
            "relations": [{"from": "a", "to": "b", "relation": "informs"}],
            "feedback": [{"from": "mem", "to": "form-strategy", "relation": "retained"}],
            "delays": [{"process": "form-strategy", "rounds": 1}],
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert plan.valid
    assert not plan.precedence_edges
    assert not plan.feedback_bindings
    assert not plan.mechanism_bindings
    assert len(plan.annotations) == 3
    assert all("annotation" in reason for _dtype, _did, reason in plan.annotations)


def test_precedence_binding_adds_schedule_edge() -> None:
    """THY-001: a relation with a precedence binding changes the compiled schedule."""
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "write-after-strategy",
                    "from": "strategy",
                    "to": "content",
                    "relation": "strategic adaptation precedes production",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "form-strategy",
                        "consumer_process": "write-article",
                        "lag_rounds": 0,
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert plan.valid
    assert ("form-strategy", "write-article", 0) in plan.precedence_edges
    assert "write-after-strategy" in plan.resolved


def test_feedback_binding_one_round_lag_requires_initial_policy() -> None:
    """THY-002: positive lag needs an explicit initial policy, else rejected."""
    plan = compile_theory_execution(
        {
            "feedback": [
                {
                    "id": "perf-to-strategy",
                    "from": "performance-state",
                    "to": "form-strategy",
                    "relation": "adapts to prior performance",
                    "execution": {
                        "kind": "feedback_context",
                        "source": {"kind": "state", "id": "performance-state"},
                        "consumer_process": "form-strategy",
                        "context_slot": "prior-performance",
                        "lag_rounds": 1,
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert "THEORY_INITIAL_POLICY_REQUIRED" in _codes(plan)
    ok = compile_theory_execution(
        {
            "feedback": [
                {
                    "id": "perf-to-strategy",
                    "from": "performance-state",
                    "to": "form-strategy",
                    "relation": "adapts to prior performance",
                    "execution": {
                        "kind": "feedback_context",
                        "source": {"kind": "state", "id": "performance-state"},
                        "consumer_process": "form-strategy",
                        "context_slot": "prior-performance",
                        "lag_rounds": 1,
                        "initial": {"policy": "declared_default", "value": {}},
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert ok.valid
    assert len(ok.feedback_bindings) == 1
    binding = ok.feedback_bindings[0]
    assert binding[1] == "performance-state"
    assert binding[2] == "form-strategy"
    assert binding[3] == "prior-performance"
    assert binding[4] == 1


def test_zero_lag_dependency_cycle_is_rejected() -> None:
    """THY-003: zero-lag precedence cycles fail; positive-lag feedback may not."""
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "a-b",
                    "from": "a",
                    "to": "b",
                    "relation": "x",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "form-strategy",
                        "consumer_process": "write-article",
                        "lag_rounds": 0,
                    },
                },
                {
                    "id": "b-a",
                    "from": "b",
                    "to": "a",
                    "relation": "y",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "write-article",
                        "consumer_process": "form-strategy",
                        "lag_rounds": 0,
                    },
                },
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert "THEORY_CYCLE_ZERO_LAG" in _codes(plan)


def test_mechanism_binding_verifies_existing_transition_only() -> None:
    """THY-004: a mechanism binding refers to one existing transition, never adds one."""
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "m1",
                    "from": "a",
                    "to": "b",
                    "relation": "realized by the counter update",
                    "execution": {
                        "kind": "mechanism_binding",
                        "mechanism": "counter-update",
                        "consumer_process": "update-relations",
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms={"counter-update"},
    )
    assert plan.valid
    assert len(plan.mechanism_bindings) == 1
    assert plan.mechanism_bindings[0][1] == "counter-update"
    # Unknown mechanism is rejected, not silently dropped.
    bad = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "m2",
                    "from": "a",
                    "to": "b",
                    "relation": "x",
                    "execution": {"kind": "mechanism_binding", "mechanism": "missing"},
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms={"counter-update"},
    )
    assert "THEORY_MECHANISM_UNKNOWN" in _codes(bad)


def test_conflicting_existing_dependency_is_rejected() -> None:
    """THY-004: theory-generated edges conflict with existing openness deps."""
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "dup",
                    "from": "x",
                    "to": "y",
                    "relation": "z",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "form-strategy",
                        "consumer_process": "write-article",
                        "lag_rounds": 0,
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
        existing_dependencies={"write-article": ["form-strategy"]},
    )
    assert "THEORY_DEPENDENCY_CONFLICT" in _codes(plan)


def test_annotation_requires_a_reason() -> None:
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "no-reason",
                    "from": "a",
                    "to": "b",
                    "relation": "x",
                    "execution": {"kind": "annotation"},
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert "THEORY_ANNOTATION_WITHOUT_REASON" in _codes(plan)


def test_unknown_process_and_unsupported_kind_fail_visibly() -> None:
    plan = compile_theory_execution(
        {
            "relations": [
                {
                    "id": "ghost",
                    "from": "a",
                    "to": "b",
                    "relation": "x",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "ghost-process",
                        "consumer_process": "write-article",
                        "lag_rounds": 0,
                    },
                },
                {
                    "id": "weird",
                    "from": "a",
                    "to": "b",
                    "relation": "x",
                    "execution": {"kind": "some-future-kind"},
                },
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    assert "THEORY_PROCESS_UNKNOWN" in _codes(plan)
    assert "THEORY_KIND_UNSUPPORTED" in _codes(plan)
    assert not plan.valid


def test_plan_is_versioned_and_serializable() -> None:
    plan = compile_theory_execution(
        {
            "feedback": [
                {
                    "id": "fb",
                    "from": "s",
                    "to": "form-strategy",
                    "relation": "r",
                    "execution": {
                        "kind": "feedback_context",
                        "source": {"kind": "state", "id": "s"},
                        "consumer_process": "form-strategy",
                        "context_slot": "slot",
                        "lag_rounds": 1,
                        "initial": {"policy": "skip_consumer", "value": {}},
                    },
                }
            ]
        },
        known_processes=PROCESSES,
        known_mechanisms=set(),
    )
    serialized = plan.to_dict()
    assert serialized["version"] == 1
    assert serialized["feedback_bindings"][0]["lag_rounds"] == 1
    assert serialized["feedback_bindings"][0]["initial"]["policy"] == "skip_consumer"

# ---------------------------------------------------------------------------
# F2 (effect level): theory precedence must control actual scheduling order
# ---------------------------------------------------------------------------


def test_theory_precedence_edge_controls_runtime_schedule(
    tmp_path: Path,
) -> None:
    """F2: theory precedence must change which process runs first."""
    from genesis.compiler import StudyCompiler
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    source = tmp_path / "package"
    source.mkdir(parents=True)
    imports = {
        "study": {"schema_version": "1.0", "study_id": "sched", "title": "x"},
        "openness": {
            "schema_version": "1.0",
            "study_id": "sched",
            "processes": [
                {
                    "id": "producer",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "public",
                    "outputs": [{"artifact_type": "result", "schema_ref": "result-schema"}],
                },
                {
                    "id": "consumer",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "public",
                    "dependencies": {"after": []},
                },
            ],
        },
        "theory": {
            "schema_version": "1.0",
            "study_id": "sched",
            "theory_family": "institutional",
            "relations": [
                {
                    "id": "producer-before-consumer",
                    "from": "producer",
                    "to": "consumer",
                    "relation": "producer output precedes consumption",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "producer",
                        "consumer_process": "consumer",
                        "lag_rounds": 0,
                    },
                }
            ],
        },
        "domain": {
            "schema_version": "1.0", "study_id": "sched",
            "artifacts": [{"id": "result", "artifact_type": "object"}],
        },
        "protocol": {"schema_version": "1.0", "study_id": "sched",
                     "time_model": {"type": "rounds", "end": 1}},
        "outcomes": {"schema_version": "1.0", "study_id": "sched", "outcomes": []},
        "models": {"schema_version": "1.0", "study_id": "sched", "models": []},
    }
    for name, value in imports.items():
        (source / f"{name}.yaml").write_text(_yaml_dump(value))
    schema_dir = source / "schemas"
    schema_dir.mkdir()
    (schema_dir / "result-schema.yaml").write_text(
        "type: object\nproperties:\n  value: {type: integer}\n"
    )

    build = StudyCompiler(source).compile(tmp_path / "sched-build")
    processes = json.loads((build.path / "processes.json").read_text())
    consumer = next(p for p in processes if p["id"] == "consumer")
    # The theory zero-lag precedence edge must be present in the compiled
    # process definitions (which the runtime scheduler reads).
    assert "producer" in consumer["dependencies"]["after"]

    seen: list[str] = []

    class Recorder:
        def execute(self, invocation):
            seen.append(invocation.process_id)
            return ProcessResult(outputs={"result": {"value": 1}})

    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {
                "producer": Recorder(),
                "consumer": Recorder(),
            }
        ),
        ContextEngine({"public": {"allow": []}}),
    )
    controller.run("sched-run", phase_limit=2)
    # Effect: the consumer may not run before the producer in the same ready
    # window even though no openness dependency declared it.
    producer_idx = seen.index("producer")
    consumer_idx = seen.index("consumer")
    assert producer_idx < consumer_idx


class _yaml_dump_placeholder:
    pass


def _yaml_dump(value):
    import yaml

    return yaml.safe_dump(value, sort_keys=False)

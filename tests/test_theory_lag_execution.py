"""THY — runtime execution of lagged precedence and state feedback (F2+).

Effect-level tests: a positive-lag precedence must actually delay the consumer
to a later phase, and a state-feedback binding must actually inject the state
of ``lag`` completed rounds earlier into the consumer's context slot — not just
be recorded in the plan.
"""

from __future__ import annotations

import json
from pathlib import Path

from genesis.service import GenesisService

PACKAGE_FILES = {
    "study": {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "title": "lag study",
    },
    "openness": {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "processes": [
            {
                "id": "producer",
                "executor": {"mode": "deterministic"},
                "context_policy": "producer-context",
            },
            {
                "id": "consumer",
                "executor": {"mode": "deterministic"},
                "context_policy": "consumer-context",
            },
        ],
    },
    "theory": {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "theory_family": "institutional",
        "relations": [
            {
                "id": "p-before-c",
                "from": "producer",
                "to": "consumer",
                "relation": "consumer waits one round after producer",
                "execution": {
                    "kind": "precedence",
                    "producer_process": "producer",
                    "consumer_process": "consumer",
                    "lag_rounds": 1,
                },
            }
        ],
    },
    "domain": {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        "visibility": [
            {"id": "producer-context", "allow": []},
            {"id": "consumer-context", "allow": []},
        ],
    },
    "protocol": {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "time_model": {"type": "rounds", "end": 3},
    },
    "outcomes": {"schema_version": "1.0", "study_id": "lag-study", "outcomes": []},
    "models": {"schema_version": "1.0", "study_id": "lag-study", "models": []},
}


def _write_package(source: Path) -> None:
    import yaml

    for name, value in PACKAGE_FILES.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))


def _compile(service: GenesisService, source: Path) -> str:
    from genesis.compiler import StudyCompiler

    build = StudyCompiler(source).compile(service.workspace / "builds" / source.name)
    return str(build.path)


def test_positive_lag_precedence_delays_consumer_phase(tmp_path: Path) -> None:
    """F2+: a lag-1 theory precedence must run the consumer one phase later."""
    import yaml

    source = tmp_path / "pkg-lag"
    source.mkdir(parents=True)
    for name, value in PACKAGE_FILES.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    seen: list[tuple[str, int]] = []

    def producer(inv):
        seen.append(("producer", int(inv.phase)))
        return {}

    def consumer(inv):
        seen.append(("consumer", int(inv.phase)))
        return {}

    service = GenesisService(tmp_path / "ws")
    build_path = _compile(service, source)
    # The compiled consumer must carry the theory delay.
    processes = json.loads((Path(build_path) / "processes.json").read_text())
    consumer_record = next(p for p in processes if p["id"] == "consumer")
    assert consumer_record["dependencies"]["after"] == ["producer"]
    # The lag belongs to the producer edge, not to the consumer as a whole, so
    # it compiles into a per-dependency delay.
    assert consumer_record["dependencies"]["delay"] == {"per_dependency": {"producer": 1}}

    service.create_run({"id": "lag-run", "study_id": "lag-study", "build": build_path})
    service.execute_run(
        "lag-run",
        executor_overrides={"producer": producer, "consumer": consumer},
    )
    phases = {pid: next(ph for pid2, ph in seen if pid2 == pid) for pid in ("producer", "consumer")}
    # Effect: consumer executes in a strictly later phase than producer.
    assert phases["consumer"] > phases["producer"]


def test_state_feedback_injects_prior_round_state(tmp_path: Path) -> None:
    """F2+: a state-feedback binding exposes the state of one completed round
    earlier through the declared context slot."""
    import yaml

    source = tmp_path / "pkg-fb"
    source.mkdir(parents=True)
    files = dict(PACKAGE_FILES)
    files["openness"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "processes": [
            {
                "id": "updater",
                "executor": {"mode": "deterministic"},
                "context_policy": "updater-context",
                "state_effects": [{"field": "counter", "op": "set"}],
            },
            {
                "id": "reader",
                "executor": {"mode": "deterministic"},
                "context_policy": "reader-context",
            },
        ],
    }
    files["theory"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "theory_family": "institutional",
        "feedback": [
            {
                "id": "counter-fb",
                "from": "counter",
                "to": "reader",
                "relation": "reader sees last round's counter",
                "execution": {
                    "kind": "feedback_context",
                    "source": {"kind": "state", "id": "counter"},
                    "consumer_process": "reader",
                    "context_slot": "last-counter",
                    "lag_rounds": 1,
                    "initial": {"policy": "declared_default", "value": {"counter": -1}},
                },
            }
        ],
    }
    files["domain"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        "visibility": [
            {"id": "updater-context", "allow": []},
            # The reader policy must declare the feedback slot to see it.
            {"id": "reader-context", "allow": ["feedback.last-counter"]},
        ],
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    from genesis.runtime import ProcessResult

    observed: list[dict[str, object]] = []
    ctr = {"v": 0}

    def updater(inv):
        ctr["v"] += 1
        return ProcessResult(state_effects={"counter": ctr["v"]})

    def reader(inv):
        context = getattr(inv.context, "data", {}) or {}
        context_dict = dict(context) if isinstance(context, dict | type(context)) else {}
        observed.append(
            {
                "phase": int(inv.phase),
                "feedback": dict(context_dict.get("feedback", {}))
                if isinstance(
                    context_dict.get("feedback"), dict | type(context_dict.get("feedback"))
                )
                else {},
            }
        )
        return {}

    service = GenesisService(tmp_path / "ws-fb")
    build_path = _compile(service, source)
    service.create_run({"id": "fb-run", "study_id": "lag-study", "build": build_path})
    result = service.execute_run(
        "fb-run",
        executor_overrides={"updater": updater, "reader": reader},
    )
    assert result["status"] == "completed"
    # On the first phase the updater runs; the reader either runs with the
    # declared default (-1) when no history exists, or sees the prior round.
    assert observed, "reader must have executed"
    # The feedback context must be visible through the slot.
    print("OBSERVED:", observed)
    first = observed[0]["feedback"]
    assert "last-counter" in first, f"feedback slot missing: {first}"
    # There is no prior round in phase 0, so the declared default applies.
    assert first["last-counter"] == {"counter": -1}


def test_state_feedback_skip_consumer_defers_until_history_exists(
    tmp_path: Path,
) -> None:
    """F2+: initial.policy skip_consumer defers the consumer until a full prior
    round exists, then the consumer sees the previous round's state."""
    import yaml

    source = tmp_path / "pkg-skip"
    source.mkdir(parents=True)
    files = dict(PACKAGE_FILES)
    files["openness"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "processes": [
            {
                "id": "updater",
                "executor": {"mode": "deterministic"},
                "context_policy": "updater-context",
                "state_effects": [{"field": "counter", "op": "set"}],
                "trigger": {"repeat": True},
            },
            {
                "id": "reader",
                "executor": {"mode": "deterministic"},
                "context_policy": "reader-context",
                "trigger": {"repeat": True},
            },
        ],
    }
    files["theory"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "theory_family": "institutional",
        "feedback": [
            {
                "id": "counter-skip",
                "from": "counter",
                "to": "reader",
                "relation": "reader defers until a full round exists",
                "execution": {
                    "kind": "feedback_context",
                    "source": {"kind": "state", "id": "counter"},
                    "consumer_process": "reader",
                    "context_slot": "last-counter",
                    "lag_rounds": 1,
                    "initial": {"policy": "skip_consumer"},
                },
            }
        ],
    }
    files["domain"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        "visibility": [
            {"id": "updater-context", "allow": []},
            {"id": "reader-context", "allow": ["feedback.last-counter"]},
        ],
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    from genesis.runtime import ProcessResult

    observed: list[dict[str, object]] = []
    ctr = {"v": 0}

    def updater(inv):
        ctr["v"] += 1
        return ProcessResult(state_effects={"counter": ctr["v"]})

    def reader(inv):
        context = getattr(inv.context, "data", {}) or {}
        context_dict = dict(context)
        feedback = dict(context_dict.get("feedback", {}))
        observed.append({"phase": int(inv.phase), "feedback": feedback})
        return {}

    service = GenesisService(tmp_path / "ws-skip")
    build_path = _compile(service, source)
    service.create_run({"id": "skip-run", "study_id": "lag-study", "build": build_path})
    result = service.execute_run(
        "skip-run",
        executor_overrides={"updater": updater, "reader": reader},
        # end=3 in protocol -> phases 0..2
    )
    assert result["status"] == "completed"
    # The consumer must not run in phase 0 (no prior round)...
    assert observed, "reader must execute after a prior round exists"
    first_phase = observed[0]["phase"]
    assert first_phase >= 1, f"reader ran too early at phase {first_phase}"
    # ...and must then see the previous round's counter value.
    first_feedback = observed[0]["feedback"]["last-counter"]
    assert first_feedback["counter"] >= 1, f"reader did not see prior state: {first_feedback}"


def test_state_feedback_ring_rebuilt_on_resume_from_persistence(
    tmp_path: Path,
) -> None:
    """F2+: the round-state ring is reconstructed from persisted history when a
    controller is created over an already-run (e.g. resumed) run, so lagged
    feedback still reads the completed prior rounds."""
    import yaml

    from genesis.runtime import ProcessResult, RunController, StateStore

    source = tmp_path / "pkg-resume"
    source.mkdir(parents=True)
    files = dict(PACKAGE_FILES)
    files["openness"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "processes": [
            {
                "id": "updater",
                "executor": {"mode": "deterministic"},
                "context_policy": "updater-context",
                "state_effects": [{"field": "counter", "op": "set"}],
            },
            {
                "id": "reader",
                "executor": {"mode": "deterministic"},
                "context_policy": "reader-context",
            },
        ],
    }
    files["theory"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "theory_family": "institutional",
        "feedback": [
            {
                "id": "cfb",
                "from": "counter",
                "to": "reader",
                "relation": "r",
                "execution": {
                    "kind": "feedback_context",
                    "source": {"kind": "state", "id": "counter"},
                    "consumer_process": "reader",
                    "context_slot": "last-counter",
                    "lag_rounds": 1,
                    "initial": {"policy": "declared_default", "value": {}},
                },
            }
        ],
    }
    files["domain"] = {
        "schema_version": "1.0",
        "study_id": "lag-study",
        "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        "visibility": [
            {"id": "updater-context", "allow": []},
            {"id": "reader-context", "allow": ["feedback.last-counter"]},
        ],
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(tmp_path / "ws-resume")
    build_path = _compile(service, source)
    service.create_run({"id": "rz", "study_id": "lag-study", "build": build_path})
    ctr = {"v": 0}

    def updater(inv):
        ctr["v"] += 1
        return ProcessResult(state_effects={"counter": ctr["v"]})

    service.execute_run("rz", executor_overrides={"updater": updater})

    # Simulate a resumed run: a fresh controller over the same persisted run.
    processes = json.loads((Path(build_path) / "processes.json").read_text())
    policies = json.loads((Path(build_path) / "context_policies.json").read_text())
    state_model = json.loads((Path(build_path) / "state_model.json").read_text())
    store = StateStore(
        {s["id"]: (int if s.get("value_type") == "integer" else str) for s in state_model},
        {s["id"]: s.get("initial", 0) for s in state_model},
    )
    controller = RunController(
        __import__("genesis.runtime", fromlist=["Scheduler"]).Scheduler(processes),
        __import__("genesis.runtime", fromlist=["ExecutorRegistry"]).ExecutorRegistry({}),
        __import__("genesis.runtime", fromlist=["ContextEngine"]).ContextEngine(policies),
        persistence=service.persistence,
        state_store=store,
    )
    controller._restore_persisted_frontier("rz")
    # The ring must contain the completed phase-0 final state for phase 1.
    slots, blocked = controller._feedback_history(
        {
            "theory_feedback": [
                {
                    "source": "counter",
                    "context_slot": "last-counter",
                    "lag_rounds": 1,
                    "initial": {"policy": "declared_default", "value": {}},
                }
            ]
        },
        1,
    )
    assert not blocked
    assert slots["last-counter"] == {"counter": 1}
    service.close()

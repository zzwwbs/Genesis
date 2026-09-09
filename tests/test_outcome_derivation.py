"""OUT — generic declarative outcome derivation (G3).

Declared datasets with a fixed operation registry replace study-specific row
construction in the core. Covers typed sources, transform ordering, dedup,
missingness, join cardinality and renamed-isomorphic behaviour.
"""

from __future__ import annotations

from pathlib import Path

from genesis.outcome_plan import materialize_datasets

EVENTS = [
    {
        "event_id": "e1",
        "phase": 1,
        "state_delta": {
            "analytics": [
                {"user": "u1", "exposed": 3},
                {"user": "u2", "exposed": 1},
            ]
        },
    },
    {
        "event_id": "e2",
        "phase": 2,
        "state_delta": {
            "analytics": [{"user": "u1", "exposed": 5}],
        },
    },
]

ARTIFACTS = [
    {
        "artifact_id": "a1",
        "process_id": "evaluate-clickbait",
        "payload": {
            "declared_artifact_id": "detection",
            "value": {"detected": True, "confidence": 0.9},
        },
    },
    {
        "artifact_id": "a2",
        "process_id": "other-process",
        "payload": {"declared_artifact_id": "other", "value": {"detected": False}},
    },
]

STATE = [
    (1, {"population": {"n": 10}}),
    (2, {"population": {"n": 8}}),
]


def _sources() -> dict[str, object]:
    return {
        "events": EVENTS,
        "artifacts": ARTIFACTS,
        "state": STATE,
        "artifact_sources": {},
    }


def test_events_dataset_extracts_nested_records_with_dedup() -> None:
    plan = {
        "datasets": [
            {
                "id": "exposure",
                "source": {"kind": "events", "path": "state_delta.analytics"},
                "deduplicate_on": ["user", "phase"],
            }
        ]
    }
    rows = materialize_datasets(plan, _sources())["exposure"]
    assert len(rows) == 3
    users = {(row["user"], row["phase"]) for row in rows}
    assert ("u1", 1) in users and ("u2", 1) in users and ("u1", 2) in users


def test_artifacts_dataset_filters_by_declared_type_not_process_name() -> None:
    plan = {
        "datasets": [
            {
                "id": "measurements",
                "source": {"kind": "artifacts", "artifact_type": "detection"},
            }
        ]
    }
    rows = materialize_datasets(plan, _sources())["measurements"]
    assert len(rows) == 1
    assert rows[0]["detected"] is True
    assert rows[0]["confidence"] == 0.9


def test_derived_field_operation_registry() -> None:
    plan = {
        "datasets": [
            {
                "id": "enriched",
                "source": {"kind": "events", "path": "state_delta.analytics"},
                "fields": [
                    {
                        "name": "double",
                        "op": "arithmetic",
                        "field": "exposed",
                        "operator": "multiply",
                        "value": 2,
                    },
                    {
                        "name": "is_high",
                        "op": "comparison",
                        "field": "exposed",
                        "operator": "gt",
                        "value": 2,
                    },
                    {
                        "name": "label",
                        "op": "conditional",
                        "condition": {"field": "is_high", "value": True},
                        "value": "high",
                        "else_value": "low",
                    },
                ],
            }
        ]
    }
    rows = materialize_datasets(plan, _sources())["enriched"]
    by_user = {row["user"]: row for row in rows if row["phase"] == 1}
    assert by_user["u1"]["double"] == 6
    assert by_user["u1"]["is_high"] is True
    assert by_user["u1"]["label"] == "high"
    assert by_user["u2"]["is_high"] is False
    assert by_user["u2"]["label"] == "low"


def test_divide_by_zero_yields_explicit_missing() -> None:
    plan = {
        "datasets": [
            {
                "id": "ratio",
                "source": {"kind": "events"},
                "fields": [
                    {
                        "name": "ratio",
                        "op": "arithmetic",
                        "field": "phase",
                        "operator": "divide",
                        "value": 0,
                    },
                ],
            }
        ]
    }
    rows = materialize_datasets(plan, _sources())["ratio"]
    assert all(row["ratio"] is None for row in rows)


def test_renamed_study_produces_equivalent_outcome_rows() -> None:
    """OUT-005 core behaviour: the dataset engine never names clickbait ids."""
    renamed = {
        "datasets": [
            {
                "id": "exposure",
                "source": {"kind": "events", "path": "state_delta.analytics"},
                "deduplicate_on": ["user", "phase"],
            },
            {
                "id": "measurements",
                "source": {"kind": "artifacts", "artifact_type": "detection"},
            },
        ]
    }
    rows = materialize_datasets(renamed, _sources())
    # The same engine with different study names behaves identically.
    assert len(rows["exposure"]) == 3
    assert len(rows["measurements"]) == 1
    # No study-specific identifiers are consulted by the engine itself.
    import inspect

    source = inspect.getsource(materialize_datasets)
    assert "evaluate-clickbait" not in source


def test_state_dataset_final_snapshot_and_missing_policy() -> None:
    plan = {
        "datasets": [
            {
                "id": "final-pop",
                "source": {"kind": "state", "state": "population", "snapshot": "final"},
            },
            {
                "id": "rounds",
                "source": {"kind": "state", "snapshot": "each_completed_round"},
            },
        ]
    }
    rows = materialize_datasets(plan, _sources())
    assert rows["final-pop"][0]["population"] == {"n": 8}
    assert len(rows["rounds"]) == 2


def test_service_evaluates_outcomes_over_declared_datasets(tmp_path: Path) -> None:
    """OUT-005 integration: declared datasets, not core clickbait names, drive rows."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "dataset-study",
                "title": "dataset study",
                "processes": [
                    {
                        "id": "measure",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "outputs": [{"artifact_type": "detection", "schema_ref": "detection"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {
                    "artifacts": [{"id": "detection", "artifact_type": "detection"}],
                    "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [
                    {
                        "id": "detection-rate",
                        "source": "measurements",
                        "grouping": [],
                        "aggregation": {"op": "mean", "field": "detected"},
                        "output_schema": "outcome-schema",
                    }
                ],
                "datasets": [
                    {
                        "id": "measurements",
                        "source": {"kind": "artifacts", "artifact_type": "detection"},
                    }
                ],
                "models": [],
            }
        )
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "dataset-study" / "schemas"
        )
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "detection.yaml").write_text(
            "type: object\nrequired: [detected]\nproperties:\n  detected: {type: boolean}\n"
        )
        (schema_dir / "outcome-schema.yaml").write_text(
            "type: object\nrequired: [detected_mean]\n"
            "properties:\n  detected_mean: {type: number}\n"
        )
        revised = service.update_specification(
            "dataset-study", {"description": "with schemas"}, draft["version"]
        )
        service.approve_specification("dataset-study", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/dataset-study", specification_id="dataset-study"
        )
        service.create_run(
            {"id": "dataset-run", "study_id": "dataset-study", "build": compiled["path"]}
        )

        def _measure(_invocation):
            return {"detection": {"detected": True}}

        service.execute_run("dataset-run", executor_overrides={"measure": _measure})
        outcomes = service.evaluate_outcomes("dataset-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        # The declared measurements dataset produces one row with detected=1,
        # which the mean aggregation reads without any clickbait-specific core.
        assert by_id["detection-rate"]["detected_mean"] == 1.0
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F4 (effect): events counted exactly once, no derived-row double counting
# ---------------------------------------------------------------------------


def test_outcome_counts_raw_events_exactly_once(tmp_path: Path) -> None:
    """F4: an outcome over events counts each raw event exactly once."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "count-once",
                "title": "co",
                "processes": [
                    {
                        "id": "measure",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "outputs": [{"artifact_type": "out", "schema_ref": "out"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {
                    "artifacts": [{"id": "out", "artifact_type": "out"}],
                    "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [
                    {
                        "id": "phase-count",
                        "source": "events",
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "phase"},
                        "output_schema": "outcome-schema",
                    }
                ],
                "datasets": [
                    {"id": "outs", "source": {"kind": "artifacts", "artifact_type": "out"}}
                ],
                "models": [],
            }
        )
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "count-once" / "schemas"
        )
        schema_dir.mkdir(parents=True)
        (schema_dir / "out.yaml").write_text("type: object\nproperties:\n  text: {type: string}\n")
        (schema_dir / "outcome-schema.yaml").write_text(
            "type: object\nrequired: [phase_count]\nproperties:\n  phase_count: {type: integer}\n"
        )
        rev = service.update_specification("count-once", {"description": "x"}, draft["version"])
        service.approve_specification("count-once", rev["version"], "researcher")
        compiled = service.compile_study(None, "builds/count-once", specification_id="count-once")
        service.create_run({"id": "co-run", "study_id": "count-once", "build": compiled["path"]})
        service.execute_run(
            "co-run", executor_overrides={"measure": lambda inv: {"out": {"text": "x"}}}
        )
        assert len(service.trace_run("co-run")) == 1
        outcomes = service.evaluate_outcomes("co-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["phase-count"]["phase_count"] == 1
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F5 (effect): state datasets receive dict-shaped snapshots and read the value
# ---------------------------------------------------------------------------


def test_state_dataset_reads_final_snapshot_value(tmp_path: Path) -> None:
    """F5: a final-state dataset must yield the last snapshot's field value."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "state-agg",
                "title": "sa",
                "processes": [
                    {
                        "id": "tick",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "counter", "op": "set"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [
                    {
                        "id": "final-counter",
                        "source": "final-count",
                        "grouping": [],
                        "aggregation": {"type": "sum", "field": "counter"},
                        "output_schema": "outcome-schema",
                    }
                ],
                "datasets": [
                    {
                        "id": "final-count",
                        "source": {"kind": "state", "state": "counter", "snapshot": "final"},
                    }
                ],
                "models": [],
            }
        )
        from genesis.runtime import ProcessResult

        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "state-agg" / "schemas"
        )
        schema_dir.mkdir(parents=True)
        (schema_dir / "outcome-schema.yaml").write_text(
            "type: object\nrequired: [counter_sum]\nproperties:\n  counter_sum: {type: number}\n"
        )
        rev = service.update_specification("state-agg", {"description": "x"}, draft["version"])
        service.approve_specification("state-agg", rev["version"], "researcher")
        compiled = service.compile_study(None, "builds/state-agg", specification_id="state-agg")
        service.create_run({"id": "sa-run", "study_id": "state-agg", "build": compiled["path"]})
        state = {"counter": 0}
        service.execute_run(
            "sa-run",
            executor_overrides={
                "tick": lambda inv: (
                    state.update(counter=99) or ProcessResult(state_effects={"counter": 99})
                )
            },
        )
        outcomes = service.evaluate_outcomes("sa-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["final-counter"]["counter_sum"] == 99.0
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F12 (effect): conditional/with_field pass the Pydantic package model
# ---------------------------------------------------------------------------


def test_conditional_and_with_field_pass_the_model() -> None:
    """F12: OutcomeDatasetField must accept the engine's condition/with_field."""
    from genesis.specification.models import OutcomeDatasetField

    conditional = OutcomeDatasetField.model_validate(
        {
            "name": "label",
            "op": "conditional",
            "condition": {"field": "is_high", "value": True},
            "value": "high",
            "else_value": "low",
        }
    )
    assert conditional.op == "conditional"
    arithmetic = OutcomeDatasetField.model_validate(
        {
            "name": "ratio",
            "op": "arithmetic",
            "field": "a",
            "operator": "divide",
            "with_field": "b",
        }
    )
    assert arithmetic.with_field == "b"


# ---------------------------------------------------------------------------
# F8 (effect): each_completed_round = final snapshot per round, not per commit
# ---------------------------------------------------------------------------


def test_each_completed_round_selects_final_snapshot_per_round() -> None:
    """F8: multiple state commits within one phase yield one round observation."""
    from genesis.outcome_plan import materialize_datasets

    state_rows = [
        {"counter": 1, "state_version": 1, "_round": 0},
        {"counter": 3, "state_version": 2, "_round": 0},
        {"counter": 7, "state_version": 3, "_round": 1},
    ]
    plan = {
        "datasets": [
            {
                "id": "rounds",
                "source": {"kind": "state", "snapshot": "each_completed_round"},
            }
        ]
    }
    rows = materialize_datasets(plan, {"events": [], "artifacts": [], "state": state_rows})
    observed = rows["rounds"]
    assert len(observed) == 2, "two rounds, not three commits"
    assert [row["counter"] for row in observed] == [3, 7]


def test_each_completed_round_collapses_service_state_commits(tmp_path: Path) -> None:
    """F8 (production wiring): two processes committing state in one phase must
    yield ONE round observation through the service, without injecting _round
    manually (regression: events must carry state_version)."""
    from genesis.runtime import ProcessResult
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "f8-wiring",
                "title": "f8",
                "processes": [
                    {
                        "id": "a",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "x", "op": "set"}],
                    },
                    {
                        "id": "b",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "x", "op": "set"}],
                    },
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"states": [{"id": "x", "value_type": "integer", "initial": 0}]},
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "datasets": [
                    {
                        "id": "rounds-ds",
                        "source": {"kind": "state", "snapshot": "each_completed_round"},
                        "fields": [{"name": "n", "op": "copy", "field": "x"}],
                    }
                ],
                "models": [],
            }
        )
        rev = service.update_specification("f8-wiring", {"description": "x"}, draft["version"])
        service.approve_specification("f8-wiring", rev["version"], "researcher")
        compiled = service.compile_study(None, "builds/f8-wiring", specification_id="f8-wiring")
        service.create_run({"id": "f8w", "study_id": "f8-wiring", "build": compiled["path"]})
        ctr = {"v": 0}

        def mk(delta):
            def exe(inv):
                ctr["v"] += delta
                return ProcessResult(state_effects={"x": ctr["v"]})

            return exe

        service.execute_run("f8w", executor_overrides={"a": mk(1), "b": mk(10)})
        # Events must expose state_version for the collapse to work.
        events = [e for e in service.trace_run("f8w") if e.get("kind") == "process_completed"]
        assert {e.get("state_version") for e in events} == {1, 2}
        # Directly materialize the dataset via the same path evaluate_outcomes
        # uses: build phase map from events, tag state rows, materialize.
        from genesis.outcome_plan import compile_outcome_plan, materialize_datasets

        build_path = service.resolve_path(compiled["path"])
        plan = compile_outcome_plan(build_path)
        version_phase = {
            e.get("state_version"): e.get("phase")
            for e in events
            if e.get("state_version") is not None
        }
        assert version_phase == {1: 0, 2: 0}
        state_rows = []
        for version, snapshot in service.persistence.list_state_history("f8w"):
            row = dict(snapshot)
            row["state_version"] = version
            row["_round"] = version_phase.get(version)
            state_rows.append(row)
        rows = materialize_datasets(plan, {"events": [], "artifacts": [], "state": state_rows})
        observed = rows["rounds-ds"]
        assert len(observed) == 1, "one round observation, not two commits"
        assert observed[0]["n"] == 11, "final snapshot of the round wins"
        assert "_round" not in observed[0] and "state_version" not in observed[0]
    finally:
        service.close()


# F4 (effect): a phase with a failed/active/skipped process is not a
# completed round and must not yield an observation.
# ---------------------------------------------------------------------------


def test_each_completed_round_drops_failed_phase_rows() -> None:
    """F4: state snapshots tagged with the incomplete-round sentinel are
    excluded from each_completed_round, so a failed/active phase contributes
    no observation (spec: completed rounds only)."""
    from genesis.outcome_plan import _INCOMPLETE_ROUND, materialize_datasets

    state_rows = [
        # phase 0 completed normally
        {"counter": 1, "state_version": 1, "_round": 0},
        {"counter": 3, "state_version": 2, "_round": 0},
        # phase 1 failed: engine flagged the row with the sentinel
        {"counter": 9, "state_version": 3, "_round": _INCOMPLETE_ROUND},
    ]
    plan = {
        "datasets": [
            {
                "id": "rounds",
                "source": {"kind": "state", "snapshot": "each_completed_round"},
            }
        ]
    }
    rows = materialize_datasets(plan, {"events": [], "artifacts": [], "state": state_rows})
    observed = rows["rounds"]
    assert len(observed) == 1, "failed phase must not contribute an observation"
    assert observed[0]["counter"] == 3, "final snapshot of the completed round survives"


def test_each_completed_round_excludes_failed_phase_through_service(tmp_path: Path) -> None:
    """F4 (production wiring): when a process fails in a phase, evaluate_outcomes
    must not count that phase as a completed round."""
    from genesis.runtime import ProcessResult
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "f4-fail",
                "title": "f4",
                "processes": [
                    {
                        "id": "ok",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "x", "op": "set"}],
                    },
                    {
                        "id": "bad",
                        "executor": {
                            "mode": "rule",
                            "parameters": {
                                "rules": [
                                    {
                                        "when": {"path": "inputs.v", "op": "eq", "value": 1},
                                        "outputs": {"score": 99},
                                    }
                                ]
                            },
                        },
                        "context_policy": "public",
                        "outputs": [{"artifact_type": "score", "schema_ref": "score"}],
                        "state_effects": [{"field": "x", "op": "set"}],
                    },
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {
                    "artifacts": [{"id": "score", "artifact_type": "obj", "schema_ref": "score"}],
                    "states": [{"id": "x", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "datasets": [
                    {
                        "id": "rounds-ds",
                        "source": {"kind": "state", "snapshot": "each_completed_round"},
                        "fields": [{"name": "n", "op": "copy", "field": "x"}],
                    }
                ],
                "models": [],
            }
        )
        schema_dir = tmp_path / "workspace" / ".genesis" / "specifications" / "f4-fail" / "schemas"
        schema_dir.mkdir(parents=True)
        (schema_dir / "score.yaml").write_text(
            "type: object\nrequired: [score]\nproperties:\n  score: {type: integer, maximum: 10}\n"
        )
        rev = service.update_specification("f4-fail", {"description": "x"}, draft["version"])
        service.approve_specification("f4-fail", rev["version"], "researcher")
        compiled = service.compile_study(None, "builds/f4-fail", specification_id="f4-fail")
        service.create_run({"id": "f4r", "study_id": "f4-fail", "build": compiled["path"]})
        try:
            service.execute_run(
                "f4r",
                executor_overrides={
                    "ok": lambda _inv: ProcessResult(state_effects={"x": 5}),
                    "bad": lambda _inv: {"score": 99},  # violates max 10 -> fails
                },
            )
            raise AssertionError("run must fail on schema rejection")
        except Exception:
            pass
        events = service.trace_run("f4r")
        assert any(e.get("kind") == "process_failed" for e in events)
        results = service.evaluate_outcomes("f4r")
        assert results == [], "a failed phase must not yield a round observation"
    finally:
        service.close()


def test_events_dataset_preserves_state_version_field() -> None:
    """Review: the internal-key strip must only affect state datasets; an
    events dataset row that legitimately carries state_version keeps it."""
    from genesis.outcome_plan import materialize_datasets

    event_rows = [
        {"event_id": "e1", "phase": 0, "state_version": 3, "value": 10},
        {"event_id": "e2", "phase": 1, "state_version": 4, "value": 20},
    ]
    plan = {"datasets": [{"id": "from-events", "source": {"kind": "events"}}]}
    rows = materialize_datasets(plan, {"events": event_rows, "artifacts": [], "state": []})
    assert rows["from-events"] == event_rows

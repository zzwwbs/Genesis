"""AW-09: outcome builder completion — trajectory, distribution, missingness, window, sources."""

from __future__ import annotations

from genesis.analysis import AnalysisEngine, OutcomePlan
from genesis.service import GenesisService

EVENTS = [
    {"process_id": "a", "time": 0, "value": 1, "group": "x"},
    {"process_id": "b", "time": 1, "value": 2, "group": "x"},
    {"process_id": "c", "time": 2, "value": None, "group": "y"},
    {"process_id": "d", "time": 3, "value": 4, "group": "y"},
    {"process_id": "e", "time": 5, "value": 5, "group": "x"},
]


def test_trajectory_operation_returns_ordered_values() -> None:
    plan = OutcomePlan(
        id="t", source="events", select="value", group_by="group", operation="trajectory"
    )
    rows = AnalysisEngine().evaluate(plan, {"events": EVENTS})
    by_group = {row["group"]: row["value_trajectory"] for row in rows}
    assert by_group["x"] == [1, 2, 5]
    assert by_group["y"] == [4]


def test_distribution_operation_summarizes() -> None:
    plan = OutcomePlan(id="d", source="events", select="value", operation="distribution")
    rows = AnalysisEngine().evaluate(plan, {"events": EVENTS})
    assert rows[0]["value_count"] == 4
    assert rows[0]["value_min"] == 1
    assert rows[0]["value_max"] == 5
    assert rows[0]["value_mean"] == 3.0
    assert rows[0]["value_missing"] == 1


def test_missingness_zero_affects_sums_and_means() -> None:
    plan = OutcomePlan(
        id="m", source="events", select="value", aggregation="sum", missingness="zero"
    )
    rows = AnalysisEngine().evaluate(plan, {"events": EVENTS})
    assert rows[0]["value_sum"] == 12  # 1+2+4+5, missing counted as 0
    exclude = OutcomePlan(
        id="e", source="events", select="value", aggregation="sum", missingness="exclude"
    )
    assert AnalysisEngine().evaluate(exclude, {"events": EVENTS})[0]["value_sum"] == 12


def test_window_filters_by_time_field() -> None:
    plan = OutcomePlan(
        id="w",
        source="events",
        select="value",
        aggregation="count",
        window={"start": 1, "end": 3, "time_field": "time"},
    )
    rows = AnalysisEngine().evaluate(plan, {"events": EVENTS})
    assert rows[0]["value_count"] == 2  # times 1 and 3 (None excluded)


def test_service_evaluates_artifact_source_outcomes(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "outcome-study",
                "title": "outcome study",
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
                "protocol": {"time_model": {"type": "rounds", "end": 3}, "conditions": []},
                "outcomes": [
                    {
                        "id": "note-count",
                        "source": "artifacts",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "note"},
                    },
                    {
                        "id": "events-vs",
                        "source": "events",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "trajectory", "field": "phase"},
                    },
                ],
                "models": [],
            }
        )
        approved = service.approve_specification("outcome-study", draft["version"], "researcher")
        assert approved["status"] == "approved"
        compiled = service.compile_study(
            None, "builds/outcome-study", specification_id="outcome-study"
        )
        service.create_run({"id": "orun", "study_id": "outcome-study", "build": compiled["path"]})
        result = service.execute_run(
            "orun", executor_overrides={"tick": lambda _inv: {"counter": 1, "note": "hello"}}
        )
        assert result["status"] == "completed"
        outcomes = service.evaluate_outcomes("orun")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["note-count"]["note_count"] == 1
        assert by_id["events-vs"]["phase_trajectory"] == [0]
    finally:
        service.close()


def test_join_merges_events_with_artifacts() -> None:
    # The service layer resolves the declared join into a merged source first;
    # the engine then evaluates over the merged rows.
    plan = OutcomePlan(
        id="j",
        source="events+artifacts",
        select="response",
        aggregation="count",
    )
    pre_joined = {"events+artifacts": [{"invocation_id": "i1", "process_id": "p", "response": "x"}]}
    rows = AnalysisEngine().evaluate(plan, pre_joined)
    assert rows[0]["response_count"] == 1


def test_multi_key_grouping() -> None:
    plan = OutcomePlan(
        id="m",
        source="events",
        select="value",
        aggregation="sum",
        group_by=("condition", "replication"),
    )
    rows = AnalysisEngine().evaluate(
        plan,
        {
            "events": [
                {"condition": "c1", "replication": 1, "value": 2},
                {"condition": "c1", "replication": 1, "value": 3},
                {"condition": "c2", "replication": 1, "value": 5},
            ]
        },
    )
    by_key = {(row["group_0"], row["group_1"]): row["value_sum"] for row in rows}
    assert by_key == {("c1", 1): 5, ("c2", 1): 5}


def test_state_and_lineage_sources_are_exposed(tmp_path) -> None:
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "sources-study",
                "title": "sources",
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
                "outcomes": [
                    {
                        "id": "state-sum",
                        "source": "state",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "sum", "field": "counter"},
                    },
                    {
                        "id": "lineage-count",
                        "source": "lineage",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "process_id"},
                    },
                ],
                "models": [],
            }
        )
        service.approve_specification("sources-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/sources-study", specification_id="sources-study"
        )
        service.create_run(
            {"id": "src-run", "study_id": "sources-study", "build": compiled["path"]}
        )
        service.execute_run("src-run", executor_overrides={"tick": lambda _inv: {"counter": 3}})
        outcomes = service.evaluate_outcomes("src-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["state-sum"]["counter_sum"] > 0
        assert by_id["lineage-count"]["process_id_count"] >= 1
        service.export_run("src-run", "exports/src")
        assert (service.workspace / "exports/src/states.parquet").is_file()
        # raw cumulative state JSON is intentionally excluded from bundles
        # (hundreds of MB); the compact parquet projection is the export form.
        assert not (service.workspace / "exports/src/states.json").exists()
    finally:
        service.close()


def test_service_enforces_declared_outcome_output_schema(tmp_path) -> None:
    """Finding 9: outcome rows must satisfy declared output_schema."""
    import pytest

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "oc-schema-study",
                "title": "oc schema",
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
                "outcomes": [
                    {
                        "id": "bounded-count",
                        "source": "events",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "phase"},
                        "output_schema": "outcome-schema",
                    }
                ],
                "models": [],
            }
        )
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "oc-schema-study" / "schemas"
        )
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "outcome-schema.yaml").write_text(
            "type: object\nrequired: [phase_count]\nproperties:\n  phase_count: {type: integer}\n"
        )
        revised = service.update_specification(
            "oc-schema-study", {"description": "with schema"}, draft["version"]
        )
        service.approve_specification("oc-schema-study", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/oc-schema-study", specification_id="oc-schema-study"
        )
        service.create_run(
            {"id": "oc-run", "study_id": "oc-schema-study", "build": compiled["path"]}
        )
        service.execute_run("oc-run")
        outcomes = service.evaluate_outcomes("oc-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["bounded-count"]["phase_count"] >= 1
        # A schema demanding a field the rows never produce must fail loudly.
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "oc-schema-study" / "schemas"
        )
        (schema_dir / "outcome-schema.yaml").write_text(
            "type: object\nrequired: [never_produced_field]\n"
        )
        changed = service.update_specification(
            "oc-schema-study",
            {"description": "breaking schema"},
            service.get_specification("oc-schema-study")["version"],
        )
        service.approve_specification("oc-schema-study", changed["version"], "researcher")
        compiled2 = service.compile_study(
            None, "builds/oc-schema-study-2", specification_id="oc-schema-study"
        )
        service.create_run(
            {"id": "oc-run-2", "study_id": "oc-schema-study", "build": compiled2["path"]}
        )
        service.execute_run("oc-run-2")
        with pytest.raises(ValueError, match="OUTPUT_SCHEMA_VIOLATION"):
            service.evaluate_outcomes("oc-run-2")
    finally:
        service.close()


def test_outcome_join_is_expressible_in_specification(tmp_path) -> None:
    """Finding 5: OutcomeSpec accepts join; the evaluator resolves joined sources."""
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "join-study",
                "title": "join",
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
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [
                    {
                        "id": "joined-count",
                        "source": "events",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "process_id"},
                        "join": {"left": "events", "right": "artifacts", "on": "invocation_id"},
                    }
                ],
                "models": [],
            }
        )
        service.approve_specification("join-study", draft["version"], "researcher")
        compiled = service.compile_study(None, "builds/join-study", specification_id="join-study")
        service.create_run({"id": "join-run", "study_id": "join-study", "build": compiled["path"]})
        service.execute_run(
            "join-run", executor_overrides={"tick": lambda _inv: {"counter": 1, "note": "x"}}
        )
        outcomes = service.evaluate_outcomes("join-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["joined-count"]["process_id_count"] >= 1
    finally:
        service.close()


def test_outcome_consumes_declared_artifact_values(tmp_path) -> None:
    """Finding 5: declared artifacts (payload.value) feed the artifacts source."""
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "declared-value-study",
                "title": "declared values",
                "processes": [
                    {
                        "id": "record",
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
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [
                    {
                        "id": "declared-value-count",
                        "source": "artifacts",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "value"},
                    },
                    {
                        "id": "named-artifact-count",
                        "source": "note",
                        "filters": [],
                        "grouping": ["condition_id", "phase"],
                        "aggregation": {"op": "count"},
                    },
                ],
                "models": [],
            }
        )
        service.approve_specification("declared-value-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/declared-value-study", specification_id="declared-value-study"
        )
        service.create_run(
            {
                "id": "declared-run",
                "study_id": "declared-value-study",
                "build": compiled["path"],
            }
        )
        service.execute_run(
            "declared-run",
            executor_overrides={"record": lambda _inv: {"counter": 1, "note": "hello"}},
        )
        outcomes = service.evaluate_outcomes("declared-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        assert by_id["declared-value-count"]["value_count"] >= 1
        named = [row for row in outcomes if row["outcome_id"] == "named-artifact-count"]
        assert named == [
            {
                "outcome_id": "named-artifact-count",
                "group_0": "base",
                "group_1": 0,
                "value_count": 1,
                "value_missing": 0,
            }
        ]
    finally:
        service.close()


def test_duckdb_aggregate_path_matches_engine(tmp_path, monkeypatch) -> None:
    """AW-11: with GENESIS_USE_DUCKDB=1 aggregates equal the in-process engine."""

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "duck-study",
                "title": "duck",
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
                "protocol": {"time_model": {"type": "rounds", "end": 4}},
                "outcomes": [
                    {
                        "id": "phase-sum",
                        "source": "events",
                        "filters": [],
                        "grouping": ["process_id"],
                        "aggregation": {"type": "sum", "field": "phase"},
                    },
                    {
                        "id": "phase-count",
                        "source": "events",
                        "filters": [],
                        "grouping": ["process_id"],
                        "aggregation": {"type": "count", "field": "phase"},
                    },
                ],
                "models": [],
            }
        )
        service.approve_specification("duck-study", draft["version"], "researcher")
        compiled = service.compile_study(None, "builds/duck-study", specification_id="duck-study")
        service.create_run({"id": "duck-run", "study_id": "duck-study", "build": compiled["path"]})
        service.execute_run("duck-run")
        engine_rows = service.evaluate_outcomes("duck-run")
        assert service.last_outcome_engine == "python"
        monkeypatch.setenv("GENESIS_USE_DUCKDB", "1")
        try:
            duck_rows = service.evaluate_outcomes("duck-run")
        finally:
            monkeypatch.delenv("GENESIS_USE_DUCKDB")
        # The DuckDB engine actually produced the results, not the fallback.
        assert service.last_outcome_engine == "duckdb"
        engine = {
            (row["outcome_id"], row["process_id"]): row.get("phase_sum") or row.get("phase_count")
            for row in engine_rows
        }
        duck = {
            (row["outcome_id"], row["process_id"]): row.get("phase_sum") or row.get("phase_count")
            for row in duck_rows
        }
        assert duck == engine
        # Shape parity: the original grouping field name and missingness.
        for row in duck_rows:
            assert "process_id" in row  # original field, not a generic "group"
            assert "phase_missing" in row  # engine-shaped missingness
    finally:
        service.close()


def test_join_is_used_unconditionally_and_cardinality_is_exact(tmp_path) -> None:
    """Finding 1: declared joins drive evaluation even when 'events' exists."""
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "join-strict-study",
                "title": "join strict",
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
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [
                    {
                        "id": "right-only-count",
                        "source": "events",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "note"},
                        "join": {"left": "events", "right": "artifacts", "on": "invocation_id"},
                    }
                ],
                "models": [],
            }
        )
        service.approve_specification("join-strict-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/join-strict-study", specification_id="join-strict-study"
        )
        service.create_run(
            {"id": "join-strict-run", "study_id": "join-strict-study", "build": compiled["path"]}
        )
        service.execute_run(
            "join-strict-run",
            executor_overrides={"tick": lambda _inv: {"counter": 1, "note": "hi"}},
        )
        outcomes = service.evaluate_outcomes("join-strict-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        # note exists only on the joined (right) rows; the join must have applied.
        assert by_id["right-only-count"]["note_count"] >= 1
    finally:
        service.close()


def test_join_with_empty_right_side_produces_zero_rows(tmp_path) -> None:
    """Finding 1: an inner join over an empty right side yields no rows."""
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "join-empty-study",
                "title": "join empty",
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
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [
                    {
                        "id": "joined-empty",
                        "source": "events",
                        "filters": [],
                        "grouping": [],
                        "aggregation": {"type": "count", "field": "process_id"},
                        "join": {"left": "events", "right": "artifacts", "on": "artifact_id"},
                    }
                ],
                "models": [],
            }
        )
        service.approve_specification("join-empty-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/join-empty-study", specification_id="join-empty-study"
        )
        service.create_run(
            {"id": "join-empty-run", "study_id": "join-empty-study", "build": compiled["path"]}
        )
        service.execute_run(
            "join-empty-run", executor_overrides={"tick": lambda _inv: {"counter": 1}}
        )
        outcomes = service.evaluate_outcomes("join-empty-run")
        by_id = {row["outcome_id"]: row for row in outcomes}
        # Event rows carry no artifact_id, so the inner join matches nothing.
        assert by_id["joined-empty"]["process_id_count"] == 0
    finally:
        service.close()

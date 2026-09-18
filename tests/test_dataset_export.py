"""Declared datasets are exported as the study's analysis tables (2026-09-14).

They were built only to feed outcomes and never written, so a researcher had to
rebuild them from the ledger and artifact payloads; their rows did not say which
cell they came from; a per-round state table lost its round; and a package's
datasets were dropped whenever its form was rebuilt.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from genesis.service import GenesisService

STUDY = {
    "id": "tables",
    "title": "tables",
    "models": [],
    "processes": [
        {
            "id": "tick",
            "executor": {"mode": "deterministic"},
            "context_policy": "public",
            "trigger": {"phase": 1, "repeat": True},
        }
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {"states": [{"id": "counter", "initial": 1, "value_type": "integer"}]},
    "protocol": {
        "time_model": {"type": "rounds", "start": 1, "end": 3},
        "factors": [{"id": "policy", "levels": ["strict", "lenient"]}],
    },
    "datasets": [{"id": "rounds", "source": {"kind": "state", "snapshot": "each_completed_round"}}],
    "outcomes": [
        {"id": "n", "source": "rounds", "aggregation": {"op": "count", "field": "counter"}}
    ],
}


def test_a_run_bundle_carries_each_dataset_with_its_cell(tmp_path: Path) -> None:
    service = GenesisService(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        service.approve_specification("tables", draft["version"], "researcher")
        build = service.compile_study(None, "builds/tables", specification_id="tables")["path"]
        service.create_run(
            {
                "id": "run",
                "study_id": "tables",
                "build": build,
                "condition_id": "policy-strict",
                "condition": {"id": "policy-strict", "factors": {"policy": "strict"}},
                "replication": 2,
            }
        )
        service.execute_run("run", executor_overrides={"tick": lambda _inv: {}})
        service.export_run("run", "exports/run")
        root = service.workspace / "exports" / "run" / "datasets"
        rows = list(csv.DictReader((root / "rounds.csv").open()))
        assert (root / "rounds.parquet").is_file()
        assert [row["phase"] for row in rows] == ["1", "2", "3"]
        assert {row["condition_id"] for row in rows} == {"policy-strict"}
        assert {row["factor_policy"] for row in rows} == {"strict"}
        assert {row["replication"] for row in rows} == {"2"}
        assert {row["counter"] for row in rows} == {"1"}
        dictionary = json.loads((root / "dictionary.json").read_text())
        assert dictionary["rounds"]["rows"] == 3
        assert dictionary["rounds"]["columns"]["factor_policy"]["origin"].startswith("engine")
        integrity = json.loads(
            (service.workspace / "exports" / "run" / "integrity.json").read_text()
        )
        assert "datasets/rounds.csv" in integrity
    finally:
        service.close()


def test_a_package_s_datasets_survive_its_form_being_rebuilt(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    service = GenesisService(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        directory = service._specification_dir("tables")
        form = GenesisService._form_from_package(
            StudyCompiler(directory)._load(), directory, "tables"
        )
        assert form["datasets"][0]["id"] == "rounds"
        assert draft["version"] == 1
    finally:
        service.close()


def test_an_experiment_export_stacks_every_cell(tmp_path: Path) -> None:
    """Each run bundle held one cell; a cross-cell comparison began by hand-stacking."""
    import pytest

    service = GenesisService(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        service.approve_specification("tables", draft["version"], "researcher")
        build = service.compile_study(None, "builds/tables", specification_id="tables")["path"]
        service.create_run({"id": "exp", "study_id": "tables", "build": build})
        result = service.execute_protocol(
            "exp", replications=2, executor_overrides={"tick": lambda _inv: {}}
        )
        assert result["status"] == "completed", result
        service.export_experiment("exp", "exports/exp")
        root = service.workspace / "exports" / "exp"
        rows = list(csv.DictReader((root / "datasets" / "rounds.csv").open()))
        cells = {(row["factor_policy"], row["replication"]) for row in rows}
        assert cells == {("strict", "1"), ("strict", "2"), ("lenient", "1"), ("lenient", "2")}
        assert len(rows) == 4 * 3
        assert {row["experiment_id"] for row in rows} == {"exp"}
        experiment = json.loads((root / "experiment.json").read_text())
        assert len(experiment["runs"]) == 4 and all(run["complete"] for run in experiment["runs"])
        outcomes = json.loads((root / "outcomes.json").read_text())
        assert {row["factor_policy"] for row in outcomes} == {"strict", "lenient"}
        integrity = json.loads((root / "integrity.json").read_text())
        assert "datasets/rounds.parquet" in integrity
        with pytest.raises(ValueError, match="EXPORT_DESTINATION"):
            service.export_experiment("exp", "exports/exp")
        with pytest.raises(ValueError, match="EXPORT_EXPERIMENT_EMPTY"):
            service.export_experiment("nothing", "exports/none")
    finally:
        service.close()


def test_a_revised_outcome_plan_applies_only_to_identical_execution(tmp_path: Path) -> None:
    """What a study measures may change after its runs; what it ran may not."""
    import pytest

    service = GenesisService(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        service.approve_specification("tables", draft["version"], "researcher")
        build = service.compile_study(None, "builds/t1", specification_id="tables")["path"]
        service.create_run({"id": "exp", "study_id": "tables", "build": build})
        service.execute_protocol("exp", executor_overrides={"tick": lambda _inv: {}})

        revised = {
            **STUDY,
            "datasets": [
                *STUDY["datasets"],
                {"id": "final", "source": {"kind": "state", "snapshot": "final"}},
            ],
        }
        current = service.get_specification("tables")
        updated = service.update_specification("tables", revised, current["version"])
        service.approve_specification("tables", updated["version"], "researcher")
        service.compile_study(None, "builds/t2", specification_id="tables")
        service.export_experiment("exp", "exports/revised", analysis_build="builds/t2")
        root = service.workspace / "exports" / "revised"
        assert (root / "datasets" / "final.csv").is_file()
        assert json.loads((root / "experiment.json").read_text())["analysis_build"].endswith("t2")

        longer = {
            **revised,
            "protocol": {
                **STUDY["protocol"],
                "time_model": {"type": "rounds", "start": 1, "end": 4},
            },
        }
        current = service.get_specification("tables")
        updated = service.update_specification("tables", longer, current["version"])
        service.approve_specification("tables", updated["version"], "researcher")
        service.compile_study(None, "builds/t3", specification_id="tables")
        with pytest.raises(ValueError, match="ANALYSIS_BUILD_MISMATCH.*protocol.json"):
            service.export_experiment("exp", "exports/longer", analysis_build="builds/t3")
    finally:
        service.close()


def test_table_rows_run_in_run_and_round_order() -> None:
    """Storage order put round 10 before round 2."""
    from genesis.service import _table_order

    rows = [
        {"run_id": "b", "phase": 1},
        {"run_id": "a", "phase": 10},
        {"run_id": "a", "phase": 2},
        {"run_id": "a"},
    ]
    assert [(r["run_id"], r.get("phase")) for r in sorted(rows, key=_table_order)] == [
        ("a", 2),
        ("a", 10),
        ("a", None),
        ("b", 1),
    ]


def test_a_protocol_passes_its_concurrency_to_every_run(tmp_path: Path) -> None:
    """max_concurrency was accepted by execute_run but not by execute_protocol,
    so a protocol always ran each cell's model calls one at a time."""
    service = GenesisService(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        service.approve_specification("tables", draft["version"], "researcher")
        build = service.compile_study(None, "builds/tables", specification_id="tables")["path"]
        service.create_run({"id": "exp", "study_id": "tables", "build": build})
        seen: list[object] = []
        original = service.execute_run

        def execute_run(run_id: str, **kwargs: object) -> dict:
            seen.append(kwargs.get("max_concurrency"))
            return original(run_id, **kwargs)  # type: ignore[arg-type]

        service.execute_run = execute_run  # type: ignore[method-assign]
        service.execute_protocol(
            "exp", executor_overrides={"tick": lambda _inv: {}}, max_concurrency=16
        )
        assert seen == [16, 16]
    finally:
        service.close()


def test_a_dataset_may_declare_a_column_named_value(tmp_path: Path) -> None:
    """Artifact rows drop their nested `value`, but not a column of that name.

    The strip was unconditional, so a dataset whose own fields step created a
    `value` column evaluated fine for outcomes while the exported table and the
    data dictionary lost it (2026-09-14 M6).
    """
    import json

    from genesis.service import GenesisService

    build = tmp_path / "build"
    build.mkdir()
    (build / "outcome_plan.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "d",
                        "source": {"kind": "artifacts", "artifact_type": "detection-result"},
                        "fields": [
                            {"name": "value", "op": "copy", "field": "detected"},
                        ],
                    }
                ],
                "outcomes": [],
            }
        )
    )
    rows = {"d": [{"artifact_id": "a1", "value": 1, "score": 3}]}
    tables, dictionary = GenesisService._dataset_tables(rows, build)
    assert "value" in tables["d"][0], "the declared column survives the artifact strip"
    assert tables["d"][0]["value"] == 1
    assert "value" in dictionary["d"]["columns"]

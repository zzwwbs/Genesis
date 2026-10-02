"""Regressions from the 2026-09-28 review.

Each test fails against the behaviour the review reproduced: a caller-supplied
dataset digest reading a file outside the store, a missing score counted as
false, a symlinked package asset reported as an internal fault, and the rest
of the confirmed defects.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.analysis import AnalysisEngine, OutcomePlan
from genesis.app import create_app
from genesis.elicitation import SpecificationPatch
from genesis.execution_manifest import build_package_closure
from genesis.information_timing import batch_dependencies, timing_diagnostics
from genesis.outcome_plan import _apply_fields, materialize_datasets
from genesis.persistence import PersistenceCoordinator
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
)
from genesis.service import GenesisService


def test_create_run_cannot_read_a_json_file_or_forge_outcomes(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text(json.dumps([{"api_key": "sk-OUTSIDE"}]))
    dictionary = outside / "dictionary.json"
    dictionary.write_text(json.dumps({"t": {"columns": {}}}))
    client = TestClient(create_app(tmp_path / "ws"))

    created = client.post(
        "/runs",
        json={
            "id": "evil",
            "imported_datasets": {
                "tables": {"t": "ab" + str(secret)},
                "dictionary": "ab" + str(dictionary),
            },
        },
    )
    assert created.status_code == 422
    assert created.json()["error"]["code"] == "RUN_CREATE"
    assert client.get("/runs/evil/datasets/t").status_code == 404

    forged = client.post("/runs", json={"id": "evil2", "imported_outcomes": [{"forged": True}]})
    assert forged.status_code == 422
    assert client.get("/runs/evil2/outcomes").status_code == 404


def test_root_files_do_not_enter_the_closure_or_its_digest(tmp_path: Path) -> None:
    def package(spec_id: str, approver: str) -> Path:
        root = tmp_path / spec_id
        (root / "prompts").mkdir(parents=True)
        (root / "study.yaml").write_text("study_id: same\n")
        (root / "prompts" / "api-key.txt").write_text("SECRET-PROMPT\n")
        (root / "notes.txt").write_text("scratch\n")
        (root / "service-account.json").write_text('{"private_key": "PEM"}\n')
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "id": spec_id,
                    "approved_by": approver,
                    "form": {"prompts": {"api-key": "SECRET-PROMPT"}},
                }
            )
        )
        return root

    first = build_package_closure(package("study-a", "ada"))
    second = build_package_closure(package("study-b", "bea"))
    paths = {asset["path"] for asset in first.manifest["assets"]}
    assert paths == {"study.yaml"}
    assert dict(first.excluded)["prompts/api-key.txt"] == "credential"
    assert "notes.txt" not in paths and "service-account.json" not in paths
    assert "metadata.json" not in paths
    assert first.digest == second.digest


def test_a_missing_score_stays_missing_in_the_proportion(tmp_path: Path) -> None:
    rows = [
        _apply_fields(
            {"score": score},
            [
                {
                    "name": "flagged",
                    "op": "comparison",
                    "field": "score",
                    "operator": "gte",
                    "value": 7,
                }
            ],
        )
        for score in (9, 8, None, None)
    ]
    assert [row["flagged"] for row in rows] == [True, True, None, None]
    outcome = AnalysisEngine().evaluate(
        OutcomePlan(
            id="flagged-by-round",
            source="detections",
            select="flagged",
            aggregation="mean",
            missingness="exclude",
        ),
        {"detections": rows},
    )[0]
    assert outcome["flagged_mean"] == 1
    assert outcome["flagged_missing"] == 2
    total = _apply_fields(
        {"score": None},
        [{"name": "total", "op": "arithmetic", "field": "score", "operator": "add", "value": 1}],
    )
    assert total["total"] is None
    assert "NaN" not in json.dumps(total)


def test_equality_on_a_missing_score_stays_missing() -> None:
    rows = [
        _apply_fields(
            {"score": score},
            [
                {
                    "name": "flagged",
                    "op": "comparison",
                    "field": "score",
                    "operator": "eq",
                    "value": 7,
                }
            ],
        )
        for score in (7, 9, None, None)
    ]
    assert [row["flagged"] for row in rows] == [True, False, None, None]
    outcome = AnalysisEngine().evaluate(
        OutcomePlan(
            id="flagged",
            source="detections",
            select="flagged",
            aggregation="mean",
            missingness="exclude",
        ),
        {"detections": rows},
    )[0]
    assert outcome["flagged_mean"] == 0.5
    assert outcome["flagged_missing"] == 2


def test_a_package_id_of_251_characters_is_a_client_error(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "ws"))
    too_long = client.post("/packages", json={"id": "a" * 251})
    assert too_long.status_code == 422
    assert too_long.json()["error"]["code"] == "PACKAGE_ID_INVALID"
    accepted = client.post("/packages", json={"id": "a" * 240})
    assert accepted.status_code == 201


def test_a_deferred_feedback_consumer_is_not_work_left() -> None:
    ran: list[str] = []

    class Executor:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            ran.append(invocation.process_id)
            return ProcessResult(outputs={})

    process = {
        "id": "p",
        "context_policy": "private",
        "trigger": {"phase": 0},
        "theory_feedback": [
            {
                "source": "counter",
                "context_slot": "last",
                "lag_rounds": 1,
                "initial": {"policy": "skip_consumer"},
            }
        ],
    }

    def controller(**kwargs):  # type: ignore[no-untyped-def]
        return RunController(
            Scheduler([process]),
            ExecutorRegistry({"p": Executor()}),
            ContextEngine({"private": {"allow": []}}),
            **kwargs,
        )

    stopped = controller(status_provider=lambda: "lease_lost")
    stopped.run("r", phase_limit=1, seed=1, terminal_phase=0)
    assert stopped.lease_lost
    assert stopped.stopped_with_work_remaining is False
    assert ran == []

    finished = controller()
    finished.run("r", phase_limit=1, seed=1, terminal_phase=0)
    assert finished.status == "completed"
    assert ran == []


def test_trace_export_stops_reconstructing_at_the_limit() -> None:
    import importlib.util

    path = Path("tools/export_traces.py")
    spec = importlib.util.spec_from_file_location("export_traces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pulled = {"n": 0}

    def events():
        for index in range(20):
            pulled["n"] += 1
            yield {"process_id": "p", "phase": index, "actors": [], "input_refs": []}

    matched, _needed = module.take_matching_events(
        events(), process="p", phase=None, prefix=None, limit=3, values="NONE"
    )
    assert len(matched) == 3
    assert pulled["n"] == 3


def test_a_symlinked_package_asset_is_a_client_error(tmp_path: Path) -> None:
    source = Path("tests/golden_studies/community_formation")
    workspace = tmp_path / "ws"
    study = workspace / "study"
    shutil.copytree(source, study)
    real = workspace / "interpret.txt"
    real.write_text((study / "prompts" / "interpret.txt").read_text())
    link = study / "prompts" / "interpret.txt"
    link.unlink()
    link.symlink_to(real)
    client = TestClient(create_app(workspace))
    response = client.post("/compile", json={"source": "study", "output": "builds/linked"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "PACKAGE_SYMLINK"


def test_opening_a_workspace_does_not_take_a_write_lock(tmp_path: Path) -> None:
    database = tmp_path / "genesis.db"
    objects = tmp_path / "objects"
    holder = PersistenceCoordinator(database, objects)
    holder.connection.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    reader = PersistenceCoordinator(database, objects)
    elapsed = time.monotonic() - started
    reader.close()
    holder.connection.execute("ROLLBACK")
    holder.close()
    assert elapsed < 1


def test_a_read_only_open_does_not_decode_the_path(tmp_path: Path) -> None:
    spaced = tmp_path / "x y"
    percent = tmp_path / "x%20y"
    for directory, marker in ((spaced, "SPACE"), (percent, "PERCENT")):
        directory.mkdir()
        database = directory / "genesis.db"
        store = PersistenceCoordinator(database, directory / "objects")
        store.connection.execute("CREATE TABLE marker(v TEXT)")
        store.connection.execute("INSERT INTO marker VALUES (?)", (marker,))
        store.close()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{database}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
    opened = PersistenceCoordinator(percent / "genesis.db", percent / "objects", read_only=True)
    try:
        assert opened.connection.execute("SELECT v FROM marker").fetchone()[0] == "PERCENT"
    finally:
        opened.close()


def test_a_read_only_close_leaves_the_wal_files(tmp_path: Path) -> None:
    """Deleting the sidecars a read-only open created dropped another process's commit."""
    database = tmp_path / "genesis.db"
    objects = tmp_path / "objects"
    writer = PersistenceCoordinator(database, objects)
    writer.connection.execute("CREATE TABLE t(v INTEGER)")
    writer.connection.execute("INSERT INTO t VALUES (1)")
    writer.close()
    for suffix in ("-wal", "-shm"):
        Path(f"{database}{suffix}").unlink(missing_ok=True)
    reader = PersistenceCoordinator(database, objects, read_only=True)
    reader.close()
    assert Path(f"{database}-wal").exists()
    assert Path(f"{database}-shm").exists()


def test_a_concurrent_pause_resumes_in_actor_order() -> None:
    fail = {"w1": True}

    class Executor:
        concurrent_safe = True

        def execute(self, invocation):  # type: ignore[no-untyped-def]
            actor = invocation.actor_ids[0]
            if fail.get(actor):
                fail[actor] = False
                raise ValueError("PROVIDER_HTTP: provider returned HTTP 402: no credit")
            return ProcessResult(outputs={})

    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": ["w1", "w2", "w3", "w4"],
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                    "information_timing": {"mode": "simultaneous", "order": "listed"},
                }
            ]
        ),
        ExecutorRegistry({"p": Executor()}),
        ContextEngine({"private": {"allow": []}}),
        max_concurrency={"p": 4},
    )
    controller.run("r", phase_limit=2, seed=1)
    assert controller.status == "paused"
    assert [group[0] for group in controller._actor_queues[("p", 1)]] == ["w1", "w2", "w3", "w4"]


def test_a_stop_ignores_a_terminal_skip_and_a_process_with_nobody_to_run() -> None:
    class Executor:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            raise AssertionError("should not run")

    def stopped(process: dict) -> bool:
        controller = RunController(
            Scheduler([process]),
            ExecutorRegistry({"p": Executor()}),
            ContextEngine({"private": {"allow": []}}),
            status_provider=lambda: "lease_lost",
        )
        controller.run("r", phase_limit=1, seed=1, terminal_phase=0)
        return controller.stopped_with_work_remaining

    assert (
        stopped(
            {
                "id": "p",
                "actors": ["w1"],
                "context_policy": "private",
                "trigger": {"phase": 0},
                "terminal_skip": True,
            }
        )
        is False
    )
    assert (
        stopped(
            {
                "id": "p",
                "actors": {"ids": [], "fan_out": True},
                "context_policy": "private",
                "trigger": {"phase": 0},
            }
        )
        is False
    )


def test_an_inline_availability_rule_requires_information_timing() -> None:
    process = {
        "id": "p",
        "actors": ["a", "b"],
        "executor": {"mode": "deterministic"},
        "context_policy": "shared",
    }
    policy = {"allow": [], "available_when": {"path": "events.kind", "op": "eq", "value": "wrote"}}
    errors, _warnings = timing_diagnostics([process], {"shared": policy})
    assert batch_dependencies(process, policy)
    assert any(error["code"] == "INFORMATION_TIMING_REQUIRED" for error in errors)


def test_a_records_own_value_stays_a_field() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {"id": "ratings", "source": {"kind": "artifacts", "artifact_type": "rating"}}
            ]
        },
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {
                        "declared_artifact_id": "rating",
                        "value": {"accuracy": 0.9, "value": 5},
                    },
                }
            ]
        },
    )["ratings"]
    assert rows[0]["accuracy"] == 0.9
    assert rows[0]["value"] == 5
    assert "rating" not in rows[0]


def test_a_ne_filter_does_not_keep_a_missing_value() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "scores",
                    "source": {"kind": "events"},
                    "where": [{"field": "score", "op": "ne", "value": 7}],
                }
            ]
        },
        {
            "events": [
                {"score": 7, "phase": 1},
                {"score": 9, "phase": 2},
                {"score": None, "phase": 3},
                {"phase": 4},
            ]
        },
    )["scores"]
    assert [row["phase"] for row in rows] == [2]


def test_a_structured_group_key_is_not_labeled_as_its_json() -> None:
    rows = AnalysisEngine().evaluate(
        OutcomePlan(id="g", source="rows", select="n", aggregation="count", group_by="k"),
        {"rows": [{"k": {"a": 2}, "n": 1}, {"k": '{"a": 2}', "n": 1}]},
    )
    assert [row["k"] for row in rows] == [{"a": 2}, '{"a": 2}']


def test_a_scalar_artifact_value_is_not_exported_twice() -> None:
    rows = materialize_datasets(
        {"datasets": [{"id": "scores", "source": {"kind": "artifacts", "artifact_type": "score"}}]},
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {"declared_artifact_id": "score", "value": 7},
                }
            ]
        },
    )["scores"]
    assert rows[0]["score"] == 7
    assert "value" not in rows[0]


def test_a_fields_step_does_not_resurrect_the_payload_copy() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "items",
                    "source": {"kind": "artifacts", "artifact_type": "item"},
                    "fields": [{"name": "score", "op": "literal", "value": 9}],
                }
            ]
        },
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {
                        "declared_artifact_id": "item",
                        "value": {"score": 1, "phase": 2},
                    },
                }
            ]
        },
    )["items"]
    assert rows[0]["score"] == 9
    assert rows[0]["phase"] == 2
    assert "value" not in rows[0]


def test_value_keeps_a_record_that_is_not_a_copy_of_the_row(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "outcome_plan.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"id": "items", "source": {"kind": "artifacts", "artifact_type": "item"}}
                ]
            }
        )
    )
    tables, _dictionary = GenesisService._dataset_tables(
        {"items": [{"phase": 2, "value": {"phase": 7}}, {"phase": 2, "value": {}}]},
        plan,
    )
    assert json.loads(tables["items"][0]["value"]) == {"phase": 7}
    assert json.loads(tables["items"][1]["value"]) == {}


def test_an_empty_evidence_target_does_not_cover_every_operation() -> None:
    patch = SpecificationPatch(
        stage_id="stage",
        base_specification_version=1,
        operations=(
            {"op": "replace", "path": "/theory/claim", "value": 1},
            {"op": "replace", "path": "/domain/actor", "value": 2},
        ),
        evidence=({"target": "", "source_turns": (1,)},),
    )
    try:
        patch.validate_evidence()
    except ValueError as exc:
        assert "ASSISTANT_EVIDENCE_INVALID" in str(exc)
    else:
        raise AssertionError("an empty evidence target was accepted")


def test_reexport_keeps_a_source_retention_label(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        service.persistence.create_run({"id": "r", "retention_policy": "purge_raw_responses"})
        assert service._export_retention(service.get_run("r"), []) == "purge_raw_responses"
    finally:
        service.close()


def test_dropping_response_format_is_recorded() -> None:
    import io
    import urllib.error

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    provider = OpenAICompatibleProvider(
        base_url="http://x", model="m", api_key_env="E", api_key="k"
    )
    calls = {"n": 0}

    def open_request(_request):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "http://x",
                400,
                "Bad Request",
                {},  # type: ignore[arg-type]
                io.BytesIO(
                    b'{"error":{"message":"unsupported_parameter: response_format '
                    b'is not supported with this model and the prompt was SECRET"}}'
                ),
            )
        return (
            json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode(),
            200,
        )

    provider._open_cancellable = open_request  # type: ignore[method-assign]
    response = provider.generate(
        ProviderRequest(
            model="m",
            prompt="SECRET",
            parameters={"response_format": {"type": "json_object"}},
        )
    )
    assert response.metadata.get("degraded_parameters") == ["response_format"]


def test_a_provider_error_keeps_its_code_and_not_the_prompt() -> None:
    import io
    import urllib.error

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    provider = OpenAICompatibleProvider(
        base_url="http://x", model="m", api_key_env="E", api_key="k"
    )

    def raising(_request):  # type: ignore[no-untyped-def]
        body = json.dumps(
            {
                "error": {
                    "code": "context_length_exceeded",
                    "message": "the prompt was SECRET and far too long",
                }
            }
        ).encode()
        raise urllib.error.HTTPError(
            "http://x",
            400,
            "Bad Request",
            {},
            io.BytesIO(body),  # type: ignore[arg-type]
        )

    provider._open_cancellable = raising  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="context_length_exceeded") as caught:
        provider.generate(ProviderRequest(model="m", prompt="SECRET"))
    assert "SECRET" not in str(caught.value)


def test_true_is_its_own_group_but_one_and_one_point_zero_are_one() -> None:
    """True is not a count of 1; 1 and 1.0 are one number, labelled as first seen."""
    rows = AnalysisEngine().evaluate(
        OutcomePlan(id="g", source="rows", select="n", aggregation="count", group_by="k"),
        {"rows": [{"k": True, "n": 1}, {"k": 1, "n": 1}, {"k": 1.0, "n": 1}]},
    )
    labelled = [(type(row["k"]).__name__, row["k"], row["n_count"]) for row in rows]
    assert labelled == [("bool", True, 1), ("int", 1, 2)]


def test_a_refused_import_deletes_only_the_build_it_created(tmp_path: Path) -> None:
    from genesis.evidence import ExportMode

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    service.create_specification(
        {
            "id": "exchange-study",
            "title": "exchange study",
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
            "protocol": {"time_model": {"type": "rounds", "end": 1}},
            "outcomes": [],
            "models": [],
        }
    )
    service.approve_specification("exchange-study", 1, "researcher")
    service.compile_study(None, "builds/exchange-study", specification_id="exchange-study")
    service.create_run(
        {"id": "xrun", "study_id": "exchange-study", "build": "builds/exchange-study"}
    )
    service.execute_run("xrun", executor_overrides={"tick": lambda _inv: {"counter": 1}})
    service.export_run("xrun", "exports/bundle", mode=ExportMode.REPRODUCIBILITY)
    service.close()

    other = tmp_path / "other"
    shutil.copytree(workspace / "exports" / "bundle", other / "imports" / "bundle")
    build_hash = json.loads((other / "imports" / "bundle" / "build_manifest.json").read_text())[
        "build_hash"
    ]
    importer = GenesisService(other)
    try:
        importer.import_run(other / "imports" / "bundle", run_id="run-a")
        kept = other / "builds" / f"run-b-imported-{build_hash[:8]}"
        kept.mkdir()
        (kept / "sentinel.txt").write_text("already here")
        with pytest.raises(ValueError):
            importer.import_run(other / "imports" / "bundle", run_id="run-b")
        assert (kept / "sentinel.txt").read_text() == "already here"
        # Nothing of the refused bundle was copied into the directory it found,
        # and the name the import took instead was cleaned up.
        assert [path.name for path in kept.iterdir()] == ["sentinel.txt"]
        assert not (other / "builds" / f"run-b-imported-{build_hash[:8]}-2").exists()
        with pytest.raises(ValueError, match="ALREADY_EXISTS"):
            importer.import_run(other / "imports" / "bundle", run_id="run-c")
        assert not (other / "builds" / f"run-c-imported-{build_hash[:8]}").exists()
    finally:
        importer.close()


def test_selecting_value_on_an_artifact_dataset_warns(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    source = Path("tests/golden_studies/community_formation")
    package = tmp_path / "pkg"
    shutil.copytree(source, package)
    (package / "outcomes.yaml").write_text(
        "schema_version: '1.0'\n"
        "study_id: community-formation\n"
        "datasets:\n"
        "- id: notes\n"
        "  source: {kind: artifacts, artifact_type: shared-artifact}\n"
        "  fields:\n"
        "  - {name: copied, op: copy, field: value}\n"
        "outcomes:\n"
        "- id: mean-value\n"
        "  source: notes\n"
        "  aggregation: {op: mean, field: value}\n"
        "  missingness: {policy: exclude}\n"
    )
    build = StudyCompiler(package).compile(tmp_path / "build")
    report = json.loads((build.path / "validation_report.json").read_text())
    codes = {item["code"] for item in report["warnings"]}
    assert "ARTIFACT_VALUE_DROPPED" in codes
    from types import SimpleNamespace

    from genesis.compiler import _advise_dropped_artifact_value

    advised = _advise_dropped_artifact_value(
        SimpleNamespace(
            datasets=[
                SimpleNamespace(
                    id="notes",
                    source=SimpleNamespace(kind="artifacts"),
                    fields=[],
                    explode=None,
                    where=[],
                    deduplicate_on=[],
                )
            ],
            outcomes=[
                SimpleNamespace(
                    id="kept",
                    source="notes",
                    aggregation={},
                    filters=[{"value": 7}],
                    grouping=["value"],
                )
            ],
        )
    )
    advised_paths = {item["path"] for item in advised}
    assert "outcomes.outcomes.kept.filters" in advised_paths
    assert "outcomes.outcomes.kept.grouping" in advised_paths
    plain = tmp_path / "plain"
    shutil.copytree(source, plain)
    plain_build = StudyCompiler(plain).compile(tmp_path / "plain-build")
    plain_report = json.loads((plain_build.path / "validation_report.json").read_text())
    assert "ARTIFACT_VALUE_DROPPED" not in {item["code"] for item in plain_report["warnings"]}


def test_a_records_nested_value_is_not_deleted_as_a_payload_copy() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {"id": "ratings", "source": {"kind": "artifacts", "artifact_type": "rating"}}
            ]
        },
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {
                        "declared_artifact_id": "rating",
                        "value": {"accuracy": 0.9, "value": {"accuracy": 0.9}},
                    },
                }
            ]
        },
    )["ratings"]
    assert rows[0]["accuracy"] == 0.9
    assert rows[0]["value"] == {"accuracy": 0.9}


def test_true_stays_distinct_from_one_outside_grouping() -> None:
    """True is not 1 anywhere a row is matched; 1 and 1.0 are one number."""
    from genesis.service import _hash_join

    deduped = materialize_datasets(
        {"datasets": [{"id": "events", "source": {"kind": "events"}, "deduplicate_on": ["k"]}]},
        {"events": [{"k": True, "n": "b"}, {"k": 1, "n": "i"}, {"k": 1.0, "n": "f"}]},
    )["events"]
    assert sorted(row["n"] for row in deduped) == ["b", "i"]

    joined = _hash_join([{"k": True, "side": "L"}], [{"k": 1, "side": "R"}, {"k": 1.0}], "k")
    assert joined == []

    counted = AnalysisEngine().evaluate(
        OutcomePlan(
            id="g",
            source="rows",
            select="n",
            aggregation="count",
            filters=({"k": 1},),
        ),
        {"rows": [{"k": True, "n": 1}, {"k": 1, "n": 1}, {"k": 1.0, "n": 1}]},
    )
    assert counted[0]["n_count"] == 2


def test_an_all_slash_evidence_target_does_not_cover_every_operation() -> None:
    patch = SpecificationPatch(
        stage_id="stage",
        base_specification_version=1,
        operations=(
            {"op": "replace", "path": "/theory/claim", "value": 1},
            {"op": "replace", "path": "/domain/actor", "value": 2},
        ),
        evidence=({"target": "//", "source_turns": (1,)},),
    )
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        patch.validate_evidence()


def test_a_named_state_field_is_the_row_for_every_snapshot() -> None:
    per_round = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "rounds",
                    "source": {
                        "kind": "state",
                        "state": "counter",
                        "snapshot": "each_completed_round",
                    },
                }
            ]
        },
        {"state": [{"counter": 1, "other": "leak", "_round": 0}]},
    )["rounds"]
    assert per_round == [{"phase": 0, "counter": 1}]

    final = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "last",
                    "source": {"kind": "state", "state": "score", "snapshot": "final"},
                }
            ]
        },
        {
            "state": [
                {"state_version": 1, "score": 5, "other": 1},
                {"state_version": 2, "other": 9},
            ]
        },
    )["last"]
    assert final == [{"score": None}]


def test_a_dotted_explode_keeps_the_containers_other_fields() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {"id": "people", "source": {"kind": "events"}, "explode": "state_delta.rows"}
            ]
        },
        {"events": [{"phase": 1, "state_delta": {"user": "u", "rows": [{"id": "a"}]}}]},
    )["people"]
    assert rows == [{"phase": 1, "user": "u", "id": "a"}]


def test_a_group_field_named_group_n_keeps_its_value() -> None:
    rows = AnalysisEngine().evaluate(
        OutcomePlan(
            id="g",
            source="rows",
            select="n",
            aggregation="count",
            group_by=("phase", "group_0"),
        ),
        {"rows": [{"phase": "p", "group_0": "g", "n": 1}]},
    )
    assert rows[0]["group_0"] == "g"
    assert rows[0]["phase"] == "p"
    assert rows[0]["group_1"] == "g"


def test_embedded_prompts_and_schemas_come_from_the_closure(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    source = Path("tests/golden_studies/community_formation")
    package = tmp_path / "pkg"
    shutil.copytree(source, package)
    prompt = (package / "prompts" / "interpret.txt").read_text()
    schema = (package / "schemas" / "interpretation-schema.yaml").read_text()
    prompt_reads = {"n": 0}
    schema_reads = {"n": 0}
    real = Path.read_text

    def wrapped(self: Path, *args: object, **kwargs: object) -> str:
        text = real(self, *args, **kwargs)
        if self.name == "interpret.txt":
            prompt_reads["n"] += 1
            if prompt_reads["n"] > 1:
                return text + "\nCHANGED"
        if self.name == "interpretation-schema.yaml":
            schema_reads["n"] += 1
            if schema_reads["n"] > 2:
                return "true\n"
        return text

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "read_text", wrapped)
    try:
        build = StudyCompiler(package).compile(tmp_path / "build")
    finally:
        monkeypatch.undo()
    templates = json.loads((build.path / "prompt_templates.json").read_text())
    schemas = json.loads((build.path / "schemas.json").read_text())
    assert templates["interpret"] == prompt
    assert "CHANGED" not in templates["interpret"]
    assert schemas["interpretation-schema"] is not True
    assert "type" in schemas["interpretation-schema"]
    assert schema.startswith("type:") or "type" in schema


def test_a_scalar_whose_artifact_id_is_value_stays() -> None:
    rows = materialize_datasets(
        {"datasets": [{"id": "scores", "source": {"kind": "artifacts", "artifact_type": "value"}}]},
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {"declared_artifact_id": "value", "value": 7},
                }
            ]
        },
    )["scores"]
    assert rows[0]["value"] == 7


def test_a_dotted_read_of_the_dropped_value_column_warns() -> None:
    from types import SimpleNamespace

    from genesis.compiler import _advise_dropped_artifact_value

    warnings = _advise_dropped_artifact_value(
        SimpleNamespace(
            datasets=[
                SimpleNamespace(
                    id="notes",
                    source=SimpleNamespace(kind="artifacts"),
                    fields=[
                        SimpleNamespace(
                            name="user",
                            field="value.user",
                            with_field=None,
                            condition=None,
                        )
                    ],
                    explode="value.rows",
                    where=[],
                    deduplicate_on=[],
                )
            ],
            outcomes=[],
        )
    )
    paths = {item["path"] for item in warnings}
    assert "outcomes.datasets.notes.fields.user" in paths
    assert "outcomes.datasets.notes.explode" in paths


def test_explode_keeps_a_sibling_object_and_the_inner_name() -> None:
    same = [{"id": "a"}]
    rows = materialize_datasets(
        {
            "datasets": [
                {"id": "people", "source": {"kind": "events"}, "explode": "state_delta.rows"}
            ]
        },
        {
            "events": [
                {
                    "phase": 1,
                    "user": "outer",
                    "state_delta": {"user": "inner", "rows": same},
                    "summary": {"rows": same, "total": 1},
                }
            ]
        },
    )["people"]
    assert rows[0]["user"] == "inner"
    assert rows[0]["summary"]["total"] == 1
    assert rows[0]["id"] == "a"

    nested = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "deeper",
                    "source": {"kind": "events"},
                    "explode": "state_delta.payload.rows",
                }
            ]
        },
        {
            "events": [
                {
                    "state_delta": {
                        "extra": "outer",
                        "payload": {"extra": "inner", "rows": [{"id": "a"}]},
                    }
                }
            ]
        },
    )["deeper"]
    assert nested[0]["extra"] == "outer"
    assert nested[0]["payload"]["extra"] == "inner"
    assert nested[0]["id"] == "a"


def test_export_keeps_a_value_that_differs_only_as_a_bool(tmp_path: Path) -> None:
    """{"phase": True} is not a copy of phase 1. {"phase": 1.0} is: the same number."""
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "outcome_plan.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "id": "items",
                        "source": {"kind": "artifacts", "artifact_type": "item"},
                    }
                ]
            }
        )
    )
    tables, _dictionary = GenesisService._dataset_tables(
        {
            "items": [
                {"phase": 1, "value": {"phase": 1.0}},
                {"phase": 1, "value": {"phase": True}},
            ]
        },
        plan,
    )
    assert tables["items"][0]["value"] is None
    assert json.loads(tables["items"][1]["value"]) == {"phase": True}


def test_a_missing_state_field_named_phase_keeps_the_round() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "rounds",
                    "source": {
                        "kind": "state",
                        "state": "phase",
                        "snapshot": "each_completed_round",
                    },
                }
            ]
        },
        {"state": [{"other": "x", "_round": 3}]},
    )["rounds"]
    assert rows == [{"phase": 3}]


def test_the_artifact_relation_does_not_copy_a_record_onto_its_id() -> None:
    from genesis.service import _artifact_row

    record = _artifact_row(
        {
            "artifact_id": "a1",
            "payload": {
                "declared_artifact_id": "rating",
                "value": {"accuracy": 0.9, "value": {"accuracy": 0.9}},
            },
        },
        "run",
        {},
        {"rating"},
    )
    assert record is not None
    assert record["accuracy"] == 0.9
    assert record["value"] == {"accuracy": 0.9}
    assert "rating" not in record

    scalar = _artifact_row(
        {
            "artifact_id": "a2",
            "payload": {"declared_artifact_id": "score", "value": 7},
        },
        "run",
        {},
        {"score"},
    )
    assert scalar is not None
    assert scalar["score"] == 7
    assert scalar["value"] == 7


# --- 2026-10-01 review: a number's spelling, and a container's engine columns ---


def test_a_score_written_as_seven_point_zero_is_still_seven() -> None:
    """A model may return a schema-valid integer as 7.0; a == 7 filter dropped it."""
    rows = [{"rating": 7, "liked": 1}, {"rating": 7.0, "liked": 1}, {"rating": 3, "liked": 0}]
    filtered = AnalysisEngine().evaluate(
        OutcomePlan(
            id="f", source="rows", select="liked", aggregation="sum", filters=[{"rating": 7}]
        ),
        {"rows": rows},
    )
    assert filtered[0]["liked_sum"] == 2
    grouped = AnalysisEngine().evaluate(
        OutcomePlan(id="g", source="rows", select="liked", aggregation="count", group_by="rating"),
        {"rows": rows},
    )
    assert [(row["rating"], row["liked_count"]) for row in grouped] == [(7, 2), (3, 1)]


def test_deduplication_and_joins_treat_one_and_one_point_zero_as_one_key() -> None:
    from genesis.service import _hash_join

    deduplicated = materialize_datasets(
        {"datasets": [{"id": "d", "source": {"kind": "events"}, "deduplicate_on": ["k"]}]},
        {
            "events": [
                {"k": 1, "payload": {}},
                {"k": 1.0, "payload": {}},
                {"k": True, "payload": {}},
            ]
        },
    )["d"]
    assert [row["k"] for row in deduplicated] == [1, True]
    joined = _hash_join([{"phase": 2, "a": 1}], [{"phase": 2.0, "b": 2}], "phase")
    assert len(joined) == 1


def test_a_nested_explode_does_not_overwrite_the_round_with_a_state_field() -> None:
    from genesis.outcome_plan import _exploded

    row = {
        "run_id": "r",
        "phase": 3,
        "state_delta": {"phase": "draft", "note": "kept", "rows": [{"user": "u1"}]},
    }
    (exploded,) = _exploded([row], "state_delta.rows")
    assert exploded["phase"] == 3
    assert exploded["state_delta_phase"] == "draft"
    assert exploded["note"] == "kept"
    assert exploded["user"] == "u1"


def test_a_final_snapshot_keeps_the_round_when_the_named_field_is_missing() -> None:
    """A missing named field on ``final`` was written as null and hid the round."""
    for field in ("phase", "score"):
        rows = materialize_datasets(
            {
                "datasets": [
                    {
                        "id": "last",
                        "source": {"kind": "state", "state": field, "snapshot": "final"},
                    }
                ]
            },
            {"state": [{"state_version": 1, "_round": 4, "other": 1}]},
        )["last"]
        assert rows[0]["phase"] == 4
        assert "other" not in rows[0]
        if field != "phase":
            assert rows[0][field] is None


def test_exploding_value_still_walks_a_list_payload() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "scores",
                    "source": {"kind": "artifacts", "artifact_type": "score"},
                    "explode": "value",
                }
            ]
        },
        {
            "artifacts": [
                {
                    "artifact_id": "a1",
                    "payload": {"declared_artifact_id": "score", "value": ["x", "y"]},
                }
            ]
        },
    )["scores"]
    assert [row["value"] for row in rows] == ["x", "y"]


def test_a_renamed_container_field_does_not_drop_an_existing_column() -> None:
    from genesis.outcome_plan import _exploded

    outer = _exploded(
        [
            {
                "phase": 3,
                "state_delta_phase": "outer",
                "state_delta": {"phase": "draft", "rows": [{"id": "r1"}]},
            }
        ],
        "state_delta.rows",
    )[0]
    assert outer["phase"] == 3
    assert outer["state_delta_phase"] == "outer"
    assert outer["from_state_delta_phase"] == "draft"

    factor = _exploded(
        [
            {
                "phase": 3,
                "factor_phase": "treatment",
                "factor": {"phase": "draft", "rows": [{"id": "r1"}]},
            }
        ],
        "factor.rows",
    )[0]
    assert factor["phase"] == 3
    assert factor["factor_phase"] == "treatment"
    assert factor["from_factor_phase"] == "draft"


def test_a_dataset_filter_keeps_true_apart_from_one_and_seven_with_seven_point_zero() -> None:
    rows = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "scores",
                    "source": {"kind": "events"},
                    "where": [{"field": "rating", "op": "eq", "value": 1}],
                }
            ]
        },
        {"events": [{"rating": True}, {"rating": 1}, {"rating": 7.0}, {"rating": 7}]},
    )["scores"]
    assert [row["rating"] for row in rows] == [1]
    matched = materialize_datasets(
        {
            "datasets": [
                {
                    "id": "scores",
                    "source": {"kind": "events"},
                    "where": [{"field": "rating", "op": "eq", "value": 7}],
                }
            ]
        },
        {"events": [{"rating": 7}, {"rating": 7.0}, {"rating": True}]},
    )["scores"]
    assert [row["rating"] for row in matched] == [7, 7.0]


# --- 2026-10-02 review: cell columns, the compile snapshot, schema links ------


def test_an_exploded_record_keeps_its_own_round() -> None:
    """The element wins, the round included: an article's `phase` is the round
    it was published in, and by-round outcomes group on it. Protecting the row's
    round put every article of a final snapshot in the last round."""
    from genesis.outcome_plan import _exploded

    row = {"run_id": "r", "phase": 40, "articles": [{"id": "a1", "phase": 3}, {"id": "a2"}]}
    first, second = _exploded([row], "articles")
    assert first["phase"] == 3
    assert second["phase"] == 40


def test_a_package_edited_while_it_compiles_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation read one study.yaml and the closure pinned another."""
    from genesis.compiler import StudyCompiler

    package = tmp_path / "pkg"
    shutil.copytree(Path("tests/golden_studies/platform_governance"), package)
    compiler = StudyCompiler(package)
    load = compiler._load

    def load_then_edit() -> dict[str, object]:
        loaded = load()
        study = package / "study.yaml"
        study.write_text(study.read_text() + "\n# edited mid-compile\n")
        return loaded

    monkeypatch.setattr(compiler, "_load", load_then_edit)
    with pytest.raises(ValueError, match="PACKAGE_CHANGED"):
        compiler.compile(tmp_path / "build")
    assert not (tmp_path / "build").exists()
    # Untouched, the same package still compiles.
    StudyCompiler(package).compile(tmp_path / "build")


def test_a_symlinked_schema_is_refused_before_it_is_parsed(tmp_path: Path) -> None:
    from genesis.compiler import StudyCompiler

    package = tmp_path / "pkg"
    shutil.copytree(Path("tests/golden_studies/platform_governance"), package)
    schema = package / "schemas" / "strategy-schema.yaml"
    schema.unlink()
    (package / "elsewhere.txt").write_text("a: [not yaml")
    try:
        schema.symlink_to("../elsewhere.txt")
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    # Parsed first, the target's broken YAML was reported as SCHEMA_INVALID.
    with pytest.raises(ValueError, match="PACKAGE_SYMLINK"):
        StudyCompiler(package).compile(tmp_path / "build")

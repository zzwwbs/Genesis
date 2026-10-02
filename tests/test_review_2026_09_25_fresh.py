"""Findings from the fresh 2026-09-25 review at c7e2d39.

Each test states the defect it pins: what the code did, and what it does now.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from genesis.compiler import StudyCompiler
from genesis.outcome_plan import check_dataset_id, compile_outcome_plan, materialize_datasets
from genesis.service import GenesisService

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden_studies" / "platform_governance"


# --- H1: a dataset id is a file name, never a path ----------------------------


@pytest.mark.parametrize("bad", ["../../../escaped", "/tmp/planted", "a/b", "Upper", ""])
def test_a_dataset_id_that_is_a_path_is_refused_everywhere(tmp_path: Path, bad: str) -> None:
    """A crafted id wrote export files outside the destination, or anywhere on disk.

    The plan is plain JSON, and an imported bundle's plan is copied verbatim and
    re-signed, so the specification's id pattern never applied to it.
    """
    build = tmp_path / "build"
    build.mkdir()
    (build / "outcome_plan.json").write_text(
        json.dumps({"datasets": [{"id": bad, "source": {"kind": "events"}}], "outcomes": []})
    )
    with pytest.raises(ValueError, match="OUTCOME_PLAN"):
        compile_outcome_plan(build)
    with pytest.raises(ValueError, match="OUTCOME_PLAN"):
        GenesisService._write_datasets(tmp_path / "exports" / "run", {bad: [{"x": 1}]}, build)
    assert not list(tmp_path.rglob("escaped*"))
    assert not Path("/tmp/planted.csv").exists()


def test_an_ordinary_dataset_id_still_exports(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    (build / "outcome_plan.json").write_text(
        json.dumps(
            {"datasets": [{"id": "feed-deliveries", "source": {"kind": "events"}}], "outcomes": []}
        )
    )
    written = GenesisService._write_datasets(
        tmp_path / "out", {"feed-deliveries": [{"x": 1}]}, build
    )
    assert (tmp_path / "out" / "feed-deliveries.csv") in written
    assert check_dataset_id("feed-deliveries") == "feed-deliveries"


# --- L3: an exploded record keeps its own value, and not its parent's ----------


def test_an_exploded_record_keeps_its_own_value_and_not_the_parent_payload(
    tmp_path: Path,
) -> None:
    """The artifact's payload copy under `value` carried the exploded list.

    Every exploded row repeated the whole collection, a record with no `value` of
    its own inherited that copy, and the export then stripped a record's real
    `value` column that outcomes still read.
    """
    dataset = {
        "id": "items",
        "source": {"kind": "artifacts", "artifact_type": "feed"},
        "explode": "items",
    }
    artifact = {
        "artifact_id": "a1",
        "payload": {
            "declared_artifact_id": "feed",
            "value": {"items": [{"id": "i1", "value": 7}, {"id": "i2"}], "n": 2},
        },
    }
    rows = materialize_datasets(
        {"datasets": [dataset]}, {"events": [], "state": [], "artifacts": [artifact]}
    )["items"]
    assert [row.get("value") for row in rows] == [7, None]
    assert all("items" not in row for row in rows)

    build = tmp_path / "build"
    build.mkdir()
    (build / "outcome_plan.json").write_text(json.dumps({"datasets": [dataset], "outcomes": []}))
    tables, _dictionary = GenesisService._dataset_tables({"items": rows}, build)
    assert [row["value"] for row in tables["items"]] == [7, None]


# --- H2: a racing first open does not rebuild a finished schema ----------------


def test_migration_1_refuses_a_schema_that_is_not_v0(tmp_path: Path) -> None:
    """A truthiness test took any non-empty schema for v0 and rebuilt it."""
    service = GenesisService(tmp_path / "ws")
    try:
        connection = service.persistence.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(ValueError, match="SCHEMA_SHAPE"):
                service.persistence._migration_1()
        finally:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
        columns = {row[1] for row in connection.execute("pragma table_info(events)")}
        assert {"commit_hash", "form"} <= columns
    finally:
        service.close()


def test_a_process_that_read_version_zero_skips_what_another_applied(tmp_path: Path) -> None:
    """Two processes opening a new workspace together both read version 0.

    The second then waited on the migration lock and ran _migration_1 against the
    finished schema: events.commit_hash, events.form and states.form were
    dropped, user_version rewound to 1, and the workspace would not open again.
    """
    workspace = tmp_path / "ws"
    first = GenesisService(workspace)
    persistence = first.persistence
    connection = persistence.connection
    before = {
        table: {row[1] for row in connection.execute(f"pragma table_info({table})")}
        for table in ("events", "states")
    }

    class _StaleVersion:
        """The version and table reads the second process took before the lock."""

        def __init__(self, inner: object) -> None:
            self.inner = inner
            self.stale = True

        def execute(self, sql: str, *args: object) -> object:
            if self.stale and sql.strip() == "PRAGMA user_version":
                self.stale = False

                class _Zero:
                    def fetchone(self) -> tuple[int]:
                        return (0,)

                return _Zero()
            return self.inner.execute(sql, *args)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    real_tables = persistence._user_tables
    reads = [0]

    def stale_tables() -> list[str]:
        reads[0] += 1
        return [] if reads[0] == 1 else real_tables()

    persistence.connection = _StaleVersion(connection)  # type: ignore[assignment]
    persistence._user_tables = stale_tables  # type: ignore[method-assign]
    try:
        persistence._migrate_database()
    finally:
        persistence.connection = connection
        persistence._user_tables = real_tables  # type: ignore[method-assign]

    for table, columns in before.items():
        assert {row[1] for row in connection.execute(f"pragma table_info({table})")} == columns
    first.close()
    GenesisService(workspace).close()  # the workspace still opens


# --- H3: a credential-named schema is refused, not silently dropped ------------


def _golden_with_schema_named(tmp_path: Path, name: str) -> Path:
    study = tmp_path / f"study-{name}"
    shutil.copytree(GOLDEN, study)
    (study / "schemas" / "strategy-schema.yaml").rename(study / "schemas" / f"{name}.yaml")
    for file in ("openness.yaml", "domain.yaml"):
        path = study / file
        path.write_text(path.read_text().replace("strategy-schema", name))
    return study


def test_a_credential_named_schema_is_refused_with_its_reason(tmp_path: Path) -> None:
    """schemas/tokens.yaml was certified, left out of schemas.json, and never applied."""
    study = _golden_with_schema_named(tmp_path, "tokens")
    with pytest.raises(ValueError, match="excluded from the build as a credential file"):
        StudyCompiler(study).compile(tmp_path / "out")


def test_an_ordinarily_named_schema_still_compiles(tmp_path: Path) -> None:
    study = _golden_with_schema_named(tmp_path, "renamed-schema")
    build = Path(StudyCompiler(study).compile(tmp_path / "out").path)
    assert "renamed-schema" in json.loads((build / "schemas.json").read_text())


# --- M1: a build with no closure bytes is not intact ---------------------------


def test_a_build_whose_closure_was_deleted_does_not_verify(tmp_path: Path) -> None:
    """`rm -rf closure/` verified clean while package_closure.json listed its assets."""
    import os

    build = Path(StudyCompiler(GOLDEN).compile(tmp_path / "build").path)
    assert StudyCompiler.verify_build(build)
    for path in build.rglob("*"):
        os.chmod(path, 0o700)
    shutil.rmtree(build / "closure")
    with pytest.raises(ValueError, match="closure/ is missing all"):
        StudyCompiler.verify_build(build)


# --- M2, M3, M4 and L4: compile-time refusals ----------------------------------


def _compile_golden_with(tmp_path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    study = tmp_path / "study"
    shutil.copytree(GOLDEN, study)
    mutate(study)
    StudyCompiler(study).compile(tmp_path / "out")


def _time_model(**changes: object):  # type: ignore[no-untyped-def]
    import yaml

    def mutate(study: Path) -> None:
        path = study / "protocol.yaml"
        protocol = yaml.safe_load(path.read_text())
        protocol["time_model"].update(changes)
        path.write_text(yaml.safe_dump(protocol))

    return mutate


def test_a_credential_named_data_source_is_refused(tmp_path: Path) -> None:
    """data/tokens.csv compiled clean, then every run was refused its population."""

    def mutate(study: Path) -> None:
        (study / "data" / "population.csv").rename(study / "data" / "tokens.csv")
        domain = study / "domain.yaml"
        domain.write_text(domain.read_text().replace("data/population.csv", "data/tokens.csv"))

    with pytest.raises(ValueError, match="excluded from the build as a credential file"):
        _compile_golden_with(tmp_path, mutate)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start": 10, "end": 5}, "contains no rounds"),
        ({"start": 0, "end": -3}, "must not be negative"),
        ({"step": 2}, "rounds always advance by 1"),
    ],
)
def test_a_time_model_the_run_would_not_honour_is_refused(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    """An inverted interval ran no rounds and reported completed; step was never read."""
    with pytest.raises(ValueError, match=message):
        _compile_golden_with(tmp_path, _time_model(**changes))


@pytest.mark.parametrize("changes", [{"start": 3, "end": 3}, {"step": 1}])
def test_a_one_round_interval_and_a_unit_step_still_compile(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    _compile_golden_with(tmp_path, _time_model(**changes))


def test_a_duplicate_dataset_id_is_refused(tmp_path: Path) -> None:
    """The later declaration silently replaced the earlier one's rows."""
    import yaml

    def mutate(study: Path) -> None:
        path = study / "outcomes.yaml"
        outcomes = yaml.safe_load(path.read_text())
        outcomes["datasets"] = [
            {"id": "rounds", "source": {"kind": "events"}},
            {"id": "rounds", "source": {"kind": "artifacts", "artifact_type": "strategy"}},
        ]
        path.write_text(yaml.safe_dump(outcomes, sort_keys=False))

    with pytest.raises(ValueError, match="duplicate ID in datasets"):
        _compile_golden_with(tmp_path, mutate)


# --- M5: a paired selector is a batch ---------------------------------------


def test_a_single_listed_id_with_per_is_classified_as_a_batch() -> None:
    """It expands to one group per inner record, so its timing must be declared."""
    from genesis.information_timing import is_batched

    paired = {"id": "p", "actors": {"ids": ["u1"], "per": {"source": "queue.${actor}"}}}
    assert is_batched(paired)
    assert not is_batched({"id": "p", "actors": {"ids": ["u1"]}})


# --- M6: a `final` state dataset streams the history ----------------------------


def test_a_final_state_dataset_holds_one_snapshot_not_the_history() -> None:
    """Listing the history first held every round's snapshot to return one row."""
    import tracemalloc

    big = "x" * 1_000_000

    def history():  # type: ignore[no-untyped-def]
        for version in range(1, 41):
            yield {"state_version": version, "big": big + str(version), "field": version}

    plan = {
        "datasets": [
            {"id": "last", "source": {"kind": "state", "state": "field", "snapshot": "final"}}
        ]
    }
    tracemalloc.start()
    rows = materialize_datasets(plan, {"events": [], "artifacts": [], "state": history})
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert rows["last"] == [{"field": 40}]
    assert peak < 10_000_000  # was 40 MB


def test_final_still_takes_the_highest_committed_version() -> None:
    plan = {"datasets": [{"id": "d", "source": {"kind": "state", "state": "f"}}]}

    def rows_of(entries):  # type: ignore[no-untyped-def]
        return materialize_datasets(
            plan, {"events": [], "artifacts": [], "state": lambda: iter(entries)}
        )["d"]

    interleaved = [
        {"state_version": 3, "f": "v3"},
        {"state_version": 5, "f": "v5"},
        {"state_version": 4, "f": "v4"},
    ]
    assert rows_of(interleaved) == [{"f": "v5"}]
    assert rows_of([{"f": "a"}, {"f": "b"}]) == [{"f": "b"}]
    assert rows_of([]) == []


# --- M7: the chain cache is bounded ---------------------------------------------


def test_the_chain_cache_is_bounded_and_cleared_on_close(tmp_path: Path) -> None:
    """It kept a decoded tail per run for the life of the process."""
    from genesis.persistence import _CHAIN_CACHE_LIMIT

    service = GenesisService(tmp_path / "ws")
    persistence = service.persistence
    for index in range(_CHAIN_CACHE_LIMIT * 3):
        persistence._chain_cache[("states", f"run-{index}")] = ((1, "ref"), {"x": index})
        persistence._chain_cache.move_to_end(("states", f"run-{index}"))
        while len(persistence._chain_cache) > _CHAIN_CACHE_LIMIT:
            persistence._chain_cache.popitem(last=False)
    assert len(persistence._chain_cache) == _CHAIN_CACHE_LIMIT
    service.close()
    assert not persistence._chain_cache


# --- M8, M9, M10: the API reports what actually happened -------------------------


def test_a_patch_removing_a_section_is_an_invalid_draft_not_a_missing_session() -> None:
    """KeyError escaped the repair loop, which only retries ValueError."""
    from types import SimpleNamespace

    from genesis.compiler import CANONICAL
    from genesis.elicitation import apply_operations
    from genesis.service import _validate_canonical_sections

    candidate = apply_operations(
        {name: {} for name in CANONICAL},
        [SimpleNamespace(op="remove", path="/study", value=None)],
    )
    with pytest.raises(ValueError, match="leaves no 'study' section") as raised:
        _validate_canonical_sections(candidate, CANONICAL)
    assert not isinstance(raised.value, KeyError)


def test_only_a_missing_resource_is_reported_as_404() -> None:
    """Any KeyError was a 404, so an internal fault read as an absent resource."""
    from genesis.app import _service_error
    from genesis.errors import NotFoundError

    assert _service_error(NotFoundError("run-1")).status_code == 404
    internal = _service_error(KeyError("id"))
    assert internal.status_code == 500
    assert b"INTERNAL_ERROR" in internal.body
    # NotFoundError is still a KeyError, so callers detecting absence are unchanged.
    assert issubclass(NotFoundError, KeyError)


def test_a_lone_string_option_is_one_item_not_one_per_character(tmp_path: Path) -> None:
    from genesis.service import _resolve_initializations

    build = tmp_path / "b"
    (build / "data").mkdir(parents=True)
    (build / "initialization.json").write_text(json.dumps({"mode": "empirical"}))
    (build / "data" / "world.json").write_text("{}")
    worlds = _resolve_initializations("data/world.json", build)  # type: ignore[arg-type]
    assert worlds == [{"id": "world", "data_source": "data/world.json"}]


# --- M11: provider responses are bounded ------------------------------------------


def test_an_oversized_provider_response_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import genesis.providers as providers

    monkeypatch.setattr(providers, "MAX_RESPONSE_BYTES", 4096)

    class Streams(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"' + b"x" * 50_000 + b'"}}]}')

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Streams)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        provider = providers.OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="m",
            api_key_env="GENESIS_UNUSED",
            api_key="k",
            timeout=10,
            max_retries=0,
        )
        with pytest.raises(ValueError, match="exceeded 4096 bytes"):
            provider.generate(providers.ProviderRequest(model="m", prompt="hi"))
    finally:
        server.shutdown()


# --- L1, L2, L5, L6 ----------------------------------------------------------------


def test_doctor_does_not_count_a_write_in_flight_as_corruption(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        service.persistence.object_store.put(b'{"a": 1}', "application/json")
        shard = next(p for p in service.persistence.object_store.root.iterdir() if p.is_dir())
        (shard / ".object-inflight").write_bytes(b"partial")
        (shard / "abandoned.tmp").write_bytes(b"partial")
        assert service.doctor()["checks"]["object_integrity_failures"] == 0
    finally:
        service.close()


def test_a_refused_package_version_leaves_no_phantom_study(tmp_path: Path) -> None:
    """The study row autocommitted, then survived the version's refusal."""
    service = GenesisService(tmp_path / "ws")
    persistence = service.persistence
    real = persistence.connection

    class RefuseVersionInsert:
        def execute(self, sql: str, *args: object) -> object:
            if sql.lstrip().startswith("INSERT INTO package_versions"):
                raise __import__("sqlite3").IntegrityError("refused")
            return real.execute(sql, *args)

        def __getattr__(self, name: str) -> object:
            return getattr(real, name)

    persistence.connection = RefuseVersionInsert()  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="PACKAGE_VERSION_EXISTS"):
            persistence.record_package_version("ghost-study", 1, "h", None, "draft", {})
    finally:
        persistence.connection = real
    assert real.execute("SELECT 1 FROM studies WHERE study_id = 'ghost-study'").fetchone() is None
    service.close()


def test_paging_a_dataset_does_not_rebuild_every_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each page re-evaluated every outcome and rebuilt every table."""
    service = GenesisService(tmp_path / "ws")
    build = tmp_path / "ws" / "builds" / "b"
    build.mkdir(parents=True)
    (build / "processes.json").write_text("[]")
    (build / "outcome_plan.json").write_text(json.dumps({"datasets": [], "outcomes": []}))
    calls = [0]

    def evaluate(run_id, *, datasets_out=None, analysis_build=None):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return []

    monkeypatch.setattr(service, "get_run", lambda run_id: {"build": str(build), "version": 1})
    monkeypatch.setattr(service, "evaluate_outcomes", evaluate)
    try:
        for _page in range(3):
            service.run_datasets("r")
        assert calls[0] == 1
        # A new commit changes the evidence, so the next page rebuilds.
        monkeypatch.setattr(service.persistence, "event_marker", lambda run_id: (1, 1))
        service.run_datasets("r")
        assert calls[0] == 2
    finally:
        service.close()


def test_the_isolation_check_honours_a_declared_measurement_use(tmp_path: Path) -> None:
    """A declared instrument was reported as a leak; an undeclared reader still is."""
    import sys

    import yaml

    sys.path.insert(0, str(ROOT / "tools"))
    from check_measurement_isolation import check_package

    (tmp_path / "domain.yaml").write_text(
        yaml.safe_dump(
            {
                "visibility": [
                    {"id": "detector-context", "allow": ["inputs"]},
                    {"id": "platform-context", "allow": ["inputs", "detection-result"]},
                    {"id": "creator-context", "allow": ["inputs", "detection-result"]},
                ]
            }
        )
    )
    detector = {
        "id": "detect",
        "measurement": True,
        "context_policy": "detector-context",
        "outputs": [{"artifact_type": "detection-result"}],
    }
    settle = {
        "id": "settle",
        "context_policy": "platform-context",
        "inputs": ["detection-result"],
        "measurement_use": [{"source": "detect", "rationale": "the score sets revenue"}],
    }
    creator = {
        "id": "publish",
        "context_policy": "creator-context",
        "inputs": ["detection-result"],
    }
    (tmp_path / "openness.yaml").write_text(
        yaml.safe_dump({"processes": [detector, settle, creator]})
    )
    violations = check_package(tmp_path)
    assert not any("settle" in item or "platform-context" in item for item in violations)
    assert any("publish" in item for item in violations)
    assert any("creator-context" in item for item in violations)


# --- 2026-09-26 review: R1 and R2 ----------------------------------------------


def test_a_missing_package_snapshot_is_a_404(tmp_path: Path) -> None:
    """The route re-wrapped NotFoundError as a plain KeyError, which became a 500."""
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    GenesisService(tmp_path / "ws").close()
    client = TestClient(create_app(tmp_path / "ws"))
    assert client.get("/specifications/nope/versions/1/snapshot").status_code == 404


def test_exploding_outside_the_payload_does_not_export_the_payload_copy(
    tmp_path: Path,
) -> None:
    dataset = {
        "id": "d",
        "source": {"kind": "artifacts", "artifact_type": "note"},
        "explode": "lineage",
    }
    artifact = {
        "artifact_id": "a1",
        "payload": {
            "declared_artifact_id": "note",
            "lineage": ["e1", "e2"],
            "value": {"score": 7},
        },
    }
    rows = materialize_datasets(
        {"datasets": [dataset]}, {"events": [], "state": [], "artifacts": [artifact]}
    )
    build = tmp_path / "b"
    build.mkdir()
    (build / "outcome_plan.json").write_text(json.dumps({"datasets": [dataset], "outcomes": []}))
    tables, _ = GenesisService._dataset_tables(rows, build)
    assert all("value" not in row or row["value"] is None for row in tables["d"])
    assert [row["score"] for row in tables["d"]] == [7, 7]


# --- 2026-09-26 review: G2, the cache key, and a data source outside data/ ------


def test_a_build_plan_declaring_a_dataset_twice_is_refused(tmp_path: Path) -> None:
    """The first declaration's rows silently vanished when the plan was loaded."""
    (tmp_path / "outcome_plan.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"id": "d", "source": {"kind": "events"}},
                    {"id": "d", "source": {"kind": "artifacts", "artifact_type": "x"}},
                ],
                "outcomes": [],
            }
        )
    )
    with pytest.raises(ValueError, match="declared twice"):
        compile_outcome_plan(tmp_path)


def test_a_recompiled_build_is_not_served_from_the_old_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = GenesisService(tmp_path / "ws")
    build = tmp_path / "ws" / "builds" / "b"
    build.mkdir(parents=True)
    (build / "processes.json").write_text("[]")
    (build / "outcome_plan.json").write_text(json.dumps({"datasets": [], "outcomes": []}))
    (build / "build_manifest.json").write_text(json.dumps({"build_hash": "one"}))
    calls = [0]

    def evaluate(run_id, *, datasets_out=None, analysis_build=None):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return []

    monkeypatch.setattr(service, "get_run", lambda run_id: {"build": str(build), "version": 1})
    monkeypatch.setattr(service, "evaluate_outcomes", evaluate)
    try:
        service.run_datasets("r")
        service.run_datasets("r")
        assert calls[0] == 1
        (build / "build_manifest.json").write_text(json.dumps({"build_hash": "two"}))
        service.run_datasets("r")
        assert calls[0] == 2
    finally:
        service.close()


def test_a_data_source_outside_data_is_refused(tmp_path: Path) -> None:
    """It compiled clean into a build that did not carry it."""

    def mutate(study: Path) -> None:
        (study / "extra").mkdir()
        (study / "data" / "population.csv").rename(study / "extra" / "population.csv")
        domain = study / "domain.yaml"
        domain.write_text(domain.read_text().replace("data/population.csv", "extra/population.csv"))

    with pytest.raises(ValueError, match="not under data/"):
        _compile_golden_with(tmp_path, mutate)


# --- 2026-09-26 review: G1 and N3 ----------------------------------------------


@pytest.mark.parametrize(
    "time_model",
    [
        {"type": "event", "end": 12.0},
        {"type": "event", "start": 2, "end": 1},
        {"type": "continuous", "start": 2, "end": 1},
    ],
)
def test_the_time_model_checks_apply_to_every_type(
    tmp_path: Path, time_model: dict[str, object]
) -> None:
    """The run reads start and end whatever the type; only rounds was checked."""
    import yaml

    def mutate(study: Path) -> None:
        path = study / "protocol.yaml"
        protocol = yaml.safe_load(path.read_text())
        protocol["time_model"] = time_model
        path.write_text(yaml.safe_dump(protocol))

    with pytest.raises(ValueError, match="TIME_MODEL_INVALID"):
        _compile_golden_with(tmp_path, mutate)


def test_a_valid_event_interval_still_compiles(tmp_path: Path) -> None:
    import yaml

    def mutate(study: Path) -> None:
        path = study / "protocol.yaml"
        protocol = yaml.safe_load(path.read_text())
        protocol["time_model"] = {"type": "event", "start": 1, "end": 3}
        path.write_text(yaml.safe_dump(protocol))

    _compile_golden_with(tmp_path, mutate)


def test_a_legacy_immediate_workspace_opened_twice_is_not_rebuilt(tmp_path: Path) -> None:
    """The second opener validated the v0 layout, waited for the lock, then
    rebuilt the schema the first had migrated -- the H2 failure, in the legacy
    branch the loop's re-read did not cover."""
    workspace = tmp_path / "ws"
    first = GenesisService(workspace)
    persistence = first.persistence
    connection = persistence.connection
    before = {
        table: {row[1] for row in connection.execute(f"pragma table_info({table})")}
        for table in ("events", "states")
    }
    immediate = ["artifacts", "checkpoints", "events", "objects", "runs", "states", "studies"]

    class _StaleVersion:
        def __init__(self, inner: object) -> None:
            self.inner, self.stale = inner, True

        def execute(self, sql: str, *args: object) -> object:
            if self.stale and sql.strip() == "PRAGMA user_version":
                self.stale = False

                class _Zero:
                    def fetchone(self) -> tuple[int]:
                        return (0,)

                return _Zero()
            return self.inner.execute(sql, *args)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self.inner, name)

    real_tables, real_validate = persistence._user_tables, persistence._validate_v0_schema
    table_reads, validations = [0], [0]

    def stale_tables() -> list[str]:
        table_reads[0] += 1
        return immediate if table_reads[0] == 1 else real_tables()

    def stale_validate(tables: set[str]) -> str:
        validations[0] += 1
        return "immediate" if validations[0] == 1 else real_validate(tables)

    persistence.connection = _StaleVersion(connection)  # type: ignore[assignment]
    persistence._user_tables = stale_tables  # type: ignore[method-assign]
    persistence._validate_v0_schema = stale_validate  # type: ignore[method-assign]
    try:
        persistence._migrate_database()
    finally:
        persistence.connection = connection
        persistence._user_tables = real_tables  # type: ignore[method-assign]
        persistence._validate_v0_schema = real_validate  # type: ignore[method-assign]

    for table, columns in before.items():
        assert {row[1] for row in connection.execute(f"pragma table_info({table})")} == columns
    first.close()
    GenesisService(workspace).close()


# --- 2026-09-26 review: N2, N4, N5, N6 -------------------------------------------


class _EventLedger:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def commit_process_result(self, event, state, artifacts, fail_after=None, **_: object) -> None:  # type: ignore[no-untyped-def]
        self.events.append(dict(event))

    def latest_json_state(self, run_id: str) -> None:
        return None

    def list_events(self, run_id: str) -> list[dict[str, object]]:
        return list(self.events)

    def list_artifacts(self, run_id: str) -> list[object]:
        return []


def test_an_in_process_resume_gives_the_paused_actor_its_turn() -> None:
    """The actor was popped before its turn and never put back."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    ran: list[str] = []
    fail = {"w2": True}

    class Executor:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            actor = invocation.actor_ids[0]
            if fail.get(actor):
                fail[actor] = False
                raise ValueError("PROVIDER_HTTP: provider returned HTTP 402: no credit")
            ran.append(actor)
            return ProcessResult(outputs={})

    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": ["w1", "w2", "w3", "w4"],
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                    "information_timing": {"mode": "sequential", "order": "listed"},
                }
            ]
        ),
        ExecutorRegistry({"p": Executor()}),
        ContextEngine({"private": {"allow": []}}),
    )
    controller.run("r", phase_limit=2, seed=1)
    assert controller.status == "paused"
    controller.resume()
    controller.run("r", phase_limit=2, seed=1)
    assert ran == ["w1", "w2", "w3", "w4"]


def test_a_resumed_concurrent_batch_runs_the_attempt_it_records() -> None:
    """Prepared as attempt 1, committed as attempt 2 (CON-012)."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
    )

    process = {
        "id": "p",
        "actors": ["a1", "a2"],
        "context_policy": "private",
        "trigger": {"phase": 1, "repeat": True},
        "retry_policy": {"max_attempts": 3},
        "information_timing": {"mode": "simultaneous", "order": "listed"},
    }
    ledger = _EventLedger()
    seen: list[tuple[tuple[str, ...], int]] = []

    class FailThenPause:
        concurrent_safe = True

        def execute(self, invocation):  # type: ignore[no-untyped-def]
            if invocation.actor_ids == ("a1",) and not ledger.events:
                return ProcessResult(status="failed", outputs={}, metadata={"code": "BAD"})
            raise ValueError("PROVIDER_HTTP: provider returned HTTP 402: no credit")

    class Succeeds:
        concurrent_safe = True

        def execute(self, invocation):  # type: ignore[no-untyped-def]
            seen.append((invocation.actor_ids, invocation.attempt))
            return ProcessResult(outputs={})

    def controller(executor):  # type: ignore[no-untyped-def]
        return RunController(
            Scheduler([dict(process)]),
            ExecutorRegistry({"p": executor}),
            ContextEngine({"private": {"allow": []}}),
            persistence=ledger,
            max_concurrency={"p": 2},
            state_store=StateStore({"x": int}, {"x": 0}),
        )

    controller(FailThenPause()).run("r", phase_limit=2, seed=1)
    controller(Succeeds()).run("r", phase_limit=2, seed=1)
    executed = [attempt for actors, attempt in seen if actors == ("a1",)]
    recorded = [
        event["attempt"]
        for event in ledger.events
        if event["actors"] == ["a1"] and event["kind"] == "process_completed"
    ]
    assert executed == recorded == [2]


@pytest.mark.parametrize(("lose_in_phase", "terminal", "remaining"), [(2, 2, False), (1, 4, True)])
def test_a_lease_lost_after_the_final_turn_is_not_work_remaining(
    lose_in_phase: int, terminal: int, remaining: bool
) -> None:
    """_next_phase advances only when a phase ends, so the terminal phase's last
    commit still read as work remaining."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    status = ["running"]

    class Executor:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            if invocation.phase == lose_in_phase:
                status[0] = "lease_lost"
            return ProcessResult(outputs={})

    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": None,
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                }
            ]
        ),
        ExecutorRegistry({"p": Executor()}),
        ContextEngine({"private": {"allow": []}}),
        status_provider=lambda: status[0],
    )
    controller.run("r", phase_limit=6, seed=1, terminal_phase=terminal)
    assert controller.lease_lost
    assert controller.stopped_with_work_remaining is remaining


def test_the_chain_cache_is_keyed_by_this_commits_own_row(tmp_path: Path) -> None:
    """Re-read after COMMIT, the tail could be another writer's row, and the next
    commit patched against the wrong base -- a key its writer never wrote."""
    from genesis.persistence import PersistenceCoordinator

    workspace = tmp_path / "ws"
    service = GenesisService(workspace)
    build = StudyCompiler(GOLDEN).compile(workspace / "builds" / "b")
    service.create_run(
        {"id": "r1", "study_id": "platform-governance", "build": str(Path(build.path))}
    )
    service.close()
    database, objects = workspace / ".genesis" / "genesis.db", workspace / ".genesis" / "objects"
    first = PersistenceCoordinator(database, objects)
    second = PersistenceCoordinator(database, objects)
    shared = {f"k{index}": "v" * 200 for index in range(300)}

    def commit(writer, number, extra):  # type: ignore[no-untyped-def]
        state = {**shared, "n": number, **extra}
        writer.commit_process_result(
            {
                "event_id": f"e{number}",
                "run_id": "r1",
                "kind": "process_completed",
                "process_id": "p",
                "phase": number,
                "state_version": number,
                "payload": {"n": number, **shared},
            },
            {
                "run_id": "r1",
                "state_version": number,
                "payload": json.dumps(state, sort_keys=True).encode(),
            },
            [],
        )

    commit(first, 1, {})
    remember = first._remember_committed

    def other_writer_lands_first(*args):  # type: ignore[no-untyped-def]
        commit(second, 3, {"z": "written-only-by-the-other"})
        remember(*args)

    first._remember_committed = other_writer_lands_first  # type: ignore[method-assign]
    commit(first, 2, {})
    first._remember_committed = remember  # type: ignore[method-assign]
    commit(first, 4, {})
    first.close()
    second.close()
    reader = PersistenceCoordinator(database, objects)
    _version, latest = reader.latest_json_state("r1")
    reader.close()
    assert "z" not in latest


# --- 2026-09-26 review: N1, N7, N9, N10 and two lows ----------------------------


@pytest.mark.parametrize("bad", ["../model-profiles", "a/b", "Upper", ""])
def test_a_package_id_must_be_a_plain_identifier(tmp_path: Path, bad: str) -> None:
    """ "../model-profiles" was written outside packages/, and "a/b" was a 500."""
    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match="PACKAGE_ID_INVALID"):
            service.create_package({"id": bad})
        assert not (tmp_path / "ws" / ".genesis" / "model-profiles.json").exists()
        assert service.create_package({"id": "fine-package"})["id"] == "fine-package"
    finally:
        service.close()


def test_an_anonymous_package_post_always_succeeds(tmp_path: Path) -> None:
    """A generated hex id starts with a digit nearly half the time, and the new id
    validation refused it -- the identical request failed nondeterministically."""
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    workspace = tmp_path / "ws"
    client = TestClient(create_app(workspace))
    codes = [
        client.post("/packages", json={"title": f"p{index}"}).status_code for index in range(40)
    ]
    assert set(codes) == {201}, codes
    listed = {package["id"] for package in client.get("/packages").json()}
    assert len(listed) == 40, sorted(listed)


def test_a_lone_artifact_id_is_one_id_not_one_per_character() -> None:
    from genesis.app import _id_list_option

    assert _id_list_option("artifact_ids", "note-x") == ("note-x",)
    assert _id_list_option("artifact_ids", ["a", "b"]) == ("a", "b")
    assert _id_list_option("artifact_ids", None) == ()
    with pytest.raises(ValueError, match="VALIDATION_ERROR"):
        _id_list_option("artifact_ids", {"a": 1})


def test_retrieving_an_artifact_the_run_does_not_have_is_not_found(tmp_path: Path) -> None:
    """A missing artifact was dropped and the retrieval reported success."""
    from genesis.errors import NotFoundError
    from genesis.replay import ReplayMode

    service = GenesisService(tmp_path / "ws")
    try:
        build = StudyCompiler(GOLDEN).compile(tmp_path / "ws" / "builds" / "b")
        service.create_run(
            {"id": "src", "study_id": "platform-governance", "build": str(Path(build.path))}
        )
        with pytest.raises(NotFoundError, match="no artifact 'note-x'"):
            service.replay_run("src", mode=ReplayMode.ARTIFACT, artifact_ids=("note-x",))
    finally:
        service.close()


def test_replay_overrides_must_be_a_mapping(tmp_path: Path) -> None:
    """A list reached overrides.items() and failed as a 500."""
    from genesis.replay import ReplayMode

    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match="overrides must map each factor"):
            service._validate_replay_request(
                ReplayMode.BRANCH, (), "e1", "because", ["not", "a", "mapping"]
            )
    finally:
        service.close()


def test_a_command_line_override_names_a_typed_level(tmp_path: Path) -> None:
    """argparse passes text, so '0.5' never equalled the declared level 0.5 and a
    numeric level could not be branched from the CLI."""
    protocol = {
        "factors": [
            {"id": "rate", "levels": [0.5, 1.0], "branchable": True},
            {"id": "flag", "levels": [True, False], "branchable": True},
        ]
    }
    service = GenesisService(tmp_path / "ws")
    try:
        factors = service._branch_factors(
            protocol, {"factors": {}}, {"rate": "0.5", "flag": "false"}
        )
        assert factors == {"rate": 0.5, "flag": False}
        assert isinstance(factors["rate"], float)
        with pytest.raises(ValueError, match="not a declared level"):
            service._branch_factors(protocol, {"factors": {}}, {"rate": "0.7"})
    finally:
        service.close()


def test_a_researcher_mode_state_field_is_not_an_empirical_envelope(tmp_path: Path) -> None:
    import yaml

    study = tmp_path / "study"
    shutil.copytree(ROOT / "tests" / "golden_studies" / "community_formation", study)
    domain = yaml.safe_load((study / "domain.yaml").read_text())
    domain["states"].append(
        {
            "id": "population",
            "value_type": "object",
            "initial": {"members": [{"id": "m1"}, {"id": "m2"}]},
            "persistence": "run",
        }
    )
    domain.setdefault("updates", []).append(
        {"state": "population", "op": "set", "declared_by": None}
    )
    domain["initialization"] = {"mode": "researcher", "state_field": "population"}
    (study / "domain.yaml").write_text(yaml.safe_dump(domain, sort_keys=False))
    openness = yaml.safe_load((study / "openness.yaml").read_text())
    openness["processes"][0]["actors"] = {"source": "population.members", "id_field": "id"}
    (study / "openness.yaml").write_text(yaml.safe_dump(openness, sort_keys=False))
    StudyCompiler(study).compile(tmp_path / "out")


def test_a_missing_build_is_not_found(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    GenesisService(tmp_path / "ws").close()
    client = TestClient(create_app(tmp_path / "ws"))
    assert client.get("/builds/builds/does-not-exist/processes").status_code == 404


# --- 2026-09-26b review: R1' and I4 --------------------------------------------


def test_a_data_source_that_is_a_symlink_to_a_credential_is_refused(tmp_path: Path) -> None:
    """A link carried a credential's bytes in under the dataset's name (R1')."""

    def mutate(study: Path) -> None:
        (study / "keys.env").write_text("API_KEY=sk-live-never-carry-me")
        population = study / "data" / "population.csv"
        population.unlink()
        os.symlink("../keys.env", population)

    with pytest.raises(ValueError, match="symbolic link"):
        _compile_golden_with(tmp_path, mutate)


def test_a_data_source_reachable_only_through_a_symlinked_directory_is_refused(
    tmp_path: Path,
) -> None:
    """A path prefix is not carriage: the closure collects no symlinked directory (I4)."""

    def mutate(study: Path) -> None:
        (study / "elsewhere").mkdir()
        (study / "data" / "population.csv").rename(study / "elsewhere" / "population.csv")
        os.symlink("../elsewhere", study / "data" / "sub")
        domain = study / "domain.yaml"
        domain.write_text(
            domain.read_text().replace("data/population.csv", "data/sub/population.csv")
        )

    with pytest.raises(ValueError, match="symbolic link"):
        _compile_golden_with(tmp_path, mutate)


def test_a_data_source_that_is_a_symlink_to_a_data_file_is_refused_too(tmp_path: Path) -> None:
    """A package carries regular files only, whatever the link points at."""

    def mutate(study: Path) -> None:
        os.symlink("population.csv", study / "data" / "alias.csv")
        domain = study / "domain.yaml"
        domain.write_text(domain.read_text().replace("data/population.csv", "data/alias.csv"))

    with pytest.raises(ValueError, match="symbolic link"):
        _compile_golden_with(tmp_path, mutate)


# --- 2026-09-26b review: I1 and I3 ---------------------------------------------


def test_a_concurrent_pause_gives_the_paused_actor_its_turn_back() -> None:
    """The concurrent loop popped the turn out of `pending` before committing it, so
    the pause handler of the serial path did not see it (I1)."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    fail = {"w1": True}
    resumed = {"value": False}
    ran: list[tuple[str, str]] = []

    class Executor:
        concurrent_safe = True

        def execute(self, invocation):  # type: ignore[no-untyped-def]
            actor = invocation.actor_ids[0]
            # Batch calls already in flight when the pause lands are discarded, so
            # an actor may legitimately run once before and once after the pause;
            # what matters is that the paused actor runs again after it.
            ran.append((actor, "after" if resumed["value"] else "before"))
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
    resumed["value"] = True
    controller.resume()
    controller.run("r", phase_limit=2, seed=1)
    assert sorted(actor for actor, when in ran if when == "after") == ["w1", "w2", "w3", "w4"]


def test_a_pause_gives_back_the_schedule_entry_it_consumed() -> None:
    """One entry per actor turn: the paused actor's retry consumed its sibling's (I3)."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
    )

    fail = {"w1": True}
    ran: list[tuple[str, int]] = []

    class Producer:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            # Two entries, one per actor of the consumer's batch.
            return ProcessResult(
                outputs={},
                scheduling_effects=[
                    {"type": "schedule", "process_id": "q", "phase": 2},
                    {"type": "schedule", "process_id": "q", "phase": 2},
                ],
            )

    class Consumer:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            actor = invocation.actor_ids[0]
            if invocation.phase == 2 and fail.get(actor):
                fail[actor] = False
                raise ValueError("PROVIDER_HTTP: provider returned HTTP 402: no credit")
            ran.append((actor, int(invocation.phase)))
            return ProcessResult(outputs={})

    processes = [
        {
            "id": "p",
            "actors": ["w1"],
            "context_policy": "private",
            "trigger": {"phase": 1, "repeat": False},
        },
        {
            "id": "q",
            "actors": ["w1", "w2"],
            "context_policy": "private",
            "repeat": True,
            "trigger": {
                "type": "condition",
                "repeat": True,
                "predicate": {"path": "flag", "op": "eq", "value": True},
            },
        },
    ]
    controller = RunController(
        Scheduler([dict(process) for process in processes]),
        ExecutorRegistry({"p": Producer(), "q": Consumer()}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({"flag": bool}, {"flag": False}),
    )
    controller.run("r", phase_limit=3, seed=1)
    assert controller.status == "paused"
    controller.resume()
    controller.run("r", phase_limit=3, seed=1)
    assert [actor for actor, _phase in ran] == ["w1", "w2"]


@pytest.mark.parametrize("limit", [1, 2])
def test_an_exhausted_attempt_cursor_fails_the_run(limit: int) -> None:
    """The serial path ran no attempt and dropped the invocation; the concurrent one
    ran an attempt past max_attempts (N1')."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
        _ReportedProcessFailure,
    )

    seen: list[tuple[tuple[str, ...], int]] = []

    class Executor:
        concurrent_safe = True

        def execute(self, invocation):  # type: ignore[no-untyped-def]
            seen.append((invocation.actor_ids, invocation.attempt))
            return ProcessResult(outputs={})

    ledger = _EventLedger()
    ledger.events.append(
        {
            "event_id": "r-p-a1-1-attempt-1",
            "invocation_id": "r-p-a1-1",
            "process_id": "p",
            "run_id": "r",
            "phase": 1,
            "attempt": 1,
            "kind": "process_failed",
            "actors": ["a1"],
            "classification": "executor_exception",
        }
    )
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": ["a1", "a2"],
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": True},
                    "information_timing": {"mode": "simultaneous", "order": "listed"},
                    "retry_policy": {"max_attempts": 1},
                }
            ]
        ),
        ExecutorRegistry({"p": Executor()}),
        ContextEngine({"private": {"allow": []}}),
        persistence=ledger,
        max_concurrency={"p": limit},
        state_store=StateStore({"x": int}, {"x": 0}),
    )
    with pytest.raises(_ReportedProcessFailure):
        controller.run("r", phase_limit=2, seed=1)
    assert controller.status == "failed"
    assert seen == [], "an attempt beyond the declared policy was executed"


def test_a_stop_at_the_phase_start_says_whether_that_phase_had_work() -> None:
    """A resume sets `_next_phase` to the last committed phase, so the phase-start
    check reported work remaining for a run with nothing left to run (I2)."""
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
    )

    class Executor:
        def execute(self, invocation):  # type: ignore[no-untyped-def]
            return ProcessResult(outputs={})

    def committed(phase: int) -> dict[str, object]:
        return {
            "event_id": f"r-p-w1-{phase}-attempt-1",
            "invocation_id": f"r-p-w1-{phase}",
            "process_id": "p",
            "run_id": "r",
            "phase": phase,
            "attempt": 1,
            "kind": "process_completed",
            "actors": ["w1"],
            "state_version": phase,
        }

    ledger = _EventLedger()
    ledger.events.extend([committed(1), committed(2)])
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "actors": ["w1"],
                    "context_policy": "private",
                    "trigger": {"phase": 1, "repeat": False},
                }
            ]
        ),
        ExecutorRegistry({"p": Executor()}),
        ContextEngine({"private": {"allow": []}}),
        persistence=ledger,
        state_store=StateStore({"x": int}, {"x": 0}),
        status_provider=lambda: "lease_lost",
    )
    controller.run("r", phase_limit=3, seed=1, terminal_phase=2)
    assert controller.lease_lost
    assert controller._stopped_at == (2, True)
    # The declared horizon is committed and nothing is ready, so the work is done
    # and the service settles the run instead of leaving it running.
    assert controller.stopped_with_work_remaining is False


def test_an_unknown_export_mode_is_refused(tmp_path: Path) -> None:
    """The branch compares exact strings, so "REPRODUCIBILITY" exported exploration
    content in a bundle whose manifest recorded the caller's spelling (N2')."""
    from fastapi.testclient import TestClient

    from genesis.app import _export_mode, create_app

    assert _export_mode(None) == "exploration"
    assert _export_mode("reproducibility") == "reproducibility"
    with pytest.raises(ValueError, match="VALIDATION_ERROR"):
        _export_mode("REPRODUCIBILITY")

    workspace = tmp_path / "ws"
    service = GenesisService(workspace)
    try:
        with pytest.raises(ValueError, match="EXPORT_MODE"):
            service.export_run("missing-run", "exports/x", mode="Reproducibility")
    finally:
        service.close()
    client = TestClient(create_app(workspace))
    response = client.post(
        "/exports", json={"run_id": "r", "output": "exports/x", "mode": "REPRODUCIBILITY"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_a_patch_form_event_is_reconstructed_for_readers() -> None:
    """The evidence tools decoded `payload_ref` alone, so a patch-form event -- most
    events of a long run -- came out with no process id (N3')."""
    import json

    from genesis.persistence import PersistenceCoordinator

    workspace = Path(__file__).resolve().parents[1] / "genesis-workspace"
    if not (workspace / ".genesis" / "genesis.db").is_file():
        pytest.skip("genesis-workspace/ is local-only")
    store = PersistenceCoordinator(
        workspace / ".genesis" / "genesis.db",
        workspace / ".genesis" / "objects",
        read_only=True,
    )
    try:
        row = store.connection.execute(
            "SELECT run_id, payload_ref FROM events WHERE form = 'patch' LIMIT 1"
        ).fetchone()
        assert row is not None, "the reference workspace records no patch-form event"
        run_id, payload_ref = str(row[0]), str(row[1])
        # Decoded on its own, a patch row is the delta it stores, not the event.
        raw = json.loads(store._read_object(payload_ref))
        assert "process_id" not in raw
        # Reconstructed, it is the event again.
        events = [event for event in store.iter_events(run_id) if event.get("process_id")]
        assert events, "no event of the run reconstructed with a process id"
    finally:
        store.close()

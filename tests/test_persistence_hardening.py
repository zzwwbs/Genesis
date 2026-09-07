"""Regression tests for durable metadata and atomic commit identity."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from genesis.persistence import ObjectStore, PersistenceCoordinator


def coordinator(tmp_path):
    return PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")


def seed_legacy_commit(tmp_path):
    event = {"event_id": "event-1", "run_id": "run-1", "kind": "completed"}
    state = {"run_id": "run-1", "state_version": 1, "payload": b"state"}
    artifacts = [{"artifact_id": "artifact-1", "run_id": "run-1", "payload": b"artifact"}]
    store = ObjectStore(tmp_path / "objects")
    event_bytes = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    event_ref = store.put(event_bytes, "application/json")
    state_ref = store.put(state["payload"])
    artifact_ref = store.put(artifacts[0]["payload"])
    checkpoint_payload = json.dumps({"phase": 1}, sort_keys=True, separators=(",", ":")).encode()
    checkpoint_ref = store.put(checkpoint_payload, "application/json")
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.executescript(
        """
        CREATE TABLE objects (
            digest TEXT PRIMARY KEY, media_type TEXT NOT NULL, size INTEGER NOT NULL
        );
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
            payload_ref TEXT NOT NULL REFERENCES objects(digest), event_hash TEXT NOT NULL
        );
        CREATE TABLE states (
            run_id TEXT NOT NULL, state_version INTEGER NOT NULL,
            payload_ref TEXT NOT NULL REFERENCES objects(digest), PRIMARY KEY(run_id, state_version)
        );
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
            payload_ref TEXT NOT NULL REFERENCES objects(digest)
        );
        CREATE TABLE checkpoints (
            checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            payload_ref TEXT NOT NULL REFERENCES objects(digest), integrity_hash TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO objects VALUES (?, ?, ?)",
        [
            (ref.digest, ref.media_type, ref.size)
            for ref in (event_ref, state_ref, artifact_ref, checkpoint_ref)
        ],
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (
            event["event_id"],
            event["run_id"],
            event["kind"],
            event_ref.digest,
            hashlib.sha256(event_bytes).hexdigest(),
        ),
    )
    connection.execute(
        "INSERT INTO states VALUES (?, ?, ?)",
        (state["run_id"], state["state_version"], state_ref.digest),
    )
    connection.execute(
        "INSERT INTO artifacts VALUES (?, ?, ?)",
        (artifacts[0]["artifact_id"], artifacts[0]["run_id"], artifact_ref.digest),
    )
    connection.execute(
        """INSERT INTO checkpoints(
               checkpoint_id, run_id, payload_ref, integrity_hash
           ) VALUES (?, ?, ?, ?)""",
        (1, "run-1", checkpoint_ref.digest, hashlib.sha256(checkpoint_payload).hexdigest()),
    )
    connection.commit()
    connection.close()
    return event, state, artifacts


def add_second_legacy_commit(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    event = {"event_id": "event-2", "run_id": "run-1", "kind": "completed"}
    state = {"run_id": "run-1", "state_version": 2, "payload": b"state-2"}
    event_bytes = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    event_ref = store.put(event_bytes, "application/json")
    state_ref = store.put(state["payload"])
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.executemany(
        "INSERT OR IGNORE INTO objects VALUES (?, ?, ?)",
        [(ref.digest, ref.media_type, ref.size) for ref in (event_ref, state_ref)],
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (
            event["event_id"],
            event["run_id"],
            event["kind"],
            event_ref.digest,
            hashlib.sha256(event_bytes).hexdigest(),
        ),
    )
    connection.execute(
        "INSERT INTO states VALUES (?, ?, ?)",
        (state["run_id"], state["state_version"], state_ref.digest),
    )
    connection.commit()
    connection.close()


def seed_immediate_prior_v0(tmp_path):
    event, state, artifacts = seed_legacy_commit(tmp_path)
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.execute("ALTER TABLE events ADD COLUMN commit_hash TEXT")
    connection.execute(
        "UPDATE events SET commit_hash = ? WHERE event_id = ?",
        ("c" * 64, event["event_id"]),
    )
    connection.executescript(
        """
        CREATE TABLE studies (
            study_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL
        );
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, status TEXT NOT NULL, version INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );
        """
    )
    study = {"id": "study-one", "status": "draft", "title": "Legacy", "version": 0}
    run = {"id": "run-one", "study_id": "study-one", "status": "paused", "version": 2}
    connection.execute(
        "INSERT INTO studies(study_id, payload_json) VALUES (?, ?)",
        (study["id"], json.dumps(study, sort_keys=True, separators=(",", ":"))),
    )
    connection.execute(
        """INSERT INTO runs(run_id, status, version, payload_json)
           VALUES (?, ?, ?, ?)""",
        (
            run["id"],
            run["status"],
            run["version"],
            json.dumps(run, sort_keys=True, separators=(",", ":")),
        ),
    )
    connection.commit()
    connection.close()
    return event, state, artifacts, study, run


def test_commit_rejects_artifact_from_a_different_run_before_writing(tmp_path):
    persistence = coordinator(tmp_path)

    with pytest.raises(ValueError, match="RUN_ID"):
        persistence.commit_process_result(
            {"event_id": "event-1", "run_id": "run-1", "kind": "completed"},
            {"run_id": "run-1", "state_version": 1, "payload": b"state"},
            [{"artifact_id": "artifact-1", "run_id": "run-2", "payload": b"artifact"}],
        )

    assert persistence.count("events") == 0
    assert persistence.count("states") == 0
    assert persistence.count("artifacts") == 0


@pytest.mark.parametrize("changed_part", ["state", "artifact_payload", "artifact_identity"])
def test_idempotent_retry_requires_the_entire_commit_to_match(tmp_path, changed_part):
    persistence = coordinator(tmp_path)
    event = {"event_id": "event-1", "run_id": "run-1", "kind": "completed"}
    state = {"run_id": "run-1", "state_version": 1, "payload": b"state"}
    artifacts = [{"artifact_id": "artifact-1", "run_id": "run-1", "payload": b"artifact"}]
    persistence.commit_process_result(event, state, artifacts)

    retry_state = dict(state)
    retry_artifacts = [dict(artifacts[0])]
    if changed_part == "state":
        retry_state["payload"] = b"different-state"
    elif changed_part == "artifact_payload":
        retry_artifacts[0]["payload"] = b"different-artifact"
    else:
        retry_artifacts[0]["artifact_id"] = "artifact-2"

    with pytest.raises(ValueError, match="IDEMPOTENCY"):
        persistence.commit_process_result(event, retry_state, retry_artifacts)


def test_identical_full_commit_retry_is_a_noop_even_after_reopen(tmp_path):
    event = {"event_id": "event-1", "run_id": "run-1", "kind": "completed"}
    state = {"run_id": "run-1", "state_version": 1, "payload": b"state"}
    artifacts = [{"artifact_id": "artifact-1", "run_id": "run-1", "payload": b"artifact"}]
    persistence = coordinator(tmp_path)
    persistence.commit_process_result(event, state, artifacts)
    persistence.close()

    reopened = coordinator(tmp_path)
    reopened.commit_process_result(event, state, artifacts)

    assert reopened.count("events") == 1
    assert reopened.count("states") == 1
    assert reopened.count("artifacts") == 1


def test_identical_legacy_retry_is_validated_and_backfills_commit_identity(tmp_path):
    event, state, artifacts = seed_legacy_commit(tmp_path)
    persistence = coordinator(tmp_path)

    persistence.commit_process_result(event, state, artifacts)

    stored = persistence.connection.execute(
        "SELECT commit_hash FROM events WHERE event_id = ?", (event["event_id"],)
    ).fetchone()
    assert stored is not None and stored[0] == persistence._commit_hash(event, state, artifacts)
    assert persistence.count("events") == persistence.count("states") == 1
    assert persistence.count("artifacts") == 1


def test_exact_populated_v0_schema_upgrades_to_strict_v2_without_data_loss(tmp_path):
    event, state, artifacts = seed_legacy_commit(tmp_path)

    persistence = coordinator(tmp_path)

    assert persistence.connection.execute("PRAGMA user_version").fetchone() == (7,)
    assert persistence.count("events") == 1
    assert persistence.count("states") == 1
    assert persistence.count("artifacts") == 1
    assert persistence.count("checkpoints") == 1
    assert persistence.restore_checkpoint("1") == {"phase": 1}
    assert persistence.connection.execute(
        "SELECT event_id, run_id, kind FROM events"
    ).fetchone() == (event["event_id"], event["run_id"], event["kind"])
    assert persistence.connection.execute(
        "SELECT run_id, state_version FROM states"
    ).fetchone() == (state["run_id"], state["state_version"])
    assert persistence.connection.execute(
        "SELECT artifact_id, run_id FROM artifacts"
    ).fetchone() == (artifacts[0]["artifact_id"], artifacts[0]["run_id"])
    assert persistence.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    for table, column in (
        ("objects", "digest"),
        ("events", "event_id"),
        ("artifacts", "artifact_id"),
    ):
        shape = {
            row[1]: (row[3], row[5])
            for row in persistence.connection.execute(f"PRAGMA table_info({table})")
        }
        assert shape[column] == (1, 1)


def test_populated_immediate_prior_seven_table_v0_upgrades_without_data_loss(tmp_path):
    event, state, artifacts, study, run = seed_immediate_prior_v0(tmp_path)

    persistence = coordinator(tmp_path)

    assert persistence.connection.execute("PRAGMA user_version").fetchone() == (7,)
    assert persistence.connection.execute(
        "SELECT commit_hash FROM events WHERE event_id = ?", (event["event_id"],)
    ).fetchone() == ("c" * 64,)
    assert persistence.connection.execute(
        "SELECT run_id, state_version FROM states"
    ).fetchone() == (state["run_id"], state["state_version"])
    assert persistence.connection.execute(
        "SELECT artifact_id, run_id FROM artifacts"
    ).fetchone() == (artifacts[0]["artifact_id"], artifacts[0]["run_id"])
    assert persistence.get_study(study["id"]) == study
    assert persistence.get_run(run["id"]) == run
    assert persistence.restore_checkpoint("1") == {"phase": 1}
    assert persistence.connection.execute("PRAGMA foreign_key_check").fetchall() == []
    for table, column in (
        ("objects", "digest"),
        ("events", "event_id"),
        ("artifacts", "artifact_id"),
        ("studies", "study_id"),
        ("runs", "run_id"),
    ):
        shape = {
            row[1]: (row[3], row[5])
            for row in persistence.connection.execute(f"PRAGMA table_info({table})")
        }
        assert shape[column] == (1, 1)


@pytest.mark.parametrize(
    "changed_part",
    [
        "state",
        "state_version",
        "state_metadata",
        "artifact",
        "artifact_identity",
        "artifact_metadata",
    ],
)
def test_mismatched_legacy_retry_is_rejected_without_backfill(tmp_path, changed_part):
    event, state, artifacts = seed_legacy_commit(tmp_path)
    persistence = coordinator(tmp_path)
    if changed_part == "state":
        state["payload"] = b"different-state"
    elif changed_part == "state_version":
        state["state_version"] = 2
    elif changed_part == "state_metadata":
        state["unverifiable"] = True
    elif changed_part == "artifact":
        artifacts[0]["payload"] = b"different-artifact"
    elif changed_part == "artifact_identity":
        artifacts[0]["artifact_id"] = "artifact-2"
    else:
        artifacts[0]["unverifiable"] = True

    with pytest.raises(ValueError, match="IDEMPOTENCY"):
        persistence.commit_process_result(event, state, artifacts)

    assert persistence.connection.execute(
        "SELECT commit_hash FROM events WHERE event_id = ?", (event["event_id"],)
    ).fetchone() == (None,)


def test_legacy_retry_with_multiple_events_and_states_is_rejected_as_ambiguous(tmp_path):
    event, state, artifacts = seed_legacy_commit(tmp_path)
    add_second_legacy_commit(tmp_path)
    persistence = coordinator(tmp_path)

    with pytest.raises(ValueError, match="IDEMPOTENCY.*ambiguous"):
        persistence.commit_process_result(event, state, artifacts)

    assert persistence.connection.execute(
        "SELECT commit_hash FROM events WHERE event_id = ?", (event["event_id"],)
    ).fetchone() == (None,)


def test_studies_are_durable_and_have_create_list_get_operations(tmp_path):
    persistence = coordinator(tmp_path)
    created = persistence.create_study({"id": "study-one", "title": "Study One"})

    assert created == {
        "id": "study-one",
        "title": "Study One",
        "status": "draft",
        "version": 0,
    }
    assert persistence.get_study("study-one") == created
    assert persistence.list_studies() == [created]
    with pytest.raises(ValueError, match="ALREADY_EXISTS"):
        persistence.create_study({"id": "study-one", "title": "Duplicate"})
    persistence.close()

    reopened = coordinator(tmp_path)
    assert reopened.get_study("study-one") == created
    assert reopened.list_studies() == [created]
    with pytest.raises(KeyError):
        reopened.get_study("missing")


def test_runs_are_durable_and_have_create_list_get_operations(tmp_path):
    persistence = coordinator(tmp_path)
    created = persistence.create_run(
        {"id": "run-one", "study_id": "study-one", "condition": "baseline"}
    )

    assert created == {
        "id": "run-one",
        "study_id": "study-one",
        "condition": "baseline",
        "status": "created",
        "version": 0,
    }
    assert persistence.get_run("run-one") == created
    assert persistence.list_runs() == [created]
    with pytest.raises(ValueError, match="ALREADY_EXISTS"):
        persistence.create_run({"id": "run-one"})
    persistence.close()

    reopened = coordinator(tmp_path)
    assert reopened.get_run("run-one") == created
    assert reopened.list_runs() == [created]
    with pytest.raises(KeyError):
        reopened.get_run("missing")


def test_create_run_rejects_conflicting_id_aliases(tmp_path):
    persistence = coordinator(tmp_path)

    with pytest.raises(ValueError, match="RUN_ID"):
        persistence.create_run({"id": "run-one", "run_id": "run-two"})

    assert persistence.list_runs() == []


@pytest.mark.parametrize("status", ["running", "paused", "completed", "failed", "cancelled"])
def test_new_runs_must_start_in_created_status(tmp_path, status):
    persistence = coordinator(tmp_path)

    with pytest.raises(ValueError, match="RUN_STATUS"):
        persistence.create_run({"id": "run-one", "status": status})

    assert persistence.list_runs() == []


def test_run_lifecycle_transitions_are_legal_and_version_checked(tmp_path):
    persistence = coordinator(tmp_path)
    persistence.create_run({"id": "run-one"})

    running = persistence.transition_run("run-one", "running", expected_version=0)
    paused = persistence.transition_run("run-one", "paused", expected_version=1)
    resumed = persistence.transition_run("run-one", "running", expected_version=2)
    completed = persistence.transition_run("run-one", "completed", expected_version=3)

    assert [running["version"], paused["version"], resumed["version"], completed["version"]] == [
        1,
        2,
        3,
        4,
    ]
    assert completed["status"] == "completed"
    with pytest.raises(ValueError, match="RUN_TRANSITION"):
        persistence.transition_run("run-one", "running", expected_version=4)


def test_run_transition_rejects_stale_version_and_preserves_record(tmp_path):
    persistence = coordinator(tmp_path)
    created = persistence.create_run({"id": "run-one"})

    with pytest.raises(ValueError, match="EXPECTED_VERSION"):
        persistence.transition_run("run-one", "running", expected_version=7)

    assert persistence.get_run("run-one") == created


@pytest.mark.parametrize(
    ("initial", "target"),
    [
        ("created", "paused"),
        ("created", "completed"),
        ("paused", "completed"),
        ("cancelled", "running"),
        ("completed", "cancelled"),
        ("failed", "running"),
    ],
)
def test_run_transition_rejects_illegal_state_changes(tmp_path, initial, target):
    persistence = coordinator(tmp_path)
    persistence.create_run({"id": "run-one"})
    version = 0
    if initial == "paused":
        persistence.transition_run("run-one", "running", expected_version=version)
        version += 1
        persistence.transition_run("run-one", "paused", expected_version=version)
        version += 1
    elif initial == "cancelled":
        persistence.transition_run("run-one", "cancelled", expected_version=version)
        version += 1
    elif initial == "completed":
        persistence.transition_run("run-one", "running", expected_version=version)
        version += 1
        persistence.transition_run("run-one", "completed", expected_version=version)
        version += 1
    elif initial == "failed":
        persistence.transition_run("run-one", "failed", expected_version=version)
        version += 1

    with pytest.raises(ValueError, match="RUN_TRANSITION"):
        persistence.transition_run("run-one", target, expected_version=version)


def test_object_store_rejects_corrupted_existing_cas_file(tmp_path):
    store = ObjectStore(tmp_path / "objects")
    ref = store.put(b"expected")
    ref.path.write_bytes(b"corrupt!")

    with pytest.raises(ValueError, match="integrity"):
        store.put(b"expected")


def test_concurrent_transitions_on_shared_coordinator_are_serialized(tmp_path):
    persistence = coordinator(tmp_path)
    persistence.create_run({"id": "run-one"})

    def transition():
        try:
            return persistence.transition_run("run-one", "running", expected_version=0)
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: transition(), range(2)))

    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum("EXPECTED_VERSION" in result for result in results if isinstance(result, str)) == 1
    assert persistence.get_run("run-one")["version"] == 1


def test_concurrent_commits_on_shared_coordinator_are_serialized(tmp_path):
    persistence = coordinator(tmp_path)

    def commit(index):
        try:
            persistence.commit_process_result(
                {"event_id": f"event-{index}", "run_id": "run-one", "kind": "completed"},
                {"run_id": "run-one", "state_version": 1, "payload": f"state-{index}".encode()},
                [],
                expected_state_version=0,
            )
            return "committed"
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(commit, range(2)))

    assert results.count("committed") == 1
    assert sum("STATE_VERSION" in result for result in results) == 1
    assert persistence.count("events") == persistence.count("states") == 1


def test_database_object_metadata_conflict_is_detected_across_live_coordinators(tmp_path):
    first = coordinator(tmp_path)
    second = coordinator(tmp_path)
    checkpoint = {"score": 1}
    encoded = json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode()
    first.commit_process_result(
        {"event_id": "event-1", "run_id": "run-one", "kind": "completed"},
        {"run_id": "run-one", "state_version": 1, "payload": encoded},
        [],
    )

    with pytest.raises(ValueError, match="metadata"):
        second.create_checkpoint("run-one", checkpoint)

    assert second.count("checkpoints") == 0


def test_database_schema_uses_explicit_current_user_version(tmp_path):
    persistence = coordinator(tmp_path)

    assert persistence.connection.execute("PRAGMA user_version").fetchone() == (7,)


def test_database_rejects_unsupported_future_schema_version(tmp_path):
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(ValueError, match="SCHEMA_VERSION"):
        coordinator(tmp_path)


def test_database_validates_shape_for_declared_schema_version(tmp_path):
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.execute("CREATE TABLE events(event_id TEXT PRIMARY KEY)")
    connection.execute("PRAGMA user_version = 2")
    connection.close()

    with pytest.raises(ValueError, match="SCHEMA_SHAPE"):
        coordinator(tmp_path)


def test_invalid_v0_database_is_rejected_without_partial_migration(tmp_path):
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.execute("CREATE TABLE events(event_id INTEGER)")
    before = connection.execute("PRAGMA table_info(events)").fetchall()
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
    connection.close()

    with pytest.raises(ValueError, match="SCHEMA_SHAPE"):
        coordinator(tmp_path)

    reopened = sqlite3.connect(tmp_path / "genesis.db")
    assert reopened.execute("PRAGMA user_version").fetchone() == (0,)
    assert reopened.execute("PRAGMA journal_mode").fetchone() == journal_mode
    assert reopened.execute("PRAGMA table_info(events)").fetchall() == before
    assert {
        row[0]
        for row in reopened.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    } == {"events"}
    reopened.close()


def test_v2_schema_rejects_unexpected_columns(tmp_path):
    persistence = coordinator(tmp_path)
    persistence.close()
    connection = sqlite3.connect(tmp_path / "genesis.db")
    connection.execute("ALTER TABLE events ADD COLUMN hidden_drift TEXT")
    connection.close()

    with pytest.raises(ValueError, match="SCHEMA_SHAPE.*unexpected"):
        coordinator(tmp_path)


def recreate_table_without_constraints(database, table, definition):
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=OFF")
    connection.execute(f"ALTER TABLE {table} RENAME TO malformed_source")
    connection.execute(definition)
    connection.execute("DROP TABLE malformed_source")
    connection.commit()
    connection.close()


@pytest.mark.parametrize("malformation", ["event_pk", "artifact_fk", "state_type"])
def test_v2_schema_rejects_malformed_keys_types_and_foreign_keys(tmp_path, malformation):
    persistence = coordinator(tmp_path)
    persistence.close()
    database = tmp_path / "genesis.db"
    if malformation == "event_pk":
        recreate_table_without_constraints(
            database,
            "events",
            """CREATE TABLE events (
                event_id TEXT NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest),
                event_hash TEXT NOT NULL, commit_hash TEXT
            )""",
        )
    elif malformation == "artifact_fk":
        recreate_table_without_constraints(
            database,
            "artifacts",
            """CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
                payload_ref TEXT NOT NULL
            )""",
        )
    else:
        recreate_table_without_constraints(
            database,
            "states",
            """CREATE TABLE states (
                run_id TEXT NOT NULL, state_version TEXT NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest),
                PRIMARY KEY(run_id, state_version)
            )""",
        )

    with pytest.raises(ValueError, match="SCHEMA_SHAPE"):
        coordinator(tmp_path)


def test_binary_state_and_artifact_reads_preserve_bytes(tmp_path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    persistence.commit_process_result(
        {"event_id": "event-1", "run_id": "run-1", "kind": "completed"},
        {"run_id": "run-1", "state_version": 1, "payload": b"raw-state"},
        [{"artifact_id": "artifact-1", "run_id": "run-1", "payload": b"raw-artifact"}],
    )
    assert persistence.latest_state("run-1")[1] == b"raw-state"
    assert persistence.list_artifacts("run-1")[0]["payload"] == b"raw-artifact"
    with pytest.raises(ValueError, match="STATE_FORMAT"):
        persistence.latest_json_state("run-1")
    persistence.close()

"""Embedded SQLite persistence and content-addressed object storage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, cast

_SCHEMA_VERSION = 7
_RUN_STATUSES = {"created", "running", "paused", "completed", "failed", "cancelled"}
_RUN_TRANSITIONS = {
    "created": {"running", "failed", "cancelled"},
    "running": {"paused", "completed", "failed", "cancelled"},
    "paused": {"running", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


@dataclass(frozen=True)
class ObjectRef:
    digest: str
    media_type: str
    size: int
    path: Path | None = None


class ObjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[str, tuple[str, int]] = {}

    def put(
        self,
        payload: bytes,
        media_type: str = "application/octet-stream",
        size: int | None = None,
    ) -> ObjectRef:
        if not isinstance(payload, bytes):
            raise ValueError("payload must be bytes")
        if size is not None and size != len(payload):
            raise ValueError("payload size does not match size")
        if not media_type or "/" not in media_type:
            raise ValueError("invalid media type")
        digest = hashlib.sha256(payload).hexdigest()
        path = self.root / digest[:2] / digest[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if (
                path.stat().st_size != len(payload)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError(f"integrity check failed for existing object {digest}")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".object-", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        existing = self._metadata.get(digest)
        if existing and existing != (media_type, len(payload)):
            raise ValueError("object metadata conflicts with existing digest")
        self._metadata[digest] = (media_type, len(payload))
        return ObjectRef(digest, media_type, len(payload), path)

    def cleanup_orphans(self) -> int:
        removed = 0
        for path in self.root.rglob("*.tmp"):
            if path.is_file():
                path.unlink()
                removed += 1
        for path in self.root.rglob(".object-*"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def collect_garbage(self, referenced: set[str]) -> int:
        removed = 0
        for path in self.root.glob("??/*"):
            digest = path.parent.name + path.name
            if digest not in referenced and path.is_file():
                path.unlink()
                self._metadata.pop(digest, None)
                removed += 1
        return removed

    def get(self, ref: ObjectRef) -> bytes:
        self.verify(ref)
        return (ref.path or self.root / ref.digest[:2] / ref.digest[2:]).read_bytes()

    def verify(self, ref: ObjectRef) -> None:
        path = ref.path or self.root / ref.digest[:2] / ref.digest[2:]
        if (
            not path.is_file()
            or path.stat().st_size != ref.size
            or hashlib.sha256(path.read_bytes()).hexdigest() != ref.digest
        ):
            raise ValueError(f"integrity check failed for object {ref.digest}")


class PersistenceCoordinator:
    def __init__(self, database: str | Path, objects: str | Path):
        self._lock = RLock()
        self.object_store = ObjectStore(objects)
        self.connection = sqlite3.connect(database, isolation_level=None, check_same_thread=False)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._migrate_database()
        self._validate_schema()
        self.connection.execute("PRAGMA journal_mode=WAL")
        for digest, media_type, size in self.connection.execute(
            "SELECT digest, media_type, size FROM objects"
        ):
            self.object_store._metadata[digest] = (media_type, size)

    def _migrate_database(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"SCHEMA_VERSION: database version {version} is newer than supported "
                f"version {_SCHEMA_VERSION}"
            )
        tables = self._user_tables()
        if version == 0 and tables:
            v0_layout = self._validate_v0_schema(tables)
            if v0_layout == "immediate":
                self.connection.execute("BEGIN IMMEDIATE")
                try:
                    self._rebuild_v0_tables(immediate=True)
                    self._validate_schema_version(2)
                    self.connection.execute("PRAGMA user_version = 2")
                    self.connection.execute("COMMIT")
                except Exception:
                    if self.connection.in_transaction:
                        self.connection.execute("ROLLBACK")
                    raise
                version = 2
        elif version > 0:
            self._validate_schema_version(version)
        while version < _SCHEMA_VERSION:
            target = version + 1
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                if target == 1:
                    self._migration_1()
                elif target == 2:
                    self._migration_2()
                elif target == 3:
                    self._migration_3()
                elif target == 4:
                    self._migration_4()
                elif target == 5:
                    self._migration_5()
                elif target == 6:
                    self._migration_6()
                elif target == 7:
                    self._migration_7()
                self._validate_schema_version(target)
                self.connection.execute(f"PRAGMA user_version = {target}")
                self.connection.execute("COMMIT")
            except Exception:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise
            version = target

    def _validate_v0_schema(self, tables: set[str]) -> str:
        early_tables = {"objects", "events", "states", "artifacts", "checkpoints"}
        immediate_tables = early_tables | {"studies", "runs"}
        if tables == early_tables:
            self._validate_schema_version(0)
            return "early"
        if tables == immediate_tables:
            self._validate_schema_version(0, immediate_v0=True)
            return "immediate"
        unexpected = sorted(tables - immediate_tables)
        missing = sorted(early_tables - tables)
        raise ValueError(
            f"SCHEMA_SHAPE: unsupported v0 table layout; missing={missing}, unexpected={unexpected}"
        )

    def _user_tables(self) -> set[str]:
        return {
            row[0]
            for row in self.connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"""
            )
        }

    def _migration_1(self) -> None:
        if self._user_tables():
            self._rebuild_v0_tables()
            return
        self._create_v1_tables()

    def _create_v1_tables(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS objects (
                digest TEXT PRIMARY KEY NOT NULL, media_type TEXT NOT NULL,
                size INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest), event_hash TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS states (
                run_id TEXT NOT NULL, state_version INTEGER NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest),
                PRIMARY KEY(run_id, state_version)
            )""",
            """CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest)
            )""",
            """CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                payload_ref TEXT NOT NULL REFERENCES objects(digest), integrity_hash TEXT NOT NULL
            )""",
        )
        for statement in statements:
            self.connection.execute(statement)

    def _rebuild_v0_tables(self, immediate: bool = False) -> None:
        self.connection.execute("ALTER TABLE objects RENAME TO objects_v0")
        self.connection.execute(
            """CREATE TABLE objects (
                digest TEXT PRIMARY KEY NOT NULL, media_type TEXT NOT NULL,
                size INTEGER NOT NULL
            )"""
        )
        self.connection.execute(
            """INSERT INTO objects(digest, media_type, size)
               SELECT digest, media_type, size FROM objects_v0"""
        )
        event_commit_column = ", commit_hash TEXT" if immediate else ""
        event_columns = (
            "event_id, run_id, kind, payload_ref, event_hash, commit_hash"
            if immediate
            else "event_id, run_id, kind, payload_ref, event_hash"
        )
        table_rebuilds = [
            (
                "events",
                f"""CREATE TABLE events (
                    event_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                    payload_ref TEXT NOT NULL REFERENCES objects(digest), event_hash TEXT NOT NULL
                    {event_commit_column}
                )""",
                event_columns,
            ),
            (
                "states",
                """CREATE TABLE states (
                    run_id TEXT NOT NULL, state_version INTEGER NOT NULL,
                    payload_ref TEXT NOT NULL REFERENCES objects(digest),
                    PRIMARY KEY(run_id, state_version)
                )""",
                "run_id, state_version, payload_ref",
            ),
            (
                "artifacts",
                """CREATE TABLE artifacts (
                    artifact_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
                    payload_ref TEXT NOT NULL REFERENCES objects(digest)
                )""",
                "artifact_id, run_id, payload_ref",
            ),
            (
                "checkpoints",
                """CREATE TABLE checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    payload_ref TEXT NOT NULL REFERENCES objects(digest),
                    integrity_hash TEXT NOT NULL
                )""",
                "checkpoint_id, run_id, payload_ref, integrity_hash",
            ),
        ]
        if immediate:
            table_rebuilds.extend(
                [
                    (
                        "studies",
                        """CREATE TABLE studies (
                            study_id TEXT PRIMARY KEY NOT NULL, payload_json TEXT NOT NULL
                        )""",
                        "study_id, payload_json",
                    ),
                    (
                        "runs",
                        """CREATE TABLE runs (
                            run_id TEXT PRIMARY KEY NOT NULL, status TEXT NOT NULL,
                            version INTEGER NOT NULL, payload_json TEXT NOT NULL
                        )""",
                        "run_id, status, version, payload_json",
                    ),
                ]
            )
        for table, create_statement, columns in table_rebuilds:
            legacy_table = f"{table}_v0"
            self.connection.execute(f"ALTER TABLE {table} RENAME TO {legacy_table}")
            self.connection.execute(create_statement)
            self.connection.execute(
                f"INSERT INTO {table}({columns}) SELECT {columns} FROM {legacy_table}"
            )
            self.connection.execute(f"DROP TABLE {legacy_table}")
        self.connection.execute("DROP TABLE objects_v0")

    def _migration_2(self) -> None:
        event_columns = self._table_columns("events")
        if "commit_hash" not in event_columns:
            self.connection.execute("ALTER TABLE events ADD COLUMN commit_hash TEXT")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS studies (
                study_id TEXT PRIMARY KEY NOT NULL, payload_json TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            )"""
        )

    def _migration_3(self) -> None:
        """AW-15: study lifecycle entities — package versions, experiments, instances."""
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS package_versions (
                study_id TEXT NOT NULL, version INTEGER NOT NULL,
                content_hash TEXT NOT NULL, parent_version INTEGER,
                status TEXT NOT NULL, payload_json TEXT NOT NULL,
                PRIMARY KEY(study_id, version)
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY NOT NULL, study_id TEXT NOT NULL,
                build_ref TEXT NOT NULL, protocol_hash TEXT NOT NULL,
                created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS process_instances (
                instance_id TEXT PRIMARY KEY NOT NULL, run_id TEXT NOT NULL,
                process_id TEXT NOT NULL, phase INTEGER NOT NULL,
                attempt INTEGER NOT NULL, status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )"""
        )

    def _migration_4(self) -> None:
        """Review finding 3/4: build records, idempotency, and lifecycle foreign keys."""
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS study_builds (
                build_hash TEXT PRIMARY KEY NOT NULL,
                study_id TEXT NOT NULL REFERENCES studies(study_id),
                package_version INTEGER NOT NULL,
                package_content_hash TEXT NOT NULL,
                compiler_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (study_id, package_version)
                    REFERENCES package_versions(study_id, version)
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS idempotency (
                idempotency_key TEXT PRIMARY KEY NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        # Runs gain an optional relational experiment reference (finding 4).
        if "experiment_id" not in self._table_columns("runs"):
            self.connection.execute("ALTER TABLE runs ADD COLUMN experiment_id TEXT")

    def _migration_5(self) -> None:
        """Finding 4 residual: experiments gain a relational build reference."""
        if "build_hash" not in self._table_columns("experiments"):
            self.connection.execute(
                """ALTER TABLE experiments ADD COLUMN build_hash TEXT
                   REFERENCES study_builds(build_hash)"""
            )

    def _migration_6(self) -> None:
        """Finding 4: rebuild runs and process_instances with lifecycle foreign keys.

        The migration runner wraps this call in one transaction.
        """
        self.connection.execute(
            """CREATE TABLE runs_v6 (
                run_id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                experiment_id TEXT,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )"""
        )
        self.connection.execute(
            """INSERT INTO runs_v6(run_id, status, version, experiment_id, payload_json)
               SELECT run_id, status, version, experiment_id, payload_json FROM runs"""
        )
        self.connection.execute("DROP TABLE runs")
        self.connection.execute("ALTER TABLE runs_v6 RENAME TO runs")
        self.connection.execute(
            """CREATE TABLE process_instances_v6 (
                instance_id TEXT PRIMARY KEY NOT NULL,
                run_id TEXT NOT NULL,
                process_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )"""
        )
        self.connection.execute(
            """INSERT INTO process_instances_v6(
                   instance_id, run_id, process_id, phase, attempt, status, payload_json
               ) SELECT instance_id, run_id, process_id, phase, attempt, status, payload_json
                 FROM process_instances"""
        )
        self.connection.execute("DROP TABLE process_instances")
        self.connection.execute("ALTER TABLE process_instances_v6 RENAME TO process_instances")

    def _migration_7(self) -> None:
        """Review P1: package versions retain immutable content-addressed snapshots."""
        if "snapshot_digest" not in self._table_columns("package_versions"):
            self.connection.execute("ALTER TABLE package_versions ADD COLUMN snapshot_digest TEXT")

    def _validate_schema(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version != _SCHEMA_VERSION:
            raise ValueError(
                f"SCHEMA_VERSION: expected {_SCHEMA_VERSION} after migration, found {version}"
            )
        self._validate_schema_version(version)

    def _validate_schema_version(self, version: int, immediate_v0: bool = False) -> None:
        primary_key_not_null = 0 if version == 0 else 1
        specs: dict[str, dict[str, tuple[str, int, int]]] = {
            "objects": {
                "digest": ("TEXT", primary_key_not_null, 1),
                "media_type": ("TEXT", 1, 0),
                "size": ("INTEGER", 1, 0),
            },
            "events": {
                "event_id": ("TEXT", primary_key_not_null, 1),
                "run_id": ("TEXT", 1, 0),
                "kind": ("TEXT", 1, 0),
                "payload_ref": ("TEXT", 1, 0),
                "event_hash": ("TEXT", 1, 0),
            },
            "states": {
                "run_id": ("TEXT", 1, 1),
                "state_version": ("INTEGER", 1, 2),
                "payload_ref": ("TEXT", 1, 0),
            },
            "artifacts": {
                "artifact_id": ("TEXT", primary_key_not_null, 1),
                "run_id": ("TEXT", 1, 0),
                "payload_ref": ("TEXT", 1, 0),
            },
            "checkpoints": {
                "checkpoint_id": ("INTEGER", 0, 1),
                "run_id": ("TEXT", 1, 0),
                "payload_ref": ("TEXT", 1, 0),
                "integrity_hash": ("TEXT", 1, 0),
            },
        }
        if version >= 4:
            specs["study_builds"] = {
                "build_hash": ("TEXT", primary_key_not_null, 1),
                "study_id": ("TEXT", 1, 0),
                "package_version": ("INTEGER", 1, 0),
                "package_content_hash": ("TEXT", 1, 0),
                "compiler_version": ("TEXT", 1, 0),
                "created_at": ("TEXT", 1, 0),
                "payload_json": ("TEXT", 1, 0),
            }
            specs["idempotency"] = {
                "idempotency_key": ("TEXT", primary_key_not_null, 1),
                "response_json": ("TEXT", 1, 0),
                "created_at": ("TEXT", 1, 0),
            }
        if version >= 3:
            specs["package_versions"] = {
                "study_id": ("TEXT", 1, 1),
                "version": ("INTEGER", 1, 2),
                "content_hash": ("TEXT", 1, 0),
                "parent_version": ("INTEGER", 0, 0),
                "status": ("TEXT", 1, 0),
                "payload_json": ("TEXT", 1, 0),
            }
            specs["experiments"] = {
                "experiment_id": ("TEXT", primary_key_not_null, 1),
                "study_id": ("TEXT", 1, 0),
                "build_ref": ("TEXT", 1, 0),
                "protocol_hash": ("TEXT", 1, 0),
                "created_at": ("TEXT", 1, 0),
                "payload_json": ("TEXT", 1, 0),
            }
            specs["process_instances"] = {
                "instance_id": ("TEXT", primary_key_not_null, 1),
                "run_id": ("TEXT", 1, 0),
                "process_id": ("TEXT", 1, 0),
                "phase": ("INTEGER", 1, 0),
                "attempt": ("INTEGER", 1, 0),
                "status": ("TEXT", 1, 0),
                "payload_json": ("TEXT", 1, 0),
            }
        if version >= 5:
            specs["experiments"]["build_hash"] = ("TEXT", 0, 0)
        if version >= 6:
            self._validate_relationship("runs", "experiment_id", "experiments")
            self._validate_relationship("process_instances", "run_id", "runs")
        if version >= 7:
            specs["package_versions"]["snapshot_digest"] = ("TEXT", 0, 0)
        if version >= 2 or immediate_v0:
            specs["events"]["commit_hash"] = ("TEXT", 0, 0)
            specs["studies"] = {
                "study_id": ("TEXT", primary_key_not_null, 1),
                "payload_json": ("TEXT", 1, 0),
            }
            specs["runs"] = {
                "run_id": ("TEXT", primary_key_not_null, 1),
                "status": ("TEXT", 1, 0),
                "version": ("INTEGER", 1, 0),
                "payload_json": ("TEXT", 1, 0),
            }
        if version >= 4:
            specs["runs"]["experiment_id"] = ("TEXT", 0, 0)
        for table, expected_columns in specs.items():
            actual = {
                row[1]: (str(row[2]).upper(), row[3], row[5])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            }
            missing = sorted(set(expected_columns) - set(actual))
            if missing:
                raise ValueError(f"SCHEMA_SHAPE: table {table} is missing columns {missing}")
            unexpected = sorted(set(actual) - set(expected_columns))
            if unexpected:
                raise ValueError(f"SCHEMA_SHAPE: table {table} has unexpected columns {unexpected}")
            for column, expected_shape in expected_columns.items():
                if actual[column] != expected_shape:
                    raise ValueError(
                        f"SCHEMA_SHAPE: {table}.{column} has shape {actual[column]}, "
                        f"expected {expected_shape}"
                    )

        for table in ("events", "states", "artifacts", "checkpoints"):
            self._validate_payload_foreign_key(table)
        for table, columns in {
            "objects": ("digest",),
            "events": ("event_id",),
            "states": ("run_id", "state_version"),
            "artifacts": ("artifact_id",),
            **(
                {
                    "studies": ("study_id",),
                    "runs": ("run_id",),
                    "package_versions": ("study_id", "version"),
                    "experiments": ("experiment_id",),
                    "process_instances": ("instance_id",),
                    "study_builds": ("build_hash",),
                    "idempotency": ("idempotency_key",),
                }
                if version >= 4
                else (
                    {
                        "studies": ("study_id",),
                        "runs": ("run_id",),
                        "package_versions": ("study_id", "version"),
                        "experiments": ("experiment_id",),
                        "process_instances": ("instance_id",),
                    }
                    if version >= 3
                    else (
                        {"studies": ("study_id",), "runs": ("run_id",)}
                        if version >= 2 or immediate_v0
                        else {}
                    )
                )
            ),
        }.items():
            self._validate_unique_index(table, columns)

    def _validate_payload_foreign_key(self, table: str) -> None:
        foreign_keys = self.connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        if not any(
            row[2] == "objects" and row[3] == "payload_ref" and row[4] == "digest"
            for row in foreign_keys
        ):
            raise ValueError(f"SCHEMA_SHAPE: {table}.payload_ref must reference objects.digest")

    def _validate_relationship(self, table: str, column: str, reference: str) -> None:
        foreign_keys = self.connection.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        valid = any(row[3] == column and row[2] == reference for row in foreign_keys)
        if not valid:
            raise ValueError(f"SCHEMA_SHAPE: {table}.{column} must reference {reference}")

    def _validate_unique_index(self, table: str, columns: tuple[str, ...]) -> None:
        for index in self.connection.execute(f"PRAGMA index_list({table})").fetchall():
            if not index[2]:
                continue
            indexed_columns = tuple(
                row[2]
                for row in self.connection.execute(f"PRAGMA index_info({index[1]})").fetchall()
            )
            if indexed_columns == columns:
                return
        raise ValueError(f"SCHEMA_SHAPE: table {table} lacks unique index on {columns}")

    def _table_columns(self, table: str) -> set[str]:
        return {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}

    def commit_process_result(
        self,
        event: dict[str, Any],
        state: dict[str, Any],
        artifacts: list[dict[str, Any]],
        fail_after: str | None = None,
        expected_state_version: int | None = None,
    ) -> None:
        with self._lock:
            self._commit_process_result(event, state, artifacts, fail_after, expected_state_version)

    def _commit_process_result(
        self,
        event: dict[str, Any],
        state: dict[str, Any],
        artifacts: list[dict[str, Any]],
        fail_after: str | None = None,
        expected_state_version: int | None = None,
    ) -> None:
        if event["run_id"] != state["run_id"]:
            raise ValueError("RUN_ID: event and state run IDs must match")
        if any(artifact["run_id"] != event["run_id"] for artifact in artifacts):
            raise ValueError("RUN_ID: artifact run IDs must match event and state run ID")
        event_bytes = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
        commit_hash = self._commit_hash(event, state, artifacts)
        event_object = self.object_store.put(event_bytes, "application/json")
        state_object = self.object_store.put(state["payload"])
        refs = [event_object, state_object]
        artifact_refs = [
            (a["artifact_id"], a["run_id"], self.object_store.put(a["payload"])) for a in artifacts
        ]
        refs.extend(ref for _, _, ref in artifact_refs)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            current = self.connection.execute(
                "SELECT coalesce(max(state_version), 0) FROM states WHERE run_id = ?",
                (state["run_id"],),
            ).fetchone()[0]
            existing = self.connection.execute(
                "SELECT event_hash, commit_hash FROM events WHERE event_id = ?",
                (event["event_id"],),
            ).fetchone()
            event_hash = hashlib.sha256(event_bytes).hexdigest()
            if existing:
                if existing[1] is None:
                    event_count = self.connection.execute(
                        "SELECT count(*) FROM events WHERE run_id = ?", (event["run_id"],)
                    ).fetchone()[0]
                    state_count = self.connection.execute(
                        "SELECT count(*) FROM states WHERE run_id = ?", (event["run_id"],)
                    ).fetchone()[0]
                    if event_count != 1 or state_count != 1:
                        self.connection.execute("ROLLBACK")
                        raise ValueError(
                            "IDEMPOTENCY: ambiguous legacy commit association cannot be proven"
                        )
                    if existing[0] != event_hash or not self._legacy_commit_matches(
                        event, event_bytes, state, artifacts
                    ):
                        self.connection.execute("ROLLBACK")
                        raise ValueError("IDEMPOTENCY: legacy process-result commit differs")
                    self.connection.execute(
                        "UPDATE events SET commit_hash = ? WHERE event_id = ?",
                        (commit_hash, event["event_id"]),
                    )
                    self.connection.execute("COMMIT")
                    return
                self.connection.execute("ROLLBACK")
                if existing[0] != event_hash or existing[1] != commit_hash:
                    raise ValueError("IDEMPOTENCY: process-result commit differs")
                return
            if expected_state_version is not None and current != expected_state_version:
                raise ValueError(
                    f"STATE_VERSION: expected {expected_state_version}, found {current}"
                )
            if state["state_version"] != current + 1:
                raise ValueError("STATE_VERSION: state versions must progress sequentially")
            for ref in refs:
                self._record_object(ref)
            self.connection.execute(
                """INSERT INTO events(
                       event_id, run_id, kind, payload_ref, event_hash, commit_hash
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event["event_id"],
                    event["run_id"],
                    event["kind"],
                    event_object.digest,
                    event_hash,
                    commit_hash,
                ),
            )
            if fail_after == "event":
                raise RuntimeError("injected failure")
            self.connection.execute(
                "INSERT INTO states(run_id, state_version, payload_ref) VALUES (?, ?, ?)",
                (state["run_id"], state["state_version"], state_object.digest),
            )
            if fail_after == "state":
                raise RuntimeError("injected failure")
            self.connection.executemany(
                """INSERT INTO artifacts(artifact_id, run_id, payload_ref)
                   VALUES (?, ?, ?)""",
                [(artifact_id, run_id, ref.digest) for artifact_id, run_id, ref in artifact_refs],
            )
            self.connection.execute("COMMIT")
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _record_object(self, ref: ObjectRef) -> None:
        existing = self.connection.execute(
            "SELECT media_type, size FROM objects WHERE digest = ?", (ref.digest,)
        ).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO objects(digest, media_type, size) VALUES (?, ?, ?)",
                (ref.digest, ref.media_type, ref.size),
            )
        elif existing != (ref.media_type, ref.size):
            raise ValueError(f"object metadata conflicts with existing digest {ref.digest}")

    @staticmethod
    def _commit_hash(
        event: dict[str, Any], state: dict[str, Any], artifacts: list[dict[str, Any]]
    ) -> str:
        """Return the identity of every logical value in an atomic process commit."""

        def with_payload_digest(value: dict[str, Any]) -> dict[str, Any]:
            payload = value.get("payload")
            if not isinstance(payload, bytes):
                raise ValueError("payload must be bytes")
            normalized = {key: item for key, item in value.items() if key != "payload"}
            normalized["payload_sha256"] = hashlib.sha256(payload).hexdigest()
            return normalized

        identity = {
            "event": event,
            "state": with_payload_digest(state),
            "artifacts": sorted(
                (with_payload_digest(artifact) for artifact in artifacts),
                key=lambda artifact: (
                    str(artifact.get("artifact_id", "")),
                    str(artifact.get("run_id", "")),
                    str(artifact["payload_sha256"]),
                ),
            ),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _legacy_commit_matches(
        self,
        event: dict[str, Any],
        event_bytes: bytes,
        state: dict[str, Any],
        artifacts: list[dict[str, Any]],
    ) -> bool:
        """Verify a pre-commit-hash row before upgrading its idempotency identity.

        Legacy artifacts did not record an event association.  Therefore a safe
        backfill is possible only when the submitted artifact set is the complete
        artifact set currently recorded for the run; ambiguous legacy histories
        are rejected rather than incorrectly declared idempotent.
        """

        event_row = self.connection.execute(
            "SELECT payload_ref FROM events WHERE event_id = ?", (event["event_id"],)
        ).fetchone()
        if set(state) != {"run_id", "state_version", "payload"} or any(
            set(artifact) != {"artifact_id", "run_id", "payload"} for artifact in artifacts
        ):
            return False
        state_row = self.connection.execute(
            "SELECT payload_ref FROM states WHERE run_id = ? AND state_version = ?",
            (state["run_id"], state["state_version"]),
        ).fetchone()
        if event_row is None or state_row is None:
            return False
        if self._read_object(event_row[0]) != event_bytes:
            return False
        if self._read_object(state_row[0]) != state["payload"]:
            return False

        stored_artifacts = self.connection.execute(
            "SELECT artifact_id, run_id, payload_ref FROM artifacts WHERE run_id = ?",
            (event["run_id"],),
        ).fetchall()
        submitted_artifacts = sorted(
            (
                artifact["artifact_id"],
                artifact["run_id"],
                hashlib.sha256(artifact["payload"]).hexdigest(),
            )
            for artifact in artifacts
        )
        stored_identities = sorted(
            (artifact_id, run_id, payload_ref)
            for artifact_id, run_id, payload_ref in stored_artifacts
        )
        if submitted_artifacts != stored_identities:
            return False
        return all(
            self._read_object(payload_ref) == artifact["payload"]
            for artifact, (_, _, payload_ref) in zip(
                sorted(artifacts, key=lambda item: (item["artifact_id"], item["run_id"])),
                stored_identities,
                strict=True,
            )
        )

    def _read_object(self, digest: str) -> bytes:
        metadata = self.connection.execute(
            "SELECT media_type, size FROM objects WHERE digest = ?", (digest,)
        ).fetchone()
        if metadata is None:
            raise ValueError(f"IDEMPOTENCY: referenced object {digest} is missing")
        return self.object_store.get(ObjectRef(digest, metadata[0], metadata[1]))

    def create_study(self, study: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._create_study(study)

    def _create_study(self, study: dict[str, Any]) -> dict[str, Any]:
        record = deepcopy(study)
        study_id = record.get("id")
        if not isinstance(study_id, str) or not study_id:
            raise ValueError("STUDY_ID: study id must be a non-empty string")
        record.setdefault("status", "draft")
        record.setdefault("version", 0)
        if record["version"] != 0:
            raise ValueError("STUDY_VERSION: a new study must have version 0")
        encoded = self._encode_record(record)
        try:
            self.connection.execute(
                "INSERT INTO studies(study_id, payload_json) VALUES (?, ?)",
                (study_id, encoded),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"ALREADY_EXISTS: study '{study_id}' already exists") from exc
        return deepcopy(record)

    def get_study(self, study_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get_study(study_id)

    def _get_study(self, study_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT payload_json FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise KeyError(study_id)
        return cast(dict[str, Any], json.loads(row[0]))

    def list_studies(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._list_studies()

    def _list_studies(self) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT payload_json FROM studies ORDER BY study_id"
            ).fetchall()
        ]

    def create_run(self, run: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return self._create_run(run)

    def _create_run(self, run: dict[str, Any]) -> dict[str, Any]:
        record = deepcopy(run)
        if "id" in record and "run_id" in record and record["id"] != record["run_id"]:
            raise ValueError("RUN_ID: id and run_id must match when both are provided")
        run_id = record.get("id", record.get("run_id"))
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("RUN_ID: run id must be a non-empty string")
        record.pop("run_id", None)
        record["id"] = run_id
        record.setdefault("status", "created")
        record.setdefault("version", 0)
        if record["status"] != "created":
            raise ValueError("RUN_STATUS: a new run must start in created status")
        if record["version"] != 0:
            raise ValueError("RUN_VERSION: a new run must have version 0")
        encoded = self._encode_record(record)
        experiment_id = record.get("experiment_id")
        if experiment_id is not None:
            self._require_experiment(str(experiment_id))
        try:
            self.connection.execute(
                "INSERT INTO runs(run_id, status, version, experiment_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, record["status"], record["version"], experiment_id, encoded),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"ALREADY_EXISTS: run '{run_id}' already exists") from exc
        return deepcopy(record)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get_run(run_id)

    def _get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT payload_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return cast(dict[str, Any], json.loads(row[0]))

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._list_runs()

    def _list_runs(self) -> list[dict[str, Any]]:
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                "SELECT payload_json FROM runs ORDER BY run_id"
            ).fetchall()
        ]

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload_ref FROM events WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
            return [json.loads(self._read_object(row[0])) for row in rows]

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """SELECT artifacts.artifact_id, artifacts.payload_ref,
                          objects.media_type, objects.size
                   FROM artifacts JOIN objects ON objects.digest = artifacts.payload_ref
                   WHERE run_id = ? ORDER BY artifact_id""",
                (run_id,),
            ).fetchall()
            return [
                {
                    "artifact_id": artifact_id,
                    "payload": self._read_object(payload_ref),
                    "media_type": media_type,
                    "size": size,
                }
                for artifact_id, payload_ref, media_type, size in rows
            ]

    def list_state_history(self, run_id: str) -> list[tuple[int, dict[str, Any]]]:
        """Every committed state snapshot for a run, oldest first (AW-09)."""
        with self._lock:
            rows = self.connection.execute(
                """SELECT states.state_version, states.payload_ref
                   FROM states WHERE run_id = ? ORDER BY state_version""",
                (run_id,),
            ).fetchall()
            history = []
            for version, payload_ref in rows:
                try:
                    decoded = json.loads(self._read_object(payload_ref))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    decoded = {}
                if isinstance(decoded, dict):
                    history.append((version, decoded))
            return history

    def latest_state(self, run_id: str) -> tuple[int, bytes, str] | None:
        with self._lock:
            row = self.connection.execute(
                """SELECT states.state_version, states.payload_ref, objects.media_type
                   FROM states JOIN objects ON objects.digest = states.payload_ref
                   WHERE run_id = ? ORDER BY state_version DESC LIMIT 1""",
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0], self._read_object(row[1]), row[2]

    def latest_json_state(self, run_id: str) -> tuple[int, dict[str, Any]] | None:
        latest = self.latest_state(run_id)
        if latest is None:
            return None
        version, payload, _media_type = latest
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("STATE_FORMAT: latest runtime state is not JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("STATE_FORMAT: latest runtime state must be a JSON object")
        return version, decoded

    def transition_run(self, run_id: str, target: str, expected_version: int) -> dict[str, Any]:
        with self._lock:
            return self._transition_run(run_id, target, expected_version)

    def attach_run_manifest(self, run_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        """Attach the frozen run manifest to a run record.

        The manifest is written once, before the first execution; re-attaching
        with equal content is a no-op, and a different manifest is rejected.
        """
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._get_run(run_id)
                existing = record.get("manifest")
                if existing is not None:
                    self.connection.execute("ROLLBACK")
                    if existing != manifest:
                        raise ValueError(
                            "RUN_MANIFEST: run manifest already frozen with different content"
                        )
                    return deepcopy(record)
                record["manifest"] = deepcopy(manifest)
                record["version"] += 1
                self.connection.execute(
                    """UPDATE runs SET version = ?, payload_json = ? WHERE run_id = ?""",
                    (record["version"], self._encode_record(record), run_id),
                )
                self.connection.execute("COMMIT")
                return deepcopy(record)
            except Exception:
                if self.connection.in_transaction:
                    self.connection.execute("ROLLBACK")
                raise

    def append_run_collection(
        self, run_id: str, field: str, item: dict[str, Any]
    ) -> dict[str, Any]:
        if field not in {"events", "artifacts", "outcomes"}:
            raise ValueError("RUN_COLLECTION: unsupported run collection")
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._get_run(run_id)
                values = record.setdefault(field, [])
                if not isinstance(values, list):
                    raise ValueError(f"RUN_COLLECTION: {field} must be a list")
                values.append(deepcopy(item))
                record["version"] += 1
                self.connection.execute(
                    """UPDATE runs SET version = ?, payload_json = ? WHERE run_id = ?""",
                    (record["version"], self._encode_record(record), run_id),
                )
                self.connection.execute("COMMIT")
                return deepcopy(item)
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

    def _transition_run(self, run_id: str, target: str, expected_version: int) -> dict[str, Any]:
        if target not in _RUN_STATUSES:
            raise ValueError(f"RUN_TRANSITION: unknown target status '{target}'")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT status, version, payload_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            status, version, payload_json = row
            if version != expected_version:
                raise ValueError(f"EXPECTED_VERSION: expected {expected_version}, found {version}")
            if target not in _RUN_TRANSITIONS[status]:
                raise ValueError(f"RUN_TRANSITION: cannot transition from {status} to {target}")
            record: dict[str, Any] = json.loads(payload_json)
            record["status"] = target
            record["version"] = version + 1
            cursor = self.connection.execute(
                """UPDATE runs SET status = ?, version = ?, payload_json = ?
                   WHERE run_id = ? AND version = ?""",
                (target, version + 1, self._encode_record(record), run_id, version),
            )
            if cursor.rowcount != 1:
                raise ValueError("EXPECTED_VERSION: run was modified concurrently")
            self.connection.execute("COMMIT")
            return record
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _encode_record(record: dict[str, Any]) -> str:
        try:
            return json.dumps(record, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("RECORD_JSON: record must contain JSON-compatible values") from exc

    def count(self, table: str) -> int:
        with self._lock:
            return self._count(table)

    def _count(self, table: str) -> int:
        if table not in {"events", "states", "artifacts", "checkpoints", "studies", "runs"}:
            raise ValueError("unknown table")
        return int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])

    def create_checkpoint(self, run_id: str, payload: dict[str, Any]) -> str:
        """Persist a checkpoint only after callers have reached an atomic boundary."""
        with self._lock:
            return self._create_checkpoint(run_id, payload)

    def _create_checkpoint(self, run_id: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ref = self.object_store.put(encoded, "application/json")
        digest = hashlib.sha256(encoded).hexdigest()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._record_object(ref)
            cursor = self.connection.execute(
                "INSERT INTO checkpoints(run_id, payload_ref, integrity_hash) VALUES (?, ?, ?)",
                (run_id, ref.digest, digest),
            )
            self.connection.execute("COMMIT")
            return str(cursor.lastrowid)
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        with self._lock:
            return self._restore_checkpoint(checkpoint_id)

    def _restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT payload_ref, integrity_hash FROM checkpoints WHERE checkpoint_id = ?",
            (int(checkpoint_id),),
        ).fetchone()
        if not row:
            raise KeyError(checkpoint_id)
        digest, expected = row
        ref_row = self.connection.execute(
            "SELECT media_type, size FROM objects WHERE digest = ?", (digest,)
        ).fetchone()
        if not ref_row:
            raise ValueError("checkpoint object missing")
        ref = ObjectRef(digest, ref_row[0], ref_row[1])
        encoded = self.object_store.get(ref)
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError("checkpoint integrity failure")
        return cast(dict[str, Any], json.loads(encoded))

    def record_package_version(
        self,
        study_id: str,
        version: int,
        content_hash: str,
        parent_version: int | None,
        status: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one immutable StudyPackageVersion row (LIFE-001/002)."""
        with self._lock:
            self._ensure_study(study_id, str(payload.get("title", study_id)))
            existing = self.connection.execute(
                """SELECT status, content_hash FROM package_versions
                   WHERE study_id = ? AND version = ?""",
                (study_id, version),
            ).fetchone()
            if existing is not None:
                if status == "approved" and existing[0] == "draft":
                    # Approval transitions the version row; content is never rewritten.
                    # The approval-time digest must equal the recorded draft digest.
                    if existing[1] != content_hash:
                        raise ValueError(
                            "APPROVAL_HASH_MISMATCH: approved content hash "
                            f"{content_hash[:12]} differs from the recorded draft hash "
                            f"{existing[1][:12]} of version {version}"
                        )
                    self.connection.execute(
                        """UPDATE package_versions SET status = ?, payload_json = ?
                           WHERE study_id = ? AND version = ?""",
                        (status, self._encode_record(payload), study_id, version),
                    )
                elif existing[1] != content_hash:
                    raise ValueError(
                        f"PACKAGE_VERSION_EXISTS: version {version} of '{study_id}' is immutable"
                    )
                return {
                    "study_id": study_id,
                    "version": version,
                    "content_hash": content_hash,
                    "parent_version": parent_version,
                    "status": status,
                }
            row = (
                study_id,
                version,
                content_hash,
                parent_version,
                status,
                self._encode_record(payload),
                str(payload.get("snapshot_digest", "")),
            )
            try:
                self.connection.execute(
                    """INSERT INTO package_versions(
                           study_id, version, content_hash, parent_version, status,
                           payload_json, snapshot_digest
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    row,
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"PACKAGE_VERSION_EXISTS: version {version} of '{study_id}' is immutable"
                ) from exc
            return {
                "study_id": study_id,
                "version": version,
                "content_hash": content_hash,
                "parent_version": parent_version,
                "status": status,
            }

    def list_package_versions(self, study_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "study_id": study_id,
                    "version": version,
                    "content_hash": content_hash,
                    "parent_version": parent_version,
                    "status": status,
                    "snapshot_digest": snapshot_digest,
                }
                for (
                    _study_id,
                    version,
                    content_hash,
                    parent_version,
                    status,
                    snapshot_digest,
                ) in (
                    self.connection.execute(
                        """SELECT study_id, version, content_hash, parent_version, status,
                               snapshot_digest
                           FROM package_versions WHERE study_id = ? ORDER BY version""",
                        (study_id,),
                    )
                )
            ]

    def create_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist one Experiment record referencing a build and protocol (LIFE-003)."""
        with self._lock:
            record = deepcopy(payload)
            experiment_id = record.get("id", record.get("experiment_id"))
            if not isinstance(experiment_id, str) or not experiment_id:
                raise ValueError("EXPERIMENT_ID: experiment id must be a non-empty string")
            self._require_study(str(record["study_id"]))
            build_hash = self._require_build(str(record["build_ref"]))
            try:
                self.connection.execute(
                    """INSERT INTO experiments(
                           experiment_id, study_id, build_ref, build_hash, protocol_hash,
                           created_at, payload_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        experiment_id,
                        str(record["study_id"]),
                        str(record["build_ref"]),
                        build_hash,
                        str(record["protocol_hash"]),
                        str(record.get("created_at", "")),
                        self._encode_record(record),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"ALREADY_EXISTS: experiment '{experiment_id}' already exists"
                ) from exc
            return deepcopy(record)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            return cast(dict[str, Any], json.loads(row[0]))

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(row[0])
                for row in self.connection.execute(
                    "SELECT payload_json FROM experiments ORDER BY experiment_id"
                ).fetchall()
            ]

    def record_process_instances(self, run_id: str, rows: list[dict[str, Any]]) -> int:
        """Bulk-record ProcessInstance rows for one run (AW-15)."""
        with self._lock:
            self._require_run(run_id)
            for row in rows:
                self.connection.execute(
                    """INSERT OR REPLACE INTO process_instances(
                           instance_id, run_id, process_id, phase, attempt, status, payload_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(row["id"]),
                        run_id,
                        str(row["process_id"]),
                        int(row["phase"]),
                        int(row["attempt"]),
                        str(row["status"]),
                        self._encode_record(row),
                    ),
                )
            return len(rows)

    def list_process_instances(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                json.loads(row[0])
                for row in self.connection.execute(
                    """SELECT payload_json FROM process_instances
                       WHERE run_id = ? ORDER BY instance_id""",
                    (run_id,),
                ).fetchall()
            ]

    def record_study_build(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist one StudyBuild lifecycle record (LIFE-002/003)."""
        with self._lock:
            record = deepcopy(payload)
            build_hash = str(record["build_hash"])
            study_id = str(record["study_id"])
            self._ensure_study(study_id, study_id)
            package_version = int(record["package_version"])
            if package_version == 0:
                # Unversioned packages (direct source compiles) get a version-0
                # lifecycle placeholder so the composite FK stays satisfiable.
                existing = self.connection.execute(
                    "SELECT 1 FROM package_versions WHERE study_id = ? AND version = 0",
                    (study_id,),
                ).fetchone()
                if existing is None:
                    self.connection.execute(
                        """INSERT INTO package_versions(
                               study_id, version, content_hash, parent_version, status,
                               payload_json
                           ) VALUES (?, ?, ?, NULL, 'draft', ?)""",
                        (
                            study_id,
                            0,
                            str(record.get("package_content_hash", "")),
                            '{"title": "' + study_id + '"}',
                        ),
                    )
            try:
                self.connection.execute(
                    """INSERT INTO study_builds(
                           build_hash, study_id, package_version, package_content_hash,
                           compiler_version, created_at, payload_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        build_hash,
                        str(record["study_id"]),
                        int(record["package_version"]),
                        str(record.get("package_content_hash", "")),
                        str(record["compiler_version"]),
                        str(record["created_at"]),
                        self._encode_record(record),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if "FOREIGN KEY" in str(exc):
                    raise ValueError(
                        f"FK_VIOLATION: build references unknown study {record['study_id']}"
                    ) from exc
                raise ValueError(f"ALREADY_EXISTS: build '{build_hash}' already recorded") from exc
            return deepcopy(record)

    def list_study_builds(self, study_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if study_id is not None:
                rows = self.connection.execute(
                    """SELECT payload_json FROM study_builds
                       WHERE study_id = ? ORDER BY created_at""",
                    (study_id,),
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT payload_json FROM study_builds ORDER BY created_at"
                ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def record_idempotency(self, key: str, response: dict[str, Any]) -> None:
        with self._lock:
            self.connection.execute(
                """INSERT OR REPLACE INTO idempotency(
                       idempotency_key, response_json, created_at
                   ) VALUES (?, ?, ?)""",
                (
                    key,
                    json.dumps(response, sort_keys=True),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_idempotency(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT response_json FROM idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return cast(dict[str, Any], json.loads(row[0]))

    def _require_study(self, study_id: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"FK_VIOLATION: unknown study '{study_id}'")

    def _ensure_study(self, study_id: str, fallback_title: str) -> None:
        """Create the implied Study record when the study is not yet registered."""
        row = self.connection.execute(
            "SELECT 1 FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is not None:
            return
        self.connection.execute(
            """INSERT INTO studies(study_id, payload_json) VALUES (?, ?)""",
            (
                study_id,
                self._encode_record(
                    {
                        "id": study_id,
                        "title": fallback_title,
                        "status": "draft",
                        "version": 0,
                    }
                ),
            ),
        )

    def _require_build(self, build_ref: str) -> str:
        """Resolve a build reference to its recorded StudyBuild hash.

        A build reference resolves by hash name or by recorded path; the
        resolved hash backs the ``experiments.build_hash`` column FK.
        """
        build_hash = Path(build_ref).name
        row = self.connection.execute(
            "SELECT 1 FROM study_builds WHERE build_hash = ?", (build_hash,)
        ).fetchone()
        if row is not None:
            return build_hash
        for row in self.connection.execute(
            "SELECT build_hash, payload_json FROM study_builds"
        ).fetchall():
            record = cast(dict[str, Any], json.loads(row[1]))
            if str(record.get("path", "")) == str(build_ref):
                return str(row[0])
        raise ValueError(f"FK_VIOLATION: unknown build '{build_hash}'")

    def _require_run(self, run_id: str) -> None:
        row = self.connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"FK_VIOLATION: unknown run '{run_id}'")

    def _require_experiment(self, experiment_id: str) -> None:
        row = self.connection.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"FK_VIOLATION: unknown experiment '{experiment_id}'")

    def schema_version(self) -> int:
        return int(self.connection.execute("PRAGMA user_version").fetchone()[0])

    @property
    def current_schema_version(self) -> int:
        return _SCHEMA_VERSION

    def retention_purge(self, run_id: str) -> int:
        """Delete durable raw-response artifact rows for a purged run (AW-20).

        The immutable event ledger is untouched; only artifact payloads that
        carry raw provider responses are removed. Payloads live in the object
        store, so rows are inspected via their object references.
        """
        with self._lock:
            rows = self.connection.execute(
                "SELECT artifact_id, payload_ref FROM artifacts WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            removed = 0
            unreferenced: list[str] = []
            for artifact_id, payload_ref in rows:
                payload = self._read_object(payload_ref)
                if b'"response"' in payload:
                    self.connection.execute(
                        "DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,)
                    )
                    removed += 1
                    unreferenced.append(payload_ref)
            # Reference-aware garbage collection: drop object rows and files that
            # are no longer used by any artifacts, events, states, or checkpoints.
            for digest in unreferenced:
                still_used = self.connection.execute(
                    """SELECT 1 FROM artifacts WHERE payload_ref = ? UNION ALL
                       SELECT 1 FROM events WHERE payload_ref = ? UNION ALL
                       SELECT 1 FROM states WHERE payload_ref = ? UNION ALL
                       SELECT 1 FROM checkpoints WHERE payload_ref = ?""",
                    (digest, digest, digest, digest),
                ).fetchone()
                if still_used is not None:
                    continue
                self.connection.execute("DELETE FROM objects WHERE digest = ?", (digest,))
                self.object_store._metadata.pop(digest, None)
                object_path = self.object_store.root / digest[:2] / digest[2:]
                if object_path.is_file():
                    object_path.unlink()
            return removed

    def backup_to(self, destination: str | Path) -> Path:
        """Create a consistent SQLite backup using the safe backup API."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = sqlite3.connect(str(target), isolation_level=None, check_same_thread=False)
        try:
            with self._lock:
                self.connection.backup(backup)
        finally:
            backup.close()
        return target

    def close(self) -> None:
        with self._lock:
            self.connection.close()

"""Declarative outcome datasets with a fixed operation registry (G3).

Named row relations are declared in ``outcomes.yaml`` and compiled into a
typed outcome plan: typed sources (events, artifacts, state), bounded
transformation steps from a fixed operation registry (no Python/SQL/shell),
explicit deduplication, and explicit missing/join semantics. The core
evaluation then builds rows from declared datasets; it never special-cases
study-specific names such as ``evaluate-clickbait`` or ``titles``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

OUTCOME_PLAN_VERSION = 1
# A dataset id names the files it is exported to, so it has to be the same
# stable kebab-case identifier the specification requires: a plain name that
# cannot climb out of, or replace, the export directory.
_DATASET_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def check_dataset_id(value: Any) -> str:
    """The dataset id, refused unless it is a single stable identifier.

    The id was spliced into export paths unchecked. A plan's ids are validated
    when a specification is written, but a build's outcome_plan.json is loaded
    as plain JSON -- and an imported bundle's plan is copied verbatim and
    re-signed -- so `../../../escaped` wrote outside the export directory, and
    an absolute id wrote anywhere on disk.
    """
    text = str(value or "")
    if not text:
        raise ValueError("OUTCOME_PLAN: dataset requires an id")
    if not _DATASET_ID.fullmatch(text):
        raise ValueError(
            f"OUTCOME_PLAN: dataset id {text!r} is not a stable lowercase kebab-case "
            "identifier; it names the dataset's export files, so it cannot contain a "
            "path"
        )
    return text


# What the outcome evaluator reads. A key outside these is never consulted, so a
# declaration using one compiled clean and evaluated as something else: every
# clickbait outcome declared ``measure: proportion`` and was counted, and seven
# named sources nothing defined and produced no rows at all.
OUTCOME_BUILTIN_SOURCES = frozenset({"events", "artifacts", "state", "lineage"})
OUTCOME_AGGREGATION_KEYS = frozenset({"op", "type", "operation", "field", "select"})
OUTCOME_AGGREGATION_OPS = frozenset({"count", "sum", "mean", "trajectory", "distribution"})
OUTCOME_MISSINGNESS_KEYS = frozenset({"policy"})
OUTCOME_MISSINGNESS_POLICIES = frozenset({"exclude", "zero"})
OUTCOME_WINDOW_KEYS = frozenset({"time_field", "start", "end"})
OUTCOME_JOIN_KEYS = frozenset({"left", "right", "on"})

# Sentinel ``_round`` annotation attached by the service to state snapshots
# committed in a phase where no round completed (a process failed, was left
# active, or was skipped). ``each_completed_round`` must never emit an
# observation for such a phase, so materialize_datasets drops these rows.
# A string is used so it can never collide with a real phase index (int).
_INCOMPLETE_ROUND = "__incomplete_round__"


def _number(value: Any) -> float | None:
    """A real number, or ``None`` when the value is missing or not numeric.

    A missing score used to become NaN. ``NaN >= 7`` is false, so the comparison
    stored False and the mean counted it; arithmetic stored a bare NaN, which
    ``json.dumps`` writes and which is not valid JSON.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and value != value:
        return None
    return float(value)


def _arithmetic(operator: str, left: Any, right: Any) -> float | None:
    a, b = _number(left), _number(right)
    if a is None or b is None:
        return None
    if operator == "add":
        return a + b
    if operator == "subtract":
        return a - b
    if operator == "multiply":
        return a * b
    if operator == "divide":
        return None if b == 0 else a / b
    return None


def _comparison(operator: str, left: Any, right: Any) -> bool | None:
    # A missing operand is not an observation. ``None == 7`` is false in Python,
    # and storing that false counted the gap: scores ``[7, 9, None, None]``
    # averaged to 0.25 with nothing missing. ``ne`` counted every gap as true.
    if left is None or right is None:
        return None
    if operator == "eq":
        # ``True == 1`` and ``7 == 7.0``. Only the first pair is two values.
        return bool(_observation_key(left) == _observation_key(right))
    if operator == "ne":
        return bool(_observation_key(left) != _observation_key(right))
    a, b = _number(left), _number(right)
    if a is None or b is None:
        return None
    if operator == "gt":
        return a > b
    if operator == "gte":
        return a >= b
    if operator == "lt":
        return a < b
    if operator == "lte":
        return a <= b
    return None


ARITHMETIC_OPS: dict[str, Any] = {
    "add": lambda a, b: _arithmetic("add", a, b),
    "subtract": lambda a, b: _arithmetic("subtract", a, b),
    "multiply": lambda a, b: _arithmetic("multiply", a, b),
    "divide": lambda a, b: _arithmetic("divide", a, b),
}
COMPARISON_OPS: dict[str, Any] = {
    "eq": lambda a, b: _comparison("eq", a, b),
    "ne": lambda a, b: _comparison("ne", a, b),
    "gt": lambda a, b: _comparison("gt", a, b),
    "gte": lambda a, b: _comparison("gte", a, b),
    "lt": lambda a, b: _comparison("lt", a, b),
    "lte": lambda a, b: _comparison("lte", a, b),
}


# Columns the store writes around a row. A payload copy repeats the row's own
# fields; these do not count. ``declared_artifact_id`` does count: it sits
# beside a nested ``value`` and is not one of that nested field's keys, which
# is how the nested field is told apart from a copy of the whole row.
_ROW_ENVELOPE = frozenset(
    {
        "artifact_id",
        "run_id",
        "time",
        "condition_id",
        "replication",
        "experiment_id",
        "invocation_id",
        "process_id",
        "producer_process",
        "producer_event",
        "phase",
        "commit_order",
        "state_version",
        "owner",
        "schema_ref",
        "visibility",
        "lifecycle_scope",
        "lineage",
        "actors",
        "instance_id",
    }
)


def _observation_key(value: Any) -> Any:
    """An identity that keeps ``True`` apart from ``1`` but not ``1`` from ``1.0``.

    ``True == 1`` and they share a hash, so a dedup set, a join index, or a
    filter took a boolean for a count. ``7`` and ``7.0`` are one number: a model
    may write a schema-valid integer score either way, and telling them apart
    split one score into two groups and dropped ``7.0`` from a ``== 7`` filter.
    Grouping uses the same rule.
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        # One tag for both, so ("num", 7) == ("num", 7.0) and they hash alike.
        return ("num", value)
    if isinstance(value, dict | list | set | tuple):
        return ("json", json.dumps(value, sort_keys=True, default=str))
    return value


def _hashable(value: Any) -> Any:
    """A deduplication key part."""
    return _observation_key(value)


def _value_repeats_row(value: Any, row: Mapping[str, Any]) -> bool:
    """Whether ``value`` is the payload already copied onto ``row``.

    A scalar is copied under the declared artifact id as well as ``value``.
    A mapping is that copy only when it still carries every non-envelope field
    of the row. A nested ``value`` can repeat some of those fields and still be
    the record's own; an empty mapping is its own value too.
    """
    declared = row.get("declared_artifact_id")
    if (
        not isinstance(value, Mapping)
        and isinstance(declared, str)
        and declared
        and declared != "value"
        and _observation_key(row.get(declared)) == _observation_key(value)
    ):
        return True
    if not isinstance(value, Mapping) or not value:
        return False
    if any(
        key not in row or _observation_key(row[key]) != _observation_key(item)
        for key, item in value.items()
    ):
        return False
    return all(
        key in value or key in _ROW_ENVELOPE or str(key).startswith("__")
        for key in row
        if key != "value"
    )


def _resolve_path(source: Mapping[str, Any], path: str) -> Any:
    """Safe dotted-path read; missing keys yield None."""
    if not path:
        return None
    value: Any = source
    for part in path.split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        elif isinstance(value, list | tuple) and part.isdigit():
            index = int(part)
            value = value[index] if index < len(value) else None
        else:
            return None
    return value


def _apply_fields(row: dict[str, Any], fields: list[Mapping[str, Any]]) -> dict[str, Any]:
    for step in fields:
        name = str(step.get("name", ""))
        op = str(step.get("op", "copy"))
        if op == "copy":
            row[name] = _resolve_path(row, str(step.get("field") or ""))
        elif op == "literal":
            row[name] = step.get("value")
        elif op == "arithmetic":
            left = _resolve_path(row, str(step.get("field") or ""))
            right = step.get("value")
            if right is None and isinstance(step.get("with_field"), str):
                right = _resolve_path(row, str(step["with_field"]))
            handler = ARITHMETIC_OPS.get(str(step.get("operator", "add")))
            if handler is None:
                operator = step.get("operator")
                raise ValueError(f"OUTCOME_PLAN: unknown arithmetic operator '{operator}'")
            row[name] = handler(left, right)
        elif op == "comparison":
            left = _resolve_path(row, str(step.get("field") or ""))
            right = step.get("value")
            if right is None and isinstance(step.get("with_field"), str):
                right = _resolve_path(row, str(step["with_field"]))
            handler = COMPARISON_OPS.get(str(step.get("operator", "eq")))
            if handler is None:
                operator = step.get("operator")
                raise ValueError(f"OUTCOME_PLAN: unknown comparison operator '{operator}'")
            result = handler(left, right)
            # bool(None) is False, which is how a missing score was counted.
            row[name] = None if result is None else bool(result)
        elif op == "conditional":
            condition = step.get("condition") or {}
            value = _resolve_path(row, str(condition.get("field") or ""))
            expected = condition.get("value")
            row[name] = step.get("value") if value == expected else step.get("else_value")
    return row


def _matches(row: Mapping[str, Any], predicate: Mapping[str, Any]) -> bool:
    """Whether one row satisfies a declared dataset filter."""
    actual = _resolve_path(row, str(predicate.get("field") or ""))
    op = str(predicate.get("op", "eq"))
    if op == "truthy":
        return bool(actual)
    if op == "falsy":
        return not bool(actual)
    expected = predicate.get("value")
    if op == "eq":
        # The same identity as grouping: ``7`` matches ``7.0``, and ``True``
        # does not match ``1``.
        return bool(_observation_key(actual) == _observation_key(expected))
    if op == "ne":
        # None != 7 is true, so a missing score was kept as "not 7" and later
        # counted. A gap is not an observation of inequality.
        if actual is None:
            return False
        return bool(_observation_key(actual) != _observation_key(expected))
    if actual is None or expected is None:
        return False
    handler = COMPARISON_OPS.get(op)
    if handler is None:
        raise ValueError(f"OUTCOME_PLAN: unknown dataset filter operator '{op}'")
    return bool(handler(actual, expected))


def _is_engine_column(key: str) -> bool:
    """A column the engine writes to place a row in its run, cell and round."""
    return key in _ROW_ENVELOPE or key.startswith("factor_")


def _named_state_row(record: Mapping[str, Any], field: str) -> dict[str, Any]:
    """One snapshot reduced to ``field``, with the round still attached.

    A missing field is null rather than the rest of the snapshot. A missing
    field named ``phase`` must not be written as null: the round is recorded
    in that column afterwards, and the null hid it. A value that is present,
    including a stored null, stays.
    """
    row: dict[str, Any] = {}
    if field in record:
        row[field] = record[field]
    if "_round" in record:
        row["_round"] = record["_round"]
    if field not in row and not (field == "phase" and "_round" in record):
        row[field] = None
    return row


def _container_column(parent: Mapping[str, Any], root: str, key: str) -> str:
    """A free name for a container field that must not occupy an engine column.

    ``{root}_{key}`` is the usual spelling. It is not used when that column
    already holds a value, or when it is itself an engine column: a container
    named ``factor`` would be written as ``factor_phase`` and replace the
    treatment. A later spelling keeps both.
    """
    stem = f"{root}_{key}"
    if stem not in parent and not _is_engine_column(stem):
        return stem
    # A suffix of ``factor_phase`` still starts with ``factor_``. The spare
    # name has to begin somewhere else.
    spare = f"from_{root}_{key}"
    name = spare
    suffix = 2
    while name in parent or _is_engine_column(name):
        name = f"{spare}_{suffix}"
        suffix += 1
    return name


def _without_leaf(container: Mapping[str, Any], parts: list[str]) -> dict[str, Any]:
    """The container with the exploded collection removed.

    The collection's immediate siblings flatten onto the caller, so exploding
    ``state_delta.rows`` keeps ``user`` as a column. A longer path keeps the
    nested object, so an outer ``extra`` is not overwritten by an inner one.
    """
    key = parts[0]
    rest = parts[1:]
    if not rest:
        return {name: item for name, item in container.items() if name != key}
    kept: dict[str, Any] = {}
    for name, item in container.items():
        if name == key and isinstance(item, Mapping):
            kept[name] = _without_leaf(item, rest)
            continue
        kept[name] = item
    return kept


def _exploded(rows: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    """One row per element of the list, or entry of the mapping, at ``path``.

    A record element's fields join its parent's, the element winning a name
    both carry; a scalar element is written under the path's last segment; a
    mapping entry's key is kept as ``<segment>_key``. The exploded value itself
    is not repeated on every row. A row with nothing at the path yields none.
    Sibling fields of the container stay on the row.
    """
    parts = [part for part in path.split(".") if part]
    if not parts:
        return []
    name = parts[-1]
    exploded: list[dict[str, Any]] = []
    for row in rows:
        value = _resolve_path(row, path)
        if isinstance(value, Mapping):
            entries: list[tuple[Any, Any]] = list(value.items())
        elif isinstance(value, list | tuple):
            entries = [(None, item) for item in value]
        else:
            continue
        # Only the payload column `value` is a copy of the exploded collection.
        # Dropping every mapping that happened to hold an equal list removed a
        # sibling such as `summary` and its `total`. The container is applied
        # after the other fields, so its own `user` wins over an outer one
        # whichever key was written first.
        parent: dict[str, Any] = {}
        root = parts[0]
        for key, item in row.items():
            if len(parts) == 1 and key == name:
                continue
            if (
                key == "value"
                and isinstance(item, Mapping)
                and name in item
                and item.get(name) == value
            ):
                continue
            if len(parts) > 1 and key == root:
                continue
            parent[key] = item
        if len(parts) > 1 and isinstance(row.get(root), Mapping):
            for key, item in _without_leaf(row[root], parts[1:]).items():
                # The container's own fields never replace an engine column. A
                # state field called `phase` put "draft" where the round number
                # was, and every outcome grouped or joined on it read the state.
                # It is kept under the container's name instead.
                if key in parent and _is_engine_column(key):
                    parent[_container_column(parent, root, key)] = item
                else:
                    parent[key] = item
        for key, element in entries:
            child = dict(parent)
            if key is not None:
                child[f"{name}_key"] = key
            if isinstance(element, Mapping):
                # The element wins every name both carry, the round included. A
                # record's own `phase` is the round it belongs to -- an article
                # carries the round it was published in -- and by-round outcomes
                # group on it. Keeping the row's round instead put every record
                # of a final snapshot in the last round.
                child.update(dict(element))
            else:
                child[name] = element
            exploded.append(child)
    return exploded


def _relation(sources: Mapping[str, Any], name: str) -> Iterable[Any]:
    """A fresh walk of one source relation, given as a sequence or a factory.

    A factory lets the service hand over the event ledger and the artifact
    store without listing them first; each dataset that reads a relation walks
    it anew rather than every dataset sharing one resident copy.
    """
    source = sources.get(name)
    if source is None:
        return ()
    relation: Iterable[Any] = source() if callable(source) else source
    return relation


_NO_ROW = object()


def _entry_version(entry: Any) -> int | None:
    """The committed version a state entry carries, or ``None``."""
    if isinstance(entry, Mapping):
        version = entry.get("state_version")
    elif isinstance(entry, tuple | list) and len(entry) == 2:
        version = entry[0]
    else:
        version = None
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return None


def materialize_datasets(
    plan: Mapping[str, Any], sources: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Compile declared datasets into named row relations from retained sources.

    ``sources`` provides the raw evidence sets: ``events``, ``artifacts``,
    ``state`` and ``artifact_sources`` keyed by declared artifact id. Datasets
    never reference study-specific identifiers in this module.
    """
    datasets = plan.get("datasets") or []
    compiled: dict[str, list[dict[str, Any]]] = {}
    for dataset in datasets:
        dataset_id = check_dataset_id(dataset.get("id"))
        source = dataset.get("source") or {}
        kind = str(source.get("kind", "events"))
        rows: list[dict[str, Any]] = []
        if kind == "events":
            path = str(source.get("path") or "")
            for event in _relation(sources, "events"):
                if not isinstance(event, Mapping):
                    continue
                records = _resolve_path(event, path) if path else [dict(event)]
                if not isinstance(records, list):
                    continue
                for record in records:
                    if isinstance(record, Mapping):
                        merged = dict(event)
                        merged.update(dict(record))
                        rows.append(merged)
        elif kind == "artifacts":
            artifact_type = str(source.get("artifact_type") or "")
            process = str(source.get("process") or "")
            for artifact in _relation(sources, "artifacts"):
                if not isinstance(artifact, Mapping):
                    continue
                payload = (
                    artifact.get("payload")
                    if isinstance(artifact.get("payload"), Mapping)
                    else None
                )
                declared_id = (
                    str(payload.get("declared_artifact_id", "")) if payload is not None else ""
                )
                if artifact_type and declared_id != artifact_type:
                    continue
                # An invocation row records process_id; a declared artifact
                # records producer_process. Reading only the first dropped
                # every declared artifact from a process-filtered dataset.
                producer = (
                    payload.get("process_id") or payload.get("producer_process")
                    if payload is not None
                    else None
                )
                if process and str(producer or "") != process:
                    continue
                row = {key: value for key, value in artifact.items() if key != "payload"}
                if payload is not None:
                    had_value = "value" in payload
                    original = payload.get("value")
                    for key, value in payload.items():
                        if key not in row:
                            row[key] = value
                    if isinstance(original, Mapping):
                        # An inner field named value wins, and it is that field,
                        # not the payload. Only a scalar payload is also stored
                        # under the declared artifact id. A mapping with no field
                        # of its own named value is the payload that was spread,
                        # so the column still repeats those fields.
                        row.update({key: value for key, value in original.items()})
                        if "value" in original:
                            row["__own_value__"] = True
                        elif original:
                            row["__mapping_payload__"] = True
                    elif had_value and not isinstance(original, list | tuple):
                        declared = str(payload.get("declared_artifact_id") or "")
                        # The scalar is copied under the declared id, and the
                        # `value` column is then the duplicate. When that id is
                        # itself `value`, the column is the artifact and stays.
                        # A list is the collection ``explode`` walks. Treating it
                        # as a scalar removed ``value``, so ``explode: value``
                        # yielded no rows.
                        if declared and declared != "value" and declared not in row:
                            row[declared] = original
                            row["__scalar_payload__"] = True
                rows.append(row)
        elif kind == "state":
            snapshot = str(source.get("snapshot", "final"))
            # Streamed, not listed: the service passes the history as a factory
            # precisely so no round's snapshot stays resident, and listing it
            # first held every snapshot at once -- 40 MB to return one `final`
            # row from 40 rounds of 1 MB state.
            state_rows = _relation(sources, "state")

            # The service supplies state history as dict-shaped snapshots
            # (with a state_version key); tolerate legacy (version, snapshot)
            # tuple entries as well. Unpacking a dict as a tuple is a
            # shape error, not a valid empty row (F5 fix).
            def _snapshot(entry: Any) -> Mapping[str, Any] | None:
                if isinstance(entry, Mapping):
                    return entry
                if (
                    isinstance(entry, tuple | list)
                    and len(entry) == 2
                    and isinstance(entry[1], Mapping)
                ):
                    return entry[1]
                return None

            if snapshot == "final":
                # The last *committed* version, not the last row: the service
                # supplies one row per round, in round order, and rounds can
                # interleave, so the final commit is not necessarily last. Each
                # candidate is projected as it arrives, so only one field is held.
                field = str(source.get("state") or "")

                def _projected(entry: Any, field: str = field) -> dict[str, Any] | None:
                    record = _snapshot(entry)
                    if record is None:
                        return None
                    if field:
                        # The named field is the row, not the rest of the
                        # snapshot. The round travels with it, as it does for
                        # each completed round: a placeholder for a missing
                        # field used to occupy ``phase`` and hide the round.
                        return _named_state_row(record, field)
                    return dict(record)

                chosen: Any = _NO_ROW
                last: Any = _NO_ROW
                best_version: int | None = None
                for entry in state_rows:
                    last = entry
                    version = _entry_version(entry)
                    if version is not None and (best_version is None or version >= best_version):
                        best_version, chosen = version, _projected(entry)
                if best_version is None and last is not _NO_ROW:
                    chosen = _projected(last)
                if chosen is not _NO_ROW and chosen is not None:
                    rows.append(chosen)
            elif snapshot == "each_completed_round":
                # F8: keep the FINAL committed snapshot of each completed
                # round, not one row per state commit (a phase may contain
                # several process invocations). Rows annotated with a
                # ``_round`` phase collapse to the last snapshot of that
                # phase; unannotated rows are each treated as their own round.
                # Rows marked with the ``_INCOMPLETE_ROUND`` sentinel come
                # from a phase in which no round completed (a process failed,
                # was left active, or was skipped) and are dropped (F4).
                round_rows: dict[Any, dict[str, Any]] = {}
                field = str(source.get("state") or "")
                for entry in state_rows:
                    record = _snapshot(entry)
                    if record is None:
                        continue
                    round_key = record.get("_round")
                    if round_key == _INCOMPLETE_ROUND:
                        continue
                    if field:
                        row = _named_state_row(record, field)
                    else:
                        row = dict(record)
                    if round_key is None:
                        round_rows.setdefault(id(row), row)
                    else:
                        round_rows[round_key] = row
                for row in round_rows.values():
                    rows.append(row)
            # The state branch is the only source where the engine injects the
            # ``_round`` annotation; drop it from the materialized rows. The
            # service's ``state_version`` annotation is likewise internal —
            # strip it here rather than letting it leak, but ONLY for state
            # datasets (event/artifact rows may legitimately carry their own
            # fields).
            # The round a snapshot closed is kept as ``phase`` when the state
            # does not already carry one: stripped, a per-round table could not
            # say which round each row was.
            rows = [
                {
                    **({"phase": row["_round"]} if "_round" in row and "phase" not in row else {}),
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in ("_round", "state_version")
                    },
                }
                for row in rows
            ]
        if kind == "artifacts":
            # Drop the payload copy before explode and derived fields. A scalar
            # payload was already written under the declared artifact id. A
            # record's own field named value is not that payload, so it stays.
            field_names = {
                str(step.get("name"))
                for step in (dataset.get("fields") or [])
                if isinstance(step, Mapping) and step.get("name")
            }
            prepared: list[dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                scalar_payload = bool(row.pop("__scalar_payload__", False))
                mapping_payload = bool(row.pop("__mapping_payload__", False))
                own_value = bool(row.pop("__own_value__", False))
                repeats = isinstance(row.get("value"), Mapping) and _value_repeats_row(
                    row["value"], row
                )
                if (
                    not own_value
                    and "value" not in field_names
                    and "value" in row
                    and (scalar_payload or mapping_payload or repeats)
                ):
                    del row["value"]
                prepared.append(row)
            rows = prepared
        explode = dataset.get("explode")
        if isinstance(explode, str) and explode:
            rows = _exploded(rows, explode)
        for step in dataset.get("fields") or []:
            if isinstance(step, Mapping):
                rows = [_apply_fields(dict(row), [step]) for row in rows]
        # Narrowing belongs to the relation: a trace seed selects rows without
        # defining an outcome over them.
        for predicate in dataset.get("where") or []:
            if isinstance(predicate, Mapping):
                rows = [row for row in rows if _matches(row, predicate)]
        deduplicate_on = dataset.get("deduplicate_on") or []
        if deduplicate_on:
            seen: set[tuple[Any, ...]] = set()
            unique: list[dict[str, Any]] = []
            for row in rows:
                # A field may hold a record or a list, which cannot be hashed;
                # its canonical JSON identifies it just as well.
                key = tuple(_hashable(row.get(key)) for key in deduplicate_on)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(row)
            rows = unique
        missing_policy = str(dataset.get("missing", "retain_null"))
        if missing_policy == "drop":
            rows = [row for row in rows if any(value is not None for value in row.values())]
        compiled[dataset_id] = rows
    return compiled


def compile_outcome_plan(build_path: Any) -> dict[str, Any]:
    """Load and normalize one build's outcome plan (legacy list or dict form)."""
    import json as _json
    from pathlib import Path

    root = build_path if isinstance(build_path, Path) else Path(str(build_path))
    raw = _json.loads((root / "outcome_plan.json").read_text())
    if isinstance(raw, list):
        return {"version": OUTCOME_PLAN_VERSION, "datasets": [], "outcomes": raw}
    if not isinstance(raw, Mapping):
        raise ValueError("OUTCOME_PLAN: outcome_plan.json must be a list or object")
    plan = dict(raw)
    plan.setdefault("version", OUTCOME_PLAN_VERSION)
    plan.setdefault("datasets", [])
    plan.setdefault("outcomes", [])
    plan.setdefault("traces", [])
    seen: set[str] = set()
    for dataset in plan["datasets"] or []:
        if isinstance(dataset, Mapping):
            dataset_id = check_dataset_id(dataset.get("id"))
            # The compiler refuses a duplicate in a written specification, but a
            # build's plan is loaded as plain JSON -- and an imported bundle's is
            # copied verbatim -- and datasets materialise keyed by id, so the
            # first declaration's rows silently vanished.
            if dataset_id in seen:
                raise ValueError(f"OUTCOME_PLAN: dataset id {dataset_id!r} is declared twice")
            seen.add(dataset_id)
    return plan


def outcome_plan_digest(plan: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

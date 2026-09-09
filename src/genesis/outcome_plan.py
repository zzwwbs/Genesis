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
from collections.abc import Mapping
from typing import Any

OUTCOME_PLAN_VERSION = 1

# Sentinel ``_round`` annotation attached by the service to state snapshots
# committed in a phase where no round completed (a process failed, was left
# active, or was skipped). ``each_completed_round`` must never emit an
# observation for such a phase, so materialize_datasets drops these rows.
# A string is used so it can never collide with a real phase index (int).
_INCOMPLETE_ROUND = "__incomplete_round__"

ARITHMETIC_OPS: dict[str, Any] = {
    "add": lambda a, b: _number(a) + _number(b),
    "subtract": lambda a, b: _number(a) - _number(b),
    "multiply": lambda a, b: _number(a) * _number(b),
    "divide": lambda a, b: None if _number(b) == 0 else _number(a) / _number(b),
}
COMPARISON_OPS: dict[str, Any] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: _number(a) > _number(b),
    "gte": lambda a, b: _number(a) >= _number(b),
    "lt": lambda a, b: _number(a) < _number(b),
    "lte": lambda a, b: _number(a) <= _number(b),
}


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


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
            row[name] = bool(handler(left, right))
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
        return bool(actual == expected)
    if op == "ne":
        return bool(actual != expected)
    if actual is None or expected is None:
        return False
    handler = COMPARISON_OPS.get(op)
    if handler is None:
        raise ValueError(f"OUTCOME_PLAN: unknown dataset filter operator '{op}'")
    return bool(handler(actual, expected))


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
        dataset_id = str(dataset.get("id", ""))
        if not dataset_id:
            raise ValueError("OUTCOME_PLAN: dataset requires an id")
        source = dataset.get("source") or {}
        kind = str(source.get("kind", "events"))
        rows: list[dict[str, Any]] = []
        if kind == "events":
            path = str(source.get("path") or "")
            for event in sources.get("events", []):
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
            for artifact in sources.get("artifacts", []):
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
                if process and (payload is None or str(payload.get("process_id", "")) != process):
                    continue
                row = {key: value for key, value in artifact.items() if key != "payload"}
                if payload is not None:
                    for key, value in payload.items():
                        if key not in row:
                            row[key] = value
                    if isinstance(payload.get("value"), Mapping):
                        row.update({key: value for key, value in payload["value"].items()})
                rows.append(row)
        elif kind == "state":
            snapshot = str(source.get("snapshot", "final"))
            state_rows = list(sources.get("state", []))

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

            if snapshot == "final" and state_rows:
                record = _snapshot(state_rows[-1])
                if record is not None:
                    record = dict(record)
                    field = str(source.get("state") or "")
                    if field and field in record:
                        record = {field: record[field]}
                    rows.append(record)
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
                for entry in state_rows:
                    record = _snapshot(entry)
                    if record is None:
                        continue
                    row = dict(record)
                    round_key = row.get("_round")
                    if round_key == _INCOMPLETE_ROUND:
                        continue
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
            rows = [
                {key: value for key, value in row.items() if key not in ("_round", "state_version")}
                for row in rows
            ]
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
                key = tuple(row.get(key) for key in deduplicate_on)
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
    return plan


def outcome_plan_digest(plan: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

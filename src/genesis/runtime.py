"""Execution contracts and deterministic local runtime primitives."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any, cast

_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def _check_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a stable lowercase kebab-case identifier")
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _event_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove purgeable provider bodies from the immutable event ledger."""
    safe = dict(_plain(metadata))
    raw = safe.pop("raw_response", None)
    parsed = safe.pop("parsed_response", None)
    attempts = safe.pop("provider_attempts", None)
    if raw is not None:
        safe["raw_response_hash"] = _hash(raw)
    if parsed is not None:
        safe["parsed_response_hash"] = _hash(parsed)
    if isinstance(attempts, list):
        safe["provider_attempts"] = [
            {
                key: value
                for key, value in attempt.items()
                if key not in {"raw_response", "parsed_response"}
            }
            for attempt in attempts
            if isinstance(attempt, Mapping)
        ]
        safe["provider_attempt_count"] = len(attempts)
    return safe


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _condition_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    token = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    if not token or not re.match(r"^[a-z]", token):
        token = f"level-{token or 'value'}"
    return token


def expand_protocol_conditions(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize explicit or factorial protocol declarations into stable cells."""
    factors = protocol.get("factors", [])
    explicit = protocol.get("conditions", [])
    if factors and explicit:
        raise ValueError("protocol must use factors or explicit conditions, not both")
    if not factors:
        return [dict(condition) for condition in explicit] or [{"id": "base"}]
    if not isinstance(factors, list):
        raise ValueError("protocol factors must be a list")
    normalized: list[tuple[str, list[Any]]] = []
    for factor in factors:
        if not isinstance(factor, Mapping):
            raise ValueError("protocol factor must be a mapping")
        factor_id = _check_id(str(factor.get("id", "")), "factor_id")
        levels = factor.get("levels")
        if not isinstance(levels, list) or not levels:
            raise ValueError("protocol factor levels must be a non-empty list")
        normalized.append((factor_id, list(levels)))
    conditions = []
    for values in product(*(levels for _factor_id, levels in normalized)):
        factor_values = {
            factor_id: value for (factor_id, _levels), value in zip(normalized, values, strict=True)
        }
        condition_id = "-".join(
            f"{factor_id}-{_condition_token(value)}" for factor_id, value in factor_values.items()
        )
        conditions.append({"id": condition_id, "factors": factor_values})
    return conditions


class _ReportedProcessFailure(RuntimeError):
    """Internal marker preventing a declared failed result being recorded twice."""


def _resolve_path(source: Mapping[str, Any], path: str) -> Any:
    if not isinstance(path, str) or not path or ".." in path or path.startswith("/"):
        raise ValueError("condition path must be a safe dotted path")
    value: Any = source
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return None
            value = value[part]
        elif isinstance(value, list | tuple) and part.isdigit():
            index = int(part)
            if index >= len(value):
                return None
            value = value[index]
        else:
            return None
    return value


def expand_actor_instances(
    process: Mapping[str, Any], state: Mapping[str, Any]
) -> list[tuple[str, ...]]:
    """Resolve a process actor declaration into deterministic invocation groups."""
    actors = process.get("actors")
    if actors is None:
        return [()]
    if isinstance(actors, list | tuple):
        ids = [_check_id(str(actor), "actor_id") for actor in actors]
        return [(actor_id,) for actor_id in ids] or [()]
    if not isinstance(actors, Mapping):
        raise ValueError("actors must be a list or actor selector")
    fan_out = bool(actors.get("fan_out", True))
    if actors.get("ids") is not None:
        raw_ids = actors.get("ids")
        if not isinstance(raw_ids, list | tuple):
            raise ValueError("actor selector ids must be a list")
        ids = [_check_id(str(actor), "actor_id") for actor in raw_ids]
    else:
        source = actors.get("source")
        if not isinstance(source, str):
            raise ValueError("actor selector requires ids or source")
        records = _resolve_path(state, source)
        if records is None:
            raise ValueError(f"actor selector source is unavailable: {source}")
        if isinstance(records, Mapping):
            records = list(records.values())
        if not isinstance(records, list | tuple):
            raise ValueError("actor selector source must resolve to a list or mapping")
        id_field = str(actors.get("id_field", "id"))
        ids = []
        for record in records:
            actor = record.get(id_field) if isinstance(record, Mapping) else record
            ids.append(_check_id(str(actor), "actor_id"))
        ids.sort()
    if len(ids) != len(set(ids)):
        raise ValueError("actor selector produced duplicate actor ids")
    if not ids:
        return []
    return [(actor_id,) for actor_id in ids] if fan_out else [tuple(ids)]


def _evaluate_predicate(predicate: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    if not isinstance(predicate, Mapping):
        raise ValueError("condition predicate must be a mapping")
    actual = _resolve_path(state, str(predicate.get("path", "")))
    operator = predicate.get("op", "eq")
    expected = predicate.get("value")
    operations: dict[str, Callable[[Any, Any], bool]] = {
        "eq": lambda left, right: left == right,
        "ne": lambda left, right: left != right,
        "gt": lambda left, right: left is not None and left > right,
        "gte": lambda left, right: left is not None and left >= right,
        "lt": lambda left, right: left is not None and left < right,
        "lte": lambda left, right: left is not None and left <= right,
        "in": lambda left, right: left in right if right is not None else False,
        "truthy": lambda left, _right: bool(left),
    }
    if operator not in operations:
        raise ValueError(f"unsupported condition operator: {operator}")
    try:
        return operations[operator](actual, expected)
    except TypeError as exc:
        raise ValueError(f"condition operands are incompatible for {operator}") from exc


def _validate_predicate(predicate: Any) -> None:
    if not isinstance(predicate, Mapping):
        raise ValueError("condition predicate must be a mapping")
    _resolve_path({}, str(predicate.get("path", "")))
    if predicate.get("op", "eq") not in {"eq", "ne", "gt", "gte", "lt", "lte", "in", "truthy"}:
        raise ValueError(f"unsupported condition operator: {predicate.get('op')}")


def _evaluate_condition(condition: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    if "all" in condition:
        children = condition["all"]
        if not isinstance(children, list | tuple) or not children:
            raise ValueError("condition all requires a non-empty list")
        return all(
            _evaluate_condition(child, state) if isinstance(child, Mapping) else False
            for child in children
        )
    if "any" in condition:
        children = condition["any"]
        if not isinstance(children, list | tuple) or not children:
            raise ValueError("condition any requires a non-empty list")
        return any(
            _evaluate_condition(child, state) if isinstance(child, Mapping) else False
            for child in children
        )
    if "not" in condition:
        child = condition["not"]
        if not isinstance(child, Mapping):
            raise ValueError("condition not requires a mapping")
        return not _evaluate_condition(child, state)
    _validate_predicate(condition)
    return _evaluate_predicate(condition, state)


def _validate_scheduling_effects(
    scheduler: Scheduler, effects: tuple[Mapping[str, Any], ...]
) -> None:
    """Validate all scheduler mutations before the process result is committed."""
    for effect in effects:
        effect_type = effect.get("type")
        if effect_type == "signal_event":
            if not isinstance(effect.get("event"), str) or not effect["event"]:
                raise ValueError("signal_event requires a non-empty event")
        elif effect_type == "schedule":
            process_id = effect.get("process_id")
            phase = effect.get("phase")
            if process_id not in scheduler.processes:
                raise ValueError(f"unknown scheduled process: {process_id}")
            if not isinstance(phase, int | float) or isinstance(phase, bool):
                raise ValueError("scheduled phase must be numeric")
        else:
            raise ValueError(f"unsupported scheduling effect: {effect_type}")


@dataclass(frozen=True)
class ProcessInvocation:
    invocation_id: str
    run_id: str
    process_id: str
    actor_ids: tuple[str, ...] = ()
    phase: int | float = 0
    time: int | float = 0
    state_version: int = 0
    inputs: Mapping[str, Any] = field(default_factory=dict)
    context: Any = None
    seed: int = 0
    attempt: int = 1
    executor_binding: Mapping[str, Any] = field(default_factory=dict)
    condition: Mapping[str, Any] = field(default_factory=dict)
    event_history: tuple[Mapping[str, Any], ...] = ()

    @property
    def temporal_position(self) -> tuple[int | float, int | float]:
        return (self.phase, self.time)

    @property
    def input_artifact_refs(self) -> Mapping[str, Any]:
        return self.inputs

    def __post_init__(self) -> None:
        for name in ("invocation_id", "run_id", "process_id"):
            _check_id(getattr(self, name), name)
        for actor in self.actor_ids:
            _check_id(actor, "actor_id")
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")
        object.__setattr__(self, "inputs", _freeze(dict(self.inputs)))
        object.__setattr__(
            self, "context", _freeze(self.context) if self.context is not None else None
        )
        object.__setattr__(self, "executor_binding", _freeze(dict(self.executor_binding)))
        object.__setattr__(self, "condition", _freeze(dict(self.condition)))
        object.__setattr__(
            self,
            "event_history",
            tuple(_freeze(dict(event)) for event in self.event_history),
        )


@dataclass(frozen=True)
class ProcessResult:
    status: str = "succeeded"
    outputs: Mapping[str, Any] = field(default_factory=dict)
    state_effects: Any = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    scheduling_effects: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed", "skipped"}:
            raise ValueError("invalid process result status")
        object.__setattr__(self, "outputs", _freeze(dict(self.outputs)))
        if isinstance(self.state_effects, list | tuple):
            # Effect-list form: [{field, op, value}, ...] consumed by StateStore.apply.
            object.__setattr__(
                self,
                "state_effects",
                [dict(item) for item in self.state_effects if isinstance(item, Mapping)],
            )
        else:
            object.__setattr__(self, "state_effects", _freeze(dict(self.state_effects)))
        object.__setattr__(self, "events", tuple(_freeze(dict(item)) for item in self.events))
        object.__setattr__(
            self,
            "scheduling_effects",
            tuple(_freeze(dict(item)) for item in self.scheduling_effects),
        )
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))

    @property
    def output_artifacts(self) -> Mapping[str, Any]:
        return self.outputs

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return self.metadata


@dataclass(frozen=True)
class ContextEnvelope:
    policy_id: str
    invocation_id: str
    data: Mapping[str, Any]
    content_hash: str
    exposures: tuple[Mapping[str, Any], ...] = ()


class ContextEngine:
    def __init__(self, policies: Mapping[str, Any]):
        if isinstance(policies, list):
            policies = {p["id"]: p for p in policies if isinstance(p, Mapping) and "id" in p}
        self.policies = dict(policies)
        for policy_id, definition in self.policies.items():
            if not isinstance(definition, Mapping):
                raise ValueError(f"context policy {policy_id} must be a mapping")
            for path in definition.get("allow", ()):
                if not isinstance(path, str) or not path or ".." in path or path.startswith("/"):
                    raise ValueError("context allow paths must be safe dotted paths")

    def build(
        self, policy_id: str, invocation: ProcessInvocation, state: Mapping[str, Any]
    ) -> ContextEnvelope:
        if policy_id not in self.policies:
            raise PermissionError(f"unknown context policy: {policy_id}")
        definition = self.policies[policy_id] or {}
        allowed = definition.get("allow", ()) if isinstance(definition, Mapping) else definition
        selected: dict[str, Any] = {}
        redact = set(definition.get("redact", ())) if isinstance(definition, Mapping) else set()
        visibility = definition.get("visibility", {}) if isinstance(definition, Mapping) else {}
        availability = definition.get("availability", {}) if isinstance(definition, Mapping) else {}
        cardinality = definition.get("cardinality", {}) if isinstance(definition, Mapping) else {}
        aggregate = definition.get("aggregate", {}) if isinstance(definition, Mapping) else {}
        source_root = {
            "state": state,
            "inputs": invocation.inputs,
            "condition": invocation.condition,
            "actor": {"ids": invocation.actor_ids},
            "events": invocation.event_history,
        }
        exposures: dict[str, dict[str, Any]] = {}
        for path in allowed or ():
            if (
                path in redact
                or visibility.get(path) == "private"
                or availability.get(path) is False
                or not self._available_when(definition, path, invocation)
            ):
                continue
            parts = tuple(path.split("."))
            namespaced = parts[0] in source_root
            source: Any = source_root if namespaced else state
            try:
                for part in parts:
                    source = source[part]
            except (KeyError, TypeError):
                continue
            value = _plain(source)
            available_when = definition.get("available_when", {})
            path_conditions = (
                available_when.get(path, {})
                if isinstance(available_when, Mapping)
                and isinstance(available_when.get(path), Mapping)
                else available_when
            )
            if (
                namespaced
                and parts == ("inputs",)
                and isinstance(value, Mapping)
                and isinstance(path_conditions, Mapping)
                and path_conditions.get("source_match_event") is True
            ):
                event_name = path_conditions.get("event")
                source_ids = {
                    str(event["source_artifact_id"])
                    for event in invocation.event_history
                    if event.get("source_artifact_id")
                    and (event.get("type") or event.get("kind") or event.get("event")) == event_name
                    and (
                        path_conditions.get("recipient_match") is not True
                        or event.get("recipient_id") in invocation.actor_ids
                        or bool(set(event.get("recipient_ids", ())) & set(invocation.actor_ids))
                    )
                }
                value = {
                    instance_id: record
                    for instance_id, record in value.items()
                    if instance_id in source_ids
                }
            if path in cardinality:
                value = _cap_cardinality(value, cardinality[path])
            if path in aggregate:
                value = _aggregate_value(value, aggregate[path])
            target_parts = parts
            target = selected
            for part in target_parts[:-1]:
                target = target.setdefault(part, {})
            if target_parts:
                target[target_parts[-1]] = value
            if namespaced and parts[0] == "inputs":
                if len(parts) == 1 and isinstance(value, Mapping):
                    instance_ids = list(value)
                elif len(parts) > 1 and parts[1] in invocation.inputs:
                    instance_ids = [parts[1]]
                else:
                    instance_ids = []
                for instance_id in instance_ids:
                    record = invocation.inputs.get(instance_id, {})
                    record = record if isinstance(record, Mapping) else {}
                    exposures[str(instance_id)] = {
                        "recipient_ids": tuple(invocation.actor_ids),
                        "source_artifact_id": str(instance_id),
                        "artifact_type": record.get("artifact_type"),
                        "phase": invocation.phase,
                        "policy_id": policy_id,
                    }
        frozen = _freeze(selected)
        frozen_exposures = tuple(
            _freeze(exposure) for _instance_id, exposure in sorted(exposures.items())
        )
        return ContextEnvelope(
            policy_id,
            invocation.invocation_id,
            frozen,
            _hash(frozen),
            frozen_exposures,
        )

    @staticmethod
    def _available_when(
        definition: Mapping[str, Any], path: str, invocation: ProcessInvocation
    ) -> bool:
        conditions = definition.get("available_when", {})
        if not isinstance(conditions, Mapping) or not conditions:
            return True
        conditions = (
            conditions.get(path, {}) if isinstance(conditions.get(path), Mapping) else conditions
        )
        phase = invocation.phase
        after = conditions.get("after_round")
        before = conditions.get("before_round")
        if after is not None and isinstance(after, int | float) and phase < after:
            return False
        if before is not None and isinstance(before, int | float) and phase >= before:
            return False
        event_name = conditions.get("event")
        if event_name is not None:
            matching_events = [
                event
                for event in invocation.event_history
                if (event.get("type") or event.get("kind") or event.get("event")) == event_name
            ]
            if conditions.get("recipient_match") is True:
                matching_events = [
                    event
                    for event in matching_events
                    if (
                        event.get("recipient_id") in invocation.actor_ids
                        or bool(set(event.get("recipient_ids", ())) & set(invocation.actor_ids))
                    )
                ]
            if not matching_events:
                return False
        predicate = conditions.get("predicate")
        if predicate is not None:
            if not isinstance(predicate, Mapping):
                raise ValueError("context availability predicate must be a mapping")
            namespace = _invocation_namespace(invocation)
            namespace["events"] = _plain(invocation.event_history)
            if not _evaluate_predicate(predicate, namespace):
                return False
        return True


def _cap_cardinality(value: Any, limit: Any) -> Any:
    """Deterministically truncate a list/dict to its declared cardinality cap."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("cardinality cap must be a non-negative integer")
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, dict):
        return dict(list(value.items())[:limit])
    return value


def _aggregate_value(value: Any, rule: Any) -> Any:
    """Reduce a collection to a scalar with a declared aggregation op."""
    if not isinstance(rule, Mapping):
        raise ValueError("aggregate rule must be a mapping with an operation")
    op = rule.get("op", "count")
    values = list(value) if isinstance(value, list) else list(value.values())
    if op == "count":
        return len(values)
    if op in {"sum", "mean"}:
        numeric = [item for item in values if isinstance(item, int | float)]
        if op == "sum":
            return sum(numeric)
        if numeric:
            return sum(numeric) / len(numeric)
        return None
    raise ValueError(f"unsupported aggregate operation: {op}")


class StateStore:
    def __init__(
        self,
        schema: Mapping[str, type] | None = None,
        initial: Mapping[str, Any] | None = None,
        reducers: Mapping[str, Callable[[Any, Any], Any]] | None = None,
    ):
        self.schema = dict(schema or {})
        self.reducers = dict(reducers or {})
        self._state = dict(initial or {})
        for key, value in self._state.items():
            if key not in self.schema or not isinstance(value, self.schema[key]):
                raise TypeError(f"invalid type for state field {key}")
        self.version = 0

    def apply(
        self,
        effects: Mapping[str, Any] | list[Mapping[str, Any]],
        declared: set[str],
        expected_version: int | None = None,
    ) -> int:
        if expected_version is not None and expected_version != self.version:
            raise ValueError(f"expected state version {expected_version}, found {self.version}")
        effects = _plain(effects)
        if isinstance(effects, list):
            candidate = self.snapshot()
            for effect in effects:
                effect_version = effect.get("expected_version")
                if effect_version is not None and effect_version != self.version:
                    raise ValueError(
                        f"expected state version {effect_version}, found {self.version}"
                    )
                field = effect.get("field")
                if field not in declared or field not in self.schema:
                    raise PermissionError("state effect is not declared in the state model")
                op, value = effect.get("op", "set"), effect.get("value")
                if op == "set":
                    candidate[field] = value
                elif op == "increment":
                    candidate[field] = candidate.get(field, 0) + value
                elif op == "append":
                    candidate[field] = candidate.get(field, []) + [value]
                elif op == "remove":
                    current = candidate.get(field)
                    if isinstance(current, list):
                        candidate[field] = [item for item in current if item != value]
                    elif isinstance(current, set):
                        candidate[field] = set(current) - {value}
                    elif isinstance(current, Mapping):
                        candidate[field] = {
                            key: item for key, item in current.items() if key != value
                        }
                    else:
                        raise TypeError("remove requires a list, set, or mapping state field")
                elif op == "custom":
                    reducer_name = effect.get("reducer")
                    if reducer_name not in self.reducers:
                        raise PermissionError("custom reducer is not declared")
                    candidate[field] = self.reducers[reducer_name](candidate.get(field), value)
                else:
                    raise ValueError(f"unknown state operation: {op}")
            effects = {k: v for k, v in candidate.items() if self._state.get(k) != v}
        if any(k not in self.schema or k not in declared for k in effects):
            raise PermissionError("state effect is not declared in the state model")
        for key, value in effects.items():
            if not isinstance(value, self.schema[key]):
                raise TypeError(f"invalid type for state field {key}")
        self._state.update(copy.deepcopy(dict(effects)))
        self.version += 1
        return self.version

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def restore(self, snapshot: Mapping[str, Any], version: int) -> None:
        """Restore a snapshot after an operation failed before durable commit."""
        candidate = dict(snapshot)
        for key, value in candidate.items():
            if key not in self.schema or not isinstance(value, self.schema[key]):
                raise TypeError(f"invalid type for state field {key}")
        if version < 0:
            raise ValueError("state version must be non-negative")
        self._state = copy.deepcopy(candidate)
        self.version = version


class ArtifactStore:
    def __init__(self, catalog: Mapping[str, Any] | None = None):
        self._artifacts: dict[str, Any] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self.catalog = dict(catalog or {})

    def put(
        self,
        artifact_id: str,
        value: Any,
        *,
        owner: str | None = None,
        schema_ref: str | None = None,
        visibility: str | None = None,
        lifecycle_scope: str | None = None,
        lineage: list[str] | None = None,
        instance_id: str | None = None,
        actors: list[str] | tuple[str, ...] | None = None,
        producer_process: str | None = None,
        producer_event: str | None = None,
        phase: int | float | None = None,
    ) -> str:
        _check_id(artifact_id, "artifact_id")
        instance_id = _check_id(instance_id or artifact_id, "artifact_instance_id")
        if self.catalog and artifact_id not in self.catalog:
            raise PermissionError(f"artifact is not declared: {artifact_id}")
        if instance_id in self._artifacts:
            raise ValueError("artifact instance is immutable")
        expected = self.catalog.get(artifact_id, {})
        supplied = {
            "owner": owner,
            "schema_ref": schema_ref,
            "visibility": visibility,
            "lifecycle_scope": lifecycle_scope,
        }
        for key, actual in supplied.items():
            if actual is not None and expected.get(key) is not None and actual != expected[key]:
                raise PermissionError(f"artifact metadata mismatch: {key}")
        supplied = {
            key: value if value is not None else expected.get(key)
            for key, value in supplied.items()
        }
        actor_ids = [_check_id(str(actor), "actor_id") for actor in (actors or [])]
        stored_value = _plain(value)
        self._artifacts[instance_id] = copy.deepcopy(stored_value)
        self._metadata[instance_id] = {
            **supplied,
            "artifact_type": artifact_id,
            "instance_id": instance_id,
            "lineage": list(lineage or []),
            "actors": actor_ids,
            "producer_process": producer_process,
            "producer_event": producer_event,
            "phase": phase,
        }
        return _hash(stored_value)

    def get(self, artifact_id: str) -> Any:
        return copy.deepcopy(self._artifacts[artifact_id])

    def metadata(self, artifact_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._metadata[artifact_id])

    def resolve(
        self,
        references: list[str] | tuple[str, ...],
        *,
        actor_ids: tuple[str, ...] = (),
        phase: int | float | None = None,
    ) -> dict[str, Any]:
        """Resolve declared artifact/process references to immutable instances.

        When same-actor instances exist for a reference, they take precedence;
        otherwise all eligible instances are returned. This permits actor-local
        state flow and deliberate many-to-one platform/recommendation processes.
        """
        resolved: dict[str, Any] = {}
        for reference in references:
            eligible = [
                instance_id
                for instance_id, metadata in self._metadata.items()
                if (
                    metadata.get("artifact_type") == reference
                    or metadata.get("producer_process") == reference
                )
                and (phase is None or metadata.get("phase") is None or metadata["phase"] <= phase)
            ]
            if actor_ids:
                matching = [
                    instance_id
                    for instance_id in eligible
                    if set(actor_ids) & set(self._metadata[instance_id].get("actors", []))
                ]
                if matching:
                    eligible = matching
            for instance_id in sorted(eligible):
                resolved[instance_id] = {
                    **copy.deepcopy(self._metadata[instance_id]),
                    "value": copy.deepcopy(self._artifacts[instance_id]),
                }
        return resolved

    def snapshot(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        return copy.deepcopy(self._artifacts), copy.deepcopy(self._metadata)

    def restore(self, snapshot: tuple[dict[str, Any], dict[str, dict[str, Any]]]) -> None:
        self._artifacts, self._metadata = copy.deepcopy(snapshot)


class DeterministicExecutor:
    def __init__(self, function: Callable[[ProcessInvocation], Mapping[str, Any]]):
        self.function = function

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(outputs=self.function(invocation))


class StochasticExecutor:
    def __init__(self, function: Callable[[ProcessInvocation, random.Random], Mapping[str, Any]]):
        self.function = function

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(outputs=self.function(invocation, random.Random(invocation.seed)))


class GenerativeExecutor:
    def __init__(
        self,
        function: Callable[[ProcessInvocation], Mapping[str, Any]],
        metadata_hook: Callable[[Mapping[str, Any]], Any] | None = None,
    ):
        self.function = function
        self.metadata_hook = metadata_hook

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        output = self.function(invocation)
        if self.metadata_hook:
            self.metadata_hook(
                {"executor": "generative", "invocation_id": invocation.invocation_id}
            )
        return ProcessResult(outputs=output, metadata={"mode": "generative"})


def _invocation_namespace(invocation: ProcessInvocation) -> dict[str, Any]:
    context = getattr(invocation.context, "data", invocation.context)
    plain_context = _plain(context or {})
    namespace = {
        "inputs": _plain(invocation.inputs),
        "context": plain_context,
        "actor": {"ids": list(invocation.actor_ids)},
        "condition": _plain(invocation.condition),
        "events": _plain(invocation.event_history),
        "phase": invocation.phase,
        "time": invocation.time,
    }
    artifacts: dict[str, list[Any]] = {}
    for record in invocation.inputs.values():
        if not isinstance(record, Mapping) or not record.get("artifact_type"):
            continue
        artifacts.setdefault(str(record["artifact_type"]), []).append(_plain(record.get("value")))
    namespace["artifacts"] = artifacts
    if isinstance(plain_context, Mapping):
        namespace.update(plain_context)
    return namespace


def _result_from_declaration(
    declaration: Mapping[str, Any], *, metadata: Mapping[str, Any]
) -> ProcessResult:
    return ProcessResult(
        status=str(declaration.get("status", "succeeded")),
        outputs=dict(declaration.get("outputs", {})),
        state_effects=dict(declaration.get("state_effects", {})),
        events=tuple(declaration.get("events", ())),
        scheduling_effects=tuple(declaration.get("scheduling_effects", ())),
        metadata={**dict(declaration.get("metadata", {})), **dict(metadata)},
    )


class RuleExecutor:
    """Evaluate ordered declarative predicates without study-specific Python."""

    def __init__(self, parameters: Mapping[str, Any]):
        rules = parameters.get("rules")
        if not isinstance(rules, list):
            raise ValueError("rule executor requires a rules list")
        self.rules = [dict(rule) for rule in rules]
        default = parameters.get("default", {})
        if not isinstance(default, Mapping):
            raise ValueError("rule executor default must be a mapping")
        self.default = dict(default)

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        namespace = _invocation_namespace(invocation)
        for index, rule in enumerate(self.rules):
            predicate = rule.get("when")
            if not isinstance(predicate, Mapping):
                raise ValueError("rule when must be a predicate mapping")
            if _evaluate_condition(predicate, namespace):
                return _result_from_declaration(
                    rule, metadata={"mode": "rule", "matched_rule": index}
                )
        return _result_from_declaration(
            self.default, metadata={"mode": "rule", "matched_rule": None}
        )


class StateTransitionExecutor:
    """Compute declared replacements for the runtime's validated state boundary."""

    def __init__(self, parameters: Mapping[str, Any]):
        operations = parameters.get("operations")
        if not isinstance(operations, list):
            raise ValueError("state-transition executor requires an operations list")
        self.operations = [dict(operation) for operation in operations]

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        namespace = _invocation_namespace(invocation)
        context = namespace.get("context", {})
        working = dict(context) if isinstance(context, Mapping) else {}
        effects: dict[str, Any] = {}
        for operation in self.operations:
            op = str(operation.get("op", ""))
            state = _check_id(str(operation.get("state", "")), "state")
            current = working.get(state)
            if "value_from" in operation:
                value = _resolve_path(namespace, str(operation["value_from"]))
            else:
                value = _plain(operation.get("value"))
            if op == "set":
                updated = value
            elif op == "increment":
                if not isinstance(current, int | float) or not isinstance(value, int | float):
                    raise TypeError("increment requires numeric current and value")
                updated = current + value
            elif op == "append":
                if not isinstance(current, list | tuple):
                    raise TypeError("append requires a list state")
                updated = [*_plain(current), value]
            elif op == "add-relation":
                if not isinstance(current, list | tuple) or not isinstance(value, Mapping):
                    raise TypeError("add-relation requires a list state and mapping value")
                updated = list(_plain(current))
                if value not in updated:
                    updated.append(dict(value))
            elif op == "remove-relation":
                if not isinstance(current, list | tuple) or not isinstance(value, Mapping):
                    raise TypeError("remove-relation requires a list state and mapping value")
                updated = [item for item in _plain(current) if item != value]
            else:
                raise ValueError(f"unsupported state-transition operation: {op}")
            working[state] = updated
            effects[state] = updated
        return ProcessResult(
            state_effects=effects,
            metadata={"mode": "state-transition", "operation_count": len(self.operations)},
        )


class CallableExecutor:
    def __init__(self, function: Callable[[ProcessInvocation], Any], mode: str):
        self.function, self.mode = function, mode

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        result = self.function(invocation)
        if isinstance(result, ProcessResult):
            return result
        if not isinstance(result, Mapping):
            raise TypeError("executor callable must return a mapping or ProcessResult")
        return ProcessResult(outputs=result, metadata={"mode": self.mode})


class RecordedArtifactExecutor:
    def __init__(self, outputs: Mapping[str, Any]):
        self.outputs = dict(outputs)

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(outputs=self.outputs, metadata={"recorded": True})


class ExecutorRegistry:
    def __init__(self, executors: Mapping[str, Any] | None = None):
        self._executors = dict(executors or {})

    def register(self, process_id: str, executor: Any, mode: str | None = None) -> None:
        self._executors[_check_id(process_id, "process_id")] = (
            CallableExecutor(executor, mode) if callable(executor) and mode else executor
        )

    def execute(self, process_id: str, invocation: ProcessInvocation) -> ProcessResult:
        return cast(ProcessResult, self._executors[process_id].execute(invocation))


def derive_seed(
    base_seed: int,
    run_id: str,
    process_id: str,
    actor_id: str = "",
    *,
    experiment_id: str = "",
    condition_id: str = "",
    replication: int = 0,
    matching_key: str | None = None,
) -> int:
    payload = (
        f"{base_seed}:{experiment_id}:{replication}:{matching_key}:{process_id}:{actor_id}"
        if matching_key is not None
        else (
            f"{base_seed}:{experiment_id}:{condition_id}:{replication}:"
            f"{run_id}:{process_id}:{actor_id}"
        )
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class ScheduledProcess:
    process_id: str
    phase: int | float


class Scheduler:
    def __init__(self, processes: list[Mapping[str, Any]]):
        self.processes: dict[str, dict[str, Any]] = {}
        for original in processes:
            p = dict(original)
            process_id = p.get("id")
            if process_id in self.processes:
                raise ValueError(f"duplicate process id: {process_id}")
            trigger = p.get("trigger", {})
            dependencies = p.get("dependencies", {})
            p.setdefault("phase", trigger.get("phase", 0) if isinstance(trigger, Mapping) else 0)
            p.setdefault(
                "after", dependencies.get("after", []) if isinstance(dependencies, Mapping) else []
            )
            p.setdefault(
                "delay", dependencies.get("delay", 0) if isinstance(dependencies, Mapping) else 0
            )
            if not isinstance(p.get("after", []), list):
                raise ValueError("dependencies.after must be a list")
            if isinstance(trigger, Mapping) and trigger.get("type") == "condition":
                _validate_predicate(trigger.get("predicate"))
            max_attempts = p.get("retry_policy", {}).get("max_attempts", 1)
            if (
                not isinstance(max_attempts, int)
                or isinstance(max_attempts, bool)
                or max_attempts < 1
            ):
                raise ValueError("retry_policy.max_attempts must be a positive integer")
            self.processes[p["id"]] = p
        for pid, process in self.processes.items():
            for dependency in process.get("after", []):
                if dependency not in self.processes:
                    raise ValueError(f"missing dependency reference: {dependency}")
            delay = process.get("delay", 0)
            if isinstance(delay, Mapping):
                delay = delay.get("rounds", 0)
            if not isinstance(delay, int | float) or isinstance(delay, bool) or delay < 0:
                raise ValueError(f"invalid dependency delay for {pid}")
        self._validate_cycles()
        self._delayed_bootstrap_edges = self._find_delayed_cycle_edges()
        self.completed: dict[str, int | float] = {}
        self.events: dict[str, int] = {}
        self._event_consumed: dict[tuple[str, str], int] = {}
        self._scheduled: list[ScheduledProcess] = []
        self._completed_occurrences: set[tuple[str, int | float]] = set()

    def _validate_cycles(self) -> None:
        graph = {
            pid: {dependency for dependency in p.get("after", []) if not p.get("delay")}
            for pid, p in self.processes.items()
        }
        while graph:
            ready = {pid for pid, deps in graph.items() if not deps}
            if not ready:
                raise ValueError("immediate dependency cycle")
            for pid in ready:
                graph.pop(pid)
            for deps in graph.values():
                deps.difference_update(ready)

    def _find_delayed_cycle_edges(self) -> set[tuple[str, str]]:
        """Return delayed dependency edges that belong to an actual dependency cycle."""
        graph = {
            process_id: set(process.get("after", []))
            for process_id, process in self.processes.items()
        }
        reachable: dict[str, set[str]] = {}
        for start in graph:
            seen: set[str] = set()
            pending = list(graph[start])
            while pending:
                node = pending.pop()
                if node in seen:
                    continue
                seen.add(node)
                pending.extend(graph[node] - seen)
            reachable[start] = seen
        edges = set()
        for process_id, process in self.processes.items():
            delay = process.get("delay", 0)
            delay_value = delay.get("rounds", 0) if isinstance(delay, Mapping) else delay
            if delay_value <= 0:
                continue
            for dependency in process.get("after", []):
                if process_id in reachable[dependency]:
                    edges.add((process_id, dependency))
        return edges

    def signal_event(self, event: str) -> None:
        self.events[event] = self.events.get(event, 0) + 1

    def schedule(self, process_id: str, phase: int | float) -> None:
        if process_id not in self.processes:
            raise ValueError(f"unknown scheduled process: {process_id}")
        if not isinstance(phase, int | float) or isinstance(phase, bool):
            raise ValueError("scheduled phase must be numeric")
        self._scheduled.append(ScheduledProcess(process_id, phase))

    def consume_scheduled(self, process_id: str, phase: int | float) -> None:
        for index, item in enumerate(self._scheduled):
            if item.process_id == process_id and item.phase <= phase:
                self._scheduled.pop(index)
                return

    def ready(
        self,
        phase: int | float,
        events: set[str] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> list[ScheduledProcess]:
        out = []
        observed_events = set(self.events) | (events or set())
        for pid, p in self.processes.items():
            repeat = bool(p.get("repeat", p.get("trigger", {}).get("repeat", False)))
            if (
                (pid, phase) in self._completed_occurrences
                or (pid in self.completed and not repeat)
                or p.get("phase", 0) > phase
            ):
                continue
            deps = p.get("after", [])
            dependencies_block = p.get("dependencies")
            if not deps and isinstance(dependencies_block, Mapping):
                declared_after = dependencies_block.get("after")
                if isinstance(declared_after, list | tuple):
                    deps = list(declared_after)
            delay = p.get("delay", 0)
            trigger = p.get("trigger", {})
            if (
                isinstance(trigger, Mapping)
                and trigger.get("type") == "event"
                and (
                    trigger.get("event") not in observed_events
                    or self.events.get(str(trigger.get("event")), 0)
                    + (1 if trigger.get("event") in (events or set()) else 0)
                    <= self._event_consumed.get((pid, str(trigger.get("event"))), 0)
                )
            ):
                continue
            if isinstance(trigger, Mapping) and trigger.get("type") == "condition":
                predicate = cast(Mapping[str, Any], trigger.get("predicate"))
                if not _evaluate_predicate(predicate, state or {}):
                    continue
            if any(
                (
                    dep not in self.completed
                    and not (
                        (delay.get("rounds", 0) if isinstance(delay, Mapping) else delay) > 0
                        and (pid, dep) in self._delayed_bootstrap_edges
                        and phase == p.get("phase", 0)
                    )
                )
                or (
                    dep in self.completed
                    and self.completed[dep]
                    + (delay.get("rounds", 0) if isinstance(delay, Mapping) else delay)
                    > phase
                )
                for dep in deps
            ):
                continue
            out.append(ScheduledProcess(pid, p.get("phase", 0)))
        out.extend(item for item in self._scheduled if item.phase <= phase)
        unique = {(item.process_id, item.phase): item for item in out}
        return sorted(unique.values(), key=lambda x: (x.phase, x.process_id))

    def complete(self, process_id: str, phase: int | float) -> None:
        self.completed[process_id] = phase
        self._completed_occurrences.add((process_id, phase))
        trigger = self.processes[process_id].get("trigger", {})
        if isinstance(trigger, Mapping) and trigger.get("type") == "event":
            event = str(trigger.get("event"))
            key = (process_id, event)
            self._event_consumed[key] = self._event_consumed.get(key, 0) + 1


class RunController:
    def __init__(
        self,
        scheduler: Scheduler,
        registry: ExecutorRegistry,
        context_engine: ContextEngine,
        *,
        state_store: StateStore | None = None,
        artifact_store: Any | None = None,
        persistence: Any | None = None,
        status_provider: Callable[[], str] | None = None,
        output_schema_validator: Callable[[str, Any], list[str]] | None = None,
    ):
        self.scheduler, self.registry, self.context_engine = scheduler, registry, context_engine
        self.state_store, self.persistence = state_store, persistence
        self.artifact_store = artifact_store
        self.output_schema_validator = output_schema_validator
        self.status_provider = status_provider
        self.status = "created"
        self.results: list[ProcessResult] = []
        self.failures: list[dict[str, Any]] = []
        self.dispatch_log: list[dict[str, Any]] = []
        self.commit_log: list[dict[str, Any]] = []
        self._run_id: str | None = None
        self._persisted_count = 0
        self._event_count = 0
        self._next_phase = 0
        self._last_event_id: str | None = None
        self._event_history: list[dict[str, Any]] = []
        self._completed_process_events: list[dict[str, Any]] = []
        self._actor_queues: dict[tuple[str, int | float], list[tuple[str, ...]]] = {}
        self._completed_actor_occurrences: set[tuple[str, int | float, tuple[str, ...]]] = set()

    def pause(self) -> None:
        if self.status in {"running", "created"}:
            self.status = "paused"

    def resume(self) -> None:
        if self.status == "paused":
            self.status = "running"

    def cancel(self) -> None:
        if self.status not in {"completed", "failed", "cancelled"}:
            self.status = "cancelled"

    def _trace_meta(self, process: Mapping[str, Any], call: ProcessInvocation) -> dict[str, Any]:
        """Provenance fields recorded with every event (PROV-002..003, trace policy)."""
        raw_trace = process.get("trace_policy", {})
        trace = raw_trace if isinstance(raw_trace, Mapping) else {}
        record_context = bool(trace.get("record_context", True))
        consumed_inputs = self._consumed_input_records(call)
        # The authorised context hash is part of the event's cryptographic identity
        # and is always recorded; the trace policy governs whether the context
        # content itself is retained (review finding 7).
        record: dict[str, Any] = {
            "context_hash": call.context.content_hash if call.context is not None else None,
            "input_refs": sorted(consumed_inputs),
            "parent_events": self._causal_parent_events(process, call),
            "executor_binding": _plain(process.get("executor", {})),
            "exposures": _plain(getattr(call.context, "exposures", ()))
            if call.context is not None
            else [],
        }
        if call.context is not None and record_context:
            record["context"] = _plain(call.context.data)
        return record

    @staticmethod
    def _consumed_input_records(call: ProcessInvocation) -> Mapping[str, Any]:
        """Return the input subset admitted to this invocation's context."""
        if call.context is None:
            return call.inputs
        context_data = getattr(call.context, "data", call.context)
        if isinstance(context_data, Mapping):
            inputs = context_data.get("inputs")
            if isinstance(inputs, Mapping):
                return inputs
        return {}

    def _causal_parent_events(
        self, process: Mapping[str, Any], call: ProcessInvocation
    ) -> list[str]:
        """Resolve causal parents from actual inputs and declared dependencies."""
        parents: list[str] = []
        for record in self._consumed_input_records(call).values():
            if isinstance(record, Mapping) and record.get("producer_event"):
                parents.append(str(record["producer_event"]))
        dependencies = process.get("dependencies", {})
        after = dependencies.get("after", []) if isinstance(dependencies, Mapping) else []
        for dependency in after if isinstance(after, list | tuple) else []:
            candidates = [
                event
                for event in self._completed_process_events
                if event.get("process_id") == dependency and event.get("phase", 0) <= call.phase
            ]
            if call.actor_ids:
                actor_matches = [
                    event
                    for event in candidates
                    if set(event.get("actors", ())) & set(call.actor_ids)
                ]
                if actor_matches:
                    candidates = actor_matches
            if candidates:
                parents.append(str(candidates[-1]["event_id"]))
        return list(dict.fromkeys(parents))

    @staticmethod
    def _state_delta(before: Mapping[str, Any] | None, after: Mapping[str, Any]) -> dict[str, Any]:
        before = before or {}
        delta = {key: value for key, value in after.items() if before.get(key) != value}
        delta.update({key: None for key in before if key not in after})
        return delta

    def preflight(
        self,
        *,
        phase_limit: int = 100,
        replications: int = 1,
        conditions: int = 1,
        max_events: int | None = None,
    ) -> dict[str, Any]:
        """Return a deterministic estimate without dispatching any executor."""
        if phase_limit < 1 or replications < 1 or conditions < 1:
            raise ValueError("phase_limit, replications, and conditions must be positive")
        process_count = len(self.scheduler.processes)
        estimated_runs = replications * conditions
        generative_processes = sum(
            1
            for process in self.scheduler.processes.values()
            if process.get("executor", {}).get("mode") == "generative"
        )
        return {
            "status": "ready",
            "phase_limit": phase_limit,
            "estimated_runs": estimated_runs,
            "estimated_process_invocations": estimated_runs * process_count,
            "estimated_generative_calls": generative_processes * replications,
            "dependencies_available": True,
            "limits": {"max_events": max_events, "replications": replications},
            "credentials_present": False,
            "warnings": ["external provider credentials are not configured"]
            if generative_processes
            else [],
        }

    def _restore_persisted_frontier(self, run_id: str) -> None:
        if (
            not self.persistence
            or self._persisted_count
            or not hasattr(self.persistence, "list_events")
        ):
            return
        events = self.persistence.list_events(run_id)
        self._persisted_count = len(events)
        if events:
            self._last_event_id = str(events[-1].get("event_id")) or None
        completed = [
            event
            for event in events
            if event.get("kind") in {"process_completed", "process_skipped"}
        ]
        if self.state_store:
            latest = self.persistence.latest_json_state(run_id)
            if latest is not None:
                version, snapshot = latest
                self.state_store.restore(snapshot, version)
        if self.artifact_store is not None and hasattr(self.persistence, "list_artifacts"):
            events_by_id = {
                str(event.get("event_id")): event for event in events if event.get("event_id")
            }
            for row in self.persistence.list_artifacts(run_id):
                try:
                    payload = json.loads(row["payload"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                declared_id = payload.get("declared_artifact_id")
                if (
                    not isinstance(declared_id, str)
                    or declared_id not in self.artifact_store.catalog
                ):
                    continue
                instance_id = str(row.get("artifact_id", ""))
                if not instance_id or instance_id in self.artifact_store._artifacts:
                    continue
                producer_event = payload.get("producer_event")
                producer = events_by_id.get(str(producer_event), {})
                self.artifact_store.put(
                    declared_id,
                    payload.get("value"),
                    owner=payload.get("owner"),
                    schema_ref=payload.get("schema_ref"),
                    visibility=payload.get("visibility"),
                    lifecycle_scope=payload.get("lifecycle_scope"),
                    lineage=list(payload.get("lineage", [])),
                    instance_id=instance_id,
                    actors=list(payload.get("actors", [])),
                    producer_process=payload.get("producer_process") or producer.get("process_id"),
                    producer_event=producer_event,
                    phase=payload.get("phase"),
                )
        if completed:
            self._next_phase = max(event.get("phase", 0) for event in completed)
        for event in events:
            if "dispatch_order" in event:
                self.dispatch_log.append(
                    {
                        "order": event["dispatch_order"],
                        "invocation_id": event.get("invocation_id"),
                        "process_id": event.get("process_id"),
                        "phase": event.get("phase", 0),
                        "attempt": event.get("attempt", 1),
                    }
                )
            if "commit_order" in event:
                self.commit_log.append(
                    {
                        "order": event["commit_order"],
                        "invocation_id": event.get("invocation_id"),
                        "process_id": event.get("process_id"),
                        "phase": event.get("phase", 0),
                        "status": (
                            "failed" if event.get("kind") == "process_failed" else "committed"
                        ),
                    }
                )
        completed_groups: dict[tuple[str, int | float], set[tuple[str, ...]]] = {}
        for event in completed:
            process_id = event.get("process_id")
            phase = event.get("phase", 0)
            actors = tuple(str(actor) for actor in event.get("actors", []))
            if process_id in self.scheduler.processes:
                self._completed_process_events.append(
                    {
                        "event_id": event.get("event_id"),
                        "process_id": process_id,
                        "phase": phase,
                        "actors": list(actors),
                    }
                )
                self._completed_actor_occurrences.add((str(process_id), phase, actors))
                completed_groups.setdefault((str(process_id), phase), set()).add(actors)
            for emitted in event.get("events", []):
                self._event_history.append(
                    {
                        **dict(emitted),
                        "producer_event": event.get("event_id"),
                        "producer_process": event.get("process_id"),
                        "actor_ids": list(event.get("actors", [])),
                        "phase": event.get("phase", 0),
                    }
                )
                event_name = emitted.get("type") or emitted.get("kind") or emitted.get("event")
                if event_name:
                    self.scheduler.signal_event(str(event_name))
            for effect in event.get("scheduling_effects", []):
                if effect.get("type") == "signal_event":
                    self.scheduler.signal_event(str(effect["event"]))
                elif effect.get("type") == "schedule":
                    self.scheduler.schedule(str(effect["process_id"]), effect["phase"])
        state_snapshot = self.state_store.snapshot() if self.state_store else {}
        for (process_id, phase), actor_groups in completed_groups.items():
            expected = set(
                expand_actor_instances(self.scheduler.processes[process_id], state_snapshot)
            )
            if expected.issubset(actor_groups):
                self.scheduler.consume_scheduled(process_id, phase)
                self.scheduler.complete(process_id, phase)
        self._event_count = len(completed)

    def _sync_persisted_state_version(self) -> None:
        if self.state_store:
            self.state_store.restore(self.state_store.snapshot(), self._persisted_count)

    @staticmethod
    def expand_runs(
        experiment_id: str, conditions: list[Mapping[str, Any]], replications: int
    ) -> list[dict[str, Any]]:
        _check_id(experiment_id, "experiment_id")
        if replications < 1:
            raise ValueError("replications must be positive")
        expanded = []
        for condition in conditions:
            condition_id = _check_id(str(condition["id"]), "condition_id")
            for replication in range(1, replications + 1):
                expanded.append(
                    {
                        "run_id": f"{experiment_id}-{condition_id}-{replication}",
                        "experiment_id": experiment_id,
                        "condition_id": condition_id,
                        "replication": replication,
                        "condition": dict(condition),
                    }
                )
        return expanded

    def _poll_external_status(self) -> bool:
        """Sync run control from persisted status (service/API cancellation)."""
        if self.status_provider is None or self.status in {"cancelled", "paused"}:
            return self.status in {"cancelled", "paused"}
        external = self.status_provider()
        if external in {"cancelled", "paused"}:
            self.status = external
            return True
        return False

    def run(
        self,
        run_id: str,
        phase_limit: int = 100,
        seed: int = 0,
        state: Mapping[str, Any] | None = None,
        max_events: int | None = None,
        *,
        experiment_id: str = "",
        condition_id: str = "",
        replication: int = 0,
        phase_start: int = 0,
        phase_end: int | None = None,
        condition: Mapping[str, Any] | None = None,
        matching: Mapping[str, Any] | None = None,
        terminal_phase: int | None = None,
        seed_identity: str | None = None,
    ) -> list[str]:
        if self.status in {"paused", "cancelled", "completed", "failed"}:
            return []
        if self._run_id is not None and self._run_id != run_id:
            raise ValueError("controller cannot be reused for a different run")
        self.status, self._run_id = "running", run_id
        if self.state_store is None and state is not None:
            # A caller may provide a lightweight initial snapshot without a
            # separately declared state schema. Infer the narrow runtime schema
            # for this convenience path; compiled studies still provide an
            # explicit schema through ``StateStore``.
            self.state_store = StateStore({key: type(value) for key, value in state.items()}, state)
        self._restore_persisted_frontier(run_id)
        if max_events is not None and self._event_count >= max_events:
            self.status = "completed"
            return []
        executed: list[str] = []
        condition = dict(condition or {})
        matching = dict(matching or {})
        stop_phase = phase_end + 1 if phase_end is not None else phase_limit
        start_phase = max(self._next_phase, phase_start)
        for phase in range(start_phase, stop_phase):
            if self._poll_external_status():
                return executed
            if self.status in {"paused", "cancelled"}:
                return executed
            executed_this_phase: set[str] = set()
            while True:
                if self._poll_external_status():
                    return executed
                scheduler_state = (
                    self.state_store.snapshot() if self.state_store else dict(state or {})
                )
                scheduler_state = {
                    **scheduler_state,
                    "condition": condition,
                    "protocol": {"phase": phase},
                }
                ready = [
                    item
                    for item in self.scheduler.ready(
                        phase,
                        state=scheduler_state,
                    )
                    if item.process_id not in executed_this_phase
                ]
                if not ready:
                    break
                item = ready[0]
                executed_this_phase.add(item.process_id)
                self.scheduler.consume_scheduled(item.process_id, phase)
                process = self.scheduler.processes[item.process_id]
                if (
                    terminal_phase is not None
                    and phase >= terminal_phase
                    and process.get("terminal_skip", False)
                ):
                    continue
                max_attempts = process.get("retry_policy", {}).get("max_attempts", 1)
                actor_key = (item.process_id, phase)
                if actor_key not in self._actor_queues:
                    actor_groups = expand_actor_instances(
                        process,
                        self.state_store.snapshot()
                        if self.state_store is not None
                        else (state or {}),
                    )
                    self._actor_queues[actor_key] = [
                        actors
                        for actors in actor_groups
                        if (item.process_id, phase, actors) not in self._completed_actor_occurrences
                    ]
                if not self._actor_queues[actor_key]:
                    self.scheduler.complete(item.process_id, phase)
                    continue
                actor_ids = self._actor_queues[actor_key].pop(0)
                actor_seed_id = "\x1f".join(actor_ids)
                actor_suffix = f"-{'-'.join(actor_ids)}" if actor_ids else ""
                for attempt in range(1, max_attempts + 1):
                    input_refs = process.get("inputs", [])
                    resolved_inputs = (
                        self.artifact_store.resolve(
                            list(input_refs), actor_ids=actor_ids, phase=phase
                        )
                        if self.artifact_store is not None and isinstance(input_refs, list | tuple)
                        else {}
                    )
                    binding = process.get("executor", {})
                    parameters = (
                        binding.get("parameters", {}) if isinstance(binding, Mapping) else {}
                    )
                    stream_id = (
                        str(parameters.get("random_stream", "conventional"))
                        if isinstance(parameters, Mapping)
                        else "conventional"
                    )
                    shared_streams = matching.get("shared_streams", [])
                    matching_key = (
                        stream_id
                        if matching.get("enabled") is True
                        and isinstance(shared_streams, list | tuple)
                        and stream_id in shared_streams
                        else None
                    )
                    call = ProcessInvocation(
                        f"{run_id}-{item.process_id}{actor_suffix}-{phase}",
                        run_id,
                        item.process_id,
                        actor_ids=actor_ids,
                        phase=phase,
                        state_version=self.state_store.version if self.state_store else 0,
                        inputs=resolved_inputs,
                        seed=derive_seed(
                            seed,
                            seed_identity or run_id,
                            item.process_id,
                            actor_seed_id,
                            experiment_id=experiment_id,
                            condition_id=condition_id,
                            replication=replication,
                            matching_key=matching_key,
                        ),
                        attempt=attempt,
                        condition=condition,
                        event_history=tuple(self._event_history),
                    )
                    policy_id = process.get("context_policy", "private")
                    call = ProcessInvocation(
                        call.invocation_id,
                        call.run_id,
                        call.process_id,
                        call.actor_ids,
                        call.phase,
                        call.time,
                        call.state_version,
                        call.inputs,
                        self.context_engine.build(
                            policy_id,
                            call,
                            self.state_store.snapshot()
                            if self.state_store is not None
                            else (state or {}),
                        ),
                        call.seed,
                        call.attempt,
                        process.get("executor", {}),
                        call.condition,
                        call.event_history,
                    )
                    dispatch_order = len(self.dispatch_log) + 1
                    self.dispatch_log.append(
                        {
                            "order": dispatch_order,
                            "invocation_id": call.invocation_id,
                            "process_id": item.process_id,
                            "phase": phase,
                            "attempt": attempt,
                        }
                    )
                    state_before = self.state_store.snapshot() if self.state_store else None
                    state_version_before = self.state_store.version if self.state_store else 0
                    artifact_before = (
                        self.artifact_store.snapshot() if self.artifact_store is not None else None
                    )
                    state_applied = False
                    persistence_committed = False
                    persistence_attempted = False
                    failure_persisted = False
                    executor_returned = False
                    failure_recorded = False
                    try:
                        if self.state_store is not None:
                            call = ProcessInvocation(
                                call.invocation_id,
                                call.run_id,
                                call.process_id,
                                call.actor_ids,
                                call.phase,
                                call.time,
                                self.state_store.version,
                                call.inputs,
                                self.context_engine.build(
                                    policy_id,
                                    call,
                                    self.state_store.snapshot(),
                                ),
                                call.seed,
                                call.attempt,
                                call.executor_binding,
                                call.condition,
                                call.event_history,
                            )
                        result = self.registry.execute(item.process_id, call)
                        executor_returned = True
                        retry_policy = process.get("retry_policy", {})
                        retry_policy = retry_policy if isinstance(retry_policy, Mapping) else {}
                        if result.status == "failed":
                            failure_policy = str(retry_policy.get("failure_policy", "fail_run"))
                            if failure_policy == "use_declared_fallback" and isinstance(
                                retry_policy.get("fallback_outputs"), Mapping
                            ):
                                result = ProcessResult(
                                    outputs=dict(retry_policy["fallback_outputs"]),
                                    metadata={
                                        **_plain(result.metadata),
                                        "fallback": True,
                                        "original_code": result.metadata.get("code"),
                                    },
                                )
                            elif failure_policy == "skip_with_event":
                                result = ProcessResult(
                                    status="skipped",
                                    outputs=dict(retry_policy.get("fallback_outputs", {})),
                                    metadata={**_plain(result.metadata), "skipped_fallback": True},
                                )
                        # F3/SCH-002: every declared artifact output — including a
                        # fallback — must satisfy its declared schema before any
                        # state or artifact commit. This is the common
                        # output-commit boundary, so all executors are covered.
                        if (
                            result.status == "succeeded"
                            and self.output_schema_validator is not None
                        ):
                            # F1: the schema comes from the executing process's
                            # own output declaration (process.outputs[].schema_ref),
                            # not from a separate domain-artifact id, so a
                            # process-declared schema is always enforced.
                            declared_outputs = process.get("outputs") or []
                            declared_schemas: list[tuple[str, str]] = []
                            for decl in declared_outputs:
                                if not isinstance(decl, Mapping):
                                    continue
                                artifact_type = decl.get("artifact_type")
                                schema_ref = decl.get("schema_ref")
                                if isinstance(artifact_type, str) and isinstance(schema_ref, str):
                                    declared_schemas.append((artifact_type, schema_ref))
                            schema_errors: list[str] = []
                            for artifact_id, schema_ref in declared_schemas:
                                if artifact_id not in (result.outputs or {}):
                                    continue
                                # Normalize frozen mappingproxies to plain
                                # JSON-able values before schema validation.
                                schema_value = _plain(result.outputs[artifact_id])
                                for message in self.output_schema_validator(
                                    schema_ref, schema_value
                                ):
                                    schema_errors.append(f"{artifact_id}: {message}")
                            if schema_errors:
                                metadata = dict(_plain(result.metadata))
                                metadata.update(
                                    {
                                        "code": "OUTPUT_VALIDATION_FAILED",
                                        "schema_valid": False,
                                        "validation_errors": schema_errors,
                                    }
                                )
                                result = ProcessResult(
                                    status="failed",
                                    outputs=result.outputs,
                                    metadata=metadata,
                                )
                        if result.status == "failed":
                            self.failures.append(
                                {
                                    "invocation_id": call.invocation_id,
                                    "attempt": attempt,
                                    "error": str(
                                        result.metadata.get(
                                            "error", result.metadata.get("code", "process failed")
                                        )
                                    ),
                                    "classification": result.metadata.get(
                                        "code", "executor_defect"
                                    ),
                                }
                            )
                            failure_recorded = True
                            if self.persistence:
                                persistence_attempted = True
                                failed_event_id = f"{call.invocation_id}-attempt-{attempt}"
                                failed_commit_order = len(self.commit_log) + 1
                                self.persistence.commit_process_result(
                                    {
                                        "event_id": failed_event_id,
                                        "invocation_id": call.invocation_id,
                                        "run_id": run_id,
                                        "kind": "process_failed",
                                        "process_id": item.process_id,
                                        "actors": list(call.actor_ids),
                                        "phase": phase,
                                        "attempt": attempt,
                                        "dispatch_order": dispatch_order,
                                        "commit_order": failed_commit_order,
                                        "metadata": _event_safe_metadata(result.metadata),
                                        **self._trace_meta(process, call),
                                        "state_delta": {},
                                    },
                                    {
                                        "run_id": run_id,
                                        "state_version": self._persisted_count + 1,
                                        "payload": json.dumps(
                                            self.state_store.snapshot() if self.state_store else {},
                                            sort_keys=True,
                                        ).encode(),
                                    },
                                    [],
                                )
                                self._persisted_count += 1
                                self._sync_persisted_state_version()
                                self._last_event_id = failed_event_id
                                self.commit_log.append(
                                    {
                                        "order": failed_commit_order,
                                        "invocation_id": call.invocation_id,
                                        "process_id": item.process_id,
                                        "phase": phase,
                                        "status": "failed",
                                    }
                                )
                            self.results.append(result)
                            if attempt == max_attempts:
                                self.status = "failed"
                                raise _ReportedProcessFailure(f"process {item.process_id} failed")
                            continue
                        if result.status == "skipped":
                            if self.persistence:
                                persistence_attempted = True
                                skipped_commit_order = len(self.commit_log) + 1
                                self.persistence.commit_process_result(
                                    {
                                        "event_id": f"{call.invocation_id}-attempt-{attempt}",
                                        "invocation_id": call.invocation_id,
                                        "run_id": run_id,
                                        "kind": "process_skipped",
                                        "process_id": item.process_id,
                                        "phase": phase,
                                        "attempt": attempt,
                                        "dispatch_order": dispatch_order,
                                        "commit_order": skipped_commit_order,
                                        "metadata": _event_safe_metadata(result.metadata),
                                        **self._trace_meta(process, call),
                                        "state_delta": {},
                                    },
                                    {
                                        "run_id": run_id,
                                        "state_version": self._persisted_count + 1,
                                        "payload": json.dumps(
                                            self.state_store.snapshot() if self.state_store else {},
                                            sort_keys=True,
                                        ).encode(),
                                    },
                                    [],
                                )
                                persistence_committed = True
                                self._persisted_count += 1
                                self._sync_persisted_state_version()
                                self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
                                self.commit_log.append(
                                    {
                                        "order": skipped_commit_order,
                                        "invocation_id": call.invocation_id,
                                        "process_id": item.process_id,
                                        "phase": phase,
                                        "status": "skipped",
                                    }
                                )
                            self.results.append(result)
                            break
                        _validate_scheduling_effects(self.scheduler, result.scheduling_effects)
                        raw_state_effects = process.get("state_effects", [])
                        declared: set[str] = set()
                        for effect_item in raw_state_effects:
                            if isinstance(effect_item, str):
                                declared.add(effect_item)
                            elif isinstance(effect_item, Mapping) and effect_item.get("field"):
                                declared.add(str(effect_item["field"]))
                        effects = result.state_effects or {
                            key: value for key, value in result.outputs.items() if key in declared
                        }
                        if self.state_store and effects:
                            self.state_store.apply(
                                effects, declared, expected_version=self.state_store.version
                            )
                            state_applied = True
                        declared_artifacts: list[dict[str, Any]] = []
                        if self.artifact_store is not None:
                            for artifact_id, value in result.outputs.items():
                                if artifact_id not in self.artifact_store.catalog:
                                    continue
                                catalog = self.artifact_store.catalog[artifact_id]
                                artifact_instance_id = (
                                    f"{artifact_id}-{call.invocation_id}-attempt-{attempt}"
                                )
                                producer_event = f"{call.invocation_id}-attempt-{attempt}"
                                consumed_input_ids = sorted(
                                    self._consumed_input_records(call).keys()
                                )
                                self.artifact_store.put(
                                    artifact_id,
                                    value,
                                    owner=catalog.get("owner"),
                                    schema_ref=catalog.get("schema_ref"),
                                    visibility=catalog.get("visibility"),
                                    lifecycle_scope=catalog.get("lifecycle_scope"),
                                    lineage=consumed_input_ids,
                                    instance_id=artifact_instance_id,
                                    actors=call.actor_ids,
                                    producer_process=item.process_id,
                                    producer_event=producer_event,
                                    phase=phase,
                                )
                                declared_artifacts.append(
                                    {
                                        "artifact_id": artifact_instance_id,
                                        "run_id": run_id,
                                        "payload": json.dumps(
                                            {
                                                "value": _plain(value),
                                                "content_hash": _hash(value),
                                                "schema_ref": catalog.get("schema_ref"),
                                                "owner": catalog.get("owner"),
                                                "visibility": catalog.get("visibility"),
                                                "lifecycle_scope": catalog.get("lifecycle_scope"),
                                                "producer_event": producer_event,
                                                "producer_process": item.process_id,
                                                "declared_artifact_id": artifact_id,
                                                "invocation_id": call.invocation_id,
                                                "phase": phase,
                                                "attempt": attempt,
                                                "actors": list(call.actor_ids),
                                                "lineage": consumed_input_ids,
                                            },
                                            sort_keys=True,
                                        ).encode(),
                                    }
                                )
                        if self.persistence:
                            persistence_attempted = True
                            trace = process.get("trace_policy", {})
                            trace = trace if isinstance(trace, Mapping) else {}
                            storage_outputs = dict(_plain(result.outputs))
                            omit_provider_bodies = trace.get("record_raw_response", True) is False
                            storage_metadata = (
                                _event_safe_metadata(result.metadata)
                                if omit_provider_bodies
                                else _plain(result.metadata)
                            )
                            if (
                                omit_provider_bodies
                                and (process.get("executor", {}).get("mode") == "generative")
                                and "response" in storage_outputs
                            ):
                                storage_outputs["response"] = "<raw-response-not-recorded>"
                            state_after = self.state_store.snapshot() if self.state_store else None
                            state_delta = (
                                self._state_delta(state_before, state_after)
                                if self.state_store and state_after is not None
                                else {}
                            )
                            payload = json.dumps(
                                {
                                    "outputs": storage_outputs,
                                    "raw_response": storage_metadata.get("raw_response"),
                                    "parsed_response": _plain(
                                        storage_metadata.get("parsed_response")
                                    ),
                                    "provider_attempts": _plain(
                                        storage_metadata.get("provider_attempts", [])
                                    ),
                                    "process_id": item.process_id,
                                    "invocation_id": call.invocation_id,
                                    "phase": phase,
                                    "attempt": attempt,
                                    "actors": list(call.actor_ids),
                                },
                                sort_keys=True,
                            ).encode()
                            state_payload = json.dumps(
                                state_after if state_after is not None else {},
                                sort_keys=True,
                            ).encode()
                            persist_version = self._persisted_count + 1
                            self.persistence.commit_process_result(
                                {
                                    "event_id": f"{call.invocation_id}-attempt-{attempt}",
                                    "invocation_id": call.invocation_id,
                                    "run_id": run_id,
                                    "kind": "process_completed",
                                    "process_id": item.process_id,
                                    "actors": list(call.actor_ids),
                                    "phase": phase,
                                    "attempt": attempt,
                                    "dispatch_order": dispatch_order,
                                    "commit_order": len(self.commit_log) + 1,
                                    "events": _plain(result.events),
                                    "metadata": _event_safe_metadata(result.metadata),
                                    "scheduling_effects": _plain(result.scheduling_effects),
                                    **self._trace_meta(process, call),
                                    "state_delta": state_delta,
                                },
                                {
                                    "run_id": run_id,
                                    "state_version": persist_version,
                                    "payload": state_payload,
                                },
                                [
                                    {
                                        "artifact_id": (f"{call.invocation_id}-attempt-{attempt}"),
                                        "run_id": run_id,
                                        "payload": payload,
                                    },
                                    *declared_artifacts,
                                ],
                            )
                            persistence_committed = True
                            self._persisted_count += 1
                            self._sync_persisted_state_version()
                            self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
                        commit_order = len(self.commit_log) + 1
                        self.commit_log.append(
                            {
                                "order": commit_order,
                                "invocation_id": call.invocation_id,
                                "process_id": item.process_id,
                                "phase": phase,
                            }
                        )
                        self._completed_process_events.append(
                            {
                                "event_id": f"{call.invocation_id}-attempt-{attempt}",
                                "process_id": item.process_id,
                                "phase": phase,
                                "actors": list(call.actor_ids),
                            }
                        )
                        self.results.append(result)
                        for emitted in result.events:
                            self._event_history.append(
                                {
                                    **_plain(emitted),
                                    "producer_event": (f"{call.invocation_id}-attempt-{attempt}"),
                                    "producer_process": item.process_id,
                                    "actor_ids": list(call.actor_ids),
                                    "phase": phase,
                                }
                            )
                            event_name = (
                                emitted.get("type") or emitted.get("kind") or emitted.get("event")
                            )
                            if event_name:
                                self.scheduler.signal_event(str(event_name))
                        for effect in result.scheduling_effects:
                            effect_type = effect.get("type")
                            if effect_type == "signal_event" and effect.get("event"):
                                self.scheduler.signal_event(str(effect["event"]))
                            elif effect_type == "schedule":
                                target = effect.get("process_id")
                                scheduled_phase = effect.get("phase")
                                self.scheduler.schedule(
                                    str(target), cast(int | float, scheduled_phase)
                                )
                        break
                    except _ReportedProcessFailure:
                        raise
                    except Exception as exc:
                        if state_applied and not persistence_committed and self.state_store:
                            self.state_store.restore(state_before or {}, state_version_before)
                        if (
                            artifact_before is not None
                            and not persistence_committed
                            and self.artifact_store is not None
                        ):
                            self.artifact_store.restore(artifact_before)
                        if not failure_recorded:
                            failure = {
                                "invocation_id": call.invocation_id,
                                "attempt": attempt,
                                "error": str(exc),
                            }
                            if not executor_returned:
                                failure["classification"] = "executor_exception"
                            self.failures.append(failure)
                            failure_recorded = True
                        if not persistence_attempted and self.persistence:
                            failed_commit_order = len(self.commit_log) + 1
                            classification = (
                                "executor_exception" if not executor_returned else "invalid_output"
                            )
                            try:
                                persistence_attempted = True
                                self.persistence.commit_process_result(
                                    {
                                        "event_id": f"{call.invocation_id}-attempt-{attempt}",
                                        "invocation_id": call.invocation_id,
                                        "run_id": run_id,
                                        "kind": "process_failed",
                                        "process_id": item.process_id,
                                        "actors": list(call.actor_ids),
                                        "phase": phase,
                                        "attempt": attempt,
                                        "dispatch_order": dispatch_order,
                                        "commit_order": failed_commit_order,
                                        "classification": classification,
                                        "error": str(exc),
                                        **self._trace_meta(process, call),
                                        "state_delta": {},
                                    },
                                    {
                                        "run_id": run_id,
                                        "state_version": self._persisted_count + 1,
                                        "payload": json.dumps(
                                            self.state_store.snapshot() if self.state_store else {},
                                            sort_keys=True,
                                        ).encode(),
                                    },
                                    [],
                                )
                            except Exception:
                                self.status = "failed"
                                raise
                            self._persisted_count += 1
                            self._sync_persisted_state_version()
                            self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
                            failure_persisted = True
                            self.commit_log.append(
                                {
                                    "order": failed_commit_order,
                                    "invocation_id": call.invocation_id,
                                    "process_id": item.process_id,
                                    "phase": phase,
                                    "status": "failed",
                                }
                            )
                        if attempt == max_attempts:
                            self.status = "failed"
                            raise
                        if (
                            persistence_attempted
                            and not persistence_committed
                            and not failure_persisted
                        ):
                            self.status = "failed"
                            raise
                executed.append(item.process_id)
                self._event_count += 1
                if max_events is not None and self._event_count >= max_events:
                    self.status = "completed"
                    return executed
                actor_key = (item.process_id, phase)
                if self._actor_queues.get(actor_key):
                    executed_this_phase.discard(item.process_id)
                    continue
                self._actor_queues.pop(actor_key, None)
                self.scheduler.complete(item.process_id, phase)
            self._next_phase = phase + 1
            if self.status in {"cancelled", "paused"}:
                break
        if self.status not in {"cancelled", "paused"}:
            self.status = "completed"
        return executed

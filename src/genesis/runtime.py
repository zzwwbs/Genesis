"""Execution contracts and deterministic local runtime primitives."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import product
from types import MappingProxyType
from typing import Any, cast

from genesis.information_timing import (
    COMPOSABLE_OPS,
    MODEL_CALL_MODES,
    batch_dependencies,
    can_ready_others,
    executor_mode,
    timing_of,
)

# measurement imports runtime lazily, inside its functions, so this does not cycle.
from genesis.measurement import withheld_sources, withhold_instances
from genesis.provider_errors import is_provider_cancellation, provider_pause_reason
from genesis.state_encoding import identical

_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

# The declared vocabulary for ``domain.states[].value_type`` and the Python
# types each admits. ``number`` accepts integers because JSON has one numeric
# type; ``integer``/``number`` reject bool despite bool subclassing int.
STATE_VALUE_TYPES: dict[str, type | tuple[type, ...]] = {
    "integer": int,
    "number": (int, float),
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
    "json": dict,
}


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


class _ProviderPause(RuntimeError):
    """A call the provider could not serve; the run pauses rather than fails."""

    def __init__(self, reason: Mapping[str, Any]) -> None:
        super().__init__(str(reason.get("error", "provider unavailable")))
        self.reason = dict(reason)


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


def _resolve_per(per: Mapping[str, Any], actor_id: str) -> dict[str, Any]:
    """The per selector with ``${actor}`` bound to this turn's outer actor."""
    resolved = dict(per)
    source = resolved.get("source")
    if isinstance(source, str):
        resolved["source"] = source.replace("${actor}", actor_id)
    return resolved


def _selector_ids(
    selector: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    what: str,
    allow_missing: bool = False,
) -> list[str]:
    """The actor ids one selector draws, in a deterministic order.

    Shared by the outer selector and by ``per``, so a nested selector resolves
    its source exactly the way a top-level one does.
    """
    if selector.get("ids") is not None:
        raw_ids = selector.get("ids")
        if not isinstance(raw_ids, list | tuple):
            raise ValueError(f"{what} ids must be a list")
        return [_check_id(str(actor), "actor_id") for actor in raw_ids]
    source = selector.get("source")
    if not isinstance(source, str):
        raise ValueError(f"{what} requires ids or source")
    records = _resolve_path(state, source)
    if records is None:
        # A per source names a path under one actor's own record. That the actor
        # has no such record yet is an ordinary state of the world -- they have
        # not acted this round -- not a broken declaration.
        if allow_missing:
            return []
        raise ValueError(f"{what} source is unavailable: {source}")
    if isinstance(records, Mapping):
        records = list(records.values())
    if not isinstance(records, list | tuple):
        raise ValueError(f"{what} source must resolve to a list or mapping")
    id_field = str(selector.get("id_field", "id"))
    ids = [
        _check_id(str(record.get(id_field) if isinstance(record, Mapping) else record), "actor_id")
        for record in records
    ]
    ids.sort()
    return ids


def actor_roles(process: Mapping[str, Any]) -> tuple[str, ...]:
    """The role name of each position in this process's actor groups.

    A process without a ``per`` selector has one role, so an output's actor
    field can name it or stay an unqualified list. A paired process has two, and
    an actor field has to say which one it means.
    """
    actors = process.get("actors")
    if not isinstance(actors, Mapping):
        return ("actor",)
    roles = [str(actors.get("role") or "actor")]
    per = actors.get("per")
    if isinstance(per, Mapping):
        roles.append(str(per.get("role") or "per"))
    return tuple(roles)


def expand_actor_instances(
    process: Mapping[str, Any], state: Mapping[str, Any]
) -> list[tuple[str, ...]]:
    """Resolve a process actor declaration into deterministic invocation groups."""
    actors = process.get("actors")
    if actors is None:
        return [()]
    if isinstance(actors, list | tuple):
        ids = [_check_id(str(actor), "actor_id") for actor in actors]
        if len(ids) != len(set(ids)):
            # Two identical groups share an invocation id, so the run aborted
            # part-way with an idempotency failure -- and a resumed run dropped
            # the duplicate instead, making a compiled declaration mean one
            # thing before a pause and another after it.
            raise ValueError("actors list contains duplicate actor ids")
        return [(actor_id,) for actor_id in ids] or [()]
    if not isinstance(actors, Mapping):
        raise ValueError("actors must be a list or actor selector")
    fan_out = bool(actors.get("fan_out", True))
    ids = _selector_ids(actors, state, what="actor selector")
    if len(ids) != len(set(ids)):
        raise ValueError("actor selector produced duplicate actor ids")
    per = actors.get("per")
    if per is not None:
        if not isinstance(per, Mapping):
            raise ValueError("actor selector per must be a mapping")
        if not fan_out:
            # A per selector exists to make one invocation per inner record; with
            # fan_out off there is a single invocation and nothing to pair it to.
            raise ValueError("actor selector per requires fan_out")
        groups: list[tuple[str, ...]] = []
        for actor_id in ids:
            resolved = _resolve_per(per, actor_id)
            inner = _selector_ids(resolved, state, what="actor selector per", allow_missing=True)
            if len(inner) != len(set(inner)):
                raise ValueError(
                    f"actor selector per produced duplicate ids for actor '{actor_id}'"
                )
            # An actor whose inner source is empty takes no turn at all: a reader
            # who opened nothing is not asked what they thought of it.
            groups.extend((actor_id, inner_id) for inner_id in inner)
        if len(groups) != len(set(groups)):
            raise ValueError("actor selector produced duplicate actor groups")
        return groups
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


def _validate_condition(condition: Any) -> None:
    """Validate a predicate or an ``all``/``any``/``not`` composition of predicates."""
    if not isinstance(condition, Mapping):
        raise ValueError("condition predicate must be a mapping")
    combinators = sorted({"all", "any", "not"} & set(condition))
    if len(combinators) > 1:
        # Only the first was ever evaluated, so the rest read as a condition
        # that was being applied and was not.
        raise ValueError(f"condition declares {', '.join(combinators)} together; use one")
    for key in ("all", "any"):
        if key in condition:
            children = condition[key]
            if not isinstance(children, list | tuple) or not children:
                raise ValueError(f"condition {key} requires a non-empty list")
            for child in children:
                _validate_condition(child)
            return
    if "not" in condition:
        _validate_condition(condition["not"])
        return
    _validate_predicate(condition)


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


def edge_delays(dependencies: Any, after: Sequence[str]) -> dict[str, int | float]:
    """Resolve the declared delay for each dependency edge of one process.

    ``dependencies.delay`` accepts two forms. ``{"rounds": N}`` (or a bare
    number) is the process-wide default applied to every edge. ``per_dependency``
    maps individual dependency ids to their own lag and takes precedence, so a
    consumer can follow one producer immediately and another with a lag —
    a distinction the theory layer needs and a single process-level number
    cannot express. Edges with no declared delay are immediate (0).
    """
    delay = dependencies.get("delay") if isinstance(dependencies, Mapping) else dependencies
    default: int | float = 0
    per_dependency: Mapping[str, Any] = {}
    if isinstance(delay, Mapping):
        rounds = delay.get("rounds")
        if isinstance(rounds, int | float) and not isinstance(rounds, bool):
            default = rounds
        candidate = delay.get("per_dependency")
        if isinstance(candidate, Mapping):
            per_dependency = candidate
    elif isinstance(delay, int | float) and not isinstance(delay, bool):
        default = delay
    resolved: dict[str, int | float] = {}
    for dependency in after:
        value = per_dependency.get(str(dependency), default)
        resolved[str(dependency)] = (
            value if isinstance(value, int | float) and not isinstance(value, bool) else 0
        )
    return resolved


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
    feedback_slots: Mapping[str, Any] = field(default_factory=dict)
    # This actor's own earlier exchanges with a process, per process id: what it
    # was given and what it answered. A policy admits them by name.
    exchanges: Mapping[str, Any] = field(default_factory=dict)

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
        object.__setattr__(self, "exchanges", _freeze(dict(self.exchanges)))
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


# The namespaces a context policy's allow entry may start with. Any other first
# segment is read as a state id.
CONTEXT_ROOTS = ("state", "inputs", "condition", "actor", "events", "feedback", "exchanges")


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
        scope = definition.get("scope", {}) if isinstance(definition, Mapping) else {}
        aggregate = definition.get("aggregate", {}) if isinstance(definition, Mapping) else {}
        project = definition.get("project", {}) if isinstance(definition, Mapping) else {}
        # CONTEXT_ROOTS names these keys; the compiler refuses an allow entry
        # that starts with anything else and is not a declared state.
        source_root = {
            "state": state,
            "inputs": invocation.inputs,
            "condition": invocation.condition,
            "actor": {"ids": invocation.actor_ids},
            "events": invocation.event_history,
            "feedback": invocation.feedback_slots,
            "exchanges": invocation.exchanges,
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
            if path in scope:
                value = _apply_scope(value, scope[path], invocation, state)
            if path in cardinality:
                value = _cap_cardinality(value, cardinality[path])
            # After the cap, so a cap may order by a field the projection drops.
            if path in project:
                value = _project_value(value, path, project[path])
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
        conditions = _availability_rule(
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
            # all/any/not are accepted everywhere else a predicate is written,
            # and the timing and measurement analyses already read them here;
            # evaluating only a bare predicate raised on a valid declaration.
            if not _evaluate_condition(predicate, namespace):
                return False
        return True


# Keys an availability rule recognises besides a predicate.
AVAILABILITY_KEYS = frozenset(
    {"after_round", "before_round", "event", "recipient_match", "source_match_event", "predicate"}
)
# The keys a predicate is written with, per the documented grammar.
PREDICATE_KEYS = frozenset({"path", "op", "value", "all", "any", "not"})


def _availability_rule(conditions: Mapping[str, Any]) -> Mapping[str, Any]:
    """An availability rule with any inline predicate moved under ``predicate``.

    The documented grammar -- and the compiler, the timing analysis and the
    measurement analysis -- write availability as a predicate directly:
    ``{path, op, value}`` or ``all``/``any``/``not``. This reader looked for a
    predicate only under ``predicate:``, so a rule written as documented matched
    none of its keys, skipped every check, and left the item available in every
    round of every condition. Round and event keys may sit beside it.
    """
    inline = {key: value for key, value in conditions.items() if key in PREDICATE_KEYS}
    if not inline or "predicate" in conditions:
        return conditions
    return {
        **{key: value for key, value in conditions.items() if key not in PREDICATE_KEYS},
        "predicate": inline,
    }


# Names the mapping key itself as the scoping field, for relations stored as
# {actor_id: rows} rather than as a list of actor-tagged rows.
SCOPE_KEY_FIELD = "__key__"


def _scope_selector(
    paths: Any, invocation: ProcessInvocation, state: Mapping[str, Any]
) -> set[Any]:
    """The set of identities a scoped field is filtered against (CTX-002).

    Each declared path resolves against the invocation and state. ``${actor}``
    expands to each acting actor in turn and the results are unioned, so a
    relation keyed by actor works for a multi-actor invocation (CTX-006).
    Nothing is added implicitly: a policy that wants the actor's own rows names
    ``actor.ids`` (CTX-003).
    """
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list | tuple):
        raise ValueError("CONTEXT_SCOPE: 'in' must be a path or a list of paths")
    namespace = {"actor": {"ids": list(invocation.actor_ids)}, "state": _plain(state)}
    selector: set[Any] = set()
    for raw in paths:
        if not isinstance(raw, str):
            raise ValueError("CONTEXT_SCOPE: selector paths must be strings")
        expansions = (
            [raw.replace("${actor}", str(actor)) for actor in invocation.actor_ids]
            if "${actor}" in raw
            else [raw]
        )
        for path in expansions:
            # An unresolved relation is the empty set, a visible outcome, not an
            # error: a study may legitimately have no relation yet (CTX-004).
            # ``*`` maps over a collection, so a selector can name a field of
            # every element -- the authors of the articles in an actor's feed --
            # rather than only a field that already holds the identity list.
            for resolved in _resolve_selector_values(namespace, path):
                collection = isinstance(resolved, list | tuple | set | frozenset)
                for item in resolved if collection else (resolved,):
                    if not isinstance(item, str | int | float | bool):
                        # Silently skipping would hand every actor an empty view
                        # for the whole run, which reads as a finding rather than
                        # as the mis-declaration it is.
                        raise ValueError(
                            f"CONTEXT_SCOPE: selector '{path}' resolved to "
                            f"{type(item).__name__}, which cannot identify a record; name a "
                            "path that resolves to identities"
                        )
                    selector.add(item)
    return selector


def _resolve_selector_values(source: Any, path: str) -> list[Any]:
    """Every value a selector path reaches; ``*`` expands list items or mapping values."""
    nodes: list[Any] = [source]
    for part in path.split("."):
        following: list[Any] = []
        for node in nodes:
            if part == "*":
                if isinstance(node, Mapping):
                    following.extend(node.values())
                elif isinstance(node, list | tuple | set | frozenset):
                    following.extend(node)
            elif isinstance(node, Mapping):
                if part in node:
                    following.append(node[part])
            elif isinstance(node, list | tuple) and part.isdigit() and int(part) < len(node):
                following.append(node[int(part)])
            elif isinstance(node, list | tuple | set | frozenset):
                # A field named over a collection reads it from each element, so
                # a relation stored as {actor: [rows]} resolves as readily as one
                # stored as a list of rows.
                following.extend(
                    entry[part] for entry in node if isinstance(entry, Mapping) and part in entry
                )
        nodes = following
    return [node for node in nodes if node is not None]


def _field_value(item: Any, field: str) -> Any:
    """A record's value at a dotted field path, or None."""
    if not isinstance(item, Mapping):
        return None
    if "." not in field:
        return item.get(field)
    return _resolve_path(item, field)


def _project_nested(item: Any, paths: frozenset[str], project: Any) -> Any | None:
    """Apply a nested projection to a record, or to each record in a list.

    A nested field is as often a list of records as a single one -- the case
    this whole feature exists for, an author's private text inside an article,
    is a list -- and recursing only into mappings made the projection a silent
    no-op there: declared, compiled, and handing over the field it promised to
    withhold. Returns None when the value is neither, leaving it to the caller.
    """
    if isinstance(item, Mapping):
        return project(item, paths)
    if isinstance(item, list | tuple) and any(
        isinstance(entry, Mapping | list | tuple) for entry in item
    ):
        # A list of lists of records is still the collection the projection
        # names. Recursing only one level made the rule a silent no-op there:
        # declared, compiled, and handing over the field it promised to
        # withhold, which is the failure this whole function exists to prevent.
        projected = []
        for entry in item:
            if isinstance(entry, Mapping):
                projected.append(project(entry, paths))
                continue
            nested = _project_nested(entry, paths, project)
            projected.append(_plain(entry) if nested is None else nested)
        return projected
    return None


def _kept_paths(record: Mapping[str, Any], paths: frozenset[str]) -> dict[str, Any]:
    """The named paths of a record, nested as they were, in the record's own order."""
    heads: dict[str, set[str]] = {}
    whole: set[str] = set()
    for path in paths:
        head, _, rest = path.partition(".")
        if rest:
            heads.setdefault(head, set()).add(rest)
        else:
            whole.add(head)
    kept: dict[str, Any] = {}
    for name, item in record.items():
        if name in whole:
            kept[name] = _plain(item)
        elif name in heads:
            nested = _project_nested(item, frozenset(heads[name]), _kept_paths)
            if nested:
                kept[name] = nested
    return kept


def _dropped_paths(record: Mapping[str, Any], paths: frozenset[str]) -> dict[str, Any]:
    """A record without the named paths, nested paths included."""
    heads: dict[str, set[str]] = {}
    whole: set[str] = set()
    for path in paths:
        head, _, rest = path.partition(".")
        if rest:
            heads.setdefault(head, set()).add(rest)
        else:
            whole.add(head)
    result: dict[str, Any] = {}
    for name, item in record.items():
        if name in whole:
            continue
        nested = (
            _project_nested(item, frozenset(heads[name]), _dropped_paths) if name in heads else None
        )
        result[name] = _plain(item) if nested is None else nested
    return result


def _project_value(value: Any, path: str, rule: Any) -> Any:
    """Keep or drop named fields of the records a context path hands over.

    A name may be a dotted path into a record, because the records a policy
    hands over are nested: the case this exists for -- an article without its
    author's private strategy text -- is exactly a nested field, and comparing
    only top-level names let such a projection compile and do nothing.

    What counts as a record is declared, not guessed. ``applies_to: records``
    (the default) projects a list's items or a mapping's values;
    ``applies_to: record`` projects the value itself. Guessing from the value's
    shape silently dropped whole rows when a mapping held lists rather than
    records. Projection never adds a field and preserves order, so the recorded
    context digest stays stable.
    """
    keep, drop, applies_to = _project_rule(rule)

    def project(record: Any, where: str) -> Any:
        if not isinstance(record, Mapping):
            raise ValueError(
                f"CONTEXT_PROJECT: '{path}' declares a projection, but {where} is "
                f"{type(record).__name__}, not a record"
            )
        if keep is not None:
            return _kept_paths(record, keep)
        return _dropped_paths(record, drop)

    if applies_to == "record":
        return project(value, "the value")
    if isinstance(value, list | tuple):
        return [project(item, "an element") for item in value]
    if isinstance(value, Mapping):
        return {key: project(item, f"the entry '{key}'") for key, item in value.items()}
    raise ValueError(
        f"CONTEXT_PROJECT: '{path}' declares a projection over records, but the value is "
        f"{type(value).__name__}; declare applies_to: record to project it directly"
    )


def _project_rule(rule: Any) -> tuple[frozenset[str] | None, frozenset[str], str]:
    """Read a projection rule as (fields to keep or None, fields to drop, target)."""
    if not isinstance(rule, Mapping):
        raise ValueError("projection rule must be a mapping with exactly one of keep or drop")
    applies_to = str(rule.get("applies_to", "records"))
    if applies_to not in {"records", "record"}:
        raise ValueError("projection 'applies_to' must be 'records' or 'record'")
    selectors = {name: value for name, value in rule.items() if name != "applies_to"}
    if len(selectors) != 1 or not set(selectors) <= {"keep", "drop"}:
        raise ValueError("projection rule must be a mapping with exactly one of keep or drop")
    ((mode, names),) = selectors.items()
    if (
        not isinstance(names, list | tuple)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(f"projection '{mode}' must be a non-empty list of field names")
    for name in names:
        if name.startswith(".") or name.endswith(".") or ".." in name:
            raise ValueError(
                f"projection '{mode}' name '{name}' is not a field or a dotted path into one"
            )
    if mode == "keep":
        return frozenset(names), frozenset(), applies_to
    return None, frozenset(names), applies_to


def _apply_scope(
    value: Any, rule: Any, invocation: ProcessInvocation, state: Mapping[str, Any]
) -> Any:
    """Keep only the elements this actor is entitled to see (CTX-001).

    Source order is preserved and nothing is deduplicated: the projection is
    recorded evidence and its digest must be stable (CTX-007).
    """
    if not isinstance(rule, Mapping):
        raise ValueError("CONTEXT_SCOPE: scope rule must be a mapping")
    unknown = set(rule) - {"field", "in"}
    if unknown:
        raise ValueError(
            f"CONTEXT_SCOPE: scope rule has unknown keys: {', '.join(sorted(unknown))}; "
            "expected field, in"
        )
    field = rule.get("field")
    if not isinstance(field, str) or not field:
        raise ValueError("CONTEXT_SCOPE: scope rule requires a 'field'")
    if "in" not in rule:
        raise ValueError("CONTEXT_SCOPE: scope rule requires 'in'")
    selector = _scope_selector(rule["in"], invocation, state)

    def entitled(item: Any) -> bool:
        found = _field_value(item, field)
        if isinstance(found, list | tuple | set | frozenset):
            # A record may name several identities -- an article with two
            # authors. Requiring a scalar dropped every such record and handed
            # the actor an empty view with nothing to say why.
            return any(
                isinstance(entry, str | int | float | bool) and entry in selector for entry in found
            )
        if found is None or isinstance(found, Mapping):
            return False
        return isinstance(found, str | int | float | bool) and found in selector

    if isinstance(value, list | tuple):
        kept = [item for item in value if entitled(item)]
        return type(value)(kept) if isinstance(value, tuple) else kept
    if isinstance(value, Mapping):
        if field == SCOPE_KEY_FIELD:
            return {key: item for key, item in value.items() if key in selector}
        return {key: item for key, item in value.items() if entitled(item)}
    return value


def _exchange_retention(policies: Mapping[str, Any]) -> dict[str, int]:
    """Per process, how many of its exchanges any policy could still read.

    Absent from the mapping means unbounded: either no cap is declared, or a
    declared cap keeps the head or orders by a field, and trimming the tail
    would discard what such a cap selects.
    """
    bounds: dict[str, int] = {}
    unbounded: set[str] = set()
    for policy in policies.values():
        if not isinstance(policy, Mapping):
            continue
        caps = policy.get("cardinality") or {}
        for path in policy.get("allow") or ():
            parts = str(path).split(".")
            if parts[0] != "exchanges" or len(parts) < 2:
                continue
            process_id = parts[1]
            rule = caps.get(str(path)) if isinstance(caps, Mapping) else None
            if rule is None:
                unbounded.add(process_id)
                continue
            try:
                limit, keep, by = _cap_rule(rule)
            except ValueError:
                unbounded.add(process_id)
                continue
            if keep != "last" or by is not None:
                unbounded.add(process_id)
                continue
            bounds[process_id] = max(bounds.get(process_id, 0), limit)
    return {
        process_id: limit for process_id, limit in bounds.items() if process_id not in unbounded
    }


def _cap_rule(rule: Any) -> tuple[int, str, str | None]:
    """Read a cardinality cap as (limit, which end, ordering field).

    A bare integer keeps the first N, which is what the cap has always done. It
    is the wrong default for an append-ordered field -- it pins every actor to
    the oldest entries and hides everything recent -- so a cap may now say which
    end it wants, and optionally which field orders the elements.
    """
    limit: Any = rule
    keep = "first"
    by: str | None = None
    if isinstance(rule, Mapping):
        # Unknown keys are rejected, as the package models reject them. A
        # mistyped "keep" would otherwise leave the cap silently taking the
        # oldest entries while the declaration reads as if it takes the newest,
        # and both produce a plausible number of rows.
        unknown = set(rule) - {"limit", "keep", "by"}
        if unknown:
            raise ValueError(
                f"cardinality cap has unknown keys: {', '.join(sorted(unknown))}; "
                "expected limit, keep, by"
            )
        limit = rule.get("limit")
        keep = str(rule.get("keep", "first"))
        raw_by = rule.get("by")
        if raw_by is not None:
            if not isinstance(raw_by, str) or not raw_by:
                raise ValueError("cardinality 'by' must be a non-empty field name")
            by = raw_by
        if keep not in {"first", "last"}:
            raise ValueError("cardinality 'keep' must be 'first' or 'last'")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("cardinality cap must be a non-negative integer")
    return limit, keep, by


def _cap_cardinality(value: Any, rule: Any) -> Any:
    """Deterministically truncate a list/dict to its declared cardinality cap.

    Whichever elements are kept, they are returned in source order: the cap
    decides *which* an actor sees, never the order they see them in.
    """
    limit, keep, by = _cap_rule(rule)
    if isinstance(value, list):
        entries: list[Any] = list(value)
    elif isinstance(value, dict):
        entries = list(value.items())
    else:
        return value
    positions = list(range(len(entries)))
    if by is not None:

        def ordering(position: int) -> tuple[Any, int]:
            entry = entries[position]
            if isinstance(value, dict):
                key, item = entry
                if by == SCOPE_KEY_FIELD:
                    return (key, position)
                entry = item
            field = entry.get(by) if isinstance(entry, Mapping) else None
            # A missing ordering value sorts before present ones rather than
            # raising, and position breaks ties so the result is stable.
            return ((field is not None, field), position)

        try:
            positions.sort(key=ordering)
        except TypeError as exc:
            raise ValueError(f"cardinality 'by' field '{by}' does not order") from exc
    # Clamped: a cap larger than the collection keeps all of it. Unclamped, the
    # start went negative and sliced from the end, so a field just past half its
    # cap kept almost nothing.
    start = max(0, len(positions) - limit)
    chosen = positions[:limit] if keep == "first" else positions[start:]
    kept = set(chosen)
    if isinstance(value, dict):
        return {key: item for position, (key, item) in enumerate(entries) if position in kept}
    return [item for position, item in enumerate(entries) if position in kept]


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
        schema: Mapping[str, type | tuple[type, ...]] | None = None,
        initial: Mapping[str, Any] | None = None,
        reducers: Mapping[str, Callable[[Any, Any], Any]] | None = None,
    ):
        self.schema = dict(schema or {})
        self.reducers = dict(reducers or {})
        self._state = dict(initial or {})
        for key, value in self._state.items():
            if key not in self.schema or not self._type_ok(self.schema[key], value):
                raise TypeError(f"invalid type for state field {key}")
        self.version = 0

    @staticmethod
    def _type_ok(expected: type | tuple[type, ...], value: Any) -> bool:
        """Whether ``value`` satisfies a declared state type.

        ``bool`` subclasses ``int``, so a numeric field would otherwise silently
        accept ``True``; only a field declared ``boolean`` may hold one.
        """
        allowed = expected if isinstance(expected, tuple) else (expected,)
        if isinstance(value, bool) and bool not in allowed:
            return False
        return isinstance(value, allowed)

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
                key = effect.get("key")
                if op == "set":
                    candidate[field] = value
                elif op == "increment":
                    current = candidate.get(field, 0)
                    if (
                        not isinstance(current, int | float)
                        or not isinstance(value, int | float)
                        or isinstance(current, bool)
                        or isinstance(value, bool)
                    ):
                        # Unchecked, "x" + "y" and ["a"] + ["b"] were recorded as
                        # successful increments of a string and a list.
                        raise TypeError("increment requires a numeric field and value")
                    candidate[field] = current + value
                elif op == "append" and key is not None:
                    current = candidate.get(field) or {}
                    if not isinstance(current, Mapping):
                        raise TypeError("keyed append requires a mapping state field")
                    entries = current.get(str(key)) or []
                    if not isinstance(entries, list):
                        raise TypeError("keyed append requires a list under the key")
                    candidate[field] = {**current, str(key): [*entries, value]}
                elif op == "append":
                    candidate[field] = list(candidate.get(field) or []) + [value]
                elif op == "put":
                    current = candidate.get(field) or {}
                    if key is None or not isinstance(current, Mapping):
                        raise TypeError("put requires a key and a mapping state field")
                    candidate[field] = {**current, str(key): value}
                elif op in {"add-relation", "remove-relation"}:
                    current = candidate.get(field) or []
                    if not isinstance(current, list) or not isinstance(value, Mapping):
                        raise TypeError(f"{op} requires a list state field and a mapping value")
                    if op == "remove-relation":
                        candidate[field] = [item for item in current if item != value]
                    elif value not in current:
                        candidate[field] = [*current, dict(value)]
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
            # ``!=`` calls 1, 1.0 and True equal; JSON writes three different
            # byte strings, and the record's identity is digested from those.
            # Filtering with it dropped a type change as "unchanged", so an
            # invalid type never reached the schema check below.
            effects = {k: v for k, v in candidate.items() if not identical(self._state.get(k), v)}
        if any(k not in self.schema or k not in declared for k in effects):
            raise PermissionError("state effect is not declared in the state model")
        for key, value in effects.items():
            if not self._type_ok(self.schema[key], value):
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
            if key not in self.schema or not self._type_ok(self.schema[key], value):
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

    # Declared lifecycle scopes that do not outlive the round that produced
    # them. Anything else (``run``, or an undeclared scope) persists for the
    # whole run, which is the conservative default.
    _ROUND_SCOPED = frozenset({"round", "phase", "event", "invocation"})

    def _in_scope(self, metadata: Mapping[str, Any], phase: int | float | None) -> bool:
        """Whether a retained instance is still live at ``phase``.

        ``lifecycle_scope`` declares how long an artifact persists (spec
        §3.3: "how long they persist"). Without this, a round-scoped artifact
        accumulated every prior instance for the whole run, so by round N a
        consumer's authorized context carried all N-1 earlier rounds.
        """
        produced = metadata.get("phase")
        if phase is None or produced is None:
            return True
        if produced > phase:
            return False
        scope = metadata.get("lifecycle_scope")
        if isinstance(scope, str) and scope in self._ROUND_SCOPED:
            return bool(produced == phase)
        return True

    def resolve(
        self,
        references: list[str] | tuple[str, ...],
        *,
        actor_ids: tuple[str, ...] = (),
        phase: int | float | None = None,
        visible: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Resolve declared artifact/process references to immutable instances.

        ``visible`` limits resolution to instances that already existed at some
        moment -- the start of a simultaneous batch (CON-008).

        Instances outside their declared ``lifecycle_scope`` are excluded.
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
                and self._in_scope(metadata, phase)
                and (visible is None or instance_id in visible)
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
    """Build the evaluation namespace for declarative (bounded) executors.

    Bounded executors read the SAME authorized context as generative ones.
    Reading ``invocation.inputs``/``event_history``/``condition`` directly would
    bypass the declared context policy, so a rule or state transition could see
    artifacts and events its policy never admitted — the policy would bind only
    the model-backed executors.
    """
    context = getattr(invocation.context, "data", invocation.context)
    plain_context = _plain(context or {})
    authorized: Mapping[str, Any] = plain_context if isinstance(plain_context, Mapping) else {}
    # Only a ContextEnvelope carries policy provenance; a bare mapping means no
    # policy was applied (direct runtime use), so it must not be read as an
    # empty authorization.
    if isinstance(invocation.context, ContextEnvelope):
        inputs = authorized.get("inputs") if isinstance(authorized.get("inputs"), Mapping) else {}
        events = authorized.get("events") or ()
        condition = (
            authorized.get("condition") if isinstance(authorized.get("condition"), Mapping) else {}
        )
    else:
        # No policy was applied (bare runtime use); fall back to the invocation.
        inputs = _plain(invocation.inputs)
        events = _plain(invocation.event_history)
        condition = _plain(invocation.condition)
    namespace = {
        "inputs": _plain(inputs),
        "context": plain_context,
        "actor": {"ids": list(invocation.actor_ids)},
        "condition": _plain(condition),
        "events": _plain(events),
        "phase": invocation.phase,
        # The documented predicate grammar, the scheduler and the measurement
        # gate all spell the round `protocol.phase`; this namespace offered only
        # `phase`, so an availability gate written as documented resolved to
        # nothing. It exposes no new information -- the value is `phase`.
        "protocol": {"phase": invocation.phase},
        "time": invocation.time,
    }
    artifacts: dict[str, list[Any]] = {}
    for record in (inputs or {}).values():
        if not isinstance(record, Mapping) or not record.get("artifact_type"):
            continue
        artifacts.setdefault(str(record["artifact_type"]), []).append(_plain(record.get("value")))
    namespace["artifacts"] = artifacts
    if isinstance(plain_context, Mapping):
        namespace.update(plain_context)
    return namespace


def _resolve_optional(source: Any, path: str) -> tuple[bool, Any]:
    """``(found, value)`` at a dotted path, telling a missing key from a null value."""
    value: Any = source
    for part in path.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif isinstance(value, list | tuple) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return False, None
    return True, value


def _declares_composable_write(declarations: Any) -> bool:
    """Whether any declared effect is more than a whole-field ``set``.

    A bare ``set`` writes the field whole, which is what handing the outputs
    over already does; only a key or an accumulating operation needs the
    declaration to be interpreted.
    """
    for declaration in declarations or ():
        if not isinstance(declaration, Mapping):
            continue
        if declaration.get("key") or str(declaration.get("op", "set")) != "set":
            return True
    return False


def model_call_effects(
    process: Mapping[str, Any], outputs: Mapping[str, Any], actor_ids: Sequence[str]
) -> list[dict[str, Any]]:
    """The state effects a model call's declarations derive from its outputs.

    A model returns outputs only, so each declared effect names the output it
    reads (``from``, defaulting to the field name, dotted into the output) and
    the operation that applies it. An output the model did not return writes
    nothing. ``key: actor`` writes under the acting actor, which is what lets
    simultaneous siblings share one mapping without overwriting each other.
    """
    effects: list[dict[str, Any]] = []
    plain_outputs = _plain(outputs)
    for declaration in process.get("state_effects") or ():
        if isinstance(declaration, str):
            if declaration in plain_outputs:
                effects.append(
                    {"field": declaration, "op": "set", "value": plain_outputs[declaration]}
                )
            continue
        if not isinstance(declaration, Mapping) or not declaration.get("field"):
            continue
        field_name = str(declaration["field"])
        # A model that returned the field as null did declare a value; only an
        # output it did not return at all writes nothing.
        present, value = _resolve_optional(plain_outputs, str(declaration.get("from", field_name)))
        if not present:
            continue
        effect: dict[str, Any] = {
            "field": field_name,
            "op": str(declaration.get("op", "set")),
            "value": value,
        }
        if declaration.get("key") == "actor":
            if len(actor_ids) != 1:
                raise ValueError(
                    f"STATE_EFFECT_KEY: '{process.get('id')}' writes '{field_name}' under the "
                    f"acting actor, which requires one actor per invocation, not {len(actor_ids)}"
                )
            effect["key"] = str(actor_ids[0])
        effects.append(effect)
    return effects


def _redacted_outputs(outputs: Any, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Outputs as the trace policy allows them to be retained."""
    kept = dict(_plain(outputs)) if isinstance(outputs, Mapping) else {}
    if trace.get("record_raw_response", True) is False and "response" in kept:
        kept["response"] = "<raw-response-not-recorded>"
    return kept


def _exchange_context(context: Any) -> dict[str, Any]:
    """A recorded context with the exchanges namespace removed.

    An exchange stores the context its invocation was given, and that context
    may itself carry earlier exchanges. Storing it whole nests each round inside
    the next, so the namespace grows exponentially in rounds however tightly the
    cap bounds the count at each level.
    """
    plain = _plain(context)
    if not isinstance(plain, dict):
        return {}
    plain.pop("exchanges", None)
    return plain


def _composes_with_siblings(effect: Mapping[str, Any], actor_ids: Sequence[str]) -> bool:
    """Whether one effect still composes when every sibling computed it from one view.

    A keyed write composes only under the acting actor's own key: under any
    other key two siblings write the same entry, and the last one silently wins.
    """
    op = str(effect.get("op", "set"))
    if op not in COMPOSABLE_OPS:
        return False
    key = effect.get("key")
    if key is None:
        return op != "put"
    return str(key) in {str(actor) for actor in actor_ids}


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
        for rule in self.rules:
            # Checked here rather than at the first invocation: a malformed rule
            # simply never matched, and the default answered for the study.
            _validate_condition(rule.get("when"))
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


def _stamp_engine_fields(
    result: ProcessResult,
    process: Mapping[str, Any],
    actor_ids: tuple[str, ...],
    phase: int | float | None,
) -> ProcessResult:
    """Write each declared actor and phase field of an output from the invocation.

    A detector drawn per article was asked to repeat back which article it
    scored, was never shown the id, and wrote "unknown", "" or the title in most
    calls -- so settlement joined its scores to nothing. A reflection written in
    round 3 dated itself round 4. The engine knows both the actor and the round;
    the output fields are set from them, and what the executor had written there
    is kept in the metadata when it differed.
    """
    declared = [
        decl
        for decl in process.get("outputs") or ()
        if isinstance(decl, Mapping) and (decl.get("actor_fields") or decl.get("phase_fields"))
    ]
    if not declared or not result.outputs:
        return result
    outputs = dict(result.outputs)
    replaced: dict[str, dict[str, Any]] = {}
    for decl in declared:
        artifact_type = str(decl.get("artifact_type"))
        value = outputs.get(artifact_type)
        if not isinstance(value, Mapping):
            continue
        actor_fields = decl.get("actor_fields")
        roles = actor_roles(process)
        if isinstance(actor_fields, Mapping):
            # A paired invocation names which role each field takes, so the
            # reader's id and the article's id land in their own fields.
            named = [str(role) for role in actor_fields.values()]
            unknown = [role for role in named if role not in roles]
            # Two fields naming one role both receive that actor, so the other
            # id is silently lost -- an article reaction attributed to the
            # reader in both of its id fields. The declaration cannot mean what
            # it says, so it is refused rather than half-applied.
            repeated = len(named) != len(set(named))
            if unknown or repeated or len(actor_ids) != len(roles):
                return replace(
                    result,
                    status="failed",
                    metadata={
                        **_plain(result.metadata),
                        "code": "OUTPUT_ACTOR_FIELD_AMBIGUOUS",
                        "error": (
                            f"output '{artifact_type}' maps actor fields to roles "
                            f"{named}, but this process declares roles {list(roles)} and "
                            f"the invocation has {len(actor_ids)} actors"
                            + ("; a role may be named only once" if repeated else "")
                        ),
                    },
                )
            written: list[tuple[Any, Any]] = [
                (name, actor_ids[roles.index(str(role))]) for name, role in actor_fields.items()
            ]
        else:
            if actor_fields and len(actor_ids) != 1:
                return replace(
                    result,
                    status="failed",
                    metadata={
                        **_plain(result.metadata),
                        "code": "OUTPUT_ACTOR_FIELD_AMBIGUOUS",
                        "error": (
                            f"output '{artifact_type}' declares actor fields, but the invocation "
                            f"has {len(actor_ids)} actors; name the role each field takes"
                        ),
                    },
                )
            written = [(name, actor_ids[0] if actor_ids else None) for name in actor_fields or ()]
        stamped = dict(_plain(value))
        written += [(name, phase) for name in decl.get("phase_fields") or ()]
        for name, engine_value in written:
            if stamped.get(name) != engine_value:
                replaced.setdefault(artifact_type, {})[str(name)] = stamped.get(name)
            stamped[str(name)] = engine_value
        outputs[artifact_type] = stamped
    metadata = dict(_plain(result.metadata))
    if replaced:
        metadata["engine_fields_replaced"] = replaced
    return replace(result, outputs=outputs, metadata=metadata)


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
        # A policy may admit a state either bare ("follows") or namespaced
        # ("state.follows"); both name the same field, so read through either.
        namespaced_state = working.get("state")
        namespaced_state = namespaced_state if isinstance(namespaced_state, Mapping) else {}
        effects: dict[str, Any] = {}
        for operation in self.operations:
            op = str(operation.get("op", ""))
            state = _check_id(str(operation.get("state", "")), "state")
            current = working[state] if state in working else namespaced_state.get(state)
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

    def get(self, process_id: str) -> Any:
        """The executor registered for a process, or ``None``."""
        return self._executors.get(process_id)

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
                _validate_condition(trigger.get("predicate"))
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
            declared = process.get("dependencies")
            if not isinstance(declared, Mapping):
                declared = {"delay": process.get("delay")}
            resolved = edge_delays(declared, process.get("after", []))
            if any(value < 0 for value in resolved.values()):
                raise ValueError(f"invalid dependency delay for {pid}")
            delay = process.get("delay", 0)
            if isinstance(delay, Mapping):
                unknown = sorted(
                    str(key)
                    for key in (delay.get("per_dependency") or {})
                    if str(key) not in set(process.get("after", []))
                )
                if unknown:
                    raise ValueError(
                        f"dependency delay for {pid} names non-dependencies: {unknown}"
                    )
                delay = delay.get("rounds", 0)
            if not isinstance(delay, int | float) or isinstance(delay, bool) or delay < 0:
                raise ValueError(f"invalid dependency delay for {pid}")
            process["_edge_delays"] = resolved
        self._validate_cycles()
        self._delayed_bootstrap_edges = self._find_delayed_cycle_edges()
        self.completed: dict[str, int | float] = {}
        self.events: dict[str, int] = {}
        self._event_consumed: dict[tuple[str, str], int] = {}
        self._scheduled: list[ScheduledProcess] = []
        self._completed_occurrences: set[tuple[str, int | float]] = set()
        # Occurrences that have committed at least one turn. Derived from committed
        # turns, so a resumed run rebuilds it from its events rather than from a
        # memory of what was asked.
        self._started_occurrences: set[tuple[str, int | float]] = set()
        # Earliest phase at which each process completed. A delayed edge asks
        # whether the producer ran at least ``lag`` rounds ago, which the
        # latest completion alone cannot answer for a repeating producer.
        self._earliest_completion: dict[str, int | float] = {}

    def _validate_cycles(self) -> None:
        # Only a delayed EDGE is temporal; a delay declared for one dependency
        # must not exempt this process's other, immediate edges from cycle
        # detection (that hid genuine zero-lag cycles).
        graph = {
            pid: {
                dependency
                for dependency in p.get("after", [])
                if not p.get("_edge_delays", {}).get(str(dependency))
            }
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
            resolved = process.get("_edge_delays", {})
            for dependency in process.get("after", []):
                if resolved.get(str(dependency), 0) <= 0:
                    continue
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

    def _untriggered(
        self, process_id: str, phase: int | float, state: Mapping[str, Any] | None
    ) -> bool:
        """Whether a repeating, condition-triggered process is not triggered now.

        Read against the same live state the producer's own readiness is read
        against, and nothing is remembered: the answer is the same however often
        it is asked, whoever asks, and whether or not the run was interrupted. An
        earlier version latched the answer for the phase, which made readiness
        depend on what had been asked before -- so a resumed run, whose scheduler
        starts empty, scheduled a producer the uninterrupted run had skipped, and
        merely deciding a concurrency limit could suppress one.

        A condition trigger is deliberately re-read as a round proceeds, so a
        process can become ready once a sibling writes the state it waits on.
        This answer therefore means "not triggered at the moment the consumer was
        considered", not "will not run this round": a producer whose trigger
        turns true later still runs, after the consumer that did not wait.
        """
        if state is None or not self._repeats(process_id):
            return False
        if (process_id, phase) in self._completed_occurrences:
            return False
        trigger = self.processes.get(process_id, {}).get("trigger", {})
        if not isinstance(trigger, Mapping) or trigger.get("type") != "condition":
            return False
        return not _evaluate_condition(cast(Mapping[str, Any], trigger.get("predicate")), state)

    def _repeats(self, process_id: str) -> bool:
        process = self.processes.get(process_id, {})
        trigger = process.get("trigger", {})
        trigger = trigger if isinstance(trigger, Mapping) else {}
        return bool(process.get("repeat", trigger.get("repeat", False)))

    def _dependency_satisfied(
        self,
        pid: str,
        process: Mapping[str, Any],
        dep: str,
        resolved_delays: Mapping[str, int | float],
        phase: int | float,
        state: Mapping[str, Any] | None = None,
    ) -> bool:
        """Whether one declared dependency edge permits ``pid`` to run at ``phase``.

        A zero-delay edge means "after the producer, in this round". When the
        producer repeats each round, "has completed at some earlier phase" is
        not enough: it let every process in a repeating chain become ready at
        once from the second round onward, so the chain ran in process-id order
        and each consumer read the PREVIOUS round's artifacts. A repeating
        producer must therefore have completed in the current phase.

        A positive-delay edge is temporal: the producer must have completed at
        least ``lag`` rounds ago. That is a question about the producer's
        HISTORY, not its latest completion — a repeating producer advances its
        latest completion every round, so ``last + lag <= phase`` could never
        become true and a lag of two or more starved the consumer forever.

        A zero-delay edge on a repeating producer whose condition trigger does
        not hold this phase is satisfied: the producer will not run this phase,
        so waiting for it would stop the consumer in every phase it is skipped
        (a reflection every third round would stop publishing in the others).
        The trigger is read against the same state that decides whether the
        producer itself is ready, so the two answers cannot differ.
        """
        delay = resolved_delays.get(str(dep), 0)
        if delay <= 0 and self._untriggered(str(dep), phase, state):
            return True
        # A producer deferred to the next round because a dependent already ran
        # without it is, for the rest of this round, not triggered. Waiting on it
        # instead would strand the dependent's remaining actors for the round --
        # the first actor went ahead without it, so the rest must too, or one
        # batch would see two different worlds.
        if delay <= 0 and self._deferred_this_round(str(dep), phase):
            return True
        if dep not in self.completed:
            # A delayed edge inside a dependency cycle bootstraps on the first
            # phase so the cycle can start at all.
            return bool(
                delay > 0
                and (pid, dep) in self._delayed_bootstrap_edges
                and phase == process.get("phase", 0)
            )
        if delay > 0:
            earliest = self._earliest_completion.get(str(dep), self.completed[dep])
            return bool(earliest + delay <= phase)
        if self._repeats(dep):
            return (str(dep), phase) in self._completed_occurrences
        return bool(self.completed[dep] <= phase)

    def _dependencies_of(
        self, process: Mapping[str, Any]
    ) -> tuple[list[str], Mapping[str, int | float]]:
        """A process's declared `after` edges and their resolved delays."""
        deps = process.get("after", [])
        dependencies_block = process.get("dependencies")
        if not deps and isinstance(dependencies_block, Mapping):
            declared_after = dependencies_block.get("after")
            if isinstance(declared_after, list | tuple):
                deps = list(declared_after)
        # Resolved once per process at construction; recompute only for a
        # process that did not pass through Scheduler.__init__.
        if "_edge_delays" in process:
            return list(deps), process["_edge_delays"]
        declared = process.get("dependencies")
        return list(deps), edge_delays(declared if isinstance(declared, Mapping) else process, deps)

    def _overtaken_by_dependent(self, process_id: str, phase: int | float) -> bool:
        """Whether a process that waits on this one has already run this round.

        A repeating, condition-triggered process that is not triggered when a
        dependent is considered does not hold that dependent back -- that is
        what lets a round proceed. But if its trigger turns true later in the
        same round, running it then would put it after a process declared to
        run after it. It waits for the next round instead. Mid-round starts are
        otherwise untouched: this applies only once something depending on the
        process, by a zero-delay edge, has committed a turn in this round.
        """
        for other_id, other in self.processes.items():
            if other_id == process_id:
                continue
            deps, delays = self._dependencies_of(other)
            if (
                process_id in deps
                and delays.get(process_id, 0) <= 0
                and (other_id, phase) in self._started_occurrences
            ):
                return True
        return False

    def _deferred_this_round(self, process_id: str, phase: int | float) -> bool:
        """A repeating, condition-triggered process held to the next round."""
        if not self._repeats(process_id) or (process_id, phase) in self._completed_occurrences:
            return False
        trigger = self.processes.get(process_id, {}).get("trigger", {})
        if not isinstance(trigger, Mapping) or trigger.get("type") != "condition":
            return False
        return self._overtaken_by_dependent(process_id, phase)

    def mark_started(self, process_id: str, phase: int | float) -> None:
        """Record that an occurrence has committed a turn."""
        self._started_occurrences.add((process_id, phase))

    def ready(
        self,
        phase: int | float,
        events: set[str] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> list[ScheduledProcess]:
        out = []
        observed_events = set(self.events) | (events or set())
        trigger_state = state or {}
        for pid, p in self.processes.items():
            repeat = bool(p.get("repeat", p.get("trigger", {}).get("repeat", False)))
            if (
                (pid, phase) in self._completed_occurrences
                or (pid in self.completed and not repeat)
                or p.get("phase", 0) > phase
            ):
                continue
            deps, resolved_delays = self._dependencies_of(p)
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
                if not _evaluate_condition(predicate, trigger_state):
                    continue
                if self._deferred_this_round(pid, phase):
                    continue
            if any(
                not self._dependency_satisfied(pid, p, dep, resolved_delays, phase, trigger_state)
                for dep in deps
            ):
                continue
            out.append(ScheduledProcess(pid, p.get("phase", 0)))
        # A scheduling effect reactivates a process that has already run: that is
        # what an executor asking for another occurrence means, and a study
        # depends on it (test_schedule_effect_reactivates_completed_process).
        out.extend(item for item in self._scheduled if item.phase <= phase)
        unique = {(item.process_id, item.phase): item for item in out}
        return sorted(unique.values(), key=lambda x: (x.phase, x.process_id))

    def complete(self, process_id: str, phase: int | float) -> None:
        self.completed[process_id] = phase
        self._completed_occurrences.add((process_id, phase))
        self._started_occurrences.add((process_id, phase))
        previous = self._earliest_completion.get(process_id)
        if previous is None or phase < previous:
            self._earliest_completion[process_id] = phase
        trigger = self.processes[process_id].get("trigger", {})
        if isinstance(trigger, Mapping) and trigger.get("type") == "event":
            event = str(trigger.get("event"))
            key = (process_id, event)
            self._event_consumed[key] = self._event_consumed.get(key, 0) + 1


@dataclass(frozen=True)
class _ExecutorRaised:
    """An exception the executor raised, carried from execution to commit.

    Wrapped rather than passed bare so an executor that *returns* an exception
    object is still treated as returning an invalid result, as it always was.
    """

    error: Exception


@dataclass(frozen=True)
class _RunScope:
    """Run-level inputs every actor turn is prepared from."""

    run_id: str
    seed: int
    seed_identity: str | None
    experiment_id: str
    condition_id: str
    replication: int
    matching: Mapping[str, Any]
    condition: Mapping[str, Any]
    state: Mapping[str, Any] | None


@dataclass(frozen=True)
class _BatchView:
    """What every actor in a simultaneous batch sees: the run as the batch began (CON-008)."""

    state: Mapping[str, Any]
    state_version: int
    event_history: tuple[Mapping[str, Any], ...]
    artifact_ids: frozenset[str] | None
    # Declared simultaneous, as opposed to an undeclared independent batch.
    simultaneous: bool


# The random stream a shuffled process's activation order is drawn from (CON-010).
ACTIVATION_ORDER_STREAM = "activation-order"


def _declared_order(process: Mapping[str, Any]) -> tuple[tuple[str, ...], ...] | None:
    """The batch order a process declaration alone reproduces, or ``None``.

    Listed ids reproduce it; a shuffled order, or actors drawn from state, do not,
    so only those batches record their order on their first event.
    """
    if timing_of(process)[1] == "shuffled":
        return None
    actors = process.get("actors")
    if isinstance(actors, Mapping) and actors.get("ids") is None:
        return None
    if isinstance(actors, Mapping) and actors.get("per") is not None:
        # A paired selector draws its inner half from state, so listed outer ids
        # do not reproduce the order either: against an empty state the
        # expansion is empty, and an empty tuple is not None, so the callers
        # that test `is None` silently stopped recording batch_actors and
        # refused to reopen an interrupted batch (CON-008/CON-010).
        return None
    try:
        return tuple(expand_actor_instances(process, {}))
    except ValueError:
        return None


@dataclass(frozen=True)
class _ActorTurn:
    """One actor group's turn in one process and phase."""

    process_id: str
    process: Mapping[str, Any]
    phase: int | float
    actor_ids: tuple[str, ...]
    max_attempts: int
    feedback_slots: dict[str, Any]
    view: _BatchView | None = None


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
        max_concurrency: Mapping[str, int] | None = None,
        cancel_event: Any | None = None,
    ):
        self.scheduler, self.registry, self.context_engine = scheduler, registry, context_engine
        # Per process: how many calls of one batch may run at once (CON-011).
        self.max_concurrency = dict(max_concurrency or {})
        # The run's cancel signal, shared with its providers: set when a
        # concurrent batch fails, so calls still in flight stop (CON-013).
        self.cancel_event = cancel_event
        # How each batched process actually ran, and why; and calls that ran but
        # were not committed because the run stopped mid-batch (CON-013, CON-015).
        self.execution_decisions: dict[str, dict[str, Any]] = {}
        self.discarded_calls: dict[str, Any] = {"calls": 0, "unfinished": 0, "usage": {}}
        # An event cap stops the run and marks it completed, which is
        # indistinguishable from reaching the declared termination. A run that
        # was cut short is not the study the researcher declared, so say so.
        self.budget_exhausted = False
        # Why the run paused itself, when a provider could not serve a call.
        self.pause_reason: dict[str, Any] | None = None
        # Set when another process took the run over mid-execution.
        self.lease_lost = False
        # The last declared phase of this run, for stopped_with_work_remaining.
        self._terminal_phase: int | None = None
        self._executed: list[str] = []
        # Invocation id -> the highest attempt already committed, so a resumed
        # run continues after it instead of re-emitting a committed attempt id.
        self._committed_attempts: dict[str, int] = {}
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
        # Simultaneous batches (CON-008): the view each open batch reads; for a
        # resumed run, the view version its already-committed siblings recorded
        # and each persisted event's state version; and the pre-run state, for a
        # batch that began before the first commit.
        self._batch_views: dict[tuple[str, int | float], _BatchView] = {}
        self._batch_view_versions: dict[tuple[str, int | float], int] = {}
        self._event_state_versions: dict[str, int] = {}
        self._initial_state: dict[str, Any] | None = None
        # Each open batch's full actor order, recorded on its first actor's events;
        # the orders a resumed run found recorded; and, per process, whether its
        # batches read from a view.
        self._batch_orders: dict[tuple[str, int | float], tuple[tuple[str, ...], ...]] = {}
        self._recorded_batch_orders: dict[tuple[str, int | float], tuple[tuple[str, ...], ...]] = {}
        self._view_decisions: dict[str, bool] = {}
        # When each open view was taken (commit-log position), so an undeclared
        # batch notices another process committing part-way through; for a
        # resumed run, the latest view version each batch's committed actors
        # read and the producer of every persisted event.
        self._view_opened_at: dict[tuple[str, int | float], int] = {}
        self._batch_view_latest: dict[tuple[str, int | float], int] = {}
        self._persisted_event_processes: list[tuple[int, str]] = []
        # Round-state ring for state-feedback bindings: snapshot the state at
        # the start of each phase, keyed by phase, so a consumer can read the
        # state as of ``lag`` completed rounds earlier.
        self._round_state_at_phase: dict[int | float, dict[str, Any]] = {}
        # (process, acting group) -> that group's own exchanges with that process.
        self._exchange_log: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        # The processes some policy asks for exchanges of. Nothing is kept for
        # any other process: a recorded context per invocation, held for the
        # whole run, is not a cost to pay for a package that cannot read it.
        self._exchange_processes: frozenset[str] = frozenset(
            str(path).split(".")[1]
            for policy in (getattr(context_engine, "policies", {}) or {}).values()
            if isinstance(policy, Mapping)
            for path in policy.get("allow") or ()
            if str(path).startswith("exchanges.") and len(str(path).split(".")) > 1
        )
        # How many exchanges per actor group are worth keeping, per process.
        # The declared cap was applied only when a prompt was built, so the log
        # itself grew for the life of the run: a policy reading "the last three
        # rounds" still held every round, at O(rounds x groups x context size).
        # Only a cap that keeps the *tail* in recorded order can be applied
        # here; one keeping the head, or ordering by a field, needs entries this
        # trim would discard, so those stay unbounded and say so.
        self._exchange_retention: dict[str, int] = _exchange_retention(
            getattr(context_engine, "policies", {}) or {}
        )
        # The protocol's first phase; the floor for "a round that actually ran".
        self._phase_start: int = 0
        # Ring entry reconstructed from an interrupted round's partial state.
        self._speculative_round_start: int | None = None
        # Committed writers of each state field, in commit order, so a lagged
        # feedback read can name the event that produced the value it saw.
        self._state_writers: dict[str, list[tuple[int | float, str]]] = {}
        # Feedback provenance for the invocation currently being dispatched.
        self._feedback_parents: tuple[str, ...] = ()

    def _feedback_history(
        self, process: Mapping[str, Any], phase: int | float
    ) -> tuple[dict[str, Any], bool]:
        """Resolve declared state-feedback context slots for one invocation.

        Returns (slots, blocked). ``slots`` maps each declared context slot to
        the state ``lag`` completed rounds before ``phase`` (the snapshot kept
        at the start of ``phase - lag``); when that history does not exist, the
        researcher-declared ``initial`` policy applies: ``declared_default``
        injects ``initial.value``, ``skip_consumer`` sets blocked=True so the
        dispatcher defers the process until enough rounds have elapsed.
        """
        slots: dict[str, Any] = {}
        blocked = False
        parents: list[str] = []
        for binding in process.get("theory_feedback", []) or []:
            if not isinstance(binding, Mapping):
                continue
            slot = str(binding.get("context_slot", ""))
            if not slot:
                continue
            lag = int(binding.get("lag_rounds", 1))
            # "Lag N completed rounds" reads the state snapshot captured at
            # the start of ``phase - lag + 1`` — the state that round
            # ``phase - lag`` finished with (the ring stores the snapshot taken
            # at the start of each phase). History exists only when that round
            # actually ran, so the floor is the protocol's FIRST phase, not 0:
            # a protocol declaring ``time_model.start: 1`` has no round 0, and
            # treating phase 0 as completed would shift every lag consumer one
            # round early and serve the pre-run snapshot as a completed round.
            history_phase: int | float = phase - lag + 1
            snapshot = (
                self._round_state_at_phase.get(history_phase)
                if (phase - lag) >= self._phase_start
                else None
            )
            if snapshot is not None:
                source = str(binding.get("source", ""))
                if source not in snapshot:
                    raise ValueError(f"THEORY_FEEDBACK_SOURCE_UNKNOWN: state '{source}' is absent")
                slots[slot] = {source: _plain(snapshot[source])}
                # The value came from the last committed write of ``source`` at
                # or before the round the snapshot describes. Recording it makes
                # the influence traceable; without it a consumer of prior-round
                # state appears to depend on nothing.
                producing_phase = phase - lag
                writers = [
                    event_id
                    for written_phase, event_id in self._state_writers.get(source, ())
                    if written_phase <= producing_phase
                ]
                if writers:
                    parents.append(writers[-1])
                continue
            initial = binding.get("initial") or {}
            policy = str(initial.get("policy") or "declared_default")
            if policy == "skip_consumer":
                blocked = True
            else:
                slots[slot] = _plain(initial.get("value", {}))
        self._feedback_parents = tuple(dict.fromkeys(parents))
        return slots, blocked

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
        # The state version the context was built from: for a simultaneous batch,
        # the version as the batch began (CON-015).
        record["view_state_version"] = call.state_version
        batch_order = self._batch_orders.get((call.process_id, call.phase))
        if (
            batch_order
            and call.attempt == 1
            and tuple(call.actor_ids) == batch_order[0]
            and _declared_order(process) is None
        ):
            # When the declaration alone cannot reproduce the order (shuffled, or
            # actors drawn from state), the first actor's first attempt records it,
            # so a resumed run reopens the batch exactly as it began (CON-008, CON-010).
            record["batch_actors"] = [list(group) for group in batch_order]
        if isinstance(process.get("information_timing"), Mapping):
            mode, order = timing_of(process)
            record["information_timing"] = {"mode": mode, "order": order}
        if call.context is not None and record_context:
            record["context"] = _plain(call.context.data)
        return record

    def _record_exchange(
        self, turn: _ActorTurn, call: ProcessInvocation, result: ProcessResult, attempt: int
    ) -> None:
        """Keep this turn's own exchange, so a later turn can be shown what it did.

        A reflection reads back what it was given and what it answered a few
        rounds ago. An exchange belongs to the actor group that made it, and is
        kept only under the same trace policy that governs whether a context is
        retained at all, so this adds no channel a policy has not admitted.
        """
        if turn.process_id not in self._exchange_processes:
            # Nothing can read this process's exchanges, and keeping every
            # recorded context for the life of the run is not free.
            return
        raw_trace = turn.process.get("trace_policy", {})
        trace = raw_trace if isinstance(raw_trace, Mapping) else {}
        entry: dict[str, Any] = {
            "phase": turn.phase,
            "attempt": attempt,
            "outputs": _redacted_outputs(_plain(result.outputs), trace),
        }
        if call.context is not None and bool(trace.get("record_context", True)):
            entry["context"] = _exchange_context(call.context.data)
        key = (turn.process_id, tuple(call.actor_ids))
        entries = self._exchange_log.setdefault(key, [])
        entries.append(entry)
        keep = self._exchange_retention.get(turn.process_id)
        if keep is not None and len(entries) > keep:
            # Recorded oldest-first, so the readable tail is the end.
            del entries[: len(entries) - keep]

    def _exchanges_for(self, turn: _ActorTurn) -> dict[str, list[dict[str, Any]]]:
        """This turn's own earlier exchanges, per process, oldest first.

        Matched on the whole acting group, not on any member: a group turn is
        one actor's worth of history, and unioning its members' exchanges would
        hand each of them the others' -- and count a joint exchange once per
        member, so a cap of three could hold one exchange three times.
        """
        acting = tuple(turn.actor_ids)
        found: dict[str, list[dict[str, Any]]] = {}
        for (process_id, actors), entries in self._exchange_log.items():
            if actors == acting:
                found.setdefault(process_id, []).extend(entries)
        return {
            process_id: sorted(
                items, key=lambda item: (item.get("phase", 0), item.get("attempt", 1))
            )
            for process_id, items in found.items()
        }

    def _restore_exchange_log(self, run_id: str, completed: list[Mapping[str, Any]]) -> None:
        """Rebuild each actor's exchange history after a resume.

        Each invocation's context is on its event; its outputs are on the
        invocation's payload row, under the same id. Without rebuilding both, a
        resumed run's reflection would see only the rounds since it resumed.

        Only invocations that completed are rebuilt. A skipped invocation
        records no exchange live, so admitting one here would give a resumed run
        a history the uninterrupted run never had.
        """
        self._exchange_log = {}
        if not self._exchange_processes:
            return
        wanted = [
            event
            for event in completed
            if event.get("kind") == "process_completed"
            and str(event.get("process_id")) in self._exchange_processes
        ]
        if not wanted:
            return
        outputs_by_id = self._recorded_invocation_outputs(run_id)
        log: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
        for event in sorted(
            wanted,
            key=lambda item: (item.get("phase", 0), item.get("commit_order", 0)),
        ):
            process_id = str(event.get("process_id"))
            process = self.scheduler.processes.get(process_id, {})
            raw_trace = process.get("trace_policy", {}) if isinstance(process, Mapping) else {}
            trace = raw_trace if isinstance(raw_trace, Mapping) else {}
            event_id = str(event.get("event_id", ""))
            entry: dict[str, Any] = {
                "phase": event.get("phase", 0),
                "attempt": int(event.get("attempt", 1)),
                # The payload row redacts a raw response only for generative
                # processes, so the live redaction is applied again here rather
                # than trusted; and a retained-but-purged invocation must say so
                # instead of reading as an actor that answered nothing.
                "outputs": _redacted_outputs(outputs_by_id.get(event_id, {}), trace),
            }
            if event_id not in outputs_by_id:
                entry["outputs_recorded"] = False
            if isinstance(event.get("context"), Mapping):
                entry["context"] = _exchange_context(event["context"])
            actors = tuple(str(actor) for actor in event.get("actors", ()) or ())
            log.setdefault((process_id, actors), []).append(entry)
        # The same bound the live path keeps. Restoring every exchange handed a
        # resumed run's models more history than the uninterrupted run shows, and
        # held all of it for the rest of the run.
        for (process_id, _actors), entries in log.items():
            keep = self._exchange_retention.get(process_id)
            if keep is not None and len(entries) > keep:
                del entries[: len(entries) - keep]
        self._exchange_log = log

    def _recorded_invocation_outputs(self, run_id: str) -> dict[str, Any]:
        """Each recorded invocation's outputs, keyed by its event id."""
        rows = getattr(self.persistence, "iter_artifacts", None) or getattr(
            self.persistence, "list_artifacts", None
        )
        found: dict[str, Any] = {}
        if rows is None:
            return found
        try:
            for row in rows(run_id):
                try:
                    payload = json.loads(row["payload"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if isinstance(payload, Mapping) and "outputs" in payload:
                    found[str(row.get("artifact_id", ""))] = payload["outputs"]
        except ValueError:
            # A purge during the read leaves the history without outputs, which
            # each entry then reports; it must not stop the resume itself.
            return found
        return found

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

    def _record_state_writes(self, event_id: str, phase: int | float, state_delta: Any) -> None:
        """Remember which committed event last wrote each state field.

        A lagged feedback binding reads a value some earlier event produced.
        Without this the reader's causal parents are empty and a trace cannot
        explain where the value it acted on came from.
        """
        if not event_id:
            return
        if isinstance(state_delta, Mapping):
            written: list[str] = [str(name) for name in state_delta]
        elif isinstance(state_delta, list | tuple):
            # Effect-list form: [{field, op, value}, ...].
            written = [
                str(item["field"])
                for item in state_delta
                if isinstance(item, Mapping) and item.get("field")
            ]
        else:
            return
        for field_name in written:
            self._state_writers.setdefault(field_name, []).append((phase, event_id))

    def _causal_parent_events(
        self, process: Mapping[str, Any], call: ProcessInvocation
    ) -> list[str]:
        """Causal parents: consumed inputs, declared dependencies, and the
        committed state writes a lagged feedback binding read."""
        parents: list[str] = list(self._feedback_parents)
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
        """What this turn changed, by identity rather than by equality.

        ``!=`` calls 1 and 1.0, True and 1, -0.0 and 0.0 unchanged, so those
        transitions were left out of the recorded delta -- which is the
        authoritative source a partial replay applies. The state history kept
        the new bytes and the ledger kept none, so a replay of the frozen prefix
        diverged from its source in exactly the bytes the commit identity is
        digested from. ``StateStore.apply`` and ``encode_patch`` already compare
        this way; this was the remaining ``!=``.
        """
        before = before or {}
        delta = {
            key: value for key, value in after.items() if not identical(before.get(key), value)
        }
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
        self._event_state_versions = {
            str(event.get("event_id")): int(event["state_version"])
            for event in events
            if event.get("event_id") and isinstance(event.get("state_version"), int)
        }
        self._persisted_event_processes = [
            (int(event["state_version"]), str(event.get("process_id")))
            for event in events
            if isinstance(event.get("state_version"), int)
        ]
        for event in events:
            order = event.get("batch_actors")
            if isinstance(order, list) and event.get("process_id") is not None:
                self._recorded_batch_orders.setdefault(
                    (str(event.get("process_id")), event.get("phase", 0)),
                    tuple(tuple(str(actor) for actor in group) for group in order),
                )
        # How far each invocation's retries already got. A resume restarted the
        # retry loop at attempt 1 and re-emitted an attempt id that was already
        # committed -- with a different dispatch order and state version, which
        # persistence rejects as a differing commit, leaving the run
        # permanently unresumable. The cursor is derived from the ledger rather
        # than persisted separately, so it survives any restart.
        for event in events:
            invocation = str(event.get("invocation_id") or "")
            attempt = event.get("attempt")
            if invocation and isinstance(attempt, int):
                self._committed_attempts[invocation] = max(
                    self._committed_attempts.get(invocation, 0), attempt
                )
        if events:
            self._last_event_id = str(events[-1].get("event_id")) or None
        completed = [
            event
            for event in events
            if event.get("kind") in {"process_completed", "process_skipped"}
        ]
        self._restore_exchange_log(run_id, completed)
        if self.state_store:
            self._initial_state = self.state_store.snapshot()
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
        # Rebuild the round-state ring so lagged state-feedback bindings keep
        # working after a pause/resume: every committed state version's phase
        # is known from the events, and the state snapshot for each phase is
        # the one in effect at that phase's start (the first commit of the
        # previous phase, or the latest state before the first commit).
        if self.state_store and hasattr(self.persistence, "list_state_history"):
            version_phase: dict[int, int] = {}
            for event in events:
                version = event.get("state_version")
                phase = event.get("phase")
                if isinstance(version, int) and isinstance(phase, int):
                    version_phase[version] = phase
            # STH-004: only the first snapshot and the FINAL snapshot of each
            # phase are needed, so the history is consumed one version at a time
            # rather than held whole — it is quadratic in population x horizon.
            initial_snapshot: dict[str, Any] = {}
            phase_final: dict[int, dict[str, Any]] = {}
            first = True
            iterate = getattr(
                self.persistence, "iter_state_history", self.persistence.list_state_history
            )
            for version, snapshot in iterate(run_id):
                if first:
                    initial_snapshot = dict(snapshot)
                    first = False
                phase = version_phase.get(version)
                if phase is None:
                    continue
                phase_final[phase] = dict(snapshot)
            ring: dict[int | float, dict[str, Any]] = {}
            for phase, final_state in phase_final.items():
                ring[int(phase) + 1] = final_state
            # ``_next_phase`` is the frontier: the phase execution re-enters.
            # Its recorded state may be only PARTIAL — the writes committed
            # when the run stopped — so the entry derived from it for the
            # FOLLOWING round is speculative. Record that, so the run loop
            # replaces it with the genuine snapshot once the round really
            # finishes, instead of preserving a stale value across resumption.
            self._speculative_round_start = int(self._next_phase) + 1
            # The earliest phase has no "previous phase"; its start state is
            # the initial snapshot (pre-first-commit).
            if events:
                earliest = min(
                    (
                        int(e.get("phase", 0))
                        for e in events
                        if e.get("kind") in {"process_completed", "process_skipped"}
                    ),
                    default=None,
                )
                if earliest is not None:
                    ring.setdefault(earliest, dict(initial_snapshot))
            sorted_phases = sorted(ring)
            for index, phase in enumerate(sorted_phases):
                # Fill any phase gap with the previous phase's final state so
                # lag lookups never miss a round that ran no state writes.
                if index > 0 and phase - sorted_phases[index - 1] > 1:
                    fill = ring[sorted_phases[index - 1]]
                    for missing in range(int(sorted_phases[index - 1]) + 1, int(phase)):
                        ring[int(missing)] = dict(fill)
            self._round_state_at_phase = dict(sorted(ring.items()))
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
                view_version = event.get("view_state_version")
                if isinstance(view_version, int) and not isinstance(view_version, bool):
                    batch_key = (str(process_id), phase)
                    self._batch_view_versions[batch_key] = min(
                        view_version, self._batch_view_versions.get(batch_key, view_version)
                    )
                    self._batch_view_latest[batch_key] = max(
                        view_version, self._batch_view_latest.get(batch_key, view_version)
                    )
                self._record_state_writes(
                    str(event.get("event_id", "")), phase, event.get("state_delta")
                )
                completed_groups.setdefault((str(process_id), phase), set()).add(actors)
                # A straight run crosses off one schedule request per actor turn,
                # when the turn is taken -- before that turn's own effects land.
                # The rebuild used to cross off one per *completed occurrence*,
                # after replaying every effect, so a batch interrupted part-way
                # kept its requests open and a resumed run took extra actor
                # turns the uninterrupted run never did. Replayed here, in commit
                # order, it counts exactly as the run did.
                self.scheduler.consume_scheduled(str(process_id), phase)
                if event.get("kind") == "process_completed":
                    self.scheduler.mark_started(str(process_id), phase)
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
            # A recorded batch order is what the batch was; the current state may
            # now expand to different actors (CON-008).
            recorded_order = self._recorded_batch_orders.get((str(process_id), phase))
            expected = (
                set(recorded_order)
                if recorded_order is not None
                else set(
                    expand_actor_instances(self.scheduler.processes[process_id], state_snapshot)
                )
            )
            if expected.issubset(actor_groups):
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

    @property
    def stopped_with_work_remaining(self) -> bool:
        """Whether stopping left declared work undone.

        A lease lost after the last declared phase had committed is not a run
        that stopped early: reporting it as one left the run 'running' with a
        complete record, and a later resume was refused for a process that was
        not running. Computed rather than set at the break, because the run has
        several exit points and only one of them is that break.
        """
        if self._terminal_phase is None:
            return True
        return int(self._next_phase or 0) <= int(self._terminal_phase)

    def _next_attempt(self, invocation_id: str) -> int:
        """The attempt number to dispatch next for one invocation.

        One past whatever the ledger already holds, so a run resumed after a
        failed attempt continues its retries instead of re-emitting a committed
        attempt id under a different dispatch order.
        """
        return self._committed_attempts.get(str(invocation_id), 0) + 1

    def _record_committed_attempt(self, invocation_id: str, attempt: int) -> None:
        self._committed_attempts[str(invocation_id)] = max(
            self._committed_attempts.get(str(invocation_id), 0), int(attempt)
        )

    def _poll_external_status(self) -> bool:
        """Sync run control from persisted status (service/API cancellation)."""
        if self.status_provider is None or self.status in {"cancelled", "paused"}:
            return self.status in {"cancelled", "paused"}
        external = self.status_provider()
        if external == "lease_lost":
            # Another process holds the run now; stop without claiming an outcome.
            self.status, self.lease_lost = "paused", True
            return True
        if external in {"cancelled", "paused"}:
            self.status = external
            return True
        return False

    def _prepare_call(self, scope: _RunScope, turn: _ActorTurn, attempt: int) -> ProcessInvocation:
        """Build one attempt's invocation and its authorized context (CON-004).

        Everything an executor may read is fixed here, on the controller's thread,
        before the executor runs.
        """
        actor_seed_id = "\x1f".join(turn.actor_ids)
        actor_suffix = f"-{'-'.join(turn.actor_ids)}" if turn.actor_ids else ""
        view = turn.view
        # A measurement used only under some conditions or rounds is not resolved
        # outside them, so the process cannot read it there at all.
        # Every declared input is resolved, and then the records a gated-off
        # measurement produced are dropped -- by producer, never by type, so
        # another process's records of the same type still arrive and the
        # consumer keeps its own.
        input_refs = turn.process.get("inputs") or []
        resolved_inputs = (
            withhold_instances(
                self.artifact_store.resolve(
                    list(input_refs),
                    actor_ids=turn.actor_ids,
                    phase=turn.phase,
                    visible=view.artifact_ids if view is not None else None,
                ),
                withheld_sources(turn.process, phase=turn.phase, condition=scope.condition),
            )
            if self.artifact_store is not None and isinstance(input_refs, list | tuple)
            else {}
        )
        binding = turn.process.get("executor", {})
        parameters = binding.get("parameters", {}) if isinstance(binding, Mapping) else {}
        stream_id = (
            str(parameters.get("random_stream", "conventional"))
            if isinstance(parameters, Mapping)
            else "conventional"
        )
        shared_streams = scope.matching.get("shared_streams", [])
        matching_key = (
            stream_id
            if scope.matching.get("enabled") is True
            and isinstance(shared_streams, list | tuple)
            and stream_id in shared_streams
            else None
        )
        call = ProcessInvocation(
            f"{scope.run_id}-{turn.process_id}{actor_suffix}-{turn.phase}",
            scope.run_id,
            turn.process_id,
            actor_ids=turn.actor_ids,
            phase=turn.phase,
            state_version=(
                view.state_version
                if view is not None
                else self.state_store.version
                if self.state_store
                else 0
            ),
            inputs=resolved_inputs,
            seed=derive_seed(
                scope.seed,
                scope.seed_identity or scope.run_id,
                turn.process_id,
                actor_seed_id,
                experiment_id=scope.experiment_id,
                condition_id=scope.condition_id,
                replication=scope.replication,
                matching_key=matching_key,
            ),
            attempt=attempt,
            condition=scope.condition,
            event_history=(view.event_history if view is not None else tuple(self._event_history)),
            feedback_slots=turn.feedback_slots,
            exchanges=self._exchanges_for(turn),
        )
        policy_id = turn.process.get("context_policy", "private")
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
                view.state
                if view is not None
                else self.state_store.snapshot()
                if self.state_store is not None
                else (scope.state or {}),
            ),
            call.seed,
            call.attempt,
            turn.process.get("executor", {}),
            call.condition,
            call.event_history,
            call.feedback_slots,
            call.exchanges,
        )
        return call

    def _log_dispatch(self, turn: _ActorTurn, call: ProcessInvocation, attempt: int) -> int:
        """Record that an attempt was dispatched and return its dispatch order."""
        dispatch_order = len(self.dispatch_log) + 1
        self.dispatch_log.append(
            {
                "order": dispatch_order,
                "invocation_id": call.invocation_id,
                "process_id": turn.process_id,
                "phase": turn.phase,
                "attempt": attempt,
            }
        )
        return dispatch_order

    def _execute_call(
        self, turn: _ActorTurn, call: ProcessInvocation
    ) -> ProcessResult | _ExecutorRaised:
        """Run the executor: the only part of a turn that reads no store (CON-004).

        An exception is returned rather than raised, so committing a turn handles
        an executor failure the same way wherever the executor ran.
        """
        try:
            return self.registry.execute(turn.process_id, call)
        except _ReportedProcessFailure:
            raise
        except Exception as exc:
            return _ExecutorRaised(exc)

    def _commit_attempt(
        self,
        turn: _ActorTurn,
        call: ProcessInvocation,
        dispatch_order: int,
        attempt: int,
        outcome: ProcessResult | _ExecutorRaised,
    ) -> bool:
        """Validate and durably record one attempt's outcome (CON-004).

        Returns ``True`` when the turn is finished and ``False`` when another
        attempt should run. Raises when the run must stop.

        The restore point for a failed commit is the store as it stands when the
        commit begins. Executors receive only a frozen invocation and cannot reach
        the controller's stores, so this equals the state before execution; it is
        also the right point once earlier turns commit while a call is running.
        """
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
            if isinstance(outcome, _ExecutorRaised):
                raise outcome.error
            result = outcome
            executor_returned = True
            retry_policy = turn.process.get("retry_policy", {})
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
            # Only a succeeded result commits outputs. A skip_with_event policy
            # keeps its fallback_outputs on the result for the trace, but the
            # skipped branch below commits no artifacts and an empty state
            # delta, so there is nothing to stamp or validate; running the guard
            # there would only turn a clean skip into a failure.
            if result.status == "succeeded":
                result = _stamp_engine_fields(result, turn.process, call.actor_ids, call.phase)
            # F3/SCH-002: every declared artifact output — including a
            # fallback — must satisfy its declared schema before any
            # state or artifact commit. This is the common
            # output-commit boundary, so all executors are covered.
            if result.status == "succeeded" and self.output_schema_validator is not None:
                # F1: the schema comes from the executing process's
                # own output declaration (process.outputs[].schema_ref),
                # not from a separate domain-artifact id, so a
                # process-declared schema is always enforced.
                declared_outputs = turn.process.get("outputs") or []
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
                    for message in self.output_schema_validator(schema_ref, schema_value):
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
                        "classification": result.metadata.get("code", "executor_defect"),
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
                            "run_id": call.run_id,
                            "kind": "process_failed",
                            "process_id": turn.process_id,
                            "actors": list(call.actor_ids),
                            "phase": turn.phase,
                            "attempt": attempt,
                            "dispatch_order": dispatch_order,
                            "commit_order": failed_commit_order,
                            "metadata": _event_safe_metadata(result.metadata),
                            **self._trace_meta(turn.process, call),
                            "state_version": self._persisted_count + 1,
                            "state_delta": {},
                        },
                        {
                            "run_id": call.run_id,
                            "state_version": self._persisted_count + 1,
                            "payload": json.dumps(
                                self.state_store.snapshot() if self.state_store else {},
                                sort_keys=True,
                            ).encode(),
                        },
                        [],
                    )
                    self._persisted_count += 1
                    self._record_committed_attempt(call.invocation_id, attempt)
                    self._sync_persisted_state_version()
                    self._last_event_id = failed_event_id
                    self.commit_log.append(
                        {
                            "order": failed_commit_order,
                            "invocation_id": call.invocation_id,
                            "process_id": turn.process_id,
                            "phase": turn.phase,
                            "status": "failed",
                        }
                    )
                self.results.append(result)
                if attempt == turn.max_attempts:
                    self.status = "failed"
                    raise _ReportedProcessFailure(f"process {turn.process_id} failed")
                return False
            if result.status == "skipped":
                if self.persistence:
                    persistence_attempted = True
                    skipped_commit_order = len(self.commit_log) + 1
                    self.persistence.commit_process_result(
                        {
                            "event_id": f"{call.invocation_id}-attempt-{attempt}",
                            "invocation_id": call.invocation_id,
                            "run_id": call.run_id,
                            "kind": "process_skipped",
                            "process_id": turn.process_id,
                            "actors": list(call.actor_ids),
                            "phase": turn.phase,
                            "attempt": attempt,
                            "dispatch_order": dispatch_order,
                            "commit_order": skipped_commit_order,
                            "metadata": _event_safe_metadata(result.metadata),
                            **self._trace_meta(turn.process, call),
                            "state_version": self._persisted_count + 1,
                            "state_delta": {},
                        },
                        {
                            "run_id": call.run_id,
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
                    self._record_committed_attempt(call.invocation_id, attempt)
                    self._sync_persisted_state_version()
                    self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
                    self.commit_log.append(
                        {
                            "order": skipped_commit_order,
                            "invocation_id": call.invocation_id,
                            "process_id": turn.process_id,
                            "phase": turn.phase,
                            "status": "skipped",
                        }
                    )
                self.results.append(result)
                return True
            _validate_scheduling_effects(self.scheduler, result.scheduling_effects)
            raw_state_effects = turn.process.get("state_effects", [])
            declared: set[str] = set()
            for effect_item in raw_state_effects:
                if isinstance(effect_item, str):
                    declared.add(effect_item)
                elif isinstance(effect_item, Mapping) and effect_item.get("field"):
                    declared.add(str(effect_item["field"]))
            effects: Any
            if result.state_effects:
                effects = result.state_effects
            elif executor_mode(turn.process) in MODEL_CALL_MODES and not result.metadata.get(
                "recorded"
            ):
                # A model call's declared operations are applied to committed
                # state, not written whole from its outputs.
                effects = model_call_effects(turn.process, result.outputs, turn.actor_ids)
            elif _declares_composable_write(raw_state_effects):
                # A process that declares a keyed or accumulating write means it
                # for its own outputs too. Writing the field whole instead lost
                # the key: fanned-out siblings overwrote one another under
                # sequential timing, and collided at commit under simultaneous
                # timing -- after every call had already run. The compiler calls
                # such a declaration composable, so the runtime has to honour it.
                effects = model_call_effects(turn.process, result.outputs, turn.actor_ids)
            else:
                effects = {key: value for key, value in result.outputs.items() if key in declared}
            # A replayed result carries recorded deltas, which reproduce the
            # recorded state exactly when applied in commit order.
            if (
                turn.view is not None
                and turn.view.simultaneous
                and not result.metadata.get("recorded")
            ):
                # Every sibling computed its writes from the same view, so a
                # whole-field write would overwrite those committed before it.
                if isinstance(effects, Mapping):
                    whole = sorted(str(key) for key in effects)
                else:
                    whole = sorted(
                        str(effect.get("field"))
                        for effect in effects
                        if not _composes_with_siblings(effect, turn.actor_ids)
                    )
                if whole:
                    raise ValueError(
                        "SIMULTANEOUS_WRITE_CONFLICT: a simultaneous batch may commit only "
                        f"composable state effects; '{turn.process_id}' wrote whole fields: "
                        f"{', '.join(whole)}"
                    )
            if self.state_store and effects:
                self.state_store.apply(effects, declared, expected_version=self.state_store.version)
                state_applied = True
            declared_artifacts: list[dict[str, Any]] = []
            if self.artifact_store is not None:
                for artifact_id, value in result.outputs.items():
                    if artifact_id not in self.artifact_store.catalog:
                        continue
                    catalog = self.artifact_store.catalog[artifact_id]
                    artifact_instance_id = f"{artifact_id}-{call.invocation_id}-attempt-{attempt}"
                    producer_event = f"{call.invocation_id}-attempt-{attempt}"
                    consumed_input_ids = sorted(self._consumed_input_records(call).keys())
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
                        producer_process=turn.process_id,
                        producer_event=producer_event,
                        phase=turn.phase,
                    )
                    declared_artifacts.append(
                        {
                            "artifact_id": artifact_instance_id,
                            "run_id": call.run_id,
                            "payload": json.dumps(
                                {
                                    "value": _plain(value),
                                    "content_hash": _hash(value),
                                    "schema_ref": catalog.get("schema_ref"),
                                    "owner": catalog.get("owner"),
                                    "visibility": catalog.get("visibility"),
                                    "lifecycle_scope": catalog.get("lifecycle_scope"),
                                    "producer_event": producer_event,
                                    "producer_process": turn.process_id,
                                    "declared_artifact_id": artifact_id,
                                    "invocation_id": call.invocation_id,
                                    "phase": turn.phase,
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
                trace = turn.process.get("trace_policy", {})
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
                    and (turn.process.get("executor", {}).get("mode") == "generative")
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
                        "parsed_response": _plain(storage_metadata.get("parsed_response")),
                        "provider_attempts": _plain(storage_metadata.get("provider_attempts", [])),
                        "process_id": turn.process_id,
                        "invocation_id": call.invocation_id,
                        "phase": turn.phase,
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
                        "run_id": call.run_id,
                        "kind": "process_completed",
                        "process_id": turn.process_id,
                        "actors": list(call.actor_ids),
                        "phase": turn.phase,
                        "attempt": attempt,
                        "dispatch_order": dispatch_order,
                        "commit_order": len(self.commit_log) + 1,
                        "events": _plain(result.events),
                        "metadata": _event_safe_metadata(result.metadata),
                        "scheduling_effects": _plain(result.scheduling_effects),
                        **self._trace_meta(turn.process, call),
                        "state_version": persist_version,
                        "state_delta": state_delta,
                    },
                    {
                        "run_id": call.run_id,
                        "state_version": persist_version,
                        "payload": state_payload,
                    },
                    [
                        {
                            "artifact_id": (f"{call.invocation_id}-attempt-{attempt}"),
                            "run_id": call.run_id,
                            "payload": payload,
                        },
                        *declared_artifacts,
                    ],
                )
                persistence_committed = True
                self._persisted_count += 1
                self._record_committed_attempt(call.invocation_id, attempt)
                self._sync_persisted_state_version()
                self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
            commit_order = len(self.commit_log) + 1
            self.commit_log.append(
                {
                    "order": commit_order,
                    "invocation_id": call.invocation_id,
                    "process_id": turn.process_id,
                    "phase": turn.phase,
                }
            )
            self._completed_process_events.append(
                {
                    "event_id": f"{call.invocation_id}-attempt-{attempt}",
                    "process_id": turn.process_id,
                    "phase": turn.phase,
                    "actors": list(call.actor_ids),
                }
            )
            self._record_exchange(turn, call, result, attempt)
            if state_applied:
                self._record_state_writes(
                    f"{call.invocation_id}-attempt-{attempt}", turn.phase, effects
                )
            self.results.append(result)
            for emitted in result.events:
                self._event_history.append(
                    {
                        **_plain(emitted),
                        "producer_event": (f"{call.invocation_id}-attempt-{attempt}"),
                        "producer_process": turn.process_id,
                        "actor_ids": list(call.actor_ids),
                        "phase": turn.phase,
                    }
                )
                event_name = emitted.get("type") or emitted.get("kind") or emitted.get("event")
                if event_name:
                    self.scheduler.signal_event(str(event_name))
            for effect in result.scheduling_effects:
                effect_type = effect.get("type")
                if effect_type == "signal_event" and effect.get("event"):
                    self.scheduler.signal_event(str(effect["event"]))
                elif effect_type == "schedule":
                    target = effect.get("process_id")
                    scheduled_phase = effect.get("phase")
                    self.scheduler.schedule(str(target), cast(int | float, scheduled_phase))
            return True
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
            # Checked before anything is recorded: the provider could not serve
            # the call, so there is no attempt to count against the process and
            # nothing to commit. Recording it as failed made the run terminal.
            pause = provider_pause_reason(exc) if not executor_returned else None
            if (
                pause is None
                and not executor_returned
                and not persistence_committed
                and is_provider_cancellation(exc)
                and self._poll_external_status()
            ):
                # The researcher paused or cancelled this run, or the lease moved
                # on: the call was aborted on purpose, so it is not an attempt
                # that failed. Recording it as one made the run terminal, and a
                # resume then re-emitted this attempt's event id.
                pause = {
                    "kind": "cancelled" if self.status == "cancelled" else "researcher_paused",
                    "status": None,
                    "error": "the call was stopped by the researcher",
                }
            if pause is not None and not persistence_committed:
                raise _ProviderPause(
                    {**pause, "process_id": turn.process_id, "phase": turn.phase}
                ) from exc
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
                classification = "executor_exception" if not executor_returned else "invalid_output"
                try:
                    persistence_attempted = True
                    self.persistence.commit_process_result(
                        {
                            "event_id": f"{call.invocation_id}-attempt-{attempt}",
                            "invocation_id": call.invocation_id,
                            "run_id": call.run_id,
                            "kind": "process_failed",
                            "process_id": turn.process_id,
                            "actors": list(call.actor_ids),
                            "phase": turn.phase,
                            "attempt": attempt,
                            "dispatch_order": dispatch_order,
                            "commit_order": failed_commit_order,
                            "classification": classification,
                            "error": str(exc),
                            **self._trace_meta(turn.process, call),
                            "state_version": self._persisted_count + 1,
                            "state_delta": {},
                        },
                        {
                            "run_id": call.run_id,
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
                self._record_committed_attempt(call.invocation_id, attempt)
                self._sync_persisted_state_version()
                self._last_event_id = f"{call.invocation_id}-attempt-{attempt}"
                failure_persisted = True
                self.commit_log.append(
                    {
                        "order": failed_commit_order,
                        "invocation_id": call.invocation_id,
                        "process_id": turn.process_id,
                        "phase": turn.phase,
                        "status": "failed",
                    }
                )
            if attempt == turn.max_attempts:
                self.status = "failed"
                raise
            if persistence_attempted and not persistence_committed and not failure_persisted:
                self.status = "failed"
                raise
        return False

    def _activation_order(
        self,
        scope: _RunScope,
        process_id: str,
        phase: int | float,
        groups: list[tuple[str, ...]],
    ) -> list[tuple[str, ...]]:
        """The seeded per-phase order of a shuffled process's actors (CON-010).

        Drawn from its own stream, so it never perturbs an executor's seed, and
        shared across matched conditions when that stream is shared, so order is
        not a confound between paired runs. A resumed run redraws the same order.
        """
        shared = scope.matching.get("shared_streams", [])
        matching_key = (
            ACTIVATION_ORDER_STREAM
            if scope.matching.get("enabled") is True
            and isinstance(shared, list | tuple)
            and ACTIVATION_ORDER_STREAM in shared
            else None
        )
        ordered = list(groups)
        random.Random(
            derive_seed(
                scope.seed,
                scope.seed_identity or scope.run_id,
                process_id,
                f"{ACTIVATION_ORDER_STREAM}@{phase}",
                experiment_id=scope.experiment_id,
                condition_id=scope.condition_id,
                replication=scope.replication,
                matching_key=matching_key,
            )
        ).shuffle(ordered)
        return ordered

    def _scheduling_stable(
        self,
        process_id: str,
        phase: int | float,
        state: Mapping[str, Any],
        executed_this_phase: set[str],
    ) -> bool:
        """Whether the scheduler would keep choosing this batch for every remaining actor.

        The serial loop re-evaluates readiness before each actor and consumes one
        scheduled entry per turn, so a scheduled entry can be what keeps a process
        ready or ahead of another. The choice cannot change part-way through when
        no entry of this process remains to consume and it is still the first
        ready process (CON-012).
        """
        if any(
            entry.process_id == process_id and entry.phase <= phase
            for entry in self.scheduler._scheduled
        ):
            return False
        ready = [
            entry
            for entry in self.scheduler.ready(phase, state=state)
            if entry.process_id == process_id or entry.process_id not in executed_this_phase
        ]
        return bool(ready) and ready[0].process_id == process_id

    def _batch_limit(
        self,
        process_id: str,
        batch_key: tuple[str, int | float],
        scheduling_stable: Callable[[], bool] | None = None,
    ) -> int:
        """How many of a batch's calls may run at once, recording why (CON-011).

        Only a batch prepared from a view, run by an executor that touches nothing
        but its invocation, may run concurrently: its actors cannot see one
        another's results, so running them together records the same run as
        running them one at a time.
        """
        view = self._batch_views.get(batch_key)
        if view is None and not self._batch_orders.get(batch_key):
            if len(self._actor_queues.get(batch_key) or ()) <= 1:
                return 1
        requested = int(self.max_concurrency.get(process_id, 1))
        if not getattr(self.registry.get(process_id), "concurrent_safe", False):
            limit, reason = 1, "its executor is not a concurrency-safe model call"
        elif view is None:
            limit, reason = (
                1,
                "a later actor can see an earlier actor's result from the same batch, "
                "and its timing is sequential",
            )
        elif not view.simultaneous and can_ready_others(
            self.scheduler.processes[process_id], self.scheduler.processes
        ):
            limit, reason = (
                1,
                "its commits can make another process ready part-way through the batch",
            )
        elif (
            requested > 1
            and not view.simultaneous
            and scheduling_stable is not None
            and not scheduling_stable()
        ):
            # A simultaneous batch is forced to finish, so only an undeclared
            # batch's turns depend on the scheduler's choice.
            limit, reason = 1, "its scheduling can change part-way through the batch"
        elif requested <= 1:
            limit, reason = 1, "max_concurrency is 1"
        else:
            limit = requested
            reason = "simultaneous" if view.simultaneous else "batch-independent"
        self.execution_decisions.setdefault(
            process_id,
            {"max_concurrency": requested, "concurrent": limit > 1, "reason": reason},
        )
        return limit

    def _stop_on_budget(self, *, work_remains: bool) -> None:
        """Record why the run stopped when its event cap is reached.

        One rule for both dispatch paths. Reaching the cap is not by itself a
        truncation -- a run that finished on its last permitted event finished
        -- so only a cap that denied work still waiting is recorded as one.
        """
        if work_remains:
            self.budget_exhausted = True
        self.status = "completed"

    def _run_batch_concurrently(
        self,
        scope: _RunScope,
        batch_key: tuple[str, int | float],
        template: _ActorTurn,
        limit: int,
        max_events: int | None,
        executed: list[str],
    ) -> bool:
        """Run a batch's remaining calls concurrently, committing in actor order (CON-012).

        Calls are prepared on this thread from the batch view and only executed on
        workers; each result is committed here once every earlier actor's has been,
        so dispatch and commit orders, state and records match a one-at-a-time run.
        Returns ``False`` when the run must stop -- paused, cancelled or out of its
        event budget -- leaving unfinished actors queued for a resume (CON-014).
        """
        queue = self._actor_queues[batch_key]
        pending: deque[tuple[_ActorTurn, ProcessInvocation, Future[Any]]] = deque()
        prepare_error: Exception | None = None
        stopped_cleanly = False
        first_turn = True
        pool = ThreadPoolExecutor(
            max_workers=limit, thread_name_prefix=f"genesis-{template.process_id}"
        )
        try:
            while pending or (queue and prepare_error is None):
                # Poll before submitting more work, as the serial loop polls before
                # preparing each actor.
                if self._poll_external_status():
                    stopped_cleanly = True
                    return False
                # Never more calls in flight than the limit (backpressure), nor more
                # than the event budget can still commit.
                budget = None if max_events is None else max_events - self._event_count
                while (
                    prepare_error is None
                    and queue
                    and len(pending) < limit
                    and (budget is None or len(pending) < budget)
                ):
                    turn = replace(template, actor_ids=queue.pop(0))
                    try:
                        call = self._prepare_call(scope, turn, 1)
                    except Exception as exc:
                        # A serial run commits every earlier actor before this one
                        # fails to prepare, so those are committed first. The actor
                        # stays queued, so a pause before the error is raised does
                        # not silently drop it.
                        queue.insert(0, turn.actor_ids)
                        prepare_error = exc
                        break
                    pending.append((turn, call, pool.submit(self._execute_call, turn, call)))
                if not pending:
                    if prepare_error is not None:
                        break
                    # The caller checks the cap before a batch starts, so this is
                    # not expected; if it is reached, the actors still queued
                    # were denied by the cap and the run says so.
                    self._stop_on_budget(work_remains=bool(queue))
                    stopped_cleanly = True
                    return False
                turn, call, future = pending.popleft()
                if not first_turn:
                    # The serial loop consumes one scheduled entry per actor turn;
                    # the first was consumed before this batch began.
                    self.scheduler.consume_scheduled(turn.process_id, turn.phase)
                first_turn = False
                self._commit_turn(scope, turn, call, future.result())
                executed.append(turn.process_id)
                self._event_count += 1
                self.scheduler.mark_started(turn.process_id, turn.phase)
                if max_events is not None and self._event_count >= max_events:
                    self._stop_on_budget(work_remains=bool(queue or pending))
                    stopped_cleanly = True
                    return False
            if prepare_error is not None:
                raise prepare_error
            stopped_cleanly = True
            return True
        finally:
            if not stopped_cleanly and self.cancel_event is not None:
                # The run is failing: stop calls still in flight instead of letting
                # them run on, and spend, after their results can no longer be used.
                self.cancel_event.set()
            self._discard_pending(pending, queue)
            pool.shutdown(wait=False, cancel_futures=True)

    def _commit_turn(
        self,
        scope: _RunScope,
        turn: _ActorTurn,
        call: ProcessInvocation,
        outcome: ProcessResult | _ExecutorRaised,
    ) -> None:
        """Commit one actor's executed attempt, then any retries, in serial order."""
        attempt = self._next_attempt(call.invocation_id)
        while True:
            dispatch_order = self._log_dispatch(turn, call, attempt)
            if self._commit_attempt(turn, call, dispatch_order, attempt, outcome):
                return
            attempt += 1
            if attempt > turn.max_attempts:
                return
            call = self._prepare_call(scope, turn, attempt)
            outcome = self._execute_call(turn, call)

    def _discard_pending(
        self,
        pending: deque[tuple[_ActorTurn, ProcessInvocation, Future[Any]]],
        queue: list[tuple[str, ...]],
    ) -> None:
        """Requeue actors whose calls will not be committed, and account for them.

        A call that already ran was paid for even though its result is dropped, so
        its usage is kept (CON-013). A call still running when the batch stopped is
        counted, but its usage is not yet known.
        """
        if not pending:
            return
        queue[0:0] = [turn.actor_ids for turn, _call, _future in pending]
        usage = self.discarded_calls["usage"]
        for _turn, _call, future in pending:
            if future.cancel():
                continue
            self.discarded_calls["calls"] += 1
            if not future.done():
                self.discarded_calls["unfinished"] += 1
                continue
            try:
                outcome = future.result()
            except BaseException:
                continue
            if isinstance(outcome, ProcessResult):
                for key, value in dict(outcome.metadata.get("usage") or {}).items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        usage[key] = usage.get(key, 0) + value
        pending.clear()

    def _reads_from_view(self, process_id: str, process: Mapping[str, Any]) -> bool:
        """Whether a process's batches are prepared from a batch view (CON-008).

        A simultaneous batch always is. So is an undeclared batch in which nothing
        its own actors write can reach a sibling through anything the package
        declares: the view is then identical to live reads for every declared
        channel, and it also closes the channels a package does not declare. A
        declared sequential batch, and an undeclared dependent one, read live.
        """
        decided = self._view_decisions.get(process_id)
        if decided is None:
            mode, _order = timing_of(process)
            if mode is not None:
                decided = mode == "simultaneous"
            else:
                policy = self.context_engine.policies.get(
                    str(process.get("context_policy", "private"))
                )
                decided = not batch_dependencies(process, policy)
            self._view_decisions[process_id] = decided
        return decided

    def _reopen_interrupted_batches(self, scope: _RunScope) -> None:
        """Reopen batches a pause interrupted, exactly as they began (CON-008, CON-010).

        The queue is the recorded actor order less the actors already committed --
        not re-derived from the current state, which those actors changed, nor
        re-shuffled. A batch that reads from a view gets its view back, and so
        runs to completion without its trigger being re-evaluated.
        """
        candidates = dict(self._recorded_batch_orders)
        for process_id, phase, _actors in list(self._completed_actor_occurrences):
            batch_key = (process_id, phase)
            process = self.scheduler.processes.get(process_id)
            if batch_key in candidates or process is None:
                continue
            declared = _declared_order(process)
            if declared is not None and len(declared) > 1:
                candidates[batch_key] = declared
        for batch_key, order in candidates.items():
            process_id, phase = batch_key
            process = self.scheduler.processes.get(process_id)
            if process is None or batch_key in self._actor_queues:
                continue
            remaining = [
                group
                for group in order
                if (process_id, phase, group) not in self._completed_actor_occurrences
            ]
            if not remaining:
                # Every recorded actor committed: the batch is done, even if the
                # current state would now expand to a different actor list.
                self.scheduler.consume_scheduled(process_id, phase)
                self._complete_occurrence(process_id, phase)
                continue
            if (process_id, phase) in self.scheduler._completed_occurrences:
                continue
            self._actor_queues[batch_key] = remaining
            self._batch_orders[batch_key] = order
            if len(order) > 1 and self._reads_from_view(process_id, process):
                mode, _order = timing_of(process)
                self._batch_views[batch_key] = self._open_batch_view(
                    scope, batch_key, simultaneous=mode == "simultaneous"
                )

    def _complete_occurrence(self, process_id: str, phase: int | float) -> None:
        """Complete a process occurrence once, however many paths reach its end."""
        if (process_id, phase) not in self.scheduler._completed_occurrences:
            self.scheduler.complete(process_id, phase)

    def _recorded_view_version(
        self, batch_key: tuple[str, int | float], *, simultaneous: bool
    ) -> int | None:
        """The view version a resumed batch must be rebuilt at, if any (consumed once).

        A simultaneous batch's actors all read the view as the batch began. An
        undeclared batch's view follows other processes' commits, so its remaining
        actors read what its last committed actor read -- or the run as it now
        stands, if another process committed after that.
        """
        earliest = self._batch_view_versions.pop(batch_key, None)
        latest = self._batch_view_latest.pop(batch_key, None)
        if simultaneous:
            return earliest
        if latest is None:
            return None
        if any(
            version > latest and producer != batch_key[0]
            for version, producer in self._persisted_event_processes
        ):
            return None
        return latest

    def _foreign_commit_since_view(self, batch_key: tuple[str, int | float]) -> bool:
        """Whether another process committed since this batch's view was taken."""
        start = self._view_opened_at.get(batch_key, len(self.commit_log))
        return any(entry.get("process_id") != batch_key[0] for entry in self.commit_log[start:])

    def _open_batch_view(
        self, scope: _RunScope, batch_key: tuple[str, int | float], *, simultaneous: bool
    ) -> _BatchView:
        """Fix what a batch's actors see when it reads from a view (CON-008).

        Normally the run as it stands when the batch begins. When a resumed run
        re-enters a batch whose first siblings already committed, the view is
        rebuilt as of the version they recorded, so the remaining actors see
        exactly what the first ones saw.
        """
        self._view_opened_at[batch_key] = len(self.commit_log)
        current = self.state_store.version if self.state_store else 0
        recorded = self._recorded_view_version(batch_key, simultaneous=simultaneous)
        if recorded is None or recorded >= current:
            return _BatchView(
                state=self.state_store.snapshot() if self.state_store else dict(scope.state or {}),
                state_version=current,
                event_history=tuple(self._event_history),
                artifact_ids=(
                    frozenset(self.artifact_store._metadata)
                    if self.artifact_store is not None
                    else None
                ),
                simultaneous=simultaneous,
            )
        # Only persisted events carry a known version; anything this session
        # committed is later than the recorded view, so it is excluded.
        versions = self._event_state_versions

        def before_view(event_id: Any) -> bool:
            version = versions.get(str(event_id))
            return version is not None and version <= recorded

        return _BatchView(
            state=self._state_at_version(scope, recorded),
            state_version=recorded,
            event_history=tuple(
                entry for entry in self._event_history if before_view(entry.get("producer_event"))
            ),
            artifact_ids=(
                frozenset(
                    instance_id
                    for instance_id, metadata in self.artifact_store._metadata.items()
                    # An artifact with no producer existed before the run.
                    if metadata.get("producer_event") is None
                    or before_view(metadata.get("producer_event"))
                )
                if self.artifact_store is not None
                else None
            ),
            simultaneous=simultaneous,
        )

    def _state_at_version(self, scope: _RunScope, version: int) -> dict[str, Any]:
        """The committed state at ``version``, read back from persistence."""
        if version <= 0:
            if self._initial_state is not None:
                return copy.deepcopy(self._initial_state)
            return dict(scope.state or {})
        history = getattr(self.persistence, "iter_state_history", None)
        if history is not None:
            iterator = history(scope.run_id)
            try:
                for candidate, snapshot in iterator:
                    if candidate == version:
                        return dict(snapshot)
                    if candidate > version:
                        break
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        raise ValueError(
            f"INFORMATION_TIMING_VIEW_UNAVAILABLE: state version {version} of run "
            f"'{scope.run_id}' cannot be read to rebuild a simultaneous batch's view"
        )

    def run(self, run_id: str, *args: Any, **kwargs: Any) -> list[str]:
        """Execute the run until it ends, pauses, is cancelled or reaches its cap.

        A call the provider cannot serve -- no credit, a rejected key, an outage
        that outlasted the retries -- pauses the run instead of failing it. The
        call is not recorded as a failure: nothing it would have produced was
        committed, and resuming runs it again. ``pause_reason`` says why.
        """
        try:
            return self._run_body(run_id, *args, **kwargs)
        except _ProviderPause as pause:
            # A cancel that aborted an in-flight call already set the status;
            # cancelled is final and must not be downgraded to paused.
            if pause.reason.get("kind") != "cancelled":
                self.status = "paused"
                self.pause_reason = pause.reason
            else:
                self.status = "cancelled"
            return list(self._executed)

    def _run_body(
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
        self._terminal_phase = terminal_phase
        self._phase_start = int(phase_start)
        if self.state_store is None and state is not None:
            # A caller may provide a lightweight initial snapshot without a
            # separately declared state schema. Infer the narrow runtime schema
            # for this convenience path; compiled studies still provide an
            # explicit schema through ``StateStore``.
            self.state_store = StateStore({key: type(value) for key, value in state.items()}, state)
        self._restore_persisted_frontier(run_id)
        if max_events is not None and self._event_count >= max_events:
            self.budget_exhausted = True
            self.status = "completed"
            return []
        executed: list[str] = []
        self._executed = executed
        condition = dict(condition or {})
        matching = dict(matching or {})
        scope = _RunScope(
            run_id=run_id,
            seed=seed,
            seed_identity=seed_identity,
            experiment_id=experiment_id,
            condition_id=condition_id,
            replication=replication,
            matching=matching,
            condition=condition,
            state=state,
        )
        self._reopen_interrupted_batches(scope)
        stop_phase = phase_end + 1 if phase_end is not None else phase_limit
        start_phase = max(self._next_phase, phase_start)
        for phase in range(start_phase, stop_phase):
            if self._poll_external_status():
                return executed
            if self.status in {"paused", "cancelled"}:
                return executed
            executed_this_phase: set[str] = set()
            # Capture the state snapshot at the START of every phase for the
            # state-feedback ring (feedback lag N reads the ring at phase - N).
            # Resuming re-enters the interrupted phase with state that is
            # already partly updated, so an existing entry — the real
            # round-start snapshot, kept in memory or rebuilt from persistence —
            # is never overwritten. Otherwise pausing mid-round changed what
            # later readers saw, making an operational interruption alter the
            # simulation's information flow.
            snapshot_now = self.state_store.snapshot() if self.state_store else dict(state or {})
            if self._speculative_round_start == phase:
                # Reconstructed from the interrupted round before it finished;
                # the round has now completed, so record its real end state.
                self._round_state_at_phase[phase] = snapshot_now
                self._speculative_round_start = None
            else:
                self._round_state_at_phase.setdefault(phase, snapshot_now)
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
                # An open simultaneous batch finishes before anything else runs,
                # and its trigger is not re-evaluated: it was evaluated once, as
                # the batch began, so a sibling's commit cannot stop the batch
                # part-way (CON-008).
                open_batch = next(
                    (
                        process_id
                        for (process_id, batch_phase), open_view in self._batch_views.items()
                        if open_view.simultaneous
                        and batch_phase == phase
                        and process_id not in executed_this_phase
                        and self._actor_queues.get((process_id, batch_phase))
                    ),
                    None,
                )
                if open_batch is not None:
                    ready = [
                        ScheduledProcess(open_batch, phase),
                        *(entry for entry in ready if entry.process_id != open_batch),
                    ]
                if not ready:
                    break
                item = ready[0]
                executed_this_phase.add(item.process_id)
                self.scheduler.consume_scheduled(item.process_id, phase)
                process = self.scheduler.processes[item.process_id]
                feedback_slots, feedback_blocked = self._feedback_history(process, phase)
                if feedback_blocked:
                    # initial.policy skip_consumer: not enough rounds of
                    # history yet — defer this process to a later phase.
                    executed_this_phase.add(item.process_id)
                    continue
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
                    timing_mode, timing_order = timing_of(process)
                    if timing_order == "shuffled":
                        actor_groups = self._activation_order(
                            scope, item.process_id, phase, actor_groups
                        )
                    self._actor_queues[actor_key] = [
                        actors
                        for actors in actor_groups
                        if (item.process_id, phase, actors) not in self._completed_actor_occurrences
                    ]
                    batched = len(actor_groups) > 1 and bool(self._actor_queues[actor_key])
                    reads_from_view = batched and self._reads_from_view(item.process_id, process)
                    # Actors drawn from state must be recorded as well as viewed
                    # or shuffled ones: they are the case where re-expanding
                    # after a pause can yield a different set, because the batch
                    # may have written the field it draws from. A literal actor
                    # list re-expands identically and needs no record.
                    drawn_from_state = (
                        isinstance(process.get("actors"), Mapping)
                        and process["actors"].get("ids") is None
                    )
                    if batched and (
                        reads_from_view or timing_order == "shuffled" or drawn_from_state
                    ):
                        self._batch_orders[actor_key] = tuple(actor_groups)
                    if reads_from_view:
                        self._batch_views[actor_key] = self._open_batch_view(
                            scope, actor_key, simultaneous=timing_mode == "simultaneous"
                        )
                if not self._actor_queues[actor_key]:
                    self._complete_occurrence(item.process_id, phase)
                    continue
                open_view = self._batch_views.get(actor_key)
                if (
                    open_view is not None
                    and not open_view.simultaneous
                    and self._foreign_commit_since_view(actor_key)
                ):
                    # Another process committed part-way through this undeclared
                    # batch. Actors read live state before batch views existed, so
                    # the remaining actors see that commit; their own siblings stay
                    # unseen, because nothing the batch writes reaches its context.
                    self._batch_views[actor_key] = self._open_batch_view(
                        scope, actor_key, simultaneous=False
                    )
                batch_limit = self._batch_limit(
                    item.process_id,
                    actor_key,
                    partial(
                        self._scheduling_stable,
                        item.process_id,
                        phase,
                        scheduler_state,
                        executed_this_phase,
                    ),
                )
                # Asked before the turn is taken, never after it is committed.
                # Checking afterwards fired on the last natural event too, so a
                # run that finished on its own was recorded as cut short; asking
                # here means the cap is only a truncation when there was still a
                # turn waiting to be denied. Asked before either dispatch path:
                # asked only on the serial one, a cap reached just before a
                # concurrent batch let the batch submit nothing and record the
                # run as finished.
                if max_events is not None and self._event_count >= max_events:
                    # Reached here only because a turn is waiting to be taken.
                    self._stop_on_budget(work_remains=True)
                    return executed
                if batch_limit > 1:
                    template = _ActorTurn(
                        process_id=item.process_id,
                        process=process,
                        phase=phase,
                        actor_ids=(),
                        max_attempts=max_attempts,
                        feedback_slots=feedback_slots,
                        view=self._batch_views.get(actor_key),
                    )
                    if not self._run_batch_concurrently(
                        scope, actor_key, template, batch_limit, max_events, executed
                    ):
                        return executed
                    self._actor_queues.pop(actor_key, None)
                    self._batch_views.pop(actor_key, None)
                    self._batch_orders.pop(actor_key, None)
                    self._view_opened_at.pop(actor_key, None)
                    self._complete_occurrence(item.process_id, phase)
                    continue
                actor_ids = self._actor_queues[actor_key].pop(0)
                turn = _ActorTurn(
                    process_id=item.process_id,
                    process=process,
                    phase=phase,
                    actor_ids=actor_ids,
                    max_attempts=max_attempts,
                    feedback_slots=feedback_slots,
                    view=self._batch_views.get(actor_key),
                )
                first_attempt = self._next_attempt(self._prepare_call(scope, turn, 1).invocation_id)
                for attempt in range(first_attempt, max_attempts + 1):
                    call = self._prepare_call(scope, turn, attempt)
                    dispatch_order = self._log_dispatch(turn, call, attempt)
                    outcome = self._execute_call(turn, call)
                    if self._commit_attempt(turn, call, dispatch_order, attempt, outcome):
                        break
                executed.append(item.process_id)
                self._event_count += 1
                self.scheduler.mark_started(item.process_id, phase)
                actor_key = (item.process_id, phase)
                if self._actor_queues.get(actor_key):
                    executed_this_phase.discard(item.process_id)
                    continue
                self._actor_queues.pop(actor_key, None)
                self._batch_views.pop(actor_key, None)
                self._batch_orders.pop(actor_key, None)
                self._view_opened_at.pop(actor_key, None)
                self._complete_occurrence(item.process_id, phase)
            self._next_phase = phase + 1
            if self.status in {"cancelled", "paused"}:
                break
        if self.status not in {"cancelled", "paused"}:
            self.status = "completed"
        return executed

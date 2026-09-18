"""Deterministic compilation of a canonical GENESIS study package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from .execution_manifest import _exclusion_reason, build_package_closure
from .information_timing import _predicate_paths, model_effect_problems, timing_diagnostics
from .measurement import measurement_diagnostics
from .providers import split_prompt_roles
from .runtime import (
    STATE_VALUE_TYPES,
    _cap_rule,
    _project_rule,
    _resolve_path,
    edge_delays,
    expand_protocol_conditions,
)
from .schema_validation import PackageSchemaCatalog, SchemaValidationError
from .specification.models import (
    DomainSpec,
    ModelsSpec,
    OpennessSpec,
    OutcomesSpec,
    ProtocolSpec,
    StrictModel,
    StudySpec,
    TheorySpec,
)
from .theory_execution import compile_theory_execution

# Keys a canonical artifact used to accept and no longer does. Strict models
# reject an unknown key with a pydantic error that says only "extra inputs are
# not permitted" -- true, but useless to a researcher whose package was valid
# last month. A retired key gets a sentence saying where the thing went instead.
#
# These are refused, not dropped: silently ignoring a declared `replications: 3`
# would quietly change how much of the study runs.
RETIRED_KEYS: dict[str, dict[str, str]] = {
    "protocol": {
        "replications": (
            "how many draws to take is chosen when the study is run, not when it "
            "is specified; remove it here and pass replications to the run"
        ),
        "budgets": (
            "an event cap guards against runaway spend rather than describing the "
            "design; remove it here and pass max_events to the run"
        ),
        "checkpoints": (
            "nothing has ever read this: the runtime takes no checkpoints from a "
            "protocol declaration, so it promised a durability policy that does "
            "not exist; remove it"
        ),
        "replay_retention": (
            "nothing has ever read this: replay keeps what the run recorded, and "
            "no retention policy is applied from here; remove it"
        ),
    },
}

CANONICAL: dict[str, type[StrictModel]] = {
    "study": StudySpec,
    "openness": OpennessSpec,
    "theory": TheorySpec,
    "domain": DomainSpec,
    "protocol": ProtocolSpec,
    "outcomes": OutcomesSpec,
    "models": ModelsSpec,
}


def _source_file_for(path: str) -> str:
    """The canonical file a diagnostic path belongs to.

    Compiler paths come in both spellings ('/protocol' and
    'openness.processes.x'); both name their section first.
    """
    first = str(path).lstrip("/").replace(".", "/").split("/", 1)[0]
    return f"{first}.yaml" if first in CANONICAL else "package"


@dataclass(frozen=True)
class ValidationRecord:
    code: str
    severity: str
    source_file: str
    json_pointer: str
    related_ids: tuple[str, ...]
    message: str
    remediation: str


class ValidationIssue(ValueError):
    def __init__(self, issues: list[ValidationRecord]):
        self.issues = issues
        super().__init__("\n".join(f"{item.code}: {item.message}" for item in issues))


def _resolve_context_policies(domain: DomainSpec) -> list[dict[str, Any]]:
    """Merge ``domain.availability`` temporal conditions into context policies.

    Review finding 9: the availability section was previously ignored; the
    context engine reads ``available_when`` from policy definitions.
    """
    availability_map: dict[str, Any] = {}
    for entry in domain.availability:
        entry_dict = entry if isinstance(entry, dict) else entry.model_dump(mode="json")
        path = entry_dict.get("path")
        when = entry_dict.get("available_when")
        if path and isinstance(when, dict):
            availability_map[str(path)] = when
    policies = []
    for policy in domain.visibility:
        if hasattr(policy, "model_dump"):
            merged = dict(policy.model_dump(mode="json"))
        elif isinstance(policy, dict):
            merged = dict(policy)
        else:
            continue
        if availability_map:
            existing = merged.get("available_when")
            if not isinstance(existing, dict):
                existing = {}
            merged["available_when"] = {**availability_map, **existing}
        policies.append(merged)
    return policies


def _validate_availability_rules(domain: DomainSpec) -> list[dict[str, str]]:
    """Refuse an availability rule the runtime would silently ignore.

    A rule the runtime does not recognise skips every check and leaves the item
    available always -- which is how gates written as bare predicates went open
    unnoticed. A key must be a rule key, a predicate key, or an allowed path
    carrying its own rule; anything else is a typo that reads as a gate.
    """
    from .runtime import AVAILABILITY_KEYS, PREDICATE_KEYS

    known = AVAILABILITY_KEYS | PREDICATE_KEYS
    errors: list[dict[str, str]] = []
    for policy in domain.visibility:
        definition = policy if isinstance(policy, dict) else policy.model_dump(mode="json")
        when = definition.get("available_when") or {}
        if not isinstance(when, Mapping):
            continue
        allowed = {str(path) for path in definition.get("allow") or ()}
        for key, rule in when.items():
            where = f"domain.visibility.{definition.get('id')}.available_when.{key}"
            if key in known:
                continue
            if key not in allowed or not isinstance(rule, Mapping):
                errors.append(
                    {
                        "code": "AVAILABILITY_RULE_INERT",
                        "severity": "error",
                        "path": where,
                        "message": (
                            f"'{key}' is neither an availability key ({', '.join(sorted(known))}) "
                            "nor a path this policy allows; the runtime would ignore it and "
                            "leave the item available always"
                        ),
                    }
                )
                continue
            stray = sorted(set(rule) - known)
            if stray:
                errors.append(
                    {
                        "code": "AVAILABILITY_RULE_INERT",
                        "severity": "error",
                        "path": where,
                        "message": (
                            f"the rule for '{key}' has keys the runtime does not read: "
                            f"{', '.join(stray)}; they would be ignored and the item left "
                            "available always"
                        ),
                    }
                )
    return errors


def _validate_context_allow(domain: DomainSpec) -> list[dict[str, str]]:
    """Refuse a context allow entry that resolves to nothing.

    The context engine reads an entry from one of its namespaces, or else as a
    state id, and silently skips a path that is not there. So an entry naming an
    artifact type, an attribute, or the protocol reads as a grant but delivers
    nothing: the clickbait detector's policy allowed only `article`, and every
    detector call was sent an empty context while scoring articles it never saw.
    """
    from .runtime import CONTEXT_ROOTS

    states = {state.id for state in domain.states}
    artifacts = {artifact.id for artifact in domain.artifacts}
    attributes = {attribute.id for attribute in domain.attributes}
    errors: list[dict[str, str]] = []
    for policy in domain.visibility:
        definition = policy if isinstance(policy, dict) else policy.model_dump(mode="json")
        for entry in definition.get("allow") or ():
            parts = str(entry).split(".")
            head = parts[0]
            if head in CONTEXT_ROOTS and (head != "state" or len(parts) == 1 or parts[1] in states):
                continue
            if head in states:
                continue
            if head == "state":
                why = f"'{parts[1]}' is not a declared state"
                hint = f"declared states: {', '.join(sorted(states)) or 'none'}"
            elif head in artifacts:
                why = f"'{head}' is an artifact type, not a state"
                hint = "artifacts a process consumes reach its model through `inputs`"
            elif head == "protocol":
                why = "the protocol is not part of a model's context"
                hint = (
                    "the round reaches a prompt through {phase}; `protocol.phase` is read "
                    "only by availability predicates"
                )
            elif head in attributes or head == "attributes":
                why = f"'{head}' is an attribute, not a state"
                hint = "actor attributes are not delivered through context; carry them in a state"
            else:
                why = f"'{head}' is neither a context namespace nor a declared state"
                hint = f"namespaces: {', '.join(CONTEXT_ROOTS)}"
            errors.append(
                {
                    "code": "CONTEXT_ALLOW_UNRESOLVED",
                    "severity": "error",
                    "path": f"domain.visibility.{definition.get('id')}.allow",
                    "message": (
                        f"allow entry '{entry}' resolves to nothing: {why}, so the policy "
                        f"delivers nothing for it; {hint}"
                    ),
                }
            )
    return errors


def _validate_prompt_context(
    source: Path, openness: OpennessSpec, domain: DomainSpec
) -> list[dict[str, str]]:
    """Refuse a model prompt that never puts its authorised context in front of it.

    The request a model receives is the rendered prompt and nothing else, and
    context enters it only through `{context}` or `{context.<path>}`. A prompt
    without either sends none of what the process's context policy allows, so
    the policy -- and every condition gate on it -- is inert for that model.
    Every clickbait prompt was a prompt specification saved as a dict repr, and
    each model answered blind. A process whose policy allows nothing -- the
    built-in private, public and none allow nothing -- may omit it.
    """
    from .information_timing import MODEL_CALL_MODES
    from .providers import _PLACEHOLDER

    grants = {
        str(policy.get("id")): bool(policy.get("allow"))
        for policy in _resolve_context_policies(domain)
    }
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        if process.executor.mode not in MODEL_CALL_MODES or not process.prompt_ref:
            continue
        if not grants.get(str(process.context_policy or "none"), False):
            continue
        path = source / "prompts" / f"{process.prompt_ref}.txt"
        if not path.is_file():
            continue  # a missing template renders as the whole context
        placeholders = {match.group(0) for match in _PLACEHOLDER.finditer(path.read_text())}
        if any(item == "{context}" or item.startswith("{context.") for item in placeholders):
            continue
        errors.append(
            {
                "code": "PROMPT_OMITS_CONTEXT",
                "severity": "error",
                "path": f"prompts/{path.name}",
                "message": (
                    f"process '{process.id}' is granted context by '{process.context_policy}', "
                    f"but its prompt '{process.prompt_ref}' contains neither {{context}} nor "
                    "{context.<path>}, so the model is sent none of it; a prompt is plain text, "
                    "and context reaches the model only through those placeholders"
                ),
            }
        )
    return errors


def _validate_context_scope(domain: DomainSpec) -> list[dict[str, str]]:
    """Reject malformed scopes and caps before a run can depend on them (CTX-008).

    Every entry carries a code. The compiler builds its validation records by
    code, and an entry without one crashed it with a KeyError instead of being
    reported -- so no malformed declaration was ever actually refused.
    """
    errors: list[dict[str, str]] = []
    state_ids = {str(state.id) for state in domain.states}

    def fail(code: str, path: str, message: str) -> None:
        errors.append({"code": code, "severity": "error", "path": path, "message": message})

    for policy in domain.visibility:
        raw = policy.model_dump(mode="json") if hasattr(policy, "model_dump") else policy
        if not isinstance(raw, dict):
            continue
        policy_id = str(raw.get("id", ""))
        allowed = {str(item) for item in (raw.get("allow") or ())}
        # A malformed cap otherwise fails during a run rather than at
        # compilation, which is where every other declaration is checked.
        cardinality = raw.get("cardinality") or {}
        if isinstance(cardinality, dict):
            for path, cap in cardinality.items():
                try:
                    _cap_rule(cap)
                except ValueError as exc:
                    fail(
                        "CONTEXT_CARDINALITY_INVALID",
                        f"domain.visibility.{policy_id}.cardinality.{path}",
                        str(exc),
                    )
        project = raw.get("project") or {}
        if not isinstance(project, dict):
            fail(
                "CONTEXT_PROJECT_INVALID",
                f"domain.visibility.{policy_id}.project",
                "project must be a mapping",
            )
            project = {}
        for path, rule in project.items():
            where = f"domain.visibility.{policy_id}.project.{path}"
            if str(path) not in allowed:
                fail(
                    "CONTEXT_PROJECT_INVALID",
                    where,
                    f"project names '{path}', which the policy does not allow",
                )
            try:
                _project_rule(rule)
            except ValueError as exc:
                fail("CONTEXT_PROJECT_INVALID", where, str(exc))
        scope = raw.get("scope") or {}
        if not isinstance(scope, dict):
            fail(
                "CONTEXT_SCOPE_INVALID",
                f"domain.visibility.{policy_id}.scope",
                "scope must be a mapping",
            )
            continue
        for path, rule in scope.items():
            where = f"domain.visibility.{policy_id}.scope.{path}"
            if str(path) not in allowed:
                fail(
                    "CONTEXT_SCOPE_INVALID",
                    where,
                    f"scope names '{path}', which the policy does not allow",
                )
            if not isinstance(rule, dict):
                fail("CONTEXT_SCOPE_INVALID", where, "scope rule must be a mapping")
                continue
            unknown = set(rule) - {"field", "in"}
            if unknown:
                fail(
                    "CONTEXT_SCOPE_INVALID",
                    where,
                    f"scope rule has unknown keys: {', '.join(sorted(unknown))}",
                )
            field = rule.get("field")
            if not isinstance(field, str) or not field:
                fail("CONTEXT_SCOPE_INVALID", where, "scope rule requires a non-empty 'field'")
            else:
                try:
                    _resolve_path({}, field)
                except ValueError as exc:
                    fail("CONTEXT_SCOPE_INVALID", where, f"scope field '{field}': {exc}")
            selectors = rule.get("in")
            if selectors is None:
                fail("CONTEXT_SCOPE_INVALID", where, "scope rule requires 'in'")
                continue
            if isinstance(selectors, str):
                selectors = [selectors]
            if not isinstance(selectors, list | tuple) or not selectors:
                fail(
                    "CONTEXT_SCOPE_INVALID",
                    where,
                    "'in' must be a selector path or a non-empty list",
                )
                continue
            for selector in selectors:
                problem = _selector_problem(selector, state_ids)
                if problem:
                    fail("CONTEXT_SCOPE_INVALID", where, problem)
    return errors


def _policy_allows(domain: DomainSpec) -> list[tuple[str, str]]:
    """Every (policy id, allowed path) a domain declares."""
    found: list[tuple[str, str]] = []
    for policy in domain.visibility:
        raw = policy.model_dump(mode="json") if hasattr(policy, "model_dump") else policy
        if not isinstance(raw, dict):
            continue
        found.extend((str(raw.get("id", "")), str(path)) for path in raw.get("allow") or ())
    return found


def _retry_fallback_outputs(process: Any) -> Any:
    """The outputs a retry policy substitutes when every attempt fails."""
    policy = getattr(process, "retry_policy", None)
    if isinstance(policy, Mapping):
        return policy.get("fallback_outputs")
    return getattr(policy, "fallback_outputs", None)


def _validate_input_producers(
    domain: DomainSpec, openness: OpennessSpec, warnings: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    """Refuse an input artifact no process produces.

    An artifact exists only because some process outputs it, so an input naming
    one nothing produces reads empty in every round of every condition. The
    process still runs, the actor still answers, and the study still reports --
    on an actor that was never shown the thing the design says it acts on. A
    creator reading a prior-performance artifact nobody writes publishes forty
    times with no feedback at all, which is not a finding about the phenomenon.
    """
    # ``outputs`` is the ordinary channel, not the only one: the runtime also
    # produces from a recorded_artifact executor's declared outputs and from a
    # retry policy's fallback outputs. Counting only the first refused packages
    # whose producer used another -- and with the check disabled they compiled,
    # ran and delivered the artifact. This refusal blocks a package outright, so
    # missing a channel costs far more than accepting one it cannot interpret.
    produced: set[str] = set()
    for process in openness.processes:
        produced.update(
            str(output.artifact_type) for output in process.outputs or () if output.artifact_type
        )
        parameters = getattr(process.executor, "parameters", None) or {}
        for channel in (parameters.get("outputs"), _retry_fallback_outputs(process)):
            if isinstance(channel, Mapping | list | tuple):
                produced.update(str(name) for name in channel)
    # An input names an artifact *id*; an output declares an artifact *type*, and
    # the two are often but not always the same string.
    declared = {
        str(artifact.id): str(artifact.artifact_type or artifact.id)
        for artifact in domain.artifacts
    }
    errors: list[dict[str, str]] = []
    warnings = [] if warnings is None else warnings
    for process in openness.processes:
        for name in process.inputs or ():
            artifact = str(name)
            # A reference to an undeclared artifact is already refused elsewhere;
            # this is the declared-but-unproduced case.
            if artifact not in declared:
                continue
            if artifact in produced:
                continue
            if declared[artifact] in produced:
                # Tolerated, because a channel this check cannot read might key
                # its output by the id. But the ordinary path keys inputs by the
                # artifact's own id, so a consumer naming an id whose only
                # producer declares the *type* receives nothing: probed, the
                # matching-id case delivers one input and this case delivers
                # none. Advisory rather than refused, because the whole point of
                # the tolerance is that this check cannot see every producer.
                warnings.append(
                    {
                        "code": "INPUT_ARTIFACT_TYPE_ONLY",
                        "severity": "warning",
                        "dependency_section": "openness",
                        "path": f"openness.processes.{process.id}.inputs/{artifact}",
                        "message": (
                            f"process '{process.id}' reads artifact '{artifact}', which nothing "
                            f"produces under that id; only its type "
                            f"'{declared[artifact]}' is produced. Unless a producer keys its "
                            "output by the id, this input is empty in every round"
                        ),
                    }
                )
                continue
            errors.append(
                {
                    "code": "INPUT_ARTIFACT_UNPRODUCED",
                    "severity": "error",
                    "dependency_section": "openness",
                    "path": f"openness.processes.{process.id}.inputs/{artifact}",
                    "message": (
                        f"process '{process.id}' reads artifact '{artifact}', which no "
                        "process produces; it would be empty in every round"
                    ),
                }
            )
    return errors


def _advise_unwritten_states(domain: DomainSpec, openness: OpennessSpec) -> list[dict[str, Any]]:
    """Flag a declared state nothing writes and initialization does not seed."""
    written = {
        str(effect.get("field"))
        for process in openness.processes
        for effect in process.state_effects or ()
        if isinstance(effect, Mapping) and effect.get("field")
    }
    seeded = str(getattr(domain.initialization, "state_field", "") or "")
    advisories: list[dict[str, Any]] = []
    for state in domain.states:
        state_id = str(state.id)
        if state_id in written or state_id == seeded:
            continue
        advisories.append(
            {
                "code": "STATE_NEVER_WRITTEN",
                "severity": "warning",
                "path": f"domain.states.{state_id}",
                "message": (
                    f"state '{state_id}' is declared but no process effect writes it and "
                    "initialization does not seed it; anything reading it, including a "
                    "theory feedback sourced from it, gets its initial value forever"
                ),
            }
        )
    return advisories


TRIGGER_KEYS = frozenset({"phase", "repeat", "type", "predicate", "event"})
# The scheduler resolves ordering from these two and nothing else.
DEPENDENCY_KEYS = frozenset({"after", "delay"})


def _validate_dependencies(openness: OpennessSpec) -> list[dict[str, str]]:
    """Refuse a dependency key the scheduler never reads.

    Ordering within a round comes from ``dependencies.after``. A block written
    ``{requires: [...], same_round_results_visible: true}`` states the round's
    chain clearly to a reader and says nothing to the scheduler, which then runs
    the processes in whatever order it likes -- distribution before publication,
    settlement before anyone has read anything. The study completes, every
    declared step having run, and measures nothing.
    """
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        block = process.dependencies if isinstance(process.dependencies, dict) else {}
        unknown = sorted(set(block) - DEPENDENCY_KEYS)
        if not unknown:
            continue
        hint = (
            " (ordering is declared with 'after')"
            if any(key in {"requires", "depends_on", "needs"} for key in unknown)
            else ""
        )
        errors.append(
            {
                "code": "DEPENDENCY_KEY_UNKNOWN",
                "severity": "error",
                "path": f"openness.processes.{process.id}.dependencies",
                "message": (
                    f"dependencies declares {', '.join(unknown)}, which the scheduler never "
                    f"reads; it reads only {', '.join(sorted(DEPENDENCY_KEYS))}{hint}"
                ),
            }
        )
    return errors


def _validate_triggers(openness: OpennessSpec) -> list[dict[str, str]]:
    """Refuse a trigger the scheduler cannot read.

    The scheduler reads exactly five keys, and it compares ``phase`` against the
    current round as a number. A trigger written ``{phase: round, rounds: 1-40}``
    reads perfectly well to a person and puts the string "round" where an integer
    belongs: the run dies on its first scheduling pass with a TypeError that
    names neither the process nor the declaration. Any other key -- ``rounds``
    most of all -- is simply never consulted, so a schedule written that way is
    silently the default one.
    """
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        trigger = process.trigger if isinstance(process.trigger, dict) else {}
        if not trigger:
            continue
        where = f"openness.processes.{process.id}.trigger"
        phase = trigger.get("phase")
        if phase is not None and not isinstance(phase, int) or isinstance(phase, bool):
            errors.append(
                {
                    "code": "TRIGGER_PHASE_INVALID",
                    "severity": "error",
                    "path": where,
                    "message": (
                        f"trigger phase is {phase!r}; it must be the integer round the "
                        "process first runs in. To run in particular rounds, use "
                        "{type: condition, predicate: {path: protocol.phase, op: in, "
                        "value: [...]}}"
                    ),
                }
            )
        unknown = sorted(set(trigger) - TRIGGER_KEYS)
        if unknown:
            errors.append(
                {
                    "code": "TRIGGER_KEY_UNKNOWN",
                    "severity": "error",
                    "path": where,
                    "message": (
                        f"trigger declares {', '.join(unknown)}, which the scheduler never "
                        f"reads; it reads only {', '.join(sorted(TRIGGER_KEYS))}"
                    ),
                }
            )
    return errors


def _validate_outcomes(outcomes: Any, domain: DomainSpec) -> list[dict[str, str]]:
    """Refuse an outcome the evaluator would skip, or evaluate as something else.

    The evaluator reads a small grammar and ignores everything outside it: an
    unknown source yields no rows, an unread aggregation key falls back to a
    count, and an aggregation with no field is skipped. The clickbait study
    declared ten outcomes that way; none was computed as written.
    """
    from .outcome_plan import (
        OUTCOME_AGGREGATION_KEYS,
        OUTCOME_AGGREGATION_OPS,
        OUTCOME_BUILTIN_SOURCES,
        OUTCOME_JOIN_KEYS,
        OUTCOME_MISSINGNESS_KEYS,
        OUTCOME_MISSINGNESS_POLICIES,
        OUTCOME_WINDOW_KEYS,
    )

    known_sources = (
        set(OUTCOME_BUILTIN_SOURCES)
        | {str(dataset.id) for dataset in outcomes.datasets}
        | {str(artifact.id) for artifact in domain.artifacts}
    )
    errors: list[dict[str, str]] = []

    def refuse(outcome_id: str, part: str, message: str, code: str) -> None:
        errors.append(
            {
                "code": code,
                "severity": "error",
                "path": f"outcomes.outcomes.{outcome_id}.{part}",
                "message": message,
            }
        )

    for outcome in outcomes.outcomes:
        oid = str(outcome.id)
        sources = outcome.source if isinstance(outcome.source, list) else [outcome.source]
        if len(sources) != 1:
            refuse(
                oid,
                "source",
                f"an outcome reads exactly one source; only the first of {sources} would be read",
                "OUTCOME_SOURCE_UNKNOWN",
            )
        join = outcome.join
        if isinstance(join, dict):
            unread = sorted(set(join) - OUTCOME_JOIN_KEYS)
            if unread:
                refuse(
                    oid,
                    "join",
                    f"join keys {unread} are never read; a join is {{left, right, on}}",
                    "OUTCOME_KEY_UNREAD",
                )
            for side, default in (("left", "events"), ("right", "artifacts")):
                name = str(join.get(side, default))
                if name not in known_sources:
                    refuse(
                        oid,
                        f"join.{side}",
                        f"join {side} '{name}' is not a source",
                        "OUTCOME_SOURCE_UNKNOWN",
                    )
        elif str(sources[0]) not in known_sources:
            refuse(
                oid,
                "source",
                f"source '{sources[0]}' is neither a built-in source "
                f"({', '.join(sorted(OUTCOME_BUILTIN_SOURCES))}), a declared artifact nor a "
                "dataset declared in outcomes.datasets, so it has no rows",
                "OUTCOME_SOURCE_UNKNOWN",
            )
        aggregation = outcome.aggregation or {}
        unread = sorted(set(aggregation) - OUTCOME_AGGREGATION_KEYS)
        if unread:
            refuse(
                oid,
                "aggregation",
                f"aggregation keys {unread} are never read; an aggregation is {{op, field}}",
                "OUTCOME_KEY_UNREAD",
            )
        op = aggregation.get("op", aggregation.get("type", aggregation.get("operation", "count")))
        if op not in OUTCOME_AGGREGATION_OPS:
            refuse(
                oid,
                "aggregation.op",
                f"aggregation '{op}' is not computed; use one of "
                f"{', '.join(sorted(OUTCOME_AGGREGATION_OPS))}, and derive anything else "
                "from the exported datasets",
                "OUTCOME_AGGREGATION_UNSUPPORTED",
            )
        if op != "count" and not (aggregation.get("field") or aggregation.get("select")):
            refuse(
                oid,
                "aggregation.field",
                f"a {op} needs a field; without one the outcome is skipped",
                "OUTCOME_FIELD_MISSING",
            )
        missingness = outcome.missingness or {}
        unread = sorted(set(missingness) - OUTCOME_MISSINGNESS_KEYS)
        if unread:
            refuse(
                oid,
                "missingness",
                f"missingness keys {unread} are never read",
                "OUTCOME_KEY_UNREAD",
            )
        policy = missingness.get("policy", "exclude")
        if policy not in OUTCOME_MISSINGNESS_POLICIES:
            refuse(
                oid,
                "missingness.policy",
                f"missingness policy '{policy}' is not applied; "
                f"use {' or '.join(sorted(OUTCOME_MISSINGNESS_POLICIES))}",
                "OUTCOME_KEY_UNREAD",
            )
        window = outcome.window or {}
        unread = sorted(set(window) - OUTCOME_WINDOW_KEYS)
        if unread:
            refuse(
                oid,
                "window",
                f"window keys {unread} are never read; a window is {{time_field, start, end}}",
                "OUTCOME_KEY_UNREAD",
            )
    return errors


def _validate_run_length(protocol: Any, warnings: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Refuse a declared run length or termination the engine would not honour.

    The runtime takes the run length from ``time_model.end`` only when it is a
    whole number, and never reads ``termination``: ``end: 12.0`` compiled clean
    and the run went on to the default 100 rounds, a model call in every one.
    """
    errors: list[dict[str, str]] = []
    time_model = protocol.time_model
    if time_model is None or time_model.type != "rounds":
        return errors

    def whole(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    for name in ("start", "end"):
        value = getattr(time_model, name)
        if value is not None and not whole(value):
            errors.append(
                {
                    "code": "TIME_MODEL_INVALID",
                    "severity": "error",
                    "path": f"protocol.time_model.{name}",
                    "message": (
                        f"time_model.{name} is {value!r}; a rounds time model counts whole "
                        "rounds, and any other value is ignored at run time"
                    ),
                }
            )
    if time_model.end is None:
        warnings.append(
            {
                "code": "TIME_MODEL_END_DEFAULT",
                "severity": "warning",
                "path": "protocol.time_model.end",
                "message": "no time_model.end is declared, so a run stops after 100 rounds",
            }
        )
    for index, rule in enumerate(protocol.termination):
        where = f"protocol.termination.{index}"
        supported = (
            isinstance(rule, dict)
            and rule.get("type") == "end_time"
            and set(rule) <= {"type", "at", "early_stopping"}
            and not rule.get("early_stopping")
        )
        if not supported:
            errors.append(
                {
                    "code": "TERMINATION_UNSUPPORTED",
                    "severity": "error",
                    "path": where,
                    "message": (
                        f"termination {rule!r} is not read by the engine; a run ends at "
                        "time_model.end, and the only termination it honours is "
                        "{type: end_time, at: <time_model.end>}"
                    ),
                }
            )
        elif rule.get("at") != time_model.end:
            errors.append(
                {
                    "code": "TERMINATION_UNSUPPORTED",
                    "severity": "error",
                    "path": where,
                    "message": (
                        f"termination ends at {rule.get('at')!r} but time_model.end is "
                        f"{time_model.end!r}; the run ends at time_model.end"
                    ),
                }
            )
    return errors


RETRY_POLICY_KEYS = frozenset({"max_attempts", "failure_policy", "fallback_outputs"})
FAILURE_POLICIES = frozenset({"fail_run", "use_declared_fallback", "skip_with_event"})


def _validate_retry_policies(openness: OpennessSpec) -> list[dict[str, str]]:
    """Refuse a retry policy the runtime would read differently than it says.

    The runtime reads max_attempts, failure_policy and fallback_outputs. The
    clickbait detector declared on_exhausted: record-missing, which nothing
    reads: a failed detection failed the whole run instead of being recorded as
    missing.
    """
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        policy = process.retry_policy or {}
        where = f"openness.processes.{process.id}.retry_policy"

        def refuse(message: str, where: str = where) -> None:
            errors.append(
                {
                    "code": "RETRY_POLICY_INVALID",
                    "severity": "error",
                    "path": where,
                    "message": message,
                }
            )

        unread = sorted(set(policy) - RETRY_POLICY_KEYS)
        if unread:
            refuse(
                f"retry_policy keys {unread} are never read; a retry policy is "
                f"{{{', '.join(sorted(RETRY_POLICY_KEYS))}}}"
            )
        attempts = policy.get("max_attempts", 1)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            refuse(f"max_attempts is {attempts!r}; it must be a whole number of at least 1")
        failure = policy.get("failure_policy", "fail_run")
        if failure not in FAILURE_POLICIES:
            refuse(
                f"failure_policy '{failure}' is not applied; use one of "
                f"{', '.join(sorted(FAILURE_POLICIES))}"
            )
        fallback = policy.get("fallback_outputs")
        if failure == "use_declared_fallback" and not isinstance(fallback, dict):
            refuse("failure_policy use_declared_fallback needs fallback_outputs to substitute")
    return errors


def _validate_retention(openness: OpennessSpec) -> list[dict[str, str]]:
    """Refuse a retention value the engine does not read as keep or purge.

    Retention was matched as a substring, so "never-purge-this" purged every raw
    provider body of its process; an unknown value now fails at compile.
    """
    from .persistence import RETENTION_KEEP, RETENTION_PURGE

    errors: list[dict[str, str]] = []
    for process in openness.processes:
        retention = process.trace_policy.retention
        if retention is None or retention in RETENTION_KEEP | RETENTION_PURGE:
            continue
        errors.append(
            {
                "code": "RETENTION_UNKNOWN",
                "severity": "error",
                "path": f"openness.processes.{process.id}.trace_policy.retention",
                "message": (
                    f"retention '{retention}' is not a value the engine reads; use one of "
                    f"{', '.join(sorted(RETENTION_PURGE))} to purge raw provider responses, "
                    f"or {', '.join(sorted(RETENTION_KEEP))} to keep them"
                ),
            }
        )
    return errors


def _phase_lists(predicate: Any) -> list[list[Any]]:
    """The value lists of every `protocol.phase in [...]` inside a predicate."""
    if isinstance(predicate, Mapping):
        found = []
        if predicate.get("path") == "protocol.phase" and predicate.get("op") == "in":
            value = predicate.get("value")
            if isinstance(value, list):
                found.append(value)
        for key in ("all", "any"):
            for item in predicate.get(key) or ():
                found.extend(_phase_lists(item))
        found.extend(_phase_lists(predicate.get("not")))
        return found
    return []


def _advise_rounds_without_repeat(openness: OpennessSpec) -> list[dict[str, Any]]:
    """Warn when a trigger lists several rounds but is not declared to repeat.

    A condition trigger without `repeat: true` fires once, the first time its
    predicate holds, so every later round in the list is ignored. The clickbait
    reflection named rounds 3, 6, ..., 39 and ran in round 3 only; creators were
    handed that one reflection at every later stage.
    """
    warnings: list[dict[str, Any]] = []
    for process in openness.processes:
        trigger = process.trigger if isinstance(process.trigger, dict) else {}
        if trigger.get("type") != "condition" or trigger.get("repeat"):
            continue
        rounds = [value for value in _phase_lists(trigger.get("predicate")) if len(value) > 1]
        if rounds:
            warnings.append(
                {
                    "code": "TRIGGER_ROUNDS_WITHOUT_REPEAT",
                    "severity": "warning",
                    "path": f"openness.processes.{process.id}.trigger",
                    "message": (
                        f"the trigger lists rounds {rounds[0]} but does not declare "
                        "repeat: true, so it fires once, in the first of them; add "
                        "repeat: true to run in each"
                    ),
                }
            )
    return warnings


def _process_actor_roles(process: Any) -> tuple[str, ...]:
    """The role name of each position in a process's actor groups."""
    actors = getattr(process, "actors", None)
    if actors is None:
        return ()
    roles = [str(getattr(actors, "role", "actor") or "actor")]
    per = getattr(actors, "per", None)
    if per is not None:
        roles.append(str(getattr(per, "role", "per") or "per"))
    return tuple(roles)


def _actor_role_problems(process: Any, names: Any) -> list[str]:
    """Refuse actor fields whose form does not match the process's actor roles.

    A paired process has two actors, so a bare list of field names has no way to
    say which id it means; an unpaired one has a single role, so a mapping may
    only name that role.
    """
    roles = _process_actor_roles(process)
    if not roles:
        return []
    if isinstance(names, Mapping):
        problems = []
        unknown = sorted({str(role) for role in names.values() if str(role) not in roles})
        if unknown:
            problems.append(
                f"actor fields name role(s) {', '.join(unknown)}, but process "
                f"'{process.id}' declares {', '.join(roles)}"
            )
        named = [str(role) for role in names.values()]
        repeated = sorted({role for role in named if named.count(role) > 1})
        if repeated:
            # Both fields would receive the same actor and the other id would be
            # lost, so the declaration cannot mean what it says.
            problems.append(
                f"actor fields name role(s) {', '.join(repeated)} more than once; each "
                "field must take a different role, or the other actor's id is lost"
            )
        return problems
    if len(roles) > 1:
        return [
            f"process '{process.id}' pairs actors ({', '.join(roles)}), so each actor "
            "field must name the role it takes rather than being a bare list"
        ]
    return []


def _validate_engine_fields(openness: OpennessSpec, source: Path) -> list[dict[str, str]]:
    """Refuse an actor or phase field the engine could not write as declared."""
    catalog = _schema_catalog(source)
    kinds = {
        "actor_fields": ("an actor id", {"string"}),
        "phase_fields": ("a round", {"integer", "number"}),
    }
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        for output in process.outputs:
            schema = catalog.get(output.schema_ref)
            properties = schema.get("properties") if isinstance(schema, dict) else None
            closed = isinstance(schema, dict) and schema.get("additionalProperties") is False
            for kind, (what, types) in kinds.items():
                names = getattr(output, kind) or []
                if not names:
                    continue
                where = f"openness.processes.{process.id}.outputs.{output.artifact_type}.{kind}"
                problems = []
                if kind == "actor_fields" and process.actors is None:
                    problems.append(
                        f"process '{process.id}' declares actor fields but no actors, "
                        "so there is no actor id to write"
                    )
                if kind == "actor_fields":
                    problems.extend(_actor_role_problems(process, names))
                fields: dict[str, Any] = properties if isinstance(properties, dict) else {}
                # A closed schema with no properties block admits no field at all,
                # so it is checked too; skipping it let the run fail mid-way.
                for name in names if (fields or closed) else ():
                    declared = fields.get(name)
                    declared_type = declared.get("type") if isinstance(declared, dict) else None
                    # A type may be a list (["string", "null"]); the field can hold
                    # the engine's value when any listed type does.
                    if isinstance(declared_type, list):
                        admitted = bool({str(item) for item in declared_type} & types)
                    else:
                        admitted = declared_type is None or declared_type in types
                    if declared is None and closed:
                        problems.append(f"schema '{output.schema_ref}' has no field '{name}'")
                    elif not admitted:
                        problems.append(
                            f"'{name}' is typed {declared_type!r} in schema "
                            f"'{output.schema_ref}', but {what} is {' or '.join(sorted(types))}"
                        )
                errors.extend(
                    {
                        "code": "OUTPUT_ENGINE_FIELD_INVALID",
                        "severity": "error",
                        "path": where,
                        "message": problem,
                    }
                    for problem in problems
                )
    return errors


# The envelope _load_empirical_data wraps an imported asset in.
EMPIRICAL_ENVELOPE = frozenset({"origin", "data_source", "rows"})


def _validate_actor_sources(
    domain: DomainSpec, openness: OpennessSpec, warnings: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    """Refuse an actor selector whose source names no declared state.

    ``expand_actor_instances`` resolves ``actors.source`` as a dotted path into
    run state, so a source naming nothing raises once the run has started and
    the build has been paid for. The usual mistake is naming the collection
    rather than the state that holds it -- ``creators`` where the state is
    ``population`` and the path is ``population.creators``.
    """
    warnings = [] if warnings is None else warnings
    states = {str(state.id) for state in domain.states}
    # A state's declared initial names the keys it starts with. A process may
    # add one before the draw, so this cannot refuse -- but naming a collection
    # that is not there is the mistake this check exists for, and saying so at
    # compile time beats aborting before round one.
    declared_keys = {
        str(state.id): set(state.initial)
        for state in domain.states
        if isinstance(state.initial, Mapping) and state.initial
    }
    seeded = str(getattr(domain.initialization, "state_field", "") or "")
    if seeded:
        states.add(seeded)
    errors: list[dict[str, str]] = []
    for process in openness.processes:
        actors = process.actors
        per = getattr(actors, "per", None) if actors is not None else None
        per_source = getattr(per, "source", None) if per is not None else None
        if isinstance(per_source, str) and per_source:
            # The segment after the root is the outer actor's own id, bound at
            # expansion; only the root has to be a state declared up front.
            per_root = per_source.partition(".")[0]
            if per_root not in states:
                known = ", ".join(sorted(states)) or "none"
                errors.append(
                    {
                        "code": "ACTOR_SOURCE_UNKNOWN",
                        "severity": "error",
                        "dependency_section": "domain",
                        "path": f"openness.processes.{process.id}.actors.per.source",
                        "message": (
                            f"per source '{per_source}' starts from '{per_root}', which is not "
                            f"a declared state (declared: {known}); name the state that holds "
                            "each actor's own records"
                        ),
                    }
                )
        source = getattr(actors, "source", None) if actors is not None else None
        if not isinstance(source, str) or not source:
            continue
        root, _, rest = source.partition(".")
        if root in states:
            # An empirically seeded state holds a fixed envelope -- origin,
            # data_source, rows -- so a deeper path into it is decidable even
            # though a deeper path into ordinary state is not. Naming the
            # collection directly ('population.creators') resolves to nothing
            # and aborts the run before its first round.
            first = rest.split(".", 1)[0] if rest else ""
            if rest and root != seeded and first and first not in declared_keys.get(root, {first}):
                warnings.append(
                    {
                        "code": "ACTOR_SOURCE_UNDECLARED_KEY",
                        "severity": "warning",
                        "dependency_section": "domain",
                        "path": f"openness.processes.{process.id}.actors.source",
                        "message": (
                            f"actor source '{source}' reads '{first}' from state '{root}', "
                            f"whose declared initial holds "
                            f"{', '.join(sorted(declared_keys[root]))}; unless a process writes "
                            "it first, the run aborts before its first round"
                        ),
                    }
                )
            if rest and root == seeded and rest.split(".", 1)[0] not in EMPIRICAL_ENVELOPE:
                errors.append(
                    {
                        "code": "ACTOR_SOURCE_UNKNOWN",
                        "severity": "error",
                        "dependency_section": "domain",
                        "path": f"openness.processes.{process.id}.actors.source",
                        "message": (
                            f"actor source '{source}' reads '{rest}' from the empirically "
                            f"seeded state '{root}', which holds only "
                            f"{', '.join(sorted(EMPIRICAL_ENVELOPE))}; the records are under "
                            f"'{root}.rows'"
                        ),
                    }
                )
            continue
        known = ", ".join(sorted(states)) or "none"
        errors.append(
            {
                "code": "ACTOR_SOURCE_UNKNOWN",
                "severity": "error",
                "dependency_section": "domain",
                "path": f"openness.processes.{process.id}.actors.source",
                "message": (
                    f"actor source '{source}' starts from '{root}', which is not a declared "
                    f"state (declared: {known}); name the state that holds the records, "
                    "as a dotted path if they are nested"
                ),
            }
        )
    return errors


def _advise_inert_deterministic_processes(openness: OpennessSpec) -> list[dict[str, Any]]:
    """Flag a deterministic process that declares work its executor cannot do.

    The mode's executor returns ``{}``. A package may still intend to supply the
    callable as a run-time override, so this is advice rather than a refusal --
    but without one, every declared output goes unproduced and every declared
    effect writes from nothing, silently.
    """
    advisories: list[dict[str, Any]] = []
    for process in openness.processes:
        if process.executor.mode != "deterministic":
            continue
        declared = []
        if process.outputs:
            declared.append(f"{len(process.outputs)} output(s)")
        if process.state_effects:
            declared.append(f"{len(process.state_effects)} state effect(s)")
        if not declared:
            continue
        advisories.append(
            {
                "code": "EXECUTOR_INERT",
                "severity": "warning",
                "path": f"openness.processes.{process.id}.executor",
                "message": (
                    f"process '{process.id}' declares {' and '.join(declared)} but its "
                    "deterministic executor produces nothing; bind mode 'computational' "
                    "with an entry_point, or supply a run-time executor override"
                ),
            }
        )
    return advisories


def _advise_dead_context_policies(
    domain: DomainSpec, openness: OpennessSpec
) -> list[dict[str, Any]]:
    """Flag a declared context policy no process binds.

    A policy is only ever reached through a process's ``context_policy``, so an
    unbound one grants nothing to anybody. On its own that is merely dead, but
    it is the visible half of a real failure: the process the policy was written
    for is bound to some *other* policy, and is therefore seeing something the
    author never intended it to see. A detector written a title-and-body policy
    and left bound to the broad platform policy still compiles, still validates,
    and quietly reads the outcomes it is supposed to be measuring.
    """
    bound = {str(process.context_policy) for process in openness.processes}
    advisories: list[dict[str, Any]] = []
    for policy in domain.visibility:
        policy_id = str(getattr(policy, "id", ""))
        if not policy_id or policy_id in bound:
            continue
        advisories.append(
            {
                "code": "CONTEXT_POLICY_UNBOUND",
                "severity": "warning",
                "path": f"domain.visibility.{policy_id}",
                "message": (
                    f"context policy '{policy_id}' is bound by no process, so it grants "
                    "nothing; check whether a process meant to use it is bound to a "
                    "different policy, or remove it"
                ),
            }
        )
    return advisories


def _advise_unfed_feedback_slots(domain: DomainSpec, theory: TheorySpec) -> list[dict[str, Any]]:
    """Flag a ``feedback.<slot>`` allowance the theory never fills.

    The opposite direction is already an error: a declared feedback binding whose
    consumer policy does not allow its slot would inject into a view nobody can
    read. This direction is the leftover -- a slot allowed, and scoped, and
    capped, that no binding writes to, so it reads as a live channel and is not.
    """
    declared = {
        str(binding.execution.context_slot)
        for binding in theory.feedback
        if binding.execution is not None and binding.execution.context_slot
    }
    advisories: list[dict[str, Any]] = []
    for policy in domain.visibility:
        policy_id = str(getattr(policy, "id", ""))
        for path in policy.allow or ():
            name = str(path)
            if not name.startswith("feedback.") or name == "feedback":
                continue
            slot = name.split(".", 1)[1]
            if slot in declared:
                continue
            advisories.append(
                {
                    "code": "FEEDBACK_SLOT_UNFED",
                    "severity": "warning",
                    "path": f"domain.visibility.{policy_id}.allow/{name}",
                    "message": (
                        f"policy '{policy_id}' allows feedback slot '{slot}', but no theory "
                        "feedback declares that context_slot, so nothing is ever injected "
                        "there"
                    ),
                }
            )
    return advisories


def _validate_condition_factors(loaded: dict[str, Any]) -> list[dict[str, str]]:
    """Refuse a ``condition.<factor>`` path naming no declared factor.

    Such a predicate resolves to nothing and is therefore false in every round
    and every cell -- so the gate it guards never opens, silently. The study
    still runs and still reports, and the treatment simply never arrives: the
    result reads as "no effect" rather than as a broken package.
    """
    protocol = loaded["protocol"]
    # Derived from the same expansion the scheduler is given, never from a
    # belief about its shape. The first version of this check assumed a factor
    # sat at the top of the condition; it sits under ``factors`` whenever the
    # protocol declares factors or nests them, so the check refused the spelling
    # that fires and accepted the one that never does -- and a study whose
    # treatment never arrived still ran, still reported, and read as a null
    # result. Deriving both sides from one function is what stops that
    # recurring.
    resolvable: set[str] = set()
    try:
        expanded = expand_protocol_conditions(protocol.model_dump(mode="json"))
    except ValueError:
        # The protocol declares both factors and explicit conditions.
        # PROTOCOL_CONDITIONS_AMBIGUOUS reports that; guessing which spelling
        # such a package would resolve is not this check's business.
        return []
    for condition in expanded:
        for key, value in condition.items():
            if key == "factors" and isinstance(value, Mapping):
                resolvable.update(f"condition.factors.{name}" for name in value)
            else:
                resolvable.add(f"condition.{key}")
    errors: list[dict[str, str]] = []

    def _check(predicate: Any, where: str) -> None:
        for path in _predicate_paths(predicate):
            text = str(path)
            if not text.startswith("condition."):
                continue
            # A deeper read into a resolvable value is fine; what cannot be
            # resolved at all is the gate that never opens.
            if any(text == known or text.startswith(f"{known}.") for known in resolvable):
                continue
            known_paths = ", ".join(sorted(resolvable)) or "none"
            errors.append(
                {
                    "code": "CONDITION_FACTOR_UNKNOWN",
                    "severity": "error",
                    "dependency_section": "protocol",
                    "path": where,
                    "message": (
                        f"reads '{text}', which resolves in no condition this "
                        f"protocol declares, so the gate it guards never opens; "
                        f"readable here: {known_paths}"
                    ),
                }
            )

    for process in loaded["openness"].processes:
        trigger = process.trigger if isinstance(process.trigger, dict) else {}
        if trigger.get("type") == "condition":
            _check(trigger.get("predicate"), f"openness.processes.{process.id}.trigger")
        for index, use in enumerate(process.measurement_use):
            _check(use.when, f"openness.processes.{process.id}.measurement_use.{index}.when")
        # A rule executor evaluates its own `when` predicates, which read the
        # condition the same way a trigger does; unchecked, a dead gate inside
        # a rule fails exactly as silently as one on the process itself.
        parameters = getattr(process.executor, "parameters", None) or {}
        for index, rule in enumerate(parameters.get("rules") or ()):
            if isinstance(rule, Mapping):
                _check(
                    rule.get("when"),
                    f"openness.processes.{process.id}.executor.parameters.rules.{index}.when",
                )
    domain = loaded["domain"]
    for policy in domain.visibility:
        for path, rule in (policy.available_when or {}).items():
            _check(rule, f"domain.visibility.{policy.id}.available_when.{path}")
    for entry in domain.availability:
        _check(entry.available_when, f"domain.availability.{entry.path}.available_when")
    return errors


def _validate_context_exchanges(domain: DomainSpec, openness: OpennessSpec) -> list[dict[str, str]]:
    """Refuse an exchanges path that names no process, or names one unusably.

    An allowed path that resolves to nothing is dropped in silence at run time,
    so a typo would hand the actor no history for the whole run while the
    package reads as if it had one.
    """
    process_ids = {str(process.id) for process in openness.processes}
    errors: list[dict[str, str]] = []
    for policy_id, path in _policy_allows(domain):
        parts = path.split(".")
        if parts[0] != "exchanges":
            continue
        where = f"domain.visibility.{policy_id}.allow/{path}"
        if len(parts) != 2 or not parts[1]:
            errors.append(
                {
                    "code": "CONTEXT_EXCHANGES_INVALID",
                    "severity": "error",
                    "path": where,
                    "message": "an exchanges path names one process: exchanges.<process id>",
                }
            )
        elif parts[1] not in process_ids:
            errors.append(
                {
                    "code": "CONTEXT_EXCHANGES_INVALID",
                    "severity": "error",
                    "path": where,
                    "message": (
                        f"'{parts[1]}' is not a declared process, so this path would hand the "
                        "actor no history at all"
                    ),
                }
            )
    return errors


def _advise_unbounded_exchanges(domain: DomainSpec) -> list[dict[str, Any]]:
    """Flag an exchanges path with no cap: it grows with every round."""
    advisories: list[dict[str, Any]] = []
    for policy in domain.visibility:
        raw = policy.model_dump(mode="json") if hasattr(policy, "model_dump") else policy
        if not isinstance(raw, dict):
            continue
        cardinality = raw.get("cardinality") or {}
        for path in raw.get("allow") or ():
            name = str(path)
            if name.split(".")[0] != "exchanges" or name in cardinality:
                continue
            advisories.append(
                {
                    "code": "CONTEXT_UNBOUNDED",
                    "severity": "warning",
                    "path": f"domain.visibility.{raw.get('id', '')}.allow/{name}",
                    "message": (
                        f"'{name}' grows by one exchange every round it runs; declare a "
                        "cardinality cap, or record that an unbounded history is intended"
                    ),
                }
            )
    return advisories


def _selector_problem(selector: Any, state_ids: set[str]) -> str | None:
    """Why a scope selector cannot resolve to anything, or None if it can.

    Checked by name, not only by syntax. ``stait.follows`` is a well-formed
    path; it resolves to nothing at run time and would hand every actor an
    empty view without any error.
    """
    if not isinstance(selector, str) or not selector:
        return "selector paths must be non-empty strings"
    try:
        _resolve_path({}, selector.replace("${actor}", "actor"))
    except ValueError as exc:
        return f"selector '{selector}': {exc}"
    parts = selector.split(".")
    if parts[0] == "actor":
        if selector != "actor.ids":
            return f"selector '{selector}': the actor namespace offers only 'actor.ids'"
        return None
    if parts[0] == "state":
        named = parts[1] if len(parts) > 1 else ""
        if named not in state_ids:
            return f"selector '{selector}': '{named}' is not a declared state field"
        return None
    return f"selector '{selector}': must start with 'actor.ids' or 'state.<field>'"


def _advise_unbounded_context(domain: DomainSpec) -> list[dict[str, Any]]:
    """Flag collection fields a policy hands over whole (CTX-008).

    Not an error: an unbounded view can be exactly the design. But it is the one
    thing that makes context grow with population rather than with anything the
    study declares, so the choice should be explicit rather than a default.

    ``state.follows`` names the same field as ``follows``; the namespace is
    resolved rather than letting the namespaced spelling escape the advisory.
    """
    collections = {
        str(state.id)
        for state in domain.states
        if str(getattr(state, "value_type", "")) in {"array", "object"}
    }
    advisories: list[dict[str, Any]] = []
    for policy in domain.visibility:
        raw = policy.model_dump(mode="json") if hasattr(policy, "model_dump") else policy
        if not isinstance(raw, dict):
            continue
        policy_id = str(raw.get("id", ""))
        scope = raw.get("scope") or {}
        cardinality = raw.get("cardinality") or {}
        aggregate = raw.get("aggregate") or {}
        for path in raw.get("allow") or ():
            name = str(path)
            parts = name.split(".")
            field = parts[1] if parts[0] == "state" and len(parts) > 1 else parts[0]
            if field not in collections:
                continue
            if name in scope or name in cardinality or name in aggregate:
                continue
            advisories.append(
                {
                    "code": "CONTEXT_UNBOUNDED",
                    "severity": "warning",
                    "path": f"domain.visibility.{policy_id}.allow/{name}",
                    "message": (
                        f"policy '{policy_id}' hands over the whole of '{name}' to every actor; "
                        "declare a scope, a cardinality cap or an aggregate, or record that an "
                        "unbounded view is intended"
                    ),
                }
            )
    return advisories


def _advise_empirical_envelope(domain: DomainSpec, openness: OpennessSpec) -> list[dict[str, Any]]:
    """Warn that an empirically seeded field arrives wrapped, not bare.

    An ``empirical`` initialization stores the loaded asset as
    ``{origin, data_source, rows}``. ``value_type: object`` admits both that
    envelope and the bare table, so nothing tells an author that consumers must
    unwrap it. One process that read the envelope as if it were the table --
    iterating it to its own keys -- overwrote a study's follow graph with
    character lists in the first round, and it never recovered.
    """
    initialization = getattr(domain, "initialization", None)
    if initialization is None:
        return []
    raw = (
        initialization.model_dump(mode="json")
        if hasattr(initialization, "model_dump")
        else initialization
    )
    if not isinstance(raw, dict) or str(raw.get("mode", "")) != "empirical":
        return []
    field = str(raw.get("state_field", "population"))

    def effect_field(effect: Any) -> str:
        if isinstance(effect, Mapping):
            return str(effect.get("field", ""))
        return str(getattr(effect, "field", "") or "")

    writers = sorted(
        {
            str(process.id)
            for process in openness.processes
            for effect in (process.state_effects or ())
            if effect_field(effect) == field
        }
    )
    return [
        {
            "code": "EMPIRICAL_ENVELOPE",
            "severity": "warning",
            "path": f"domain.initialization/{field}",
            "message": (
                f"'{field}' is seeded empirically, so it holds "
                "{origin, data_source, rows} rather than the loaded table; every "
                "reader must unwrap it"
                + (f" (written by: {', '.join(writers)})" if writers else "")
            ),
        }
    ]


def _schema_catalog(source: Path) -> dict[str, Any]:
    """Parse every schema asset under ``schemas/`` into a JSON Schema catalog (AW-18)."""
    schema_dir = source / "schemas"
    if not schema_dir.is_dir():
        return {}
    catalog: dict[str, Any] = {}
    for path in sorted(schema_dir.glob("*")):
        if not path.is_file() or path.suffix not in {".yaml", ".yml", ".json"}:
            continue
        # The closure's exclusion rule applies here too: a credential-like file in
        # schemas/ must not be parsed into the build's schemas.json (H2).
        if _exclusion_reason(Path("schemas") / path.name) is not None:
            continue
        try:
            value = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"SCHEMA_INVALID: schema {path.name} is not valid YAML/JSON") from exc
        # F13: boolean schemas (true/false) are valid under the package dialect
        # and must be accepted here, matching the runtime catalog validator.
        if not isinstance(value, (dict, bool)):
            raise ValueError(
                f"SCHEMA_INVALID: schema {path.name} must be an object or boolean schema"
            )
        catalog[path.stem] = value
    return catalog


def _data_manifest(source: Path) -> dict[str, str]:
    """sha256 digests of every empirical data asset under ``data/`` (AW-06)."""
    data_dir = source / "data"
    if not data_dir.is_dir():
        return {}
    return {
        path.relative_to(data_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(data_dir.rglob("*"))
        if _exclusion_reason(Path("data") / path.relative_to(data_dir)) is None
        if path.is_file()
    }


class StudyCompiler:
    compiler_version = "1.0"

    def __init__(
        self,
        source: str | Path,
        *,
        theory_templates: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self.source = Path(source)
        self.theory_templates = {
            str(key): dict(value) for key, value in (theory_templates or {}).items()
        }

    def _load(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        errors: list[str] = []
        for name, model in CANONICAL.items():
            path = self.source / f"{name}.yaml"
            if not path.is_file():
                errors.append(f"MISSING_ARTIFACT: {path.name}")
                continue
            try:
                document = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError as exc:
                errors.append(f"SCHEMA_INVALID:{name}: {exc}")
                continue
            retired = RETIRED_KEYS.get(name, {})
            if isinstance(document, Mapping) and retired:
                found = sorted(set(document) & set(retired))
                for key in found:
                    errors.append(f"SPEC_KEY_RETIRED:{name}.{key}: {retired[key]}")
                if found:
                    # Validate without them, so the retirement is the only thing
                    # reported rather than being buried under "extra inputs are
                    # not permitted" for the same key.
                    document = {k: v for k, v in document.items() if k not in found}
            try:
                loaded[name] = model.model_validate(document)
            except ValidationError as exc:
                errors.append(f"SCHEMA_INVALID:{name}: {exc}")
        if errors:
            raise ValueError("\n".join(errors))
        study_ids = {item.study_id for item in loaded.values()}
        if len(study_ids) != 1:
            raise ValueError("STUDY_ID_MISMATCH: canonical artifacts must share study_id")
        return loaded

    @staticmethod
    def _ids(items: list[Any]) -> set[str]:
        return {item.id for item in items if hasattr(item, "id")}

    def _validate(self, loaded: dict[str, Any]) -> list[dict[str, str]]:
        openness = loaded["openness"]
        domain = loaded["domain"]
        policies = {"private", "public", "none"}
        for policy in domain.visibility:
            policy_id = (
                policy.get("id") if isinstance(policy, dict) else getattr(policy, "id", None)
            )
            if policy_id:
                policies.add(str(policy_id))
        errors: list[dict[str, str]] = []
        process_ids = self._ids(openness.processes)
        model_ids = self._ids(loaded["models"].models)
        artifact_ids = self._ids(domain.artifacts)
        seen: set[str] = set()
        catalogs = (
            ("models", loaded["models"].models),
            ("artifacts", domain.artifacts),
            ("attributes", domain.attributes),
            ("outcomes", loaded["outcomes"].outcomes),
        )
        for label, items in catalogs:
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                errors.append(
                    {
                        "code": "DUPLICATE_ID",
                        "path": f"/{label}",
                        "message": f"duplicate ID in {label}",
                    }
                )
        protocol = loaded["protocol"]
        if protocol.factors and protocol.conditions:
            # expand_protocol_conditions refuses this at run time, so a package
            # that compiled cleanly could only fail once a run was attempted.
            errors.append(
                {
                    "code": "PROTOCOL_CONDITIONS_AMBIGUOUS",
                    "path": "/protocol",
                    "message": (
                        "protocol declares both factors and explicit conditions; "
                        "declare one or the other"
                    ),
                }
            )
        graph: dict[str, set[str]] = {p: set() for p in process_ids}
        for process in openness.processes:
            if process.id in seen:
                errors.append(
                    {
                        "code": "DUPLICATE_ID",
                        "path": f"openness.processes.{process.id}",
                        "message": f"duplicate process id '{process.id}'",
                    }
                )
            seen.add(process.id)
            if process.context_policy not in policies:
                errors.append(
                    {
                        "code": "REF_CONTEXT_POLICY",
                        "dependency_section": "domain",
                        "path": f"openness.processes.{process.id}.context_policy",
                        "message": f"unknown context policy '{process.context_policy}'",
                    }
                )
            if process.executor.model_profile and process.executor.model_profile not in model_ids:
                errors.append(
                    {
                        "code": "REF_MODEL_PROFILE",
                        "dependency_section": "models",
                        "path": f"openness.processes.{process.id}.executor.model_profile",
                        "message": f"unknown model profile '{process.executor.model_profile}'",
                    }
                )
            if process.prompt_ref and not self._prompt_exists(process.prompt_ref):
                variants = self._prompt_variants(process.prompt_ref)
                message = (
                    f"prompt '{process.prompt_ref}' must be prompts/{process.prompt_ref}.txt: "
                    f"prompts are plain-text templates (found {', '.join(variants)})"
                    if variants
                    else f"prompt '{process.prompt_ref}' is not present in prompts/"
                )
                errors.append(
                    {
                        "code": "REF_PROMPT",
                        "path": f"openness.processes.{process.id}.prompt_ref",
                        "message": message,
                    }
                )
            for output in process.outputs:
                if output.schema_ref not in artifact_ids and not self._schema_exists(
                    output.schema_ref
                ):
                    errors.append(
                        {
                            "code": "REF_SCHEMA",
                            "dependency_section": "schemas",
                            "path": f"openness.processes.{process.id}.outputs",
                            "message": f"unknown output schema '{output.schema_ref}'",
                        }
                    )
            for input_ref in process.inputs:
                if input_ref not in artifact_ids and input_ref not in process_ids:
                    errors.append(
                        {
                            "code": "REF_INPUT",
                            "dependency_section": "domain",
                            "path": f"openness.processes.{process.id}.inputs",
                            "message": f"unknown input '{input_ref}'",
                        }
                    )
            dependencies = (
                process.dependencies.get("after", [])
                if isinstance(process.dependencies, dict)
                else []
            )
            if not isinstance(dependencies, list) or any(
                not isinstance(dep, str) for dep in dependencies
            ):
                errors.append(
                    {
                        "code": "DEPENDENCIES_INVALID",
                        "path": f"/processes/{process.id}/dependencies/after",
                        "message": "dependencies.after must be a list of process IDs",
                    }
                )
                dependencies = []
            delay = (
                process.dependencies.get("delay")
                if isinstance(process.dependencies, dict)
                else None
            )
            if delay is not None:

                def _positive_rounds(value: Any) -> bool:
                    return (
                        isinstance(value, int | float)
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value > 0
                    )

                rounds = delay.get("rounds") if isinstance(delay, dict) else None
                per_dependency = delay.get("per_dependency") if isinstance(delay, dict) else None
                has_per_dependency = isinstance(per_dependency, dict) and bool(per_dependency)
                # Either form is valid on its own: a process-wide ``rounds`` or a
                # ``per_dependency`` map giving individual edges their own lag.
                if not _positive_rounds(rounds) and not has_per_dependency:
                    errors.append(
                        {
                            "code": "DELAY_INVALID",
                            "path": f"openness.processes.{process.id}/dependencies/delay",
                            "message": "edge delay must be a positive number",
                        }
                    )
                if isinstance(per_dependency, dict):
                    for dep_id, value in per_dependency.items():
                        if str(dep_id) not in {str(item) for item in dependencies}:
                            errors.append(
                                {
                                    "code": "DELAY_INVALID",
                                    "path": (
                                        f"openness.processes.{process.id}"
                                        f"/dependencies/delay/per_dependency/{dep_id}"
                                    ),
                                    "message": (
                                        f"'{dep_id}' is not a declared dependency of '{process.id}'"
                                    ),
                                }
                            )
                        elif value != 0 and not _positive_rounds(value):
                            errors.append(
                                {
                                    "code": "DELAY_INVALID",
                                    "path": (
                                        f"openness.processes.{process.id}"
                                        f"/dependencies/delay/per_dependency/{dep_id}"
                                    ),
                                    "message": "edge delay must be zero or a positive number",
                                }
                            )
                elif per_dependency is not None:
                    errors.append(
                        {
                            "code": "DELAY_INVALID",
                            "path": (
                                f"openness.processes.{process.id}/dependencies/delay/per_dependency"
                            ),
                            "message": "per_dependency must map dependency ids to rounds",
                        }
                    )
            resolved_delays = edge_delays(process.dependencies, dependencies)
            for dep in dependencies:
                if dep not in process_ids:
                    errors.append(
                        {
                            "code": "REF_PROCESS",
                            "path": f"openness.processes.{process.id}",
                            "message": f"unknown dependency '{dep}'",
                        }
                    )
                else:
                    # A declared temporal delay breaks an immediate cycle, but
                    # only on the edge that actually carries it.
                    if not resolved_delays.get(str(dep)):
                        graph[process.id].add(dep)
            if process.executor.mode == "deterministic":
                # A deterministic executor is a no-op that returns {}; the engine
                # never resolves a callable for it. Declaring one reads as an
                # implementation and is decoration, so the process runs, produces
                # nothing, and the study reports on a mechanism that never fired.
                ignored = sorted(
                    key
                    for key in ("function", "entry_point")
                    if process.executor.parameters.get(key)
                )
                if ignored:
                    errors.append(
                        {
                            "code": "EXECUTOR_FUNCTION_IGNORED",
                            "path": f"openness.processes.{process.id}.executor.parameters",
                            "message": (
                                f"deterministic executor declares {', '.join(ignored)}, which "
                                "this mode ignores; use mode 'computational' with "
                                "parameters.entry_point in module:attribute form, or supply "
                                "the callable as a run-time executor override"
                            ),
                        }
                    )
            if process.executor.mode in {"stochastic", "computational"}:
                required_key = (
                    "function" if process.executor.mode == "stochastic" else "entry_point"
                )
                reference = str(process.executor.parameters.get(required_key, "") or "")
                if not reference or ":" not in reference:
                    errors.append(
                        {
                            "code": "EXECUTOR_UNAVAILABLE",
                            "path": f"openness.processes.{process.id}.executor.parameters",
                            "message": (
                                f"{process.executor.mode} executor requires parameters."
                                f"{required_key} in module:attribute form"
                            ),
                        }
                    )
        initialization = domain.initialization
        init_mode = (
            initialization.get("mode")
            if isinstance(initialization, dict)
            else getattr(initialization, "mode", "")
        )
        if str(init_mode or "") == "empirical":
            data_source = str(
                (
                    initialization.get("data_source")
                    if isinstance(initialization, dict)
                    else getattr(initialization, "data_source", None)
                )
                or ""
            )
            if not data_source:
                errors.append(
                    {
                        "code": "DATA_SOURCE_MISSING",
                        "path": "domain.initialization.data_source",
                        "message": "empirical initialization requires a data_source",
                    }
                )
            else:
                asset = self.source / data_source
                if not asset.resolve().is_relative_to(self.source.resolve()):
                    # Compiled clean before, yet the build embeds only data the
                    # package contains, so the run could never load it.
                    errors.append(
                        {
                            "code": "DATA_SOURCE_INVALID",
                            "path": "domain.initialization.data_source",
                            "message": (
                                f"data_source '{data_source}' is outside the package; the "
                                "build carries only data inside it, so put the asset under "
                                "data/"
                            ),
                        }
                    )
                elif not asset.is_file():
                    errors.append(
                        {
                            "code": "DATA_SOURCE_MISSING",
                            "path": "domain.initialization.data_source",
                            "message": (
                                "empirical initialization references missing data asset: "
                                f"{data_source}"
                            ),
                        }
                    )
                elif asset.suffix not in {".csv", ".parquet", ".json"}:
                    errors.append(
                        {
                            "code": "DATA_SOURCE_INVALID",
                            "path": "domain.initialization.data_source",
                            "message": f"unsupported empirical data format: {asset.suffix}",
                        }
                    )
        # Kahn's algorithm; delayed edges are explicitly temporal and do not form immediate cycles.
        pending = {key: set(value) for key, value in graph.items()}
        while pending:
            ready = {key for key, deps in pending.items() if not deps}
            if not ready:
                errors.append(
                    {
                        "code": "GRAPH_IMMEDIATE_CYCLE",
                        "path": "openness.processes",
                        "message": "immediate process dependency cycle",
                    }
                )
                break
            for key in ready:
                pending.pop(key)
            for deps in pending.values():
                deps.difference_update(ready)
        return errors

    @staticmethod
    def _dict_ids(items: list[Any]) -> set[str]:
        result: set[str] = set()
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                result.add(str(item["id"]))
            elif hasattr(item, "id") and item.id:
                result.add(str(item.id))
        return result

    def _validate_theory(
        self, loaded: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Cross-layer consistency checks of Section 5.6 (XL-001..XL-005)."""
        openness = loaded["openness"]
        domain = loaded["domain"]
        theory = loaded["theory"]
        process_ids = self._ids(openness.processes)
        artifact_ids = self._dict_ids(domain.artifacts)
        state_ids = self._dict_ids(domain.states)
        attribute_ids = self._ids(domain.attributes)
        domain_ids = artifact_ids | state_ids | attribute_ids
        function_ids = {mapping.theory_function for mapping in theory.process_mappings}
        mapped_processes = {mapping.process for mapping in theory.process_mappings}
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        # A state's declared value_type is the runtime's type contract. An
        # unrecognized name used to fall back to ``object``, which accepts any
        # value: the contract silently disappeared while compilation and the
        # integrity check both passed.
        declared_datasets = {str(dataset.id) for dataset in loaded["outcomes"].datasets}
        for trace in loaded["outcomes"].traces:
            seed = str(trace.seed.dataset)
            if seed not in declared_datasets:
                errors.append(
                    {
                        "code": "TRACE_SEED_UNKNOWN",
                        "path": f"outcomes.traces.{trace.id}/seed/dataset",
                        "message": (
                            f"trace '{trace.id}' seeds from dataset '{seed}', which is not "
                            "declared in outcomes.datasets"
                        ),
                    }
                )
            unknown_labels = sorted(set(trace.labels) - process_ids)
            if unknown_labels:
                errors.append(
                    {
                        "code": "TRACE_LABEL_UNKNOWN",
                        "path": f"outcomes.traces.{trace.id}/labels",
                        "message": f"labels name processes that do not exist: {unknown_labels}",
                    }
                )
        for state in domain.states:
            declared = str(getattr(state, "value_type", "object") or "object")
            if declared not in STATE_VALUE_TYPES:
                errors.append(
                    {
                        "code": "STATE_VALUE_TYPE",
                        "path": f"domain.states.{getattr(state, 'id', '')}/value_type",
                        "message": (
                            f"unknown value_type '{declared}'; expected one of "
                            f"{sorted(STATE_VALUE_TYPES)}"
                        ),
                    }
                )
        for process in openness.processes:
            if process.executor.mode != "generative":
                continue
            missing = []
            if not process.openness_rationale:
                missing.append("openness_rationale")
            if not process.closure_rationale:
                missing.append("closure_rationale")
            if not process.context_policy:
                missing.append("context_policy")
            if not process.outputs:
                missing.append("outputs")
            if missing:
                errors.append(
                    {
                        "code": "OPENNESS_INCOMPLETE",
                        "path": f"openness.processes.{process.id}",
                        "message": f"generative process is missing: {', '.join(missing)}",
                    }
                )
        for mapping in theory.process_mappings:
            if not mapping.theory_function:
                errors.append(
                    {
                        "code": "THEORY_FUNCTION_UNMAPPED",
                        "path": f"theory.process_mappings.{mapping.process}",
                        "message": "theory function must be non-empty",
                    }
                )
            if mapping.process not in process_ids:
                errors.append(
                    {
                        "code": "THEORY_FUNCTION_UNMAPPED",
                        "path": f"theory.process_mappings.{mapping.process}",
                        "message": (
                            f"theoretical function '{mapping.theory_function}' maps to "
                            f"unknown process '{mapping.process}'"
                        ),
                    }
                )
        for mechanism in domain.mechanisms:
            mechanism_id = getattr(mechanism, "id", None) or (
                mechanism.get("id") if isinstance(mechanism, dict) else None
            )
            if not mechanism_id:
                errors.append(
                    {
                        "code": "MECHANISM_UNBOUND",
                        "path": "domain.mechanisms",
                        "message": "each mechanism requires a stable id",
                    }
                )
                continue
            implements = getattr(mechanism, "implements", None)
            if implements is None and isinstance(mechanism, dict):
                implements = mechanism.get("implements")
            if implements is None:
                continue
            if implements not in function_ids and implements not in process_ids:
                errors.append(
                    {
                        "code": "MECHANISM_UNBOUND",
                        "path": f"domain.mechanisms.{mechanism_id}",
                        "message": (
                            f"mechanism '{mechanism_id}' implements undeclared function "
                            f"or process '{implements}'"
                        ),
                    }
                )
        for feedback in theory.feedback:
            if feedback.target not in process_ids:
                errors.append(
                    {
                        "code": "FEEDBACK_CONTEXT_MISSING",
                        "path": f"theory.feedback.{feedback.source}",
                        "dependency_section": "openness",
                        "message": (
                            f"feedback target '{feedback.target}' is not a declared process"
                        ),
                    }
                )
            if feedback.source not in domain_ids:
                errors.append(
                    {
                        "code": "FEEDBACK_CONTEXT_MISSING",
                        "path": f"theory.feedback.{feedback.source}",
                        "dependency_section": "domain",
                        "message": (
                            f"feedback source '{feedback.source}' is not declared in the "
                            "domain (states, artifacts, or attributes)"
                        ),
                    }
                )
        template = self.theory_templates.get(theory.theory_family)
        if template is not None:
            missing_functions = sorted(set(template["functions"]) - function_ids)
            if missing_functions:
                errors.append(
                    {
                        "code": "THEORY_FUNCTION_MISSING",
                        "path": "theory.yaml",
                        "message": (
                            f"template '{theory.theory_family}' declares mandatory "
                            f"functions that are not mapped: {', '.join(missing_functions)}"
                        ),
                    }
                )
        for process in openness.processes:
            if process.executor.mode != "generative":
                continue
            for output in process.outputs:
                if not self._schema_exists(output.schema_ref):
                    errors.append(
                        {
                            "code": "OUTPUT_SCHEMA_MISSING",
                            "dependency_section": "schemas",
                            "path": f"openness.processes.{process.id}.outputs",
                            "message": (
                                f"generative output schema '{output.schema_ref}' has no "
                                "usable schema file; structured-output validation cannot "
                                "apply to this process"
                            ),
                        }
                    )
        for process in openness.processes:
            if process.executor.mode == "generative" and process.id not in mapped_processes:
                warnings.append(
                    {
                        "code": "OPENNESS_UNJUSTIFIED",
                        "severity": "warning",
                        "path": f"openness.processes.{process.id}",
                        "message": (
                            f"generative process '{process.id}' is not referenced by any "
                            "theory process mapping; justify its openness in the theory layer"
                        ),
                    }
                )
        errors.extend(_validate_context_scope(domain))
        errors.extend(_validate_availability_rules(domain))
        errors.extend(_validate_context_exchanges(domain, openness))
        # A trigger reads state fields bare, beside "condition" and "protocol".
        # A path written "state.<field>" resolves to nothing there, so the
        # trigger was silently false in every round.
        for process in openness.processes:
            trigger = process.trigger if isinstance(process.trigger, dict) else {}
            if trigger.get("type") != "condition":
                continue
            for path in _predicate_paths(trigger.get("predicate")):
                if path.split(".")[0] == "state":
                    errors.append(
                        {
                            "code": "TRIGGER_PATH_INVALID",
                            "severity": "error",
                            "path": f"openness.processes.{process.id}.trigger",
                            "message": (
                                f"trigger reads '{path}', but a trigger reads state fields "
                                f"directly: name '{path.split('.', 1)[1]}'"
                            ),
                        }
                    )
        errors.extend(_validate_condition_factors(loaded))
        errors.extend(_validate_input_producers(domain, openness, warnings))
        errors.extend(_validate_actor_sources(domain, openness, warnings))
        errors.extend(_validate_triggers(openness))
        errors.extend(_validate_retention(openness))
        errors.extend(_validate_retry_policies(openness))
        errors.extend(_validate_outcomes(loaded["outcomes"], domain))
        errors.extend(_validate_run_length(loaded["protocol"], warnings))
        errors.extend(_validate_engine_fields(openness, self.source))
        warnings.extend(_advise_rounds_without_repeat(openness))
        errors.extend(_validate_dependencies(openness))
        warnings.extend(_advise_inert_deterministic_processes(openness))
        warnings.extend(_advise_unwritten_states(domain, openness))
        warnings.extend(_advise_dead_context_policies(domain, openness))
        warnings.extend(_advise_unfed_feedback_slots(domain, loaded["theory"]))
        warnings.extend(_advise_unbounded_exchanges(domain))
        warnings.extend(_advise_unbounded_context(domain))
        warnings.extend(_advise_empirical_envelope(domain, openness))
        timing_policies: dict[str, Mapping[str, Any]] = {
            name: {"allow": []} for name in ("private", "public", "none")
        }
        timing_policies.update(
            {str(policy.get("id")): policy for policy in _resolve_context_policies(domain)}
        )
        timing_errors, timing_warnings = timing_diagnostics(
            [process.model_dump(mode="json") for process in openness.processes], timing_policies
        )
        errors.extend(timing_errors)
        warnings.extend(timing_warnings)
        # A malformed role template would otherwise raise at the first model
        # call, after the run has started and that call has been paid for.
        for prompt_path in sorted((self.source / "prompts").glob("*.txt")):
            if _exclusion_reason(Path("prompts") / prompt_path.name) is not None:
                continue
            try:
                split_prompt_roles(prompt_path.read_text())
            except ValueError as exc:
                errors.append(
                    {
                        "code": "PROMPT_TEMPLATE_INVALID",
                        "severity": "error",
                        "path": f"prompts/{prompt_path.name}",
                        "message": str(exc),
                    }
                )
        errors.extend(_validate_prompt_context(self.source, openness, domain))
        errors.extend(_validate_context_allow(domain))
        # Feedback bindings are merged into processes after validation, so the
        # state a slot carries is passed in rather than read from the process.
        feedback_reads: dict[str, set[str]] = {}
        for binding in getattr(loaded["theory"], "feedback", None) or []:
            execution = getattr(binding, "execution", None)
            source = getattr(execution, "source", None) if execution is not None else None
            consumer = (
                getattr(execution, "consumer_process", None) if execution is not None else None
            )
            if isinstance(source, dict) and str(source.get("kind")) == "state" and source.get("id"):
                target = str(consumer or getattr(binding, "target", ""))
                feedback_reads.setdefault(target, set()).add(str(source["id"]))
        measurement_errors, measurement_warnings = measurement_diagnostics(
            [process.model_dump(mode="json") for process in openness.processes],
            timing_policies,
            feedback_reads,
        )
        errors.extend(measurement_errors)
        warnings.extend(measurement_warnings)
        state_types = {str(state.id): str(state.value_type) for state in domain.states}
        for process in openness.processes:
            for problem in model_effect_problems(process.model_dump(mode="json"), state_types):
                errors.append(
                    {
                        "code": "STATE_EFFECT_INVALID",
                        "severity": "error",
                        "path": f"openness.processes.{process.id}.state_effects",
                        "message": f"process '{process.id}': {problem}",
                    }
                )
        return errors, warnings

    def _compile_theory_execution(self, loaded: dict[str, Any]) -> Any:
        """Compile explicit theory execution bindings into a versioned plan.

        The plan is a derived, inspectable artifact: it records precedence
        edges, feedback-context bindings, verified mechanism bindings and
        annotation-only declarations, plus coverage and validation issues.
        """
        openness = loaded["openness"]
        domain = loaded["domain"]
        theory = loaded["theory"]
        process_ids = self._ids(openness.processes)
        mechanism_ids = self._ids(domain.mechanisms) if hasattr(domain, "mechanisms") else set()
        existing_dependencies: dict[str, list[str]] = {}
        for process in openness.processes:
            dependencies = process.dependencies
            after = (
                dependencies.get("after", []) if isinstance(dependencies, dict) else dependencies
            )
            if isinstance(after, list):
                existing_dependencies[str(process.id)] = [str(dep) for dep in after]
        theory_dict = theory.model_dump(mode="json")
        return compile_theory_execution(
            theory_dict,
            known_processes=process_ids,
            known_mechanisms=mechanism_ids,
            existing_dependencies=existing_dependencies,
        )

    def _prompt_exists(self, prompt_ref: str) -> bool:
        # Prompts are plain-text templates, and only prompts/<ref>.txt is compiled
        # into prompt_templates.json. Accepting other suffixes here let a package
        # compile while its template silently went missing (H1).
        path = self.source / "prompts" / f"{prompt_ref}.txt"
        return path.is_file() and _exclusion_reason(Path("prompts") / path.name) is None

    def _prompt_variants(self, prompt_ref: str) -> list[str]:
        """Prompt files for ``prompt_ref`` that are not the compiled ``.txt`` form."""
        prompt_dir = self.source / "prompts"
        return [
            f"prompts/{prompt_ref}{suffix}"
            for suffix in ("", ".yaml", ".yml", ".json")
            if (prompt_dir / f"{prompt_ref}{suffix}").is_file()
        ]

    def _schema_exists(self, schema_ref: str) -> bool:
        schema_dir = self.source / "schemas"
        suffixes = ("", ".yaml", ".yml", ".json")
        return any((schema_dir / f"{schema_ref}{suffix}").is_file() for suffix in suffixes)

    def compile(self, output: str | Path) -> StudyBuild:
        loaded = self._load()
        errors = self._validate(loaded)
        theory_errors, theory_warnings = self._validate_theory(loaded)
        errors.extend(theory_errors)
        # Compile-time theory execution plan (G2/THY-001): researcher-approved
        # execution bindings compile into schedule edges, context bindings and
        # verified mechanisms; unsupported or unresolved bindings fail here.
        theory_plan = self._compile_theory_execution(loaded)
        for issue in theory_plan.issues:
            if issue.code != "THEORY_ANNOTATION_WITHOUT_REASON":
                errors.append(
                    {
                        "code": issue.code,
                        "severity": "error",
                        "path": f"/theory/{issue.declaration_type}/{issue.declaration_id or ''}",
                        "message": issue.message,
                    }
                )
        # Executable state-feedback bindings must be visible to the consumer:
        # its context policy must expose every declared feedback slot under the
        # ``feedback`` namespace, otherwise the injected value would silently
        # never reach the process (XL-004/THY-006).
        if theory_plan.feedback_bindings:
            resolved_policies = _resolve_context_policies(loaded["domain"])
            policies_by_id = {str(policy.get("id", "")): policy for policy in resolved_policies}
            for (
                declaration_id,
                _source_id,
                consumer,
                slot,
                _lag,
                _initial,
            ) in theory_plan.feedback_bindings:
                consumer_process = next(
                    (p for p in loaded["openness"].processes if str(p.id) == str(consumer)),
                    None,
                )
                policy_id = (
                    str(consumer_process.context_policy) if consumer_process is not None else ""
                )
                policy = policies_by_id.get(policy_id, {})
                allow = policy.get("allow", ()) if isinstance(policy, Mapping) else ()
                if f"feedback.{slot}" not in allow and "feedback" not in allow:
                    errors.append(
                        {
                            "code": "THEORY_FEEDBACK_POLICY_MISSING",
                            "severity": "error",
                            "path": f"/theory/feedback/{declaration_id}",
                            "message": (
                                f"feedback binding '{declaration_id}' injects slot "
                                f"'{slot}' into consumer '{consumer}' but its context "
                                f"policy '{policy_id}' does not declare "
                                f"'feedback.{slot}' in allow; add it so the injected "
                                "prior-round state is actually visible to the process"
                            ),
                        }
                    )
        # Compile-time schema checks: every declared schema must be a valid
        # Draft 2020-12 schema in the package dialect (SCH-001/004).
        try:
            PackageSchemaCatalog(_schema_catalog(self.source))
        except SchemaValidationError as exc:
            errors.append(
                {
                    "code": exc.code,
                    "severity": "error",
                    "path": f"/schemas/{exc.schema_id}",
                    "message": str(exc),
                }
            )
        if errors:
            raise ValidationIssue(
                [
                    ValidationRecord(
                        e["code"],
                        e.get("severity", "error"),
                        # Hard-coding openness.yaml blamed one layer for every
                        # compile error, including errors in the others.
                        _source_file_for(e.get("path", "/")),
                        e.get("path", "/"),
                        tuple(e.get("related_ids", ())),
                        e["message"],
                        "Update the referenced canonical artifact or declaration.",
                    )
                    for e in errors
                ]
            )
        canonical = {name: model.model_dump(mode="json") for name, model in loaded.items()}
        # The build source hash covers every execution-relevant input: canonical
        # models (including extension namespaces), normalized prompts, parsed
        # schemas, and empirical data digests (review finding 1).
        source_hash = hashlib.sha256(
            json.dumps(
                {
                    **canonical,
                    # Filtered as the build's templates and closure are: hashing a
                    # prompt file the build excludes made build identity depend on
                    # content the build does not carry.
                    "prompts": {
                        path.stem: path.read_text()
                        for path in sorted((self.source / "prompts").glob("*.txt"))
                        if _exclusion_reason(Path("prompts") / path.name) is None
                    },
                    "schemas": _schema_catalog(self.source),
                    "data_manifest": _data_manifest(self.source),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        build_hash = hashlib.sha256(f"{self.compiler_version}:{source_hash}".encode()).hexdigest()
        target = Path(output)
        if target.exists():
            raise ValueError("BUILD_EXISTS: refusing unsafe overwrite")
        # Each entry reports the delay of THAT edge, not the process's delay
        # block copied onto every edge, so the inspectable graph matches what
        # the scheduler enforces.
        theory_functions = {
            mapping.process: mapping.theory_function
            for mapping in loaded["theory"].process_mappings
        }
        # THY-004: theory edges apply to the *compiled process definitions*
        # (processes.json), which is the schedule the runtime Scheduler reads.
        # Zero-lag edges become ordinary dependencies; positive-lag edges use
        # the scheduler's native per-process delay (dependencies.delay) so the
        # consumer does not run until ``lag`` rounds after the producer.
        theory_after: dict[str, list[str]] = {}
        # Lag belongs to the individual precedence relation, so it is recorded
        # per producer edge. Collapsing it into one process-level number made
        # two relations into the same consumer share the larger lag and imposed
        # that lag on the consumer's unrelated dependencies.
        theory_lag: dict[str, dict[str, int]] = {}
        for producer, consumer, lag in theory_plan.precedence_edges:
            theory_after.setdefault(str(consumer), []).append(str(producer))
            if lag > 0:
                edges = theory_lag.setdefault(str(consumer), {})
                edges[str(producer)] = max(edges.get(str(producer), 0), int(lag))
        # Executable state-feedback bindings ride on the consumer process so
        # the runtime can inject prior-round state into the declared context
        # slot (source validated against the domain in _validate_theory).
        theory_feedback: dict[str, list[dict[str, Any]]] = {}
        for (
            declaration_id,
            source_id,
            consumer,
            slot,
            lag,
            initial,
        ) in theory_plan.feedback_bindings:
            theory_feedback.setdefault(consumer, []).append(
                {
                    "declaration_id": declaration_id,
                    "source": source_id,
                    "context_slot": slot,
                    "lag_rounds": lag,
                    "initial": initial,
                }
            )
        compiled_processes = []
        for process in loaded["openness"].processes:
            record = process.model_dump(mode="json")
            if process.id in theory_functions:
                record["theory_function"] = theory_functions[process.id]
            theory_deps = theory_after.get(str(process.id), [])
            if theory_deps:
                added = [dep for dep in theory_deps if dep != str(process.id)]
                current_after = list(record.get("dependencies", {}).get("after", []))
                merged = sorted(set(current_after).union(added))
                record.setdefault("dependencies", {})
                record["dependencies"]["after"] = merged
                lags = theory_lag.get(str(process.id), {})
                if lags:
                    existing_delay = record["dependencies"].get("delay") or {}
                    if not isinstance(existing_delay, dict):
                        existing_delay = {}
                    per_dependency = dict(existing_delay.get("per_dependency") or {})
                    for producer, lag in lags.items():
                        per_dependency[producer] = max(
                            int(per_dependency.get(producer, 0) or 0), int(lag)
                        )
                    merged_delay: dict[str, Any] = {"per_dependency": per_dependency}
                    rounds = existing_delay.get("rounds")
                    if isinstance(rounds, int | float) and not isinstance(rounds, bool):
                        merged_delay["rounds"] = rounds
                    record["dependencies"]["delay"] = merged_delay
            feedback = theory_feedback.get(str(process.id))
            if feedback:
                record.setdefault("theory_feedback", []).extend(feedback)
            compiled_processes.append(record)
        # The inspectable graph is derived from the FINAL compiled records —
        # the same dependencies and delays processes.json hands the scheduler.
        # Building it from the pre-merge declarations omitted every theory edge
        # that carries a lag, so the graph reported processes as independent
        # while the runtime enforced an ordering between them.
        process_graph: dict[str, list[dict[str, Any]]] = {}
        for record in compiled_processes:
            dependencies = record.get("dependencies") or {}
            after = sorted(str(dep) for dep in dependencies.get("after", []))
            resolved = edge_delays(dependencies, after)
            process_graph[str(record["id"])] = [
                {
                    "dependency": dep,
                    "delayed": bool(resolved.get(dep)),
                    "delay": {"rounds": resolved[dep]} if resolved.get(dep) else None,
                }
                for dep in after
            ]
        # The declared edges were checked for immediate cycles, and the theory
        # edges separately, but never the two merged: a declared a->b and a
        # zero-lag theory b->a compiled, and the scheduler refused the build only
        # when a run was created from it.
        pending = {
            pid: {
                edge["dependency"]
                for edge in edges
                if not edge["delayed"] and edge["dependency"] in process_graph
            }
            for pid, edges in process_graph.items()
        }
        while pending:
            ready = {pid for pid, deps in pending.items() if not deps}
            if not ready:
                cycle = sorted(pending)
                raise ValidationIssue(
                    [
                        ValidationRecord(
                            "GRAPH_IMMEDIATE_CYCLE",
                            "error",
                            "openness.yaml",
                            "openness.processes",
                            tuple(cycle),
                            "declared dependencies and zero-lag theory edges together form "
                            "an immediate cycle; these processes are in it or wait on it: "
                            f"{', '.join(cycle)}",
                            "Give one edge in the cycle a lag, or remove it.",
                        )
                    ]
                )
            for pid in ready:
                pending.pop(pid)
            for deps in pending.values():
                deps.difference_update(ready)
        files = {
            "processes.json": compiled_processes,
            "model_profiles.json": [
                profile.model_dump(mode="json") for profile in loaded["models"].models
            ],
            "prompt_templates.json": {
                path.stem: path.read_text()
                for path in sorted((self.source / "prompts").glob("*.txt"))
                if _exclusion_reason(Path("prompts") / path.name) is None
            },
            "process_graph.json": process_graph,
            "context_policies.json": _resolve_context_policies(loaded["domain"]),
            "state_model.json": [
                state.model_dump(mode="json") for state in loaded["domain"].states
            ],
            "artifact_catalog.json": [
                a.model_dump(mode="json") for a in loaded["domain"].artifacts
            ],
            "outcome_plan.json": {
                "datasets": [d.model_dump(mode="json") for d in loaded["outcomes"].datasets],
                "outcomes": [o.model_dump(mode="json") for o in loaded["outcomes"].outcomes],
                "traces": [t.model_dump(mode="json") for t in loaded["outcomes"].traces],
            },
            "protocol.json": loaded["protocol"].model_dump(mode="json"),
            "initialization.json": (
                loaded["domain"].initialization.model_dump(mode="json")
                if hasattr(loaded["domain"].initialization, "model_dump")
                else loaded["domain"].initialization
            ),
            "data_manifest.json": _data_manifest(self.source),
            "schemas.json": _schema_catalog(self.source),
            "validation_report.json": {
                "valid": True,
                "errors": [],
                "warnings": [
                    {
                        "code": item["code"],
                        "severity": "warning",
                        "path": item.get("path", "/"),
                        "message": item["message"],
                    }
                    for item in theory_warnings
                ],
            },
        }
        files["theory_execution_plan.json"] = theory_plan.to_dict()
        manifest = {
            "build_hash": build_hash,
            "compiler_version": self.compiler_version,
            "source_hash": source_hash,
            "study_id": loaded["study"].study_id,
            "canonical_artifacts": sorted(canonical),
        }
        # Pin the approved package into an immutable content-addressed closure
        # (spec §2.1): original bytes, per-member digests/sizes/media types.
        # Export and replay later read this closure, never the editable package.
        package_closure = build_package_closure(self.source)
        manifest["package_closure_digest"] = package_closure.digest
        if package_closure.excluded:
            manifest["package_files_excluded"] = [
                {"path": path, "reason": reason} for path, reason in package_closure.excluded
            ]
        files["package_closure.json"] = package_closure.manifest
        files["build_manifest.json"] = manifest
        # The temporary build exists only while it is written, and is removed on
        # any failure. Removing its children one level deep raised on the nested
        # closure and data directories, replacing the original error and leaking
        # the directory; a validation error raised after it was made leaked it too.
        temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            integrity = {}
            # Only data the closure admits is embedded, so a file excluded from the
            # closure (an .env, an unsupported type) never reaches the build either.
            for asset in package_closure.manifest["assets"]:
                asset_path = Path(asset["path"])
                if not asset_path.parts or asset_path.parts[0] != "data":
                    continue
                destination = temp / asset_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((self.source / asset_path).read_bytes())
                destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            # Preserve original package asset bytes inside the build's closure dir.
            closure_root = self.source
            for asset in package_closure.manifest["assets"]:
                source_asset = closure_root / Path(asset["path"])
                destination = temp / "closure" / Path(asset["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source_asset.read_bytes())
                destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            for filename, value in files.items():
                path = temp / filename
                path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
                path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                integrity[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            integrity["manifest_hash"] = integrity["build_manifest.json"]
            embedded_data = temp / "data"
            if embedded_data.is_dir():
                for asset in sorted(embedded_data.rglob("*")):
                    if asset.is_file():
                        integrity[f"data/{asset.relative_to(embedded_data).as_posix()}"] = (
                            hashlib.sha256(asset.read_bytes()).hexdigest()
                        )
            integrity_path = temp / "integrity_manifest.json"
            integrity_path.write_text(json.dumps(integrity, sort_keys=True, indent=2) + "\n")
            integrity_path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.replace(temp, target)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return StudyBuild(loaded["study"].study_id, build_hash, target, manifest)

    @staticmethod
    def _verify_closure(root: Path, authenticated: bool) -> None:
        """Authenticate the copied package bytes under ``closure/``.

        The integrity manifest covered only the build's own files, so a tampered
        ``closure/study.yaml`` verified clean and was exported as the study that
        ran. ``package_closure.json`` is itself in the manifest and records each
        asset's digest, so every closure member is checked against it, and a
        member it does not list is refused.
        """
        closure_dir = root / "closure"
        present = (
            {
                path.relative_to(closure_dir).as_posix()
                for path in closure_dir.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            if closure_dir.is_dir()
            else set()
        )
        if not authenticated:
            if present:
                raise ValueError("BUILD_INTEGRITY: closure present without package_closure.json")
            return
        closure = json.loads((root / "package_closure.json").read_text())
        listed: dict[str, str] = {
            str(asset["path"]): str(asset["digest"]) for asset in closure.get("assets", [])
        }
        if not present:
            return  # a build compiled without copying its closure carries none
        unexpected = sorted(present - set(listed))
        if unexpected:
            raise ValueError(f"BUILD_INTEGRITY: unexpected closure member {unexpected[0]}")
        for relative, digest in listed.items():
            member = closure_dir / relative
            if member.is_symlink() or not member.is_file():
                raise ValueError(f"BUILD_INTEGRITY: closure/{relative} missing")
            hasher = hashlib.sha256()
            with member.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != digest:
                raise ValueError(f"BUILD_INTEGRITY: closure/{relative}")

    @staticmethod
    def verify_build(path: str | Path) -> bool:
        root = Path(path)
        try:
            expected = json.loads((root / "integrity_manifest.json").read_text())
            required = {
                "build_manifest.json",
                "processes.json",
                "process_graph.json",
                "context_policies.json",
                "state_model.json",
                "artifact_catalog.json",
                "outcome_plan.json",
                "protocol.json",
                "validation_report.json",
            }
            optional = {
                "model_profiles.json",
                "prompt_templates.json",
                "initialization.json",
                "data_manifest.json",
                "schemas.json",
                "package_closure.json",
                "theory_execution_plan.json",
            }
            expected_keys = set(expected) - {"manifest_hash"}
            if not expected_keys <= (required | optional):
                unknown = expected_keys - (required | optional)
                closure_unknown = [
                    name
                    for name in unknown
                    if str(name).startswith("closure/") or str(name).startswith("data/")
                ]
                if len(closure_unknown) != len(unknown):
                    raise ValueError("BUILD_INTEGRITY: unexpected expected-file set")
            if not required <= expected_keys:
                raise ValueError("BUILD_INTEGRITY: incomplete expected-file set")
            for name, digest in expected.items():
                if name == "manifest_hash":
                    continue
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f"BUILD_INTEGRITY: {name}")
            if expected["manifest_hash"] != expected["build_manifest.json"]:
                raise ValueError("BUILD_INTEGRITY: manifest authentication failed")
            StudyCompiler._verify_closure(root, "package_closure.json" in expected)
            return True
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("BUILD_INTEGRITY: malformed or incomplete build") from exc


class StudyBuild:
    def __init__(self, study_id: str, build_hash: str, path: Path, manifest: dict[str, Any]):
        self.study_id, self.build_hash, self.path, self.manifest = (
            study_id,
            build_hash,
            path,
            manifest,
        )

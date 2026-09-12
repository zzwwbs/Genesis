"""Information timing: when same-batch results become visible (CON-005..CON-010).

A process that runs several invocations in one phase forms a *batch*. The process
is *batch-dependent* when something its own invocations produce in the batch can
reach a later sibling's context or scheduling. Only then does the order in which
siblings commit change what an actor knows, so only then must a study declare
``information_timing.mode``: timing is a research decision, and the system never
makes it silently.

The analysis reads the same declarations the runtime enforces. Executors see only
their authorized context (``_invocation_namespace``), and the state store refuses
undeclared writes, so declared reads and writes bound what can flow. Anything the
analysis cannot resolve counts as dependent.

A scope does not make a read private: a field scoped to ``actor.ids`` may name
the record's recipient rather than its writer (an inbox, a message addressed to
another actor), so a sibling can still write what an actor sees.

Channels a package does not declare -- a callable executor reading the raw event
history, artifacts an executor outputs without declaring them -- cannot be seen
statically. The runtime closes them instead: an undeclared batch this analysis
finds independent is prepared from its batch view, which for every declared
channel is identical to reading live.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

# Executors that build their prompt only from the authorized context and emit no
# events of their own.
MODEL_CALL_MODES = frozenset({"generative", "semantic-evaluator"})
# First path segments that name an invocation namespace rather than a state field.
CONTEXT_NAMESPACES = frozenset(
    {"state", "inputs", "condition", "actor", "events", "feedback", "exchanges"}
)
# Scheduler condition paths outside the state.
SCHEDULER_NAMESPACES = frozenset({"condition", "protocol"})
# State operations that apply to the latest value, so effects computed from one
# shared view still compose when committed in order. ``put`` and a keyed
# ``append`` compose because each actor writes under its own key.
COMPOSABLE_OPS = frozenset(
    {"append", "increment", "remove", "custom", "put", "add-relation", "remove-relation"}
)
# Operations a model call's declared state effect may name. A model returns only
# outputs; the runtime applies each declared effect to the committed state.
MODEL_EFFECT_OPS = frozenset(
    {"set", "append", "increment", "remove", "put", "add-relation", "remove-relation"}
)
# Effect keys a model call may declare: which output feeds the effect, and which
# state key it writes under.
MODEL_EFFECT_KEYS = frozenset({"field", "op", "from", "key"})
# Operations that write under a key, and so compose only when the key is the
# acting actor's own: any other key is one entry two siblings both write.
KEYED_OPS = frozenset({"put"})
WHOLE_STATE = "*"


def timing_of(process: Mapping[str, Any]) -> tuple[str | None, str]:
    """The declared ``(mode, order)``; mode is ``None`` when undeclared."""
    raw = process.get("information_timing")
    if not isinstance(raw, Mapping):
        return None, "listed"
    mode = raw.get("mode")
    return (str(mode) if mode else None), str(raw.get("order") or "listed")


def executor_mode(process: Mapping[str, Any]) -> str:
    binding = process.get("executor")
    if not isinstance(binding, Mapping):
        return "deterministic"
    return str(binding.get("mode", "deterministic"))


def is_batched(process: Mapping[str, Any]) -> bool:
    """Whether a process may run more than one invocation per phase."""
    actors = process.get("actors")
    if actors is None:
        return False
    if isinstance(actors, list | tuple):
        return len(actors) > 1
    if isinstance(actors, Mapping):
        if actors.get("fan_out", True) is False:
            return False
        ids = actors.get("ids")
        if isinstance(ids, list | tuple):
            return len(ids) > 1
        return True  # drawn from state: the size is unknown until the run
    return True


def written_fields(process: Mapping[str, Any]) -> set[str]:
    fields: set[str] = set()
    for effect in process.get("state_effects") or ():
        if isinstance(effect, str):
            fields.add(effect)
        elif isinstance(effect, Mapping) and effect.get("field"):
            fields.add(str(effect["field"]))
    return fields


def _state_field(path: str) -> str | None:
    """The state field a context path reads, ``WHOLE_STATE``, or ``None``."""
    parts = path.split(".")
    if parts[0] == "state":
        return parts[1] if len(parts) > 1 and parts[1] else WHOLE_STATE
    if parts[0] in CONTEXT_NAMESPACES:
        return None
    return parts[0]


def _predicate_paths(condition: Any) -> Iterator[str]:
    if not isinstance(condition, Mapping):
        return
    for key in ("all", "any"):
        children = condition.get(key)
        if isinstance(children, list | tuple):
            for child in children:
                yield from _predicate_paths(child)
    if "not" in condition:
        yield from _predicate_paths(condition["not"])
    if "path" in condition:
        yield str(condition["path"])


def _availability_conditions(policy: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every availability block a policy applies, flat or per path."""
    when = policy.get("available_when")
    if not isinstance(when, Mapping) or not when:
        return []
    return [when, *(value for value in when.values() if isinstance(value, Mapping))]


def batch_dependencies(process: Mapping[str, Any], policy: Mapping[str, Any] | None) -> list[str]:
    """Why a later sibling could see an earlier sibling's same-batch result.

    Empty when it cannot, in which case sequential and simultaneous timing give
    identical results.
    """
    if not is_batched(process):
        return []
    policy = policy if isinstance(policy, Mapping) else {}
    writes = written_fields(process)
    emits_events = executor_mode(process) not in MODEL_CALL_MODES
    reasons: list[str] = []

    def written(field: str | None) -> list[str]:
        if field is None:
            return []
        if field == WHOLE_STATE:
            return sorted(writes)
        return [field] if field in writes else []

    scope = policy.get("scope")
    scope = scope if isinstance(scope, Mapping) else {}

    allowed = [str(path) for path in policy.get("allow") or ()]
    for path in allowed:
        for field in written(_state_field(path)):
            reasons.append(f"its context can include '{path}', and its own actors write '{field}'")
        if emits_events and path.split(".")[0] == "events":
            reasons.append(f"its context can include '{path}', and its own actors emit events")
        # Exchanges are appended when a turn commits, and each actor group acts
        # once per batch, so today a sibling's exchange cannot reach another. A
        # process reading its OWN exchanges depends on that, so it must say what
        # it assumes rather than inherit it silently.
        if path.split(".")[0] == "exchanges" and path.split(".")[-1] == str(process.get("id")):
            reasons.append(
                f"its context can include '{path}', which its own actors write as they commit"
            )

    if scope:
        for path, rule in scope.items():
            selectors = rule.get("in") if isinstance(rule, Mapping) else None
            if isinstance(selectors, str):
                selectors = [selectors]
            for selector in selectors if isinstance(selectors, list | tuple) else ():
                parts = str(selector).split(".")
                if parts[0] == "state" and len(parts) > 1:
                    for field in written(parts[1]):
                        reasons.append(
                            f"its scope for '{path}' is selected by '{selector}', and its own "
                            f"actors write '{field}'"
                        )

    for conditions in _availability_conditions(policy):
        uses_events = bool(conditions.get("event")) or bool(conditions.get("source_match_event"))
        uses_events = uses_events or any(
            path.split(".")[0] == "events" for path in _predicate_paths(conditions.get("predicate"))
        )
        if uses_events and emits_events:
            reasons.append("its context availability depends on events its own actors emit")

    produced = {str(process.get("id"))} | {
        str(output.get("artifact_type"))
        for output in process.get("outputs") or ()
        if isinstance(output, Mapping) and output.get("artifact_type")
    }
    for reference in process.get("inputs") or ():
        if str(reference) in produced:
            reasons.append(f"its inputs include '{reference}', which its own actors produce")

    trigger = process.get("trigger")
    if isinstance(trigger, Mapping):
        if trigger.get("type") == "condition":
            for path in _predicate_paths(trigger.get("predicate")):
                head = path.split(".")[0]
                if head in SCHEDULER_NAMESPACES:
                    continue
                for field in written(head):
                    reasons.append(
                        f"its trigger condition reads '{path}', and its own actors write '{field}'"
                    )
        if trigger.get("type") == "event" and emits_events:
            reasons.append("it is triggered by events, and its own actors emit events")

    return list(dict.fromkeys(reasons))


def can_ready_others(
    process: Mapping[str, Any], processes: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Whether a process's own commits could make another process ready mid-batch.

    Such a process would run between two of the batch's actors, so the batch
    cannot run all its calls at once. A model call writes only its declared
    fields; any other executor may also emit events or scheduling effects, so it
    is assumed able to (CON-011).
    """
    if executor_mode(process) not in MODEL_CALL_MODES:
        return True
    writes = written_fields(process)
    process_id = str(process.get("id"))
    for other_id, other in processes.items():
        if str(other_id) == process_id or not isinstance(other, Mapping):
            continue
        trigger = other.get("trigger")
        if isinstance(trigger, Mapping) and trigger.get("type") == "condition":
            heads = {path.split(".")[0] for path in _predicate_paths(trigger.get("predicate"))}
            if heads & writes:
                return True
    return False


def declared_ops(process: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Each declared write as ``(field, op)``; a bare field name means ``set``."""
    found: list[tuple[str, str]] = []
    for effect in process.get("state_effects") or ():
        if isinstance(effect, str):
            found.append((effect, "set"))
        elif isinstance(effect, Mapping) and effect.get("field"):
            found.append((str(effect["field"]), str(effect.get("op", "set"))))
    return found


def _acts_per_actor(process: Mapping[str, Any]) -> bool:
    """Whether every invocation of this process acts for exactly one actor."""
    actors = process.get("actors")
    if actors is None:
        return False
    if isinstance(actors, list | tuple):
        return bool(actors)
    if isinstance(actors, Mapping):
        return actors.get("fan_out", True) is not False
    return False


def model_effect_problems(
    process: Mapping[str, Any], states: Mapping[str, str] | None = None
) -> list[str]:
    """Why a process's declared state effects cannot be applied as declared.

    A model call's effects are applied by the runtime from its outputs, so each
    must name a supported operation, an output to read, and -- for keyed
    operations -- the acting actor as its key. Other executors compute their own
    writes, so ``from`` and ``key`` would be silently ignored there.
    """
    problems: list[str] = []
    model_call = executor_mode(process) in MODEL_CALL_MODES
    declared_outputs = [
        str(output.get("artifact_type"))
        for output in process.get("outputs") or ()
        if isinstance(output, Mapping) and output.get("artifact_type")
    ]
    outputs = set(declared_outputs)
    # What the executor actually hands back: a model's response is stored under
    # the first declared output, and under "response" as well.
    emitted = {declared_outputs[0], "response"} if declared_outputs else {"response"}
    for effect in process.get("state_effects") or ():
        if isinstance(effect, str):
            # The bare form names a field and nothing else, so the runtime reads
            # the output of that name and sets the field. Unchecked, a name that
            # is not an output wrote nothing, every round, in silence.
            if model_call and effect not in outputs:
                problems.append(
                    f"effect on '{effect}' reads output '{effect}', but the process declares "
                    f"outputs {sorted(outputs) or 'none'}"
                )
            if model_call and states and effect not in states:
                problems.append(f"effect writes '{effect}', which the domain does not declare")
            continue
        if not isinstance(effect, Mapping):
            continue
        field = str(effect.get("field", ""))
        if not model_call:
            ignored = sorted({"from", "key"} & set(effect))
            if ignored:
                problems.append(
                    f"effect on '{field}' declares {', '.join(ignored)}, which only a model "
                    "call's effects use; this executor computes its own writes"
                )
            continue
        unknown = sorted(set(effect) - MODEL_EFFECT_KEYS)
        if unknown:
            problems.append(f"effect on '{field}' has unknown keys: {', '.join(unknown)}")
        op = str(effect.get("op", "set"))
        if op not in MODEL_EFFECT_OPS:
            problems.append(
                f"effect on '{field}' uses op '{op}'; a model call supports "
                f"{', '.join(sorted(MODEL_EFFECT_OPS))}"
            )
        source = str(effect.get("from", field))
        if source.split(".")[0] not in outputs:
            problems.append(
                f"effect on '{field}' reads output '{source}', but the process declares "
                f"outputs {sorted(outputs) or 'none'}"
            )
        elif source.split(".")[0] not in emitted:
            # A model call returns one value, stored under the first declared
            # output. An effect reading any other declared output compiled and
            # then wrote nothing, every round, in silence.
            problems.append(
                f"effect on '{field}' reads output '{source}', but a model call returns only "
                f"'{sorted(emitted)[0]}', its first declared output"
            )
        key = effect.get("key")
        if op == "put" and key is None:
            problems.append(f"effect on '{field}' uses put, which requires key: actor")
        if key is not None and not _acts_per_actor(process):
            # Caught here rather than at the first commit, which is after the
            # run has started and the first call has been paid for.
            problems.append(
                f"effect on '{field}' writes under the acting actor, so the process must run "
                "one invocation per actor; declare actors that fan out"
            )
        if key is not None and key != "actor":
            problems.append(f"effect on '{field}' has key '{key}'; only 'actor' is supported")
        if key is not None and op not in {"put", "append"}:
            problems.append(f"effect on '{field}' declares a key, which only put and append use")
        # A package whose domain declares no states yet is incomplete, not wrong:
        # the guided route approves the openness layer before the domain layer,
        # so judging a field against an empty domain made it impossible to
        # declare any state effect there. The check still applies once the
        # domain exists, which is the case at compile.
        if states:
            value_type = states.get(field)
            if value_type is None:
                problems.append(f"effect writes '{field}', which the domain does not declare")
            elif key is not None and value_type not in {"object", "json"}:
                problems.append(
                    f"effect on '{field}' writes under a key, so '{field}' must be an object, "
                    f"not {value_type}"
                )
            elif (
                key is None
                and op in {"append", "add-relation", "remove-relation"}
                and (value_type != "array")
            ):
                problems.append(
                    f"effect on '{field}' uses {op}, so '{field}' must be an array, "
                    f"not {value_type}"
                )
    return problems


def whole_field_writes(process: Mapping[str, Any]) -> list[str]:
    """Declared writes that replace a field whole rather than composing."""
    if executor_mode(process) in MODEL_CALL_MODES:
        # A model call's declared effects are applied with their operations, so
        # only those that set a field replace it whole.
        whole = sorted({field for field, op in declared_ops(process) if op not in COMPOSABLE_OPS})
        if not whole:
            return []
        fields = ", ".join(f"'{field}'" for field in whole)
        return [f"its model call, whose effects set {fields} whole"]
    if executor_mode(process) == "state-transition":
        return ["its state-transition executor, which writes whole fields"]
    found: list[str] = []
    for effect in process.get("state_effects") or ():
        if isinstance(effect, str):
            found.append(f"'{effect}' (no operation declared, so it is written whole)")
        elif isinstance(effect, Mapping):
            op = str(effect.get("op", "set"))
            if op not in COMPOSABLE_OPS:
                found.append(f"'{effect.get('field')}' (op '{op}')")
            elif op in KEYED_OPS and effect.get("key") != "actor":
                # A keyed write composes only under the acting actor's own key.
                # Counting it composable here let such a process compile and
                # then be refused at its first commit, after the run had begun.
                found.append(
                    f"'{effect.get('field')}' (op '{op}' under a key that is not the acting actor)"
                )
    return found


def timing_diagnostics(
    processes: Iterable[Mapping[str, Any]], policies: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Compiler errors and advisories for information timing (CON-007, CON-009)."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for process in processes:
        process_id = str(process.get("id"))
        path = f"openness.processes.{process_id}.information_timing"
        mode, _order = timing_of(process)
        policy = policies.get(str(process.get("context_policy", "private")))
        reasons = batch_dependencies(process, policy)
        if reasons and mode is None:
            errors.append(
                {
                    "code": "INFORMATION_TIMING_REQUIRED",
                    "severity": "error",
                    "path": path,
                    "message": (
                        f"process '{process_id}' must declare information_timing.mode "
                        "(sequential or simultaneous), because a later actor could see an "
                        "earlier actor's result from the same batch: " + "; ".join(reasons)
                    ),
                }
            )
        elif mode is not None and not reasons:
            warnings.append(
                {
                    "code": "INFORMATION_TIMING_UNUSED",
                    "severity": "warning",
                    "path": path,
                    "message": (
                        f"process '{process_id}' declares information_timing.mode '{mode}', "
                        "but no actor can see a sibling's result from the same batch, so the "
                        "declaration does not change results"
                    ),
                }
            )
        if mode == "simultaneous" and is_batched(process):
            conflicts = whole_field_writes(process)
            if conflicts:
                errors.append(
                    {
                        "code": "SIMULTANEOUS_WRITE_CONFLICT",
                        "severity": "error",
                        "path": f"openness.processes.{process_id}.state_effects",
                        "message": (
                            f"process '{process_id}' is simultaneous, so every actor computes "
                            "its writes from the same view; these would overwrite earlier "
                            f"siblings' writes: {', '.join(conflicts)}. Use an operation that "
                            "composes (append, increment, remove, custom)"
                        ),
                    }
                )
    return errors, warnings

"""Measurement isolation: what a measurement observes must not steer behaviour.

A process declared ``measurement: true`` produces a research observable. If its
artifacts or state writes reach a process that is not itself a measurement, the
observable has become part of the mechanism it measures, and results can no
longer say what the measure alone would have shown. That is sometimes the
design -- a detector's score is both the outcome measure and, under governance,
the input to a sanction -- so the use must be declared on the consuming process
(``measurement_use``), with a rationale and, where the design applies it only in
some conditions or rounds, a ``when`` predicate the runtime enforces.

What is checked: a consumer's declared inputs; its context policy's allowed
paths, scope selectors and availability predicates; its trigger predicate; the
state its actor selector is drawn from; the state a declared feedback slot
carries; and a policy admitting the measurement's own exchanges.

What is not: channels no declaration describes. An executor that emits events or
scheduling effects, or that commits an artifact its process never declared as an
output, can still steer behaviour -- which is why a measurement declaring no
outputs at all is reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from genesis.information_timing import WHOLE_STATE, _state_field, written_fields


def produced_refs(process: Mapping[str, Any]) -> set[str]:
    """The references a process's artifacts resolve under: its id and output types."""
    refs = {str(process.get("id"))}
    for output in process.get("outputs") or ():
        if isinstance(output, Mapping) and output.get("artifact_type"):
            refs.add(str(output["artifact_type"]))
    return refs


def measurement_diagnostics(
    processes: Iterable[Mapping[str, Any]],
    policies: Mapping[str, Mapping[str, Any]],
    feedback: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Compiler errors and advisories for measurement isolation.

    ``feedback`` maps a consumer process to the state fields a declared theory
    feedback slot injects into its context. Those bindings are merged after this
    runs, so they are passed in rather than read from the process.
    """
    from genesis.runtime import _validate_condition

    listed = [process for process in processes if isinstance(process, Mapping)]
    measurements = {str(p.get("id")): p for p in listed if p.get("measurement") is True}
    feedback_reads = {
        str(key): {str(item) for item in value} for key, value in (feedback or {}).items()
    }
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for process_id, measurement in measurements.items():
        if not (measurement.get("outputs") or ()):
            warnings.append(
                _issue(
                    "MEASUREMENT_OUTPUTS_UNDECLARED",
                    f"openness.processes.{process_id}.outputs",
                    f"measurement '{process_id}' declares no outputs, so an artifact it commits "
                    "cannot be recognised as its own and a process reading that artifact will "
                    "not be reported",
                    severity="warning",
                )
            )
    for process in listed:
        process_id = str(process.get("id"))
        path = f"openness.processes.{process_id}.measurement_use"
        uses = {
            str(use.get("source")): use
            for use in process.get("measurement_use") or ()
            if isinstance(use, Mapping)
        }
        for source, use in uses.items():
            if source not in measurements:
                errors.append(
                    _issue(
                        "MEASUREMENT_USE_INVALID",
                        path,
                        f"process '{process_id}' declares a use of '{source}', which is not a "
                        "measurement process",
                    )
                )
            if use.get("when") is not None:
                try:
                    _validate_condition(use["when"])
                except ValueError as exc:
                    errors.append(
                        _issue(
                            "MEASUREMENT_USE_INVALID",
                            path,
                            f"process '{process_id}' use of '{source}': {exc}",
                        )
                    )
        if process.get("measurement") is True:
            continue
        inputs = {str(ref) for ref in process.get("inputs") or ()}
        policy = policies.get(str(process.get("context_policy", "private"))) or {}
        allowed = [str(item) for item in policy.get("allow") or ()]
        # A scope selector and an availability predicate read state just as an
        # allowed path does: a feed scoped by the detector's scores is steered
        # by the measurement even though the policy never admits that field.
        allowed += _policy_state_reads(policy)
        # A trigger predicate and an actor selector read state as surely as a
        # context path: one decides whether the process runs at all, the other
        # who acts.
        allowed += _process_state_reads(process)
        carried = feedback_reads.get(process_id, set())
        exchanges_of = {
            str(item).split(".")[1]
            for item in allowed
            if str(item).startswith("exchanges.") and len(str(item).split(".")) > 1
        }
        for source, measurement in measurements.items():
            artifact_reads = sorted(inputs & produced_refs(measurement))
            if source in exchanges_of:
                # An exchange carries what the measurement was given and what it
                # answered, which is its output by another route.
                artifact_reads.append(f"exchanges.{source}")
            writes = written_fields(measurement)
            state_reads = sorted(
                {
                    field
                    for item in allowed
                    for field in (
                        writes if _state_field(item) == WHOLE_STATE else {_state_field(item)}
                    )
                    if field in writes
                }
                | (carried & writes)
            )
            if not artifact_reads and not state_reads:
                if source in uses:
                    warnings.append(
                        _issue(
                            "MEASUREMENT_USE_UNUSED",
                            path,
                            f"process '{process_id}' declares a use of '{source}' but reads "
                            "none of its artifacts or state",
                            severity="warning",
                        )
                    )
                continue
            reads = ", ".join(
                [f"input '{ref}'" for ref in artifact_reads]
                + [f"state '{field}'" for field in state_reads]
            )
            declared_use = uses.get(source)
            if declared_use is None:
                errors.append(
                    _issue(
                        "MEASUREMENT_LEAK",
                        f"openness.processes.{process_id}",
                        f"process '{process_id}' reads measurement '{source}' ({reads}); a "
                        "measurement must not steer behaviour unless the use is declared in "
                        "measurement_use with a rationale",
                    )
                )
            elif declared_use.get("when") is not None and state_reads:
                errors.append(
                    _issue(
                        "MEASUREMENT_USE_INVALID",
                        path,
                        f"process '{process_id}' limits its use of '{source}' with 'when', but "
                        f"reads its state ({', '.join(state_reads)}), which cannot be withheld "
                        "per condition; read it as an input artifact instead",
                    )
                )
    return errors, warnings


def gated_input_refs(
    process: Mapping[str, Any],
    processes: Mapping[str, Mapping[str, Any]],
    *,
    phase: Any,
    condition: Mapping[str, Any],
) -> list[Any]:
    """A process's declared inputs, less measurement sources whose use does not apply now."""
    from genesis.runtime import _evaluate_condition

    refs = process.get("inputs") or []
    if not isinstance(refs, list | tuple):
        return list(refs) if isinstance(refs, Iterable) else []
    gated = [
        use
        for use in process.get("measurement_use") or ()
        if isinstance(use, Mapping) and use.get("when") is not None
    ]
    if not gated:
        return list(refs)
    namespace = {"condition": condition_namespace(condition), "protocol": {"phase": phase}}
    consumer_id = str(process.get("id"))
    withheld: set[str] = set()
    for use in gated:
        if _evaluate_condition(use["when"], namespace):
            continue
        source_id = str(use.get("source"))
        source = processes.get(source_id, {"id": source_id})
        # Only what this measurement alone produces: another process may declare
        # the same artifact type, and withholding that would starve the consumer
        # of a producer the declaration says nothing about. The consumer's own
        # outputs are not such a producer -- counting them let a consumer that
        # declares the same artifact type defeat its own gate.
        others: set[str] = set()
        for other_id, other in processes.items():
            if str(other_id) not in {source_id, consumer_id} and isinstance(other, Mapping):
                others |= produced_refs(other)
        withheld |= produced_refs(source) - others
    return [ref for ref in refs if str(ref) not in withheld]


def condition_namespace(condition: Mapping[str, Any]) -> dict[str, Any]:
    """A run condition as a ``when`` predicate reads it.

    A factor-based protocol resolves a condition to ``{id, factors}``, so a
    predicate written the obvious way -- ``condition.governance`` -- matched
    nothing and the gate stayed shut in exactly the condition it named. The
    factors are exposed alongside the condition's own keys.
    """
    from genesis.runtime import _plain

    plain = _plain(condition)
    if not isinstance(plain, dict):
        return {}
    factors = plain.get("factors")
    return {**(factors if isinstance(factors, dict) else {}), **plain}


def _process_state_reads(process: Mapping[str, Any]) -> list[str]:
    """State paths a process reads outside its context: trigger and actor source."""
    from genesis.information_timing import _predicate_paths

    found: list[str] = []
    trigger = process.get("trigger")
    if isinstance(trigger, Mapping) and trigger.get("type") == "condition":
        found.extend(_predicate_paths(trigger.get("predicate")))
    actors = process.get("actors")
    if isinstance(actors, Mapping) and actors.get("source"):
        found.append(str(actors["source"]))
    return found


def _policy_state_reads(policy: Mapping[str, Any]) -> list[str]:
    """State paths a policy reads outside ``allow``: scope selectors and availability."""
    from genesis.information_timing import _availability_conditions, _predicate_paths

    found: list[str] = []
    scope = policy.get("scope")
    if isinstance(scope, Mapping):
        for rule in scope.values():
            selectors = rule.get("in") if isinstance(rule, Mapping) else None
            if isinstance(selectors, str):
                selectors = [selectors]
            for selector in selectors if isinstance(selectors, list | tuple) else ():
                found.append(str(selector).replace("${actor}", "actor"))
    for conditions in _availability_conditions(policy):
        found.extend(_predicate_paths(conditions.get("predicate")))
    return found


def _issue(code: str, path: str, message: str, *, severity: str = "error") -> dict[str, str]:
    return {"code": code, "severity": severity, "path": path, "message": message}

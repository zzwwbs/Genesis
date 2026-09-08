"""Bounded theory-to-execution operationalization (G2).

Researchers approve explicit operational bindings; the compiler translates a
small supported set and reports coverage. Nothing in this module interprets
theory prose as executable mathematics: declarations without an explicit
``execution`` binding are classified as annotations and only recorded in the
coverage report (spec §6.1, §6.3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PLAN_VERSION = 1


@dataclass(frozen=True)
class TheoryExecutionIssue:
    code: str
    declaration_type: str
    declaration_id: str | None
    message: str


@dataclass
class TheoryExecutionPlan:
    """Versioned compiled theory execution plan and coverage report (THY-001)."""

    version: int = PLAN_VERSION
    # (producer_process, consumer_process, lag_rounds)
    precedence_edges: list[tuple[str, str, int]] = field(default_factory=list)
    # (declaration_id, source_id, consumer_process, context_slot, lag_rounds, initial)
    feedback_bindings: list[tuple[str, str, str, str, int, dict[str, Any]]] = field(
        default_factory=list
    )
    # (declaration_id, mechanism_id, applicable_process)
    mechanism_bindings: list[tuple[str, str, str]] = field(default_factory=list)
    annotations: list[tuple[str, str, str]] = field(default_factory=list)
    # Resolved operational declarations keyed by declaration id -> description.
    resolved: dict[str, str] = field(default_factory=dict)
    issues: list[TheoryExecutionIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "precedence_edges": [
                {"producer": a, "consumer": b, "lag_rounds": lag}
                for a, b, lag in self.precedence_edges
            ],
            "feedback_bindings": [
                {
                    "declaration_id": declaration_id,
                    "source": source_id,
                    "consumer_process": consumer,
                    "context_slot": context_slot,
                    "lag_rounds": lag,
                    "initial": initial,
                }
                for declaration_id, source_id, consumer, context_slot, lag, initial in (
                    self.feedback_bindings
                )
            ],
            "mechanism_bindings": [
                {"declaration_id": d, "mechanism": mechanism, "process": process}
                for d, mechanism, process in self.mechanism_bindings
            ],
            "annotations": [
                {"declaration_type": dtype, "declaration_id": did, "reason": reason}
                for dtype, did, reason in self.annotations
            ],
            "resolved": dict(self.resolved),
            "issues": [
                {"code": issue.code, "declaration_type": issue.declaration_type,
                 "declaration_id": issue.declaration_id, "message": issue.message}
                for issue in self.issues
            ],
        }

    @property
    def valid(self) -> bool:
        return not any(
            issue.code not in {"THEORY_ANNOTATION_WITHOUT_REASON"} for issue in self.issues
        )


def _declaration_id(kind: str, index: int, declared: str | None) -> str:
    if declared:
        return str(declared)
    return f"{kind}-{index}"


def compile_theory_execution(
    theory: Mapping[str, Any],
    *,
    known_processes: set[str],
    known_mechanisms: set[str],
    existing_dependencies: Mapping[str, list[str]] | None = None,
) -> TheoryExecutionPlan:
    """Compile a theory declaration into an executable plan + coverage report.

    ``known_processes``/``known_mechanisms`` are the compiled study's declared
    processes and mechanism/transition ids. ``existing_dependencies`` maps
    process id -> existing declared dependencies (from openness) so
    theory-generated edges are unified and conflicts rejected (THY-004).
    """
    plan = TheoryExecutionPlan()
    existing = {
        str(process): set(str(dep) for dep in deps)
        for process, deps in (existing_dependencies or {}).items()
    }

    def require_process(decl_type: str, decl_id: str, process: str | None) -> bool:
        if process is None or process not in known_processes:
            plan.issues.append(
                TheoryExecutionIssue(
                    "THEORY_PROCESS_UNKNOWN",
                    decl_type,
                    decl_id,
                    f"execution binding references unknown process '{process}'",
                )
            )
            return False
        return True

    # Relations, feedback and delays may each carry an execution binding.
    declarations: list[tuple[str, str, Mapping[str, Any]]] = []
    for index, relation in enumerate(theory.get("relations", []) or []):
        if isinstance(relation, Mapping):
            declarations.append(
                ("relation", _declaration_id("relation", index, relation.get("id")), relation)
            )
    for index, feedback in enumerate(theory.get("feedback", []) or []):
        if isinstance(feedback, Mapping):
            declarations.append(
                ("feedback", _declaration_id("feedback", index, feedback.get("id")), feedback)
            )
    for index, delay in enumerate(theory.get("delays", []) or []):
        if isinstance(delay, Mapping):
            declarations.append(
                ("delay", _declaration_id("delay", index, delay.get("id")), delay)
            )

    for decl_type, decl_id, declaration in declarations:
        binding = declaration.get("execution")
        if binding is None:
            plan.annotations.append(
                (decl_type, decl_id, "no explicit execution binding; treated as annotation")
            )
            continue
        kind = str(binding.get("kind", "")) if isinstance(binding, Mapping) else ""
        if kind == "annotation":
            reason = str(binding.get("reason") or "")
            if not reason:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_ANNOTATION_WITHOUT_REASON",
                        decl_type,
                        decl_id,
                        "annotation execution kind requires a reason",
                    )
                )
            plan.annotations.append((decl_type, decl_id, reason))
            continue
        if kind == "precedence":
            producer = binding.get("producer_process")
            consumer = binding.get("consumer_process")
            lag = binding.get("lag_rounds", 0)
            if not require_process(decl_type, decl_id, producer) or not require_process(
                decl_type, decl_id, consumer
            ):
                continue
            if not isinstance(lag, int) or lag < 0:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_LAG_INVALID",
                        decl_type,
                        decl_id,
                        "precedence lag_rounds must be a nonnegative integer",
                    )
                )
                continue
            if lag == 0:
                existing_for_consumer = existing.get(str(consumer), set())
                if str(producer) in existing_for_consumer:
                    plan.issues.append(
                        TheoryExecutionIssue(
                            "THEORY_DEPENDENCY_CONFLICT",
                            decl_type,
                            decl_id,
                            f"theory precedence duplicates an existing dependency "
                            f"'{producer}' -> '{consumer}'",
                        )
                    )
                    continue
            plan.precedence_edges.append((str(producer), str(consumer), int(lag)))
            plan.resolved[decl_id] = f"precedence {producer} -> {consumer} (lag {lag})"
            continue
        if kind == "feedback_context":
            consumer = binding.get("consumer_process")
            source = binding.get("source") or {}
            source_id = str(source.get("id") or "") if isinstance(source, Mapping) else ""
            context_slot = binding.get("context_slot")
            lag = binding.get("lag_rounds", 0)
            initial = binding.get("initial") or {}
            if not require_process(decl_type, decl_id, consumer):
                continue
            if not source_id or not context_slot:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_BINDING_INCOMPLETE",
                        decl_type,
                        decl_id,
                        "feedback_context requires source.id and context_slot",
                    )
                )
                continue
            if not isinstance(lag, int) or lag < 0:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_LAG_INVALID",
                        decl_type,
                        decl_id,
                        "feedback_context lag_rounds must be a nonnegative integer",
                    )
                )
                continue
            if lag > 0 and not initial:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_INITIAL_POLICY_REQUIRED",
                        decl_type,
                        decl_id,
                        "positive-lag feedback requires an explicit initial policy "
                        "for history that does not exist yet",
                    )
                )
                continue
            plan.feedback_bindings.append(
                (decl_id, source_id, str(consumer), str(context_slot), int(lag), dict(initial))
            )
            plan.resolved[decl_id] = (
                f"feedback_context {source_id} -> {consumer} slot "
                f"{context_slot} (lag {lag})"
            )
            continue
        if kind == "mechanism_binding":
            mechanism = binding.get("mechanism")
            process = binding.get("consumer_process") or declaration.get("target")
            mechanism_id = str(mechanism) if mechanism else ""
            if mechanism_id not in known_mechanisms:
                plan.issues.append(
                    TheoryExecutionIssue(
                        "THEORY_MECHANISM_UNKNOWN",
                        decl_type,
                        decl_id,
                        f"mechanism binding references unknown mechanism '{mechanism_id}'",
                    )
                )
                continue
            if process is not None and not require_process(decl_type, decl_id, str(process)):
                continue
            plan.mechanism_bindings.append(
                (decl_id, mechanism_id, str(process) if process else "")
            )
            plan.resolved[decl_id] = (
                f"mechanism_binding verified -> '{mechanism_id}' (no new transition)"
            )
            continue
        plan.issues.append(
            TheoryExecutionIssue(
                "THEORY_KIND_UNSUPPORTED",
                decl_type,
                decl_id,
                f"unsupported execution kind '{kind}'",
            )
        )
    # THY-003: reject zero-lag precedence/feedback dependency cycles.
    _reject_zero_lag_cycles(plan)
    return plan


def _reject_zero_lag_cycles(plan: TheoryExecutionPlan) -> None:
    """Reject zero-lag cycles; positive-lag edges cannot form same-round cycles."""
    graph: dict[str, list[str]] = {}
    for producer, consumer, lag in plan.precedence_edges:
        if lag == 0:
            graph.setdefault(str(consumer), []).append(str(producer))
    for declaration_id, _source, consumer, _slot, lag, _initial in plan.feedback_bindings:
        if lag == 0:
            graph.setdefault(str(consumer), []).append(declaration_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, path: list[str]) -> None:
        if node in visiting:
            plan.issues.append(
                TheoryExecutionIssue(
                    "THEORY_CYCLE_ZERO_LAG",
                    "execution",
                    None,
                    "zero-lag dependency cycle: " + " -> ".join([*path, node]),
                )
            )
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, ()):
            visit(str(dependency), [*path, node])
        visiting.discard(node)
        visited.add(node)

    for node in graph:
        visit(node, [])
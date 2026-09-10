"""Deterministic compilation of a canonical GENESIS study package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from .execution_manifest import build_package_closure
from .runtime import STATE_VALUE_TYPES, _cap_rule, _resolve_path, edge_delays
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

# Protocol budget keys the runtime actually enforces.
ENFORCED_BUDGETS = frozenset({"max_events"})

CANONICAL: dict[str, type[StrictModel]] = {
    "study": StudySpec,
    "openness": OpennessSpec,
    "theory": TheorySpec,
    "domain": DomainSpec,
    "protocol": ProtocolSpec,
    "outcomes": OutcomesSpec,
    "models": ModelsSpec,
}


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
                loaded[name] = model.model_validate(yaml.safe_load(path.read_text()) or {})
            except (ValidationError, yaml.YAMLError) as exc:
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
                errors.append(
                    {
                        "code": "REF_PROMPT",
                        "path": f"openness.processes.{process.id}.prompt_ref",
                        "message": f"prompt '{process.prompt_ref}' is not present in prompts/",
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
                if not asset.is_file():
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
        # ``budgets`` sits next to enforced limits, so an unrecognised key
        # reads as a constraint the runtime will apply. It will not: only the
        # keys below are enforced, and the rest are inert annotations.
        protocol_budgets = getattr(loaded["protocol"], "budgets", {}) or {}
        if isinstance(protocol_budgets, dict):
            for key in sorted(protocol_budgets):
                if str(key) not in ENFORCED_BUDGETS:
                    warnings.append(
                        {
                            "code": "BUDGET_NOT_ENFORCED",
                            "severity": "warning",
                            "path": f"protocol.budgets/{key}",
                            "message": (
                                f"budget '{key}' is recorded but not enforced by the runtime; "
                                f"enforced budgets are {sorted(ENFORCED_BUDGETS)}"
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
        warnings.extend(_advise_unbounded_context(domain))
        warnings.extend(_advise_empirical_envelope(domain, openness))
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
        prompt_dir = self.source / "prompts"
        suffixes = ("", ".yaml", ".yml", ".json", ".txt")
        return any((prompt_dir / f"{prompt_ref}{suffix}").is_file() for suffix in suffixes)

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
                        "openness.yaml",
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
                    "prompts": {
                        path.stem: path.read_text()
                        for path in sorted((self.source / "prompts").glob("*.txt"))
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
        temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
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
        files = {
            "processes.json": compiled_processes,
            "model_profiles.json": [
                profile.model_dump(mode="json") for profile in loaded["models"].models
            ],
            "prompt_templates.json": {
                path.stem: path.read_text()
                for path in sorted((self.source / "prompts").glob("*.txt"))
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
        files["package_closure.json"] = package_closure.manifest
        files["build_manifest.json"] = manifest
        integrity = {}
        data_dir = self.source / "data"
        if data_dir.is_dir():
            for asset in sorted(data_dir.rglob("*")):
                if not asset.is_file():
                    continue
                relative = asset.relative_to(data_dir)
                destination = temp / "data" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(asset.read_bytes())
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
        try:
            os.replace(temp, target)
        except Exception:
            for child in temp.iterdir():
                if child.is_dir():
                    for nested in child.rglob("*"):
                        if nested.is_file():
                            nested.unlink()
                child.unlink()
            temp.rmdir()
            raise
        return StudyBuild(loaded["study"].study_id, build_hash, target, manifest)

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

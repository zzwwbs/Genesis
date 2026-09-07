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


def _schema_catalog(source: Path) -> dict[str, dict[str, Any]]:
    """Parse every schema asset under ``schemas/`` into a JSON Schema catalog (AW-18)."""
    schema_dir = source / "schemas"
    if not schema_dir.is_dir():
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for path in sorted(schema_dir.glob("*")):
        if not path.is_file() or path.suffix not in {".yaml", ".yml", ".json"}:
            continue
        try:
            value = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"SCHEMA_INVALID: schema {path.name} is not valid YAML/JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"SCHEMA_INVALID: schema {path.name} must be an object")
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
                rounds = delay.get("rounds") if isinstance(delay, dict) else None
                invalid_delay = (
                    not isinstance(rounds, int | float)
                    or isinstance(rounds, bool)
                    or not math.isfinite(rounds)
                    or rounds <= 0
                )
                if invalid_delay:
                    errors.append(
                        {
                            "code": "DELAY_INVALID",
                            "path": f"openness.processes.{process.id}/dependencies/delay",
                            "message": "edge delay must be a positive number",
                        }
                    )
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
                    # A declared temporal delay breaks an immediate cycle.
                    if not process.dependencies.get("delay"):
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
        return errors, warnings

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
        process_graph = {
            p.id: [
                {
                    "dependency": dep,
                    "delayed": bool(p.dependencies.get("delay")),
                    "delay": p.dependencies.get("delay"),
                }
                for dep in sorted(p.dependencies.get("after", []))
            ]
            for p in loaded["openness"].processes
        }
        theory_functions = {
            mapping.process: mapping.theory_function
            for mapping in loaded["theory"].process_mappings
        }
        compiled_processes = []
        for process in loaded["openness"].processes:
            record = process.model_dump(mode="json")
            if process.id in theory_functions:
                record["theory_function"] = theory_functions[process.id]
            compiled_processes.append(record)
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
            "outcome_plan.json": [o.model_dump(mode="json") for o in loaded["outcomes"].outcomes],
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
        manifest = {
            "build_hash": build_hash,
            "compiler_version": self.compiler_version,
            "source_hash": source_hash,
            "study_id": loaded["study"].study_id,
            "canonical_artifacts": sorted(canonical),
        }
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
            }
            expected_keys = set(expected) - {"manifest_hash"}
            if not expected_keys <= (required | optional):
                unknown = expected_keys - (required | optional)
                if any(not str(name).startswith("data/") for name in unknown):
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

"""Data-driven interactive elicitation engine (IEL-001..IEL-034).

Workflow packages under ``workflows/`` declare stages, questions, instructions,
owned paths, checklist ownership, completion rules, invalidation rules, and
templates. The registry validates packages; the session engine owns the
in-memory state machine; the assistant contracts are strict Pydantic models.
Canonical YAML and package versions remain owned by the existing
specification subsystem.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from genesis.specification.models import StableId

WORKFLOW_COMPLETION_RULES = frozenset(
    {"required_checklist_complete", "no_consequential_ambiguity", "canonical_schema_valid"}
)
CANONICAL_SECTIONS = frozenset(
    {
        "study",
        "openness",
        "theory",
        "domain",
        "protocol",
        "outcomes",
        "models",
        "openness.processes",
        "theory.process_mappings",
        "domain.mechanisms",
        "domain.artifacts",
        "domain.states",
        "openness.inputs",
        "protocol.conditions",
        "outcomes.outcomes",
        "prompts",
        "schemas",
    }
)


def known_checklist_ids() -> set[str]:
    from genesis.checklists import CHECKLIST_ITEMS

    return {str(item["id"]) for item in CHECKLIST_ITEMS}


class CompletionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ClarificationPolicy(BaseModel):
    """Server-enforced question budget for one workflow stage."""

    model_config = ConfigDict(extra="forbid")
    max_turns: int = Field(default=5, ge=1, le=8)


class CriticalDecision(BaseModel):
    """One workflow-declared decision that can affect executable configuration."""

    model_config = ConfigDict(extra="forbid")
    id: StableId
    question: str
    rationale: str
    target_paths: tuple[str, ...]
    required: bool = True
    default_allowed: bool = False
    default_value: Any = None


class WorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StableId
    title: str
    opening_question: str
    instructions: str
    owned_paths: tuple[str, ...]
    clarification: ClarificationPolicy = Field(default_factory=ClarificationPolicy)
    critical_decisions: tuple[CriticalDecision, ...] = ()
    checklist_items: tuple[StableId, ...] = ()
    ambiguity_topics: tuple[StableId, ...] = ()
    completion_rules: tuple[CompletionRule, ...] = ()
    templates: dict[str, str] = Field(default_factory=dict)
    depends_on: tuple[StableId, ...] = ()
    approval_required: bool = True


class TheoryTemplate(BaseModel):
    """A file-defined set of theoretical functions and elicitation questions."""

    model_config = ConfigDict(extra="forbid")
    id: StableId
    functions: tuple[StableId, ...]
    questions: tuple[str, ...] = ()


class WorkflowDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StableId
    version: str
    title: str
    instructions: str = ""
    instructions_path: str | None = None
    session_persistence: Literal["memory", "disk"] = "disk"
    stages: tuple[WorkflowStage, ...]
    invalidation: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    theory_templates: dict[str, TheoryTemplate] = Field(default_factory=dict)

    def stage(self, stage_id: str) -> WorkflowStage:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise KeyError(stage_id)

    def next_stage(self, stage_id: str) -> WorkflowStage | None:
        for index, stage in enumerate(self.stages):
            if stage.id == stage_id and index + 1 < len(self.stages):
                return self.stages[index + 1]
        return None

    def instructions_text(self) -> str:
        if self.instructions:
            return self.instructions
        return ""

    def dependants_of(self, stage_id: str) -> tuple[str, ...]:
        """Return transitive dependants in declared workflow order."""
        self.stage(stage_id)
        dependent_ids = {stage_id}
        result: list[str] = []
        for stage in self.stages:
            if stage.id == stage_id:
                continue
            if any(dependency in dependent_ids for dependency in stage.depends_on):
                dependent_ids.add(stage.id)
                result.append(stage.id)
        return tuple(result)


def apply_operations(
    current: dict[str, Any], operations: tuple[Any, ...] | list[Any]
) -> dict[str, Any]:
    """Apply add/replace/remove operations over dict/list projections (IEL-021).

    Paths are slash segments; ``-`` appends to a list; integer segments index a
    list. Leaves are created on add/replace (upsert); removal requires the
    target to exist.
    """
    import copy as _copy

    result: dict[str, Any] = _copy.deepcopy(current)
    for operation in operations:
        op = str(operation.op)
        raw_path = str(operation.path)
        if not raw_path.startswith("/"):
            raise ValueError(f"PATCH_PATH_FORBIDDEN: path '{raw_path}' must start with '/'")
        segments = [segment for segment in raw_path.split("/") if segment]
        if not segments:
            raise ValueError(f"PATCH_PATH_FORBIDDEN: empty path '{raw_path}'")
        node: Any = result
        for index, segment in enumerate(segments):
            is_last = index == len(segments) - 1
            if isinstance(node, list):
                if segment == "-":
                    if not is_last:
                        raise ValueError(
                            f"PATCH_PATH_FORBIDDEN: '-' must be the last segment in '{raw_path}'"
                        )
                    if op == "remove":
                        if not node:
                            raise ValueError(f"PATCH_PATH_FORBIDDEN: list is empty at '{raw_path}'")
                        node.pop()
                    else:
                        node.append(operation.value)
                    break
                if not segment.isdigit():
                    raise ValueError(f"PATCH_PATH_FORBIDDEN: invalid list index '{segment}'")
                position = int(segment)
                if position >= len(node):
                    if op == "remove":
                        raise ValueError(f"PATCH_PATH_FORBIDDEN: no element at '{raw_path}'")
                    if not is_last:
                        raise ValueError(f"PATCH_PATH_FORBIDDEN: unknown target '{raw_path}'")
                    node.append(operation.value)
                    break
                if is_last:
                    if op == "remove":
                        del node[position]
                    else:
                        node[position] = operation.value
                    break
                node = node[position]
            else:
                if not isinstance(node, dict):
                    raise ValueError(f"PATCH_PATH_FORBIDDEN: cannot descend into '{raw_path}'")
                if is_last:
                    if op == "remove":
                        if segment not in node:
                            raise ValueError(f"PATCH_PATH_FORBIDDEN: no key '{segment}'")
                        del node[segment]
                    else:
                        node[segment] = operation.value
                    break
                if segment not in node or not isinstance(node[segment], (dict, list)):
                    if segment not in node and op == "add":
                        node[segment] = {}
                    else:
                        raise ValueError(f"PATCH_PATH_FORBIDDEN: unknown target '{raw_path}'")
                node = node[segment]
    return result


class PatchPreviewService:
    """Deterministic patch preview: applies, canonicalizes, inspects, diffs (IEL-024)."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def preview(
        self,
        *,
        patch: Any,
        stage: WorkflowStage,
        workflow: WorkflowDefinition,
        current_form: dict[str, Any],
        source_turn_ids: set[int],
        package_hash: str,
        live_directory: Path | None = None,
    ) -> dict[str, Any]:
        import difflib
        import tempfile as _tempfile

        patch.validate_turn_references(source_turn_ids)
        patch.validate_owned_paths(stage.owned_paths)
        current_canonical = self.service._canonical_specification(current_form)
        current_canonical["schemas"] = current_form.get("schemas", {})
        candidate: dict[str, Any] = apply_operations(current_canonical, patch.operations)
        self.service._validate_schema_files(candidate.get("schemas", {}))
        for section, original in current_canonical.items():
            if not isinstance(original, dict) or not isinstance(candidate.get(section), dict):
                continue
            if section == "schemas":
                continue
            known = set(original) | {"schema_version", "study_id"}
            sanitised = {key: value for key, value in candidate[section].items() if key in known}
            if "schema_version" in original and "schema_version" not in sanitised:
                sanitised["schema_version"] = original["schema_version"]
            if "study_id" in original:
                # Identity is derived from the specification, never the patch.
                sanitised["study_id"] = original["study_id"]
            candidate[section] = sanitised
        yaml_files = {
            f"{name}.yaml": yaml.safe_dump(value, sort_keys=False)
            for name, value in candidate.items()
        }
        with _tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Overlay the candidate on the live package so cross-layer checks and
            # checklist rules evaluate the FULL candidate-aware package.
            if live_directory is not None and live_directory.is_dir():
                import shutil as _shutil

                for path in live_directory.rglob("*"):
                    if not path.is_file() or path.name in {"metadata.json", "checklist.json"}:
                        continue
                    relative = path.relative_to(live_directory)
                    target = tmp_path / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _shutil.copy2(path, target)
            for name, text in yaml_files.items():
                (tmp_path / name).write_text(text)
            schema_dir = tmp_path / "schemas"
            schema_dir.mkdir(exist_ok=True)
            for existing in schema_dir.iterdir():
                if existing.is_file() and existing.suffix in {".json", ".yaml", ".yml"}:
                    if (
                        existing.stem not in candidate.get("schemas", {})
                        or existing.suffix != ".json"
                    ):
                        existing.unlink()
            for schema_id, content in candidate.get("schemas", {}).items():
                (schema_dir / f"{schema_id}.json").write_text(json.dumps(content))
            prompts = candidate.get("prompts")
            prompt_dir = tmp_path / "prompts"
            if isinstance(prompts, dict) and prompt_dir.exists():
                for existing in prompt_dir.glob("*.txt"):
                    if existing.stem not in prompts:
                        existing.unlink()
            if isinstance(prompts, dict) and prompts:
                prompt_dir.mkdir(exist_ok=True)
                for prompt_id, content in prompts.items():
                    (prompt_dir / f"{prompt_id}.txt").write_text(str(content))
            validation = self._inspect(tmp_path, stage=stage, workflow=workflow)
            from genesis.compiler import StudyCompiler as _Compiler
            from genesis.compiler import ValidationIssue

            # Run the FULL compiler validation (cross-layer checks such as
            # THEORY_FUNCTION_MISSING) over the candidate so errors surface in
            # the preview instead of only at approval.
            try:
                _Compiler(
                    tmp_path,
                    theory_templates=self.service._workflow_registry.theory_templates(),
                ).compile(tmp_path / "build")
                validation = dict(validation)
                validation["compile_errors"] = []
            except ValidationIssue as exc:
                validation = dict(validation)
                validation["compile_errors"] = [
                    {
                        "code": getattr(issue, "code", "COMPILE"),
                        "message": getattr(issue, "message", str(issue)),
                    }
                    for issue in exc.issues
                ]
            loaded = _Compiler(
                tmp_path,
                theory_templates=self.service._workflow_registry.theory_templates(),
            )._load()
            candidate_form = self.service._form_from_package(
                loaded, tmp_path, current_form.get("id", "")
            )
        diffs = {}
        for name in sorted(set(current_canonical) | set(candidate)):
            before = yaml.safe_dump(current_canonical.get(name, {}), sort_keys=False).splitlines()
            after = yaml.safe_dump(candidate.get(name, {}), sort_keys=False).splitlines()
            difference = list(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"{name}.yaml (current)",
                    tofile=f"{name}.yaml (candidate)",
                    lineterm="",
                )
            )
            if difference:
                diffs[f"{name}.yaml"] = "\n".join(difference)
        return {
            "candidate_form": candidate_form,
            "yaml_files": yaml_files,
            "diffs": diffs,
            "validation": validation,
            "base_package_version": patch.base_specification_version,
            "package_hash": package_hash,
            "evidence": [item.model_dump(mode="json") for item in patch.evidence],
            "assumptions": [item.model_dump(mode="json") for item in patch.assumptions],
            "unresolved_questions": list(patch.unresolved_questions),
            "checklist_changes": [{"item_id": item} for item in patch.affected_checklist_items],
        }

    def _inspect(
        self,
        package_dir: Path,
        *,
        stage: WorkflowStage,
        workflow: WorkflowDefinition,
    ) -> dict[str, Any]:
        from genesis.assistant import StudyAssistant
        from genesis.checklists import evaluate_rules

        report = (
            StudyAssistant(theory_templates=self.service._workflow_registry.theory_templates())
            .inspect_package(package_dir)
            .as_dict()
        )
        issues = report.get("issues", []) if isinstance(report, dict) else []
        current_index = next(
            index for index, item in enumerate(workflow.stages) if item.id == stage.id
        )
        downstream_sections = {
            path.split("/", 2)[1]
            for later_stage in workflow.stages[current_index + 1 :]
            for path in later_stage.owned_paths
            if path.startswith("/") and len(path.split("/", 2)) > 1
        }
        errors = []
        warnings = [issue for issue in issues if issue.get("severity") == "warning"]
        for issue in issues:
            if issue.get("severity") != "error":
                continue
            dependency = issue.get("dependency_section")
            if dependency in downstream_sections:
                deferred = dict(issue)
                deferred["severity"] = "warning"
                deferred["message"] = (
                    f"{issue.get('message', '')} (deferred until the {dependency} stage)"
                )
                warnings.append(deferred)
            else:
                errors.append(issue)
        checklist = {}
        try:
            checklist = evaluate_rules(package_dir)
        except Exception:
            checklist = {}
        return {
            "errors": [
                {
                    "code": item.get("code"),
                    "path": item.get("json_pointer", item.get("path", "")),
                    "message": item.get("message", ""),
                }
                for item in errors
            ],
            "warnings": [
                {
                    "code": item.get("code"),
                    "path": item.get("json_pointer", item.get("path", "")),
                    "message": item.get("message", ""),
                }
                for item in warnings
            ],
            "checklist": {key: value for key, value in checklist.items() if isinstance(value, str)},
            "summary": report.get("summary", "") if isinstance(report, dict) else "",
        }


class ElicitationAssistant:
    """Assembles stage-aware requests and validates assistant responses (IEL-008)."""

    def __init__(self, response_schema: dict[str, Any] | None = None) -> None:
        self.response_schema = response_schema

    def assemble_evaluation_request(
        self,
        workflow: WorkflowDefinition,
        stage: WorkflowStage,
        session: ElicitationSession,
        projections: dict[str, str] | None = None,
        checklist_state: dict[str, Any] | None = None,
        pending_answer: str | None = None,
        clarification_state: dict[str, Any] | None = None,
    ) -> str:
        clarification_state = clarification_state or {}
        lines = [
            workflow.instructions_text().strip(),
            "",
            "## Current stage",
            f"{stage.title} — {stage.id}",
            stage.instructions.strip(),
            "",
            "## Stage ambiguity topics",
            ", ".join(stage.ambiguity_topics),
            "",
            "## Simulation-critical decisions",
            _render_critical_decisions(
                stage,
                clarification_state.get("decision_coverage", {}),
            ),
            "",
            "## Clarification budget",
            f"Turns used: {clarification_state.get('turns_used', 0)}",
            (
                "Turns remaining: "
                f"{clarification_state.get('turns_remaining', stage.clarification.max_turns)}"
            ),
            f"Pending researcher turn: {len(session.turns) + 1}",
            "",
            "## Draft target templates",
            _render_templates(stage.templates, workflow),
            "",
            "## Registered theory templates",
            _render_theory_templates(workflow, stage),
            "",
            "## Approved upstream projections",
        ]
        projections = projections or {}
        if projections:
            for section, text in projections.items():
                lines.append(f"### {section}\n{text}")
        else:
            lines.append("(none yet)")
        lines += [
            "",
            "## Relevant checklist state",
            _render_checklist(checklist_state or {}, stage.checklist_items),
            "",
            "## Conversation turns (this stage)",
            _render_turns(session, stage.id),
            "",
            "## New researcher answer",
            pending_answer or "(no new answer)",
            "",
            "## Your task",
            (
                "Evaluate the researcher's most recent answer. Decide whether the stage "
                "needs clarification or is ready to draft. If clarifying, ask exactly one "
                "primary question with exactly three editable suggestions for the FIRST "
                "consequential ambiguity. Every ambiguity object MUST include all fields: "
                "id, decision_id, target_paths, question, reason, consequential "
                "(true/false), suggestions (exactly three entries). A blocking ambiguity "
                "must reference a configured required simulation-critical decision and "
                "stay within its target paths. Do not ask about undeclared or optional "
                "details after required decisions are covered. Return a complete "
                "decision_coverage snapshot. When ready_to_draft, assumptions are "
                "allowed as documented closures - never mark genuinely outstanding "
                "ambiguities as assumptions (use the ambiguities field instead). "
                "Never approve anything yourself. Respond with ONLY "
                "the evaluation object itself; do not echo schema metadata such as $schema "
                "or $id, and add no prose."
            ),
            "",
            "## Response schema",
            _json_schema_text(self.response_schema or {}),
        ]
        return "\n".join(lines)

    def parse_evaluation(
        self, text: str, existing_turn_ids: set[int], local_map: dict[int, int] | None = None
    ) -> AssistantEvaluation:
        payload = _extract_json_object(text)
        if payload is None:
            raise ValueError("ASSISTANT_OUTPUT_INVALID: evaluation is not valid JSON")
        evaluation = AssistantEvaluation.model_validate(payload)
        evaluation.validate_turn_references(existing_turn_ids, local_map)
        return evaluation


def _render_templates(templates: dict[str, str], workflow: WorkflowDefinition) -> str:
    if not templates:
        return "(none)"
    return "\n".join(f"- {key}: {path}" for key, path in templates.items())


def _render_critical_decisions(stage: WorkflowStage, coverage: dict[str, Any]) -> str:
    if not stage.critical_decisions:
        return "(legacy workflow: no critical decisions declared)"
    return "\n".join(
        (
            f"- {decision.id} [{'required' if decision.required else 'optional'}] "
            f"-> {', '.join(decision.target_paths)}: "
            f"{coverage.get(decision.id, 'unresolved')}\n"
            f"  Question: {decision.question}\n"
            f"  Runtime relevance: {decision.rationale}"
        )
        for decision in stage.critical_decisions
    )


def _render_theory_templates(workflow: WorkflowDefinition, stage: WorkflowStage) -> str:
    if "/theory" not in stage.owned_paths or not workflow.theory_templates:
        return "(not applicable to this stage)"
    sections: list[str] = []
    for template in workflow.theory_templates.values():
        sections.append(
            f"### {template.id}\n"
            f"Required functions: {', '.join(template.functions)}\n"
            + "\n".join(f"- {question}" for question in template.questions)
        )
    return "\n\n".join(sections)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from assistant text, tolerating markdown fences.

    Models frequently wrap responses in ```json fences or add prose around
    the object; transport leniency here does not relax the strict
    contract validation applied afterwards.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        if start < 0:
            return None
        cleaned = cleaned[start:]
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(cleaned):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(cleaned[: index + 1])
                except json.JSONDecodeError:
                    return None
                if isinstance(payload, dict):
                    # Models sometimes echo schema metadata; strip the envelope
                    # keys at transport level (contract validation stays strict).
                    payload.pop("$schema", None)
                    payload.pop("$id", None)
                    return payload
                return None
    return None


def _json_schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True)


def _render_checklist(checklist_state: dict[str, Any], items: tuple[str, ...]) -> str:
    lines = []
    for item in items:
        entry = checklist_state.get(item)
        status = entry.get("status") if isinstance(entry, dict) else "unresolved"
        lines.append(f"- {item}: {status}")
    return "\n".join(lines)


def _render_turns(session: ElicitationSession, stage_id: str | None = None) -> str:
    lines = []
    for turn in session.turns:
        if stage_id is not None and turn.stage_id != stage_id:
            continue
        lines.append(f"turn {turn.id} question: {turn.question}")
        lines.append(f"turn {turn.id} answer ({turn.response_mode}): {turn.answer or ''}")
        evaluation = turn.evaluation or {}
        decision_ids = [
            str(item.get("decision_id"))
            for item in evaluation.get("decision_coverage", [])
            if isinstance(item, dict) and item.get("decision_id")
        ]
        if decision_ids:
            lines.append(f"turn {turn.id} evaluated decisions: {', '.join(decision_ids)}")
    return "\n".join(lines)


COMPLETION_REQUIREMENTS = {
    "required_checklist_complete": "checklist_complete",
    "no_consequential_ambiguity": "no_consequential_ambiguity",
    "canonical_schema_valid": "canonical_schema_valid",
}


def completion_checks(
    *,
    checklist_state: dict[str, str],
    checklist_items: tuple[str, ...],
    any_consequential_ambiguity: bool,
    canonical_errors: list[dict[str, Any]],
) -> dict[str, bool]:
    """Compute generic completion flags for a stage (IEL-023)."""
    required = [
        item
        for item in checklist_items
        if checklist_state.get(item) not in {"complete", "not_applicable"}
    ]
    return {
        "checklist_complete": not required,
        "no_consequential_ambiguity": not any_consequential_ambiguity,
        "canonical_schema_valid": not canonical_errors,
    }


def require_completion_rules(
    stage: WorkflowStage,
    checks: dict[str, bool],
) -> list[str]:
    """Execute the stage's configured completion rules generically."""
    failures: list[str] = []
    for rule in stage.completion_rules:
        requirement = COMPLETION_REQUIREMENTS.get(rule.rule)
        if requirement is None:
            raise ValueError(f"WORKFLOW_INVALID: unknown completion rule '{rule.rule}'")
        if not checks[requirement]:
            failures.append(rule.rule)
    return failures


class WorkflowRegistry:
    """Discovers workflow packages, validates them, and serves immutable definitions."""

    def __init__(
        self,
        root: str | Path,
        checklist_ids: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.checklist_ids = (
            set(checklist_ids) if checklist_ids is not None else known_checklist_ids()
        )
        self._definitions: dict[str, WorkflowDefinition] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.root.is_dir():
            return
        if (self.root / "workflow.yaml").is_file():
            definition = self._load_package(self.root)
            self._definitions[definition.id] = definition
        for manifest_path in sorted(self.root.glob("*/workflow.yaml")):
            if manifest_path.parent == self.root:
                continue
            definition = self._load_package(manifest_path.parent)
            self._definitions[definition.id] = definition

    def _load_package(self, package_dir: Path) -> WorkflowDefinition:
        manifest = yaml.safe_load((package_dir / "workflow.yaml").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("WORKFLOW_INVALID: workflow.yaml must be a mapping")
        stage_ids = tuple(str(item) for item in manifest.get("stages", ()))
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError(f"WORKFLOW_INVALID: duplicate stage ids in {package_dir}")
        stages: list[WorkflowStage] = []
        for stage_id in stage_ids:
            stage_path = None
            for pattern in (
                f"{stage_id}.yaml",
                f"*-{stage_id}.yaml",
                f"*{stage_id}*.yaml",
            ):
                candidates = sorted((package_dir / "stages").glob(pattern))
                if candidates:
                    stage_path = candidates[0]
                    break
            if stage_path is None:
                raise ValueError(
                    f"WORKFLOW_STAGE_INVALID: missing stage file for '{stage_id}' in {package_dir}"
                )
            data = yaml.safe_load(stage_path.read_text())
            if not isinstance(data, dict):
                raise ValueError(f"WORKFLOW_STAGE_INVALID: {stage_path} must be a mapping")
            stage = self._validate_stage(WorkflowStage.model_validate(data), package_dir)
            stages.append(stage)
        self._validate_dependencies(tuple(stages))
        invalidation_raw = manifest.get("invalidation", {})
        invalidation: dict[str, tuple[str, ...]] = {}
        for key, targets in invalidation_raw.items():
            if key not in CANONICAL_SECTIONS:
                raise ValueError(f"WORKFLOW_INVALID: undeclared invalidation key '{key}'")
            targets = tuple(str(item) for item in targets)
            for target in targets:
                if target not in CANONICAL_SECTIONS:
                    raise ValueError(f"WORKFLOW_INVALID: undeclared invalidation target '{target}'")
            invalidation[key] = targets
        instructions = ""
        instructions_path = manifest.get("instructions") or "instructions.md"
        if instructions_path:
            instructions_file = package_dir / str(instructions_path)
            if not instructions_file.is_file():
                raise ValueError(f"WORKFLOW_INVALID: missing instructions file {instructions_file}")
            instructions = instructions_file.read_text()
        theory_templates = self._load_theory_templates(package_dir, manifest)
        return WorkflowDefinition(
            id=str(manifest.get("id")),
            version=str(manifest.get("version", "1.0")),
            title=str(manifest.get("title", "")),
            instructions=instructions,
            instructions_path=str(instructions_path),
            session_persistence=cast(Any, str(manifest.get("session_persistence", "memory"))),
            stages=tuple(stages),
            invalidation=invalidation,
            theory_templates=theory_templates,
        )

    @staticmethod
    def _validate_dependencies(stages: tuple[WorkflowStage, ...]) -> None:
        stage_ids = {stage.id for stage in stages}
        graph = {stage.id: tuple(stage.depends_on) for stage in stages}
        for stage in stages:
            for dependency in stage.depends_on:
                if dependency not in stage_ids:
                    raise ValueError(
                        f"WORKFLOW_INVALID: unknown dependency '{dependency}' for stage "
                        f"'{stage.id}'"
                    )
                if dependency == stage.id:
                    raise ValueError(
                        f"WORKFLOW_INVALID: dependency cycle includes stage '{stage.id}'"
                    )
        visiting: set[str] = set()
        visited: set[str] = set()

        def _visit(stage_id: str) -> None:
            if stage_id in visiting:
                raise ValueError(f"WORKFLOW_INVALID: dependency cycle includes stage '{stage_id}'")
            if stage_id in visited:
                return
            visiting.add(stage_id)
            for dependency in graph[stage_id]:
                _visit(dependency)
            visiting.remove(stage_id)
            visited.add(stage_id)

        for stage in stages:
            _visit(stage.id)
        positions = {stage.id: index for index, stage in enumerate(stages)}
        for stage in stages:
            for dependency in stage.depends_on:
                if positions[dependency] >= positions[stage.id]:
                    raise ValueError(
                        f"WORKFLOW_INVALID: dependency '{dependency}' must precede stage "
                        f"'{stage.id}'"
                    )

    @staticmethod
    def _load_theory_templates(
        package_dir: Path, manifest: dict[str, Any]
    ) -> dict[str, TheoryTemplate]:
        configured = manifest.get("theory_templates")
        if configured is None:
            return {}
        template_dir = package_dir / str(configured)
        if not template_dir.is_dir():
            raise ValueError(f"WORKFLOW_INVALID: missing theory template directory '{configured}'")
        templates: dict[str, TheoryTemplate] = {}
        for path in sorted(template_dir.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"WORKFLOW_INVALID: theory template {path} must be a mapping")
            template = TheoryTemplate.model_validate(payload)
            if template.id != path.stem:
                raise ValueError(
                    f"WORKFLOW_INVALID: theory template id '{template.id}' does not match "
                    f"file '{path.name}'"
                )
            if template.id in templates:
                raise ValueError(f"WORKFLOW_INVALID: duplicate theory template '{template.id}'")
            templates[template.id] = template
        return templates

    def _validate_stage(self, stage: WorkflowStage, package_dir: Path) -> WorkflowStage:
        for owned in stage.owned_paths:
            if not owned.startswith("/"):
                raise ValueError(
                    f"WORKFLOW_STAGE_INVALID: owned path '{owned}' must start with '/'"
                )
        for item_id in stage.checklist_items:
            if item_id not in self.checklist_ids:
                raise ValueError(f"WORKFLOW_STAGE_INVALID: unknown checklist id '{item_id}'")
        decision_ids = [decision.id for decision in stage.critical_decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError(
                f"WORKFLOW_STAGE_INVALID: duplicate critical decision id in '{stage.id}'"
            )
        for decision in stage.critical_decisions:
            if not decision.question.strip() or not decision.rationale.strip():
                raise ValueError(
                    f"WORKFLOW_STAGE_INVALID: critical decision '{decision.id}' needs "
                    "a question and rationale"
                )
            if decision.required and not decision.target_paths:
                raise ValueError(
                    f"WORKFLOW_STAGE_INVALID: required critical decision "
                    f"'{decision.id}' needs target paths"
                )
            for target in decision.target_paths:
                if not any(
                    target == owned or target.startswith(owned + "/") for owned in stage.owned_paths
                ):
                    raise ValueError(
                        f"WORKFLOW_STAGE_INVALID: critical decision '{decision.id}' target "
                        f"'{target}' is outside owned paths"
                    )
            if decision.default_value is not None and not decision.default_allowed:
                raise ValueError(
                    f"WORKFLOW_STAGE_INVALID: critical decision '{decision.id}' declares "
                    "a default without default_allowed"
                )
        for rule in stage.completion_rules:
            if rule.rule not in WORKFLOW_COMPLETION_RULES:
                raise ValueError(f"WORKFLOW_STAGE_INVALID: unknown completion rule '{rule.rule}'")
        for template_path in stage.templates.values():
            if not (package_dir / template_path).is_file():
                raise ValueError(f"WORKFLOW_STAGE_INVALID: missing template file '{template_path}'")
        return stage

    def get(self, workflow_id: str) -> WorkflowDefinition:
        try:
            definition = self._definitions[workflow_id]
        except KeyError as exc:
            raise KeyError(f"WORKFLOW_INVALID: unknown workflow '{workflow_id}'") from exc
        return definition.model_copy(deep=True)

    def list(self) -> list[WorkflowDefinition]:
        return [definition.model_copy(deep=True) for definition in self._definitions.values()]

    def theory_templates(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for definition in self._definitions.values():
            for template_id, template in definition.theory_templates.items():
                if template_id in result:
                    raise ValueError(
                        f"WORKFLOW_INVALID: duplicate theory template '{template_id}' "
                        "across workflow packages"
                    )
                result[template_id] = template.model_dump(mode="json")
        return result


class Suggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: str


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    source_turns: tuple[int, ...] = ()


class Ambiguity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StableId
    decision_id: StableId | None = None
    target_paths: tuple[str, ...] = ()
    question: str
    reason: str
    consequential: bool = True
    suggestions: tuple[Suggestion, ...] = ()

    @model_validator(mode="after")
    def _check_suggestion_count(self) -> Ambiguity:
        if self.consequential and len(self.suggestions) != 3:
            raise ValueError(
                f"ASSISTANT_OUTPUT_INVALID: consequential ambiguity '{self.id}' "
                "must carry exactly three suggestions"
            )
        return self


class ChecklistUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_id: StableId
    proposed_status: Literal["unresolved", "partial", "complete", "not_applicable"]


class DecisionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: StableId
    status: Literal["unresolved", "covered", "defaulted"]
    evidence_turns: tuple[int, ...] = ()


class AssistantEvaluation(BaseModel):
    """Typed assistant evaluation of a researcher answer (IEL-008)."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["needs_clarification", "ready_to_draft"]
    summary: str
    evidence: tuple[EvidenceClaim, ...] = ()
    assumptions: tuple[str, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    decision_coverage: tuple[DecisionCoverage, ...] = ()
    checklist_updates: tuple[ChecklistUpdate, ...] = ()

    @model_validator(mode="after")
    def _check_shape(self) -> AssistantEvaluation:
        if self.status == "needs_clarification" and not self.ambiguities:
            raise ValueError(
                "ASSISTANT_OUTPUT_INVALID: needs_clarification requires at least one ambiguity"
            )
        if self.status == "ready_to_draft" and any(
            ambiguity.consequential for ambiguity in self.ambiguities
        ):
            raise ValueError(
                "ASSISTANT_OUTPUT_INVALID: ready_to_draft must not carry consequential ambiguity"
            )
        # Note: assumptions ARE allowed on ready_to_draft - they are documented
        # closures (the patch path requires "evidence or listed assumptions"), and
        # routine model emissions (e.g. assumed defaults) must not block approval.
        # Genuinely unresolved status is carried by ambiguities, which are
        # consequential-blocked above.
        return self

    def validate_turn_references(
        self, existing_turn_ids: set[int], local_map: dict[int, int] | None = None
    ) -> None:
        local_map = local_map or {}
        for claim in self.evidence:
            normalized: list[int] = []
            for turn in claim.source_turns:
                if turn in existing_turn_ids:
                    normalized.append(turn)
                elif turn in local_map:
                    normalized.append(local_map[turn])
                else:
                    raise ValueError(
                        f"ASSISTANT_EVIDENCE_INVALID: evidence cites unknown turn {turn}"
                    )
            claim.source_turns = tuple(sorted(set(normalized)))

    def validate_decisions(
        self,
        stage: WorkflowStage,
        existing_turn_ids: set[int],
        local_map: dict[int, int] | None = None,
    ) -> None:
        """Validate that assistant uncertainty is bounded by workflow decisions."""
        local_map = local_map or {}
        self.validate_turn_references(existing_turn_ids, local_map)
        for coverage in self.decision_coverage:
            normalized = []
            for turn in coverage.evidence_turns:
                if turn in existing_turn_ids:
                    normalized.append(turn)
                elif turn in local_map:
                    normalized.append(local_map[turn])
                else:
                    raise ValueError(
                        "ASSISTANT_EVIDENCE_INVALID: decision coverage cites unknown turns "
                        + ", ".join(
                            str(item)
                            for item in sorted(
                                set(coverage.evidence_turns) - existing_turn_ids - set(local_map)
                            )
                        )
                    )
            coverage.evidence_turns = tuple(sorted(set(normalized)))
        configured = {decision.id: decision for decision in stage.critical_decisions}
        seen: set[str] = set()
        statuses: dict[str, str] = {}
        for coverage in self.decision_coverage:
            if coverage.decision_id in seen:
                raise ValueError(
                    f"ASSISTANT_DECISION_INVALID: duplicate coverage for '{coverage.decision_id}'"
                )
            seen.add(coverage.decision_id)
            decision = configured.get(coverage.decision_id)
            if decision is None:
                raise ValueError(
                    f"ASSISTANT_DECISION_INVALID: undeclared decision '{coverage.decision_id}'"
                )
            if coverage.status == "covered" and not coverage.evidence_turns:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: covered decision "
                    f"'{coverage.decision_id}' requires researcher evidence"
                )
            if coverage.status == "defaulted" and not decision.default_allowed:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: decision "
                    f"'{coverage.decision_id}' does not allow a default"
                )

            statuses[coverage.decision_id] = coverage.status

        consequential = [item for item in self.ambiguities if item.consequential]
        if self.status == "needs_clarification" and len(consequential) != 1:
            raise ValueError(
                "ASSISTANT_DECISION_INVALID: ordinary clarification requires exactly "
                "one consequential decision ambiguity"
            )
        remaining_required = {
            decision.id
            for decision in stage.critical_decisions
            if decision.required and statuses.get(decision.id) not in {"covered", "defaulted"}
        }
        for ambiguity in consequential:
            decision = configured.get(str(ambiguity.decision_id or ""))
            if decision is None:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: undeclared decision "
                    f"'{ambiguity.decision_id or ambiguity.id}'"
                )
            if not decision.required:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: optional decision "
                    f"'{decision.id}' cannot block clarification"
                )
            if not ambiguity.target_paths:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: ambiguity for decision "
                    f"'{decision.id}' requires target paths"
                )
            outside = [path for path in ambiguity.target_paths if path not in decision.target_paths]
            if outside:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: ambiguity targets outside decision "
                    f"'{decision.id}': {', '.join(outside)}"
                )
            if remaining_required and decision.id not in remaining_required:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: cannot ask again about already covered "
                    f"decision '{decision.id}' while required decisions remain unresolved"
                )

        if self.status == "ready_to_draft":
            if remaining_required:
                raise ValueError(
                    "ASSISTANT_DECISION_INVALID: required decisions remain unresolved: "
                    + ", ".join(sorted(remaining_required))
                )

    @property
    def next_question(self) -> str | None:
        for ambiguity in self.ambiguities:
            if ambiguity.consequential:
                return ambiguity.question
        return None

    @property
    def next_suggestions(self) -> tuple[Suggestion, ...]:
        for ambiguity in self.ambiguities:
            if ambiguity.consequential:
                return ambiguity.suggestions
        return ()


class SpecificationPatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["add", "replace", "remove"]
    path: str
    value: Any = None


class SpecificationPatchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    source_turns: tuple[int, ...] = ()


class SpecificationPatchAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    statement: str


class SpecificationPatch(BaseModel):
    """Field-level, evidence-linked patch proposal (IEL-020..IEL-022)."""

    model_config = ConfigDict(extra="forbid")
    stage_id: StableId
    base_specification_version: int
    operations: tuple[SpecificationPatchOperation, ...]
    evidence: tuple[SpecificationPatchEvidence, ...] = ()
    assumptions: tuple[SpecificationPatchAssumption, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    affected_checklist_items: tuple[StableId, ...] = ()

    def validate_turn_references(
        self, existing_turn_ids: set[int], local_map: dict[int, int] | None = None
    ) -> None:
        local_map = local_map or {}
        for item in self.evidence:
            normalized: list[int] = []
            for turn in item.source_turns:
                if turn in existing_turn_ids:
                    normalized.append(turn)
                elif turn in local_map:
                    normalized.append(local_map[turn])
                else:
                    raise ValueError(f"ASSISTANT_EVIDENCE_INVALID: patch cites unknown turn {turn}")
            item.source_turns = tuple(sorted(set(normalized)))

    def validate_owned_paths(self, owned_paths: tuple[str, ...]) -> None:
        for operation in self.operations:
            owned = any(
                operation.path == path or operation.path.startswith(path.rstrip("/") + "/")
                for path in owned_paths
            )
            if not owned:
                raise ValueError(
                    f"PATCH_PATH_FORBIDDEN: operation path '{operation.path}' is "
                    "outside the stage's owned paths"
                )

    def validate_checklist_items(self, checklist_items: tuple[str, ...]) -> None:
        allowed = set(checklist_items)
        unknown = [item for item in self.affected_checklist_items if item not in allowed]
        if unknown:
            raise ValueError(
                "ASSISTANT_OUTPUT_INVALID: unknown or unowned checklist item(s): "
                + ", ".join(unknown)
            )

    def validate_base_version(self, current_version: int) -> None:
        if self.base_specification_version != current_version:
            raise ValueError(
                f"PATCH_BASE_STALE: patch targets version "
                f"{self.base_specification_version}, current is {current_version}"
            )

    def validate_evidence(self) -> None:
        """IEL-022/023: consequential values carry evidence or explicit assumptions."""
        if not self.operations:
            return
        if not self.evidence and not self.assumptions:
            raise ValueError(
                "ASSISTANT_EVIDENCE_INVALID: operations need evidence or listed assumptions"
            )
        operation_paths = [operation.path for operation in self.operations]

        def _covers(target: str, operation_path: str) -> bool:
            return (
                target == operation_path
                or target.startswith(operation_path.rstrip("/") + "/")
                or operation_path.startswith(target.rstrip("/") + "/")
            )

        for item in self.evidence:
            if not any(_covers(item.target, path) for path in operation_paths):
                raise ValueError(
                    f"ASSISTANT_EVIDENCE_INVALID: evidence target '{item.target}' "
                    "does not correspond to any patch operation"
                )
        for assumption in self.assumptions:
            if not any(_covers(assumption.target, path) for path in operation_paths):
                raise ValueError(
                    f"ASSISTANT_EVIDENCE_INVALID: assumption target '{assumption.target}' "
                    "does not correspond to any patch operation"
                )
        for path in operation_paths:
            covered = any(_covers(item.target, path) for item in self.evidence) or any(
                _covers(item.target, path) for item in self.assumptions
            )
            if not covered:
                raise ValueError(
                    f"ASSISTANT_EVIDENCE_INVALID: uncovered operation '{path}' needs "
                    "supporting turns or an operation-scoped assumption"
                )


class ElicitationTurn(BaseModel):
    """One immutable conversation turn (IEL-004..IEL-007)."""

    model_config = ConfigDict(extra="forbid")
    id: int
    stage_id: str
    question: str
    suggestions: tuple[tuple[str, str], ...] = ()  # (label, value)
    answer: str | None = None
    response_mode: Literal["suggested", "edited", "free_form"] = "free_form"
    evaluation: dict[str, Any] | None = None
    provider: str | None = None
    model: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class StageProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    status: Literal[
        "not_started",
        "clarifying",
        "limit_reached",
        "draft_ready",
        "awaiting_approval",
        "approved",
        "needs_review",
        "cancelled",
    ] = "not_started"
    turn_count: int = 0
    decision_coverage: dict[str, Literal["unresolved", "covered", "defaulted"]] = Field(
        default_factory=dict
    )
    limit_reached: bool = False
    review_mode: bool = False
    asked_questions: list[str] = Field(default_factory=list)
    deferred_questions: list[str] = Field(default_factory=list)
    accepted_revision: int | None = None
    accepted_summary: str | None = None


class ElicitationSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    specification_id: str
    workflow_id: str
    workflow_version: str
    model_profile_id: str
    researcher_id: str
    current_stage: str
    status: Literal[
        "active",
        "awaiting_answer",
        "awaiting_approval",
        "revision_requested",
        "completed",
        "cancelled",
    ] = "awaiting_answer"
    base_specification_version: int
    current_question: str = ""
    current_suggestions: tuple[tuple[str, str], ...] = ()
    stages: dict[str, StageProgress] = Field(default_factory=dict)
    turns: list[ElicitationTurn] = Field(default_factory=list)
    pending_patch: dict[str, Any] | None = None
    pending_preview: dict[str, Any] | None = None
    last_assistant_attempts: list[dict[str, Any]] = Field(default_factory=list)
    invalidations: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def turn_ids(self) -> set[int]:
        return {turn.id for turn in self.turns}


class ElicitationSessionStore:
    """Sessions with atomic JSON persistence and a shared local-process lock."""

    def __init__(self, path: Path | None = None) -> None:
        self._sessions: dict[str, ElicitationSession] = {}
        self._idempotency: OrderedDict[tuple[str, str], tuple[str, dict[str, Any]]] = OrderedDict()
        self._idempotency_limit = 128
        self._lock = threading.RLock()
        self._lock_depth = 0
        self._path = path
        with self._access():
            pass

    def _reload(self) -> None:
        if self._path is not None and self._path.exists():
            saved = json.loads(self._path.read_text())
            self._sessions = {
                key: ElicitationSession.model_validate(value)
                for key, value in saved["sessions"].items()
            }
            self._idempotency.clear()
            for session_id, key, digest, result in saved.get("idempotency", []):
                self._idempotency[(session_id, key)] = (digest, result)

    @contextmanager
    def _access(self) -> Iterator[None]:
        """Reload once per outer operation; keep nested mutations under one lock.

        Lock a stable sidecar, not the JSON inode replaced by atomic saves.
        Holding this across run_idempotent also serializes replay checks across
        service instances. This uses the local Unix filesystem's advisory locks.
        """
        with self._lock:
            if self._lock_depth or self._path is None:
                yield
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._path.with_name(self._path.name + ".lock")
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                self._lock_depth += 1
                try:
                    self._reload()
                    yield
                finally:
                    self._lock_depth -= 1
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "sessions": {
                key: value.model_dump(mode="json") for key, value in self._sessions.items()
            },
            "idempotency": [
                [sid, key, digest, result]
                for (sid, key), (digest, result) in self._idempotency.items()
            ],
        }
        fd, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=".sessions-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def list_sessions(self) -> list[ElicitationSession]:
        with self._access():
            return [session.model_copy(deep=True) for session in self._sessions.values()]

    @property
    def idempotency_size(self) -> int:
        with self._access():
            return len(self._idempotency)

    def get_idempotency(
        self, session_id: str, key: str, payload_hash: str
    ) -> dict[str, Any] | None:
        with self._access():
            record = self._idempotency.get((session_id, key))
            if record is None:
                return None
            stored_hash, result = record
            if stored_hash != payload_hash:
                raise ValueError(
                    "IDEMPOTENCY_CONFLICT: the key was already used with a different payload"
                )
            self._idempotency.move_to_end((session_id, key))
            return deepcopy(result)

    def record_idempotency(
        self,
        session_id: str,
        key: str,
        payload_hash: str,
        result: dict[str, Any],
    ) -> None:
        with self._access():
            existing = self._idempotency.get((session_id, key))
            if existing is not None and existing[0] != payload_hash:
                raise ValueError(
                    "IDEMPOTENCY_CONFLICT: the key was already used with a different payload"
                )
            self._idempotency[(session_id, key)] = (payload_hash, deepcopy(result))
            self._idempotency.move_to_end((session_id, key))
            while len(self._idempotency) > self._idempotency_limit:
                self._idempotency.popitem(last=False)
            self._save()

    def run_idempotent(
        self,
        session_id: str,
        key: str | None,
        payload_hash: str,
        mutation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Serialize one mutation and cache a bounded replay-safe result."""
        with self._access():
            if key:
                cached = self.get_idempotency(session_id, key, payload_hash)
                if cached is not None:
                    return cached
            result = mutation()
            if key:
                self.record_idempotency(session_id, key, payload_hash, result)
            return deepcopy(result)

    def put(self, session: ElicitationSession) -> None:
        with self._access():
            self._sessions[session.session_id] = session.model_copy(deep=True)
            self._save()

    def get(self, session_id: str) -> ElicitationSession:
        with self._access():
            try:
                return self._sessions[session_id].model_copy(deep=True)
            except KeyError as exc:
                raise KeyError(
                    f"ELICITATION_SESSION_EXPIRED: no session '{session_id}' "
                    "(session not found in this workspace)"
                ) from exc

    def update(self, session_id: str, mutator: Any) -> ElicitationSession:
        """Mutate inside the lock; returns a deep copy."""
        with self._access():
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(
                    f"ELICITATION_SESSION_EXPIRED: no session '{session_id}' "
                    "(session not found in this workspace)"
                )
            mutator(session)
            session.updated_at = datetime.now(UTC).isoformat()
            self._sessions[session_id] = session
            self._save()
            return session.model_copy(deep=True)

    def delete(self, session_id: str) -> None:
        with self._access():
            self._sessions.pop(session_id, None)
            self._idempotency = OrderedDict(
                (key, value) for key, value in self._idempotency.items() if key[0] != session_id
            )
            self._save()

    def set_answer_metadata(
        self, session_id: str, turn_id: int, provider: str | None, model: str | None
    ) -> None:
        def _mutate(session: ElicitationSession) -> None:
            for turn in session.turns:
                if turn.id == turn_id:
                    turn.provider = provider
                    turn.model = model

        self.update(session_id, _mutate)

    def put_turn_evaluation(self, session_id: str, turn_id: int, evaluation: Any) -> None:
        payload = (
            evaluation.model_dump(mode="json") if hasattr(evaluation, "model_dump") else evaluation
        )

        def _mutate(session: ElicitationSession) -> None:
            for turn in session.turns:
                if turn.id == turn_id:
                    turn.evaluation = payload

        self.update(session_id, _mutate)

    def rename_to_completed(self, session_id: str) -> ElicitationSession:
        def _mutate(session: ElicitationSession) -> None:
            session.status = "completed"
            session.stages[session.current_stage].status = "approved"

        return self.update(session_id, _mutate)

    def put_base_version(self, session_id: str, version: int) -> None:
        def _mutate(session: ElicitationSession) -> None:
            session.base_specification_version = int(version)

        self.update(session_id, _mutate)

    def append_asked_question(self, session_id: str, stage_id: str, question: str) -> None:
        def _mutate(session: ElicitationSession) -> None:
            progress = session.stages.get(stage_id)
            if progress is not None and question not in progress.asked_questions:
                progress.asked_questions.append(question)

        self.update(session_id, _mutate)

    def put_invalidations(self, session_id: str, invalidations: list[dict[str, Any]]) -> None:
        def _mutate(session: ElicitationSession) -> None:
            session.invalidations = list(invalidations)

        self.update(session_id, _mutate)

    def mark_stages_needs_review(self, session_id: str, entries: list[tuple[str, str]]) -> None:
        def _mutate(session: ElicitationSession) -> None:
            for stage_id, _reason in entries:
                progress = session.stages.get(stage_id)
                if progress is None or progress.status in {"not_started", "needs_review"}:
                    continue
                progress.status = "needs_review"

        self.update(session_id, _mutate)

    def mark_stage_needs_review(self, session_id: str, stage_id: str, reason: str) -> None:
        def _mutate(session: ElicitationSession) -> None:
            progress = session.stages.get(stage_id)
            if progress is not None and progress.status == "approved":
                progress.status = "needs_review"

        self.update(session_id, _mutate)

    def put_pending_preview(self, session_id: str, preview: dict[str, Any] | None) -> None:
        def _mutate(session: ElicitationSession) -> None:
            session.pending_preview = preview

        self.update(session_id, _mutate)

    def put_pending_patch(self, session_id: str, patch: dict[str, Any] | None) -> None:
        def _mutate(session: ElicitationSession) -> None:
            session.pending_patch = patch

        self.update(session_id, _mutate)

    def record_deferred_questions(
        self, session_id: str, stage_id: str, questions: list[str]
    ) -> None:
        def _mutate(session: ElicitationSession) -> None:
            progress = session.stages.get(stage_id)
            if progress is None:
                return
            for question in questions:
                if question not in progress.deferred_questions:
                    progress.deferred_questions.append(question)

        self.update(session_id, _mutate)

    def put_assistant_attempts(self, session_id: str, attempts: list[dict[str, Any]]) -> None:
        def _mutate(session: ElicitationSession) -> None:
            session.last_assistant_attempts = [dict(item) for item in attempts]

        self.update(session_id, _mutate)


_SESSION_STATUSES = frozenset(
    {
        "active",
        "awaiting_answer",
        "awaiting_approval",
        "revision_requested",
        "completed",
        "cancelled",
    }
)
_STAGE_STATUSES = frozenset(
    {
        "not_started",
        "clarifying",
        "limit_reached",
        "draft_ready",
        "awaiting_approval",
        "approved",
        "needs_review",
        "cancelled",
    }
)


class ElicitationEngine:
    """Owns the session state machine; contains no study-specific questions."""

    def __init__(self, registry: WorkflowRegistry, store: ElicitationSessionStore) -> None:
        self.registry = registry
        self.store = store

    def start_session(
        self,
        *,
        specification_id: str,
        workflow_id: str,
        model_profile_id: str,
        researcher_id: str,
        base_specification_version: int = 1,
        session_id: str | None = None,
    ) -> ElicitationSession:
        workflow = self.registry.get(workflow_id)
        if not workflow.stages:
            raise ValueError(f"WORKFLOW_INVALID: workflow '{workflow_id}' has no stages")
        if session_id:
            try:
                return self.store.get(session_id)
            except KeyError:
                pass
        first = workflow.stages[0]
        session = ElicitationSession(
            session_id=str(session_id or uuid.uuid4()),
            specification_id=specification_id,
            workflow_id=workflow_id,
            workflow_version=workflow.version,
            model_profile_id=model_profile_id,
            researcher_id=researcher_id,
            current_stage=first.id,
            current_question=first.opening_question,
            base_specification_version=int(base_specification_version),
            stages={
                stage.id: StageProgress(
                    id=stage.id,
                    decision_coverage={
                        decision.id: "unresolved" for decision in stage.critical_decisions
                    },
                )
                for stage in workflow.stages
            },
        )
        session.stages[first.id].status = "clarifying"
        self.store.put(session)
        return session

    def require_session(self, session_id: str) -> ElicitationSession:
        return self.store.get(session_id)

    def stage_for(self, session: ElicitationSession) -> tuple[WorkflowDefinition, Any]:
        workflow = self.registry.get(session.workflow_id)
        return workflow, workflow.stage(session.current_stage)

    def record_researcher_answer(
        self,
        session: ElicitationSession,
        *,
        answer: str,
        response_mode: Literal["suggested", "edited", "free_form"] = "free_form",
        question: str | None = None,
        suggestions: tuple[tuple[str, str], ...] = (),
    ) -> ElicitationTurn:
        turn = ElicitationTurn(
            id=len(session.turns) + 1,
            stage_id=session.current_stage,
            question=question or session.current_question,
            suggestions=tuple(suggestions),
            answer=answer,
            response_mode=response_mode,
        )
        session.turns = [*session.turns, turn]
        if not session.stages[session.current_stage].review_mode:
            session.stages[session.current_stage].turn_count += 1
        self.store.update(
            session.session_id,
            lambda stored: stored.__dict__.update(session.model_copy(deep=True).__dict__),
        )
        return turn

    def transition(
        self,
        session: ElicitationSession,
        session_status: str,
        *,
        stage_status: str | None = None,
        question: str | None = None,
        suggestions: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        if session_status not in _SESSION_STATUSES:
            raise ValueError(f"ELICITATION_STATE_CONFLICT: unknown status '{session_status}'")
        if stage_status is not None and stage_status not in _STAGE_STATUSES:
            raise ValueError(f"ELICITATION_STATE_CONFLICT: unknown stage status '{stage_status}'")
        if session_status == "completed":
            raise ValueError(
                "ELICITATION_STATE_CONFLICT: completed is set only by final stage approval"
            )
        if session.status in {"cancelled", "completed"} and session_status != session.status:
            raise ValueError(
                f"ELICITATION_STATE_CONFLICT: cannot move session from "
                f"{session.status} to {session_status}"
            )
        allowed = {
            ("awaiting_answer", "awaiting_answer"),
            ("awaiting_answer", "awaiting_approval"),
            ("awaiting_approval", "revision_requested"),
            ("awaiting_approval", "awaiting_answer"),
            ("awaiting_approval", "awaiting_approval"),
            ("awaiting_answer", "awaiting_approval"),
            ("revision_requested", "awaiting_answer"),
            ("revision_requested", "awaiting_approval"),
            ("active", "awaiting_answer"),
        }
        if (session.status, session_status) not in allowed and session_status not in {"cancelled"}:
            raise ValueError(
                f"ELICITATION_STATE_CONFLICT: cannot move session from "
                f"{session.status} to {session_status}"
            )
        session.status = cast(Any, session_status)
        if stage_status is not None:
            session.stages[session.current_stage].status = cast(Any, stage_status)
        if question is not None:
            session.current_question = question
        if suggestions is not None:
            session.current_suggestions = suggestions
        self.store.update(
            session.session_id,
            lambda stored: stored.__dict__.update(session.model_copy(deep=True).__dict__),
        )

    def cancel(self, session_id: str) -> ElicitationSession:
        def _mutate(session: ElicitationSession) -> None:
            session.status = "cancelled"
            for progress in session.stages.values():
                if progress.status not in {"approved"}:
                    progress.status = "cancelled"

        return self.store.update(session_id, _mutate)

    def mark_approved(
        self, session: ElicitationSession, *, revision: int, summary: str | None = None
    ) -> None:
        def _mutate(stored: ElicitationSession) -> None:
            progress = stored.stages[stored.current_stage]
            progress.status = "approved"
            progress.accepted_revision = int(revision)
            if summary:
                progress.accepted_summary = summary

        self.store.update(session.session_id, _mutate)

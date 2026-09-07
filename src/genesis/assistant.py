"""Bounded, local study-assistance services.

The assistant only reads package/build material.  It produces reviewable guidance;
it never writes YAML, approves a proposal, or presents generated content as evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compiler import StudyCompiler


@dataclass(frozen=True)
class AssistantResponse:
    """Stable response envelope for assistant inspection operations."""

    kind: str
    valid: bool
    summary: dict[str, Any]
    issues: list[dict[str, Any]]
    guidance: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    approval_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "valid": self.valid,
            "summary": self.summary,
            "issues": self.issues,
            "guidance": self.guidance,
            "suggestions": self.suggestions,
            "approval_required": self.approval_required,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


def _suggestion(code: str, message: str, source_refs: list[str]) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "provenance": {"origin": "assistant_proposed", "source_refs": source_refs},
    }


def _validation_issue(item: Mapping[str, Any], *, severity: str = "error") -> dict[str, Any]:
    """Normalize compiler diagnostics into the public assistant issue contract."""

    raw_path = str(item.get("json_pointer") or item.get("path") or "")
    pointer = raw_path if raw_path.startswith("/") else f"/{raw_path.replace('.', '/')}"
    first_segment = pointer.removeprefix("/").split("/", 1)[0]
    source_file = (
        f"{first_segment}.yaml"
        if first_segment
        in {"study", "openness", "theory", "domain", "protocol", "outcomes", "models"}
        else "package"
    )
    normalized = {
        "code": str(item.get("code") or "PACKAGE_INVALID"),
        "severity": severity,
        "source_file": source_file,
        "json_pointer": pointer,
        "message": str(item.get("message") or "package validation failed"),
    }
    if item.get("dependency_section"):
        normalized["dependency_section"] = str(item["dependency_section"])
    return normalized


class StudyAssistant:
    """Deterministic inspection and preflight guidance for a study workspace."""

    def __init__(self, theory_templates: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.theory_templates = theory_templates

    def inspect_package(self, source: str | Path) -> AssistantResponse:
        """Validate a package in place without creating or changing any files."""

        root = Path(source)
        issues: list[dict[str, Any]] = []
        try:
            compiler = StudyCompiler(root, theory_templates=self.theory_templates)
            loaded = compiler._load()  # read-only schema and package checks
            issues.extend(_validation_issue(item) for item in compiler._validate(loaded))
            theory_errors, theory_warnings = compiler._validate_theory(loaded)
            issues.extend(_validation_issue(item) for item in theory_errors)
            issues.extend(_validation_issue(item, severity="warning") for item in theory_warnings)
        except (OSError, ValueError) as exc:
            for line in str(exc).splitlines() or [str(exc)]:
                code, _, message = line.partition(":")
                issues.append(
                    {
                        "code": code.strip() or "PACKAGE_INVALID",
                        "severity": "error",
                        "source_file": "package",
                        "json_pointer": "",
                        "message": message.strip() or line,
                    }
                )

        suggestions = [
            _suggestion(
                "REVIEW_VALIDATION_ISSUE",
                f"Resolve validation issue {item['code']} and rerun inspection.",
                [str(item.get("source_file") or item.get("path") or "package")],
            )
            for item in issues
        ]
        if not issues:
            suggestions.append(
                _suggestion(
                    "RESEARCHER_REVIEW",
                    "Review the validated package and record researcher decisions before approval.",
                    ["package"],
                )
            )
        return AssistantResponse(
            kind="package_validation",
            valid=not any(i.get("severity") != "warning" for i in issues),
            summary={"path": str(root), "issue_count": len(issues)},
            issues=issues,
            guidance=[],
            suggestions=suggestions,
        )

    # Friendly service aliases used by callers that expose validation/preflight verbs.
    validate_package = inspect_package

    def inspect_build(self, manifest: str | Path) -> AssistantResponse:
        """Read a build manifest and return deterministic preflight guidance."""

        path = Path(manifest)
        build_root: Path | None = None
        if path.is_dir():
            build_root = path
            path = path / "build_manifest.json"
        issues: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        try:
            if build_root is not None:
                StudyCompiler.verify_build(build_root)
            loaded = json.loads(path.read_text())
            if not isinstance(loaded, dict):
                raise ValueError("manifest must be a JSON object")
            data = loaded
            for field in ("study_id", "build_hash"):
                if not data.get(field):
                    issues.append({"code": "MANIFEST_FIELD_MISSING", "message": f"missing {field}"})
            report = data.get("validation_report")
            if isinstance(report, dict) and report.get("valid") is False:
                issues.append({"code": "BUILD_INVALID", "message": "validation report is invalid"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            code = (
                "BUILD_INTEGRITY"
                if str(exc).startswith("BUILD_INTEGRITY")
                else "BUILD_MANIFEST_INVALID"
            )
            issues.append({"code": code, "message": str(exc)})

        guidance = [
            {
                "code": "PREFLIGHT_REVIEW",
                "message": (
                    "Check dependency availability, estimates, limits, and credential "
                    "presence before running."
                ),
                "provenance": {"origin": "assistant_proposed", "source_refs": [str(path)]},
            }
        ]
        return AssistantResponse(
            kind="build_preflight",
            valid=not issues,
            summary={
                "path": str(path),
                "study_id": data.get("study_id"),
                "build_hash": data.get("build_hash"),
            },
            issues=issues,
            guidance=guidance,
            suggestions=[],
        )

    preflight = inspect_build

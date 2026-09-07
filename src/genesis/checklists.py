"""AW-02: persisted elicitation checklists and deterministic completion rules.

Checklist items translate the three-layer research design decisions into
explicit workflow state (Section 6 of the main specification). Statuses are
evaluated deterministically from the canonical package; the researcher may
override statuses, and finalisation fails while required items are unresolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ITEM_STATES = ("unresolved", "partial", "complete", "not_applicable")

CHECKLIST_ITEMS: list[dict[str, Any]] = [
    {
        "id": "l1-openness-need",
        "layer": 1,
        "question": (
            "Is the substantive form genuinely unresolved ex ante and "
            "consequential to the research question?"
        ),
        "required": True,
        "rule": "generative-or-not-applicable",
        "write_targets": ["openness.yaml"],
    },
    {
        "id": "l1-selective-closure",
        "layer": 1,
        "question": (
            "Why should the process remain generative rather than fixed, stochastic, or bounded?"
        ),
        "required": True,
        "rule": "closure-rationales-present",
        "write_targets": ["openness.yaml"],
    },
    {
        "id": "l1-informational-position",
        "layer": 1,
        "question": "What information may legitimately condition generation?",
        "required": True,
        "rule": "context-policies-declared",
        "write_targets": ["openness.yaml", "domain.yaml"],
    },
    {
        "id": "l1-output-control",
        "layer": 1,
        "question": "What structure must the generated output satisfy?",
        "required": True,
        "rule": "generative-outputs-declared",
        "write_targets": ["openness.yaml"],
    },
    {
        "id": "l1-model-dependence",
        "layer": 1,
        "question": ("Which model, prompt, and inference configuration realises the open process?"),
        "required": True,
        "rule": "generative-bindings-complete",
        "write_targets": ["openness.yaml", "models.yaml"],
    },
    {
        "id": "l1-traceability",
        "layer": 1,
        "question": "What must be preserved to reconstruct the realised generation?",
        "required": True,
        "rule": "trace-policies-declared",
        "write_targets": ["openness.yaml"],
    },
    {
        "id": "l2-theory-logic",
        "layer": 2,
        "question": "Which theoretical functions organise the generative dynamics?",
        "required": False,
        "rule": "theory-mappings-present",
        "write_targets": ["theory.yaml"],
    },
    {
        "id": "l2-process-relations",
        "layer": 2,
        "question": "How do generated outputs acquire consequences and feed back?",
        "required": False,
        "rule": "theory-relations-present",
        "write_targets": ["theory.yaml"],
    },
    {
        "id": "l3-domain-substance",
        "layer": 3,
        "question": (
            "Which actors, attributes, states, artifacts, and mechanisms instantiate the process?"
        ),
        "required": False,
        "rule": "domain-declared",
        "write_targets": ["domain.yaml"],
    },
    {
        "id": "l3-initialization",
        "layer": 3,
        "question": "How is the domain initialised and grounded?",
        "required": False,
        "rule": "initialization-declared-or-empty",
        "write_targets": ["domain.yaml"],
    },
    {
        "id": "proto-conditions",
        "layer": 4,
        "question": ("Which conditions are varied and held constant, and how many replications?"),
        "required": False,
        "rule": "protocol-conditions-declared",
        "write_targets": ["protocol.yaml"],
    },
    {
        "id": "proto-observables",
        "layer": 4,
        "question": "Which theoretical observables are defined before execution?",
        "required": False,
        "rule": "outcomes-declared",
        "write_targets": ["outcomes.yaml"],
    },
]


def default_checklist() -> list[dict[str, Any]]:
    items = []
    for definition in CHECKLIST_ITEMS:
        item = {key: value for key, value in definition.items() if key != "rule"}
        item["status"] = "unresolved"
        item["evidence"] = []
        item["researcher_confirmation"] = False
        items.append(item)
    return items


def evaluate_rules(spec_dir: str | Path) -> dict[str, str]:
    """Deterministically evaluate item statuses from the canonical package."""
    from genesis.compiler import StudyCompiler

    root = Path(spec_dir)
    statuses: dict[str, str] = {}
    loader = StudyCompiler(root)
    try:
        loaded = loader._load()
    except ValueError:
        return {item["id"]: "unresolved" for item in CHECKLIST_ITEMS}
    openness = loaded["openness"]
    domain = loaded["domain"]
    theory = loaded["theory"]
    protocol = loaded["protocol"]
    outcomes = loaded["outcomes"]
    generative = [
        process for process in openness.processes if process.executor.mode == "generative"
    ]
    has_generative = bool(generative)
    for item in CHECKLIST_ITEMS:
        rule = item["rule"]
        if rule == "generative-or-not-applicable":
            statuses[item["id"]] = "complete" if has_generative else "not_applicable"
        elif rule == "closure-rationales-present":
            statuses[item["id"]] = (
                "complete"
                if has_generative and all(bool(process.closure_rationale) for process in generative)
                else ("not_applicable" if not has_generative else "unresolved")
            )
        elif rule == "context-policies-declared":
            statuses[item["id"]] = (
                "complete"
                if openness.processes
                and all(bool(process.context_policy) for process in openness.processes)
                else "not_applicable"
            )
        elif rule == "generative-outputs-declared":
            statuses[item["id"]] = (
                "complete"
                if has_generative and all(bool(process.outputs) for process in generative)
                else ("not_applicable" if not has_generative else "unresolved")
            )
        elif rule == "generative-bindings-complete":
            statuses[item["id"]] = (
                "complete"
                if has_generative
                and all(
                    bool(process.executor.model_profile) and bool(process.prompt_ref)
                    for process in generative
                )
                else ("not_applicable" if not has_generative else "unresolved")
            )
        elif rule == "trace-policies-declared":
            statuses[item["id"]] = "complete" if has_generative else "not_applicable"
        elif rule == "theory-mappings-present":
            if str(getattr(theory, "theory_family", "") or "") in {"exploratory", "custom"}:
                statuses[item["id"]] = "not_applicable"
            else:
                statuses[item["id"]] = "complete" if theory.process_mappings else "unresolved"
        elif rule == "theory-relations-present":
            if str(getattr(theory, "theory_family", "") or "") in {"exploratory", "custom"}:
                statuses[item["id"]] = "not_applicable"
            else:
                statuses[item["id"]] = (
                    "complete" if (theory.relations or theory.feedback) else "unresolved"
                )
        elif rule == "domain-declared":
            statuses[item["id"]] = (
                "complete"
                if (
                    domain.actors
                    or domain.attributes
                    or domain.states
                    or domain.artifacts
                    or domain.mechanisms
                )
                else "unresolved"
            )
        elif rule == "initialization-declared-or-empty":
            statuses[item["id"]] = "complete" if domain.initialization else "not_applicable"
        elif rule == "protocol-conditions-declared":
            statuses[item["id"]] = (
                "complete" if protocol.conditions or protocol.replications > 1 else "not_applicable"
            )
        elif rule == "outcomes-declared":
            statuses[item["id"]] = "complete" if outcomes.outcomes else "unresolved"
        else:
            statuses[item["id"]] = "unresolved"
    return statuses


def checklist_record(spec_dir: str | Path) -> list[dict[str, Any]]:
    """Merge the persisted checklist with the freshly evaluated statuses."""
    path = Path(spec_dir) / "checklist.json"
    if path.is_file():
        try:
            persisted = json.loads(path.read_text())
            if isinstance(persisted, list):
                items = persisted
            elif isinstance(persisted, dict) and isinstance(persisted.get("items"), list):
                items = persisted["items"]
            else:
                items = default_checklist()
        except (OSError, json.JSONDecodeError):
            items = default_checklist()
    else:
        items = default_checklist()
    statuses = evaluate_rules(spec_dir)
    by_id = {str(item["id"]): item for item in items}
    for item in CHECKLIST_ITEMS:
        record = by_id.get(item["id"])
        if record is None:
            record = {key: value for key, value in item.items() if key != "rule"}
            record["status"] = statuses.get(item["id"], "unresolved")
            record["evidence"] = []
            record["researcher_confirmation"] = False
            items.append(record)
            by_id[item["id"]] = record
        if not record.get("manual"):
            record["status"] = statuses.get(item["id"], record.get("status", "unresolved"))
    return items


def persist_checklist(spec_dir: str | Path, items: list[dict[str, Any]]) -> None:
    path = Path(spec_dir) / "checklist.json"
    path.write_text(json.dumps(items, indent=2, sort_keys=True) + "\n")

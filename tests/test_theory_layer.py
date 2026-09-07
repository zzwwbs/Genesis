"""AW-01: typed theory layer and cross-layer consistency checks (XL-001..005)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from genesis.compiler import StudyCompiler, ValidationIssue
from genesis.elicitation import WorkflowRegistry, known_checklist_ids
from genesis.specification import TheorySpec


def _write_package(
    root: Path,
    *,
    theory: dict | None = None,
    openness: dict | None = None,
    domain: dict | None = None,
    study: dict | None = None,
    protocol: dict | None = None,
    outcomes: dict | None = None,
    models: dict | None = None,
) -> Path:
    source = root / "package"
    source.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "study": study
        or {"schema_version": "1.0", "study_id": "platform-governance", "title": "x"},
        "openness": openness
        or {"schema_version": "1.0", "study_id": "platform-governance", "processes": []},
        "theory": theory
        or {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "theory_family": "institutional",
        },
        "domain": domain or {"schema_version": "1.0", "study_id": "platform-governance"},
        "protocol": protocol
        or {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "time_model": {"type": "rounds"},
        },
        "outcomes": outcomes
        or {"schema_version": "1.0", "study_id": "platform-governance", "outcomes": []},
        "models": models or {"schema_version": "1.0", "study_id": "platform-governance"},
    }
    for name, value in artifacts.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


VSR_THEORY = {
    "schema_version": "1.0",
    "study_id": "platform-governance",
    "theory_family": "variation-selection-retention",
    "constructs": [
        {"id": "creator-strategy", "theory_role": "variation_unit"},
        {"id": "performance-history", "theory_role": "retention_state"},
    ],
    "process_mappings": [
        {"process": "formulate-strategy", "theory_function": "variation"},
        {"process": "platform-evaluation", "theory_function": "selection"},
        {"process": "performance-settlement", "theory_function": "retention"},
        {"process": "strategy-feedback", "theory_function": "feedback"},
    ],
    "relations": [
        {"from": "formulate-strategy", "to": "platform-evaluation", "relation": "produces_variant"}
    ],
    "feedback": [
        {
            "from": "performance-history",
            "to": "formulate-strategy",
            "relation": "conditions_variation",
        }
    ],
    "observables": [{"id": "strategy-diversity", "definition": "distinct retained strategies"}],
}


def _vsr_package(root: Path) -> Path:
    return _write_package(
        root,
        theory=VSR_THEORY,
        openness={
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "processes": [
                {
                    "id": "formulate-strategy",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "private",
                },
                {
                    "id": "platform-evaluation",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "private",
                },
                {
                    "id": "performance-settlement",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "private",
                },
                {
                    "id": "strategy-feedback",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "private",
                },
            ],
        },
        domain={
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "states": [{"id": "performance-history", "persistence": "across_rounds"}],
            "artifacts": [{"id": "creator-strategy", "artifact_type": "strategy"}],
            "mechanisms": [
                {"id": "platform-recommendation", "implements": "selection"},
                {"id": "strategy-update", "implements": "retention"},
            ],
        },
    )


def test_typed_theory_model_accepts_from_to_relations_and_roles() -> None:
    value = TheorySpec.model_validate(VSR_THEORY)
    assert value.constructs[0].theory_role == "variation_unit"
    assert value.process_mappings[1].theory_function == "selection"
    assert value.relations[0].source == "formulate-strategy"
    assert value.relations[0].target == "platform-evaluation"
    assert value.feedback[0].source == "performance-history"
    assert value.observables[0].definition.startswith("distinct")
    dumped = value.model_dump(mode="json")
    assert dumped["relations"][0]["source"] == "formulate-strategy"


def test_typed_theory_model_rejects_non_stable_ids() -> None:
    bad = {**VSR_THEORY, "constructs": [{"id": "Not Stable"}]}
    with pytest.raises(ValidationError, match="id"):
        TheorySpec.model_validate(bad)


def test_valid_vsr_package_compiles(tmp_path: Path) -> None:
    source = _vsr_package(tmp_path)
    build = StudyCompiler(source).compile(tmp_path / "vsr-build")
    assert build.study_id == "platform-governance"
    report = json.loads((build.path / "validation_report.json").read_text())
    assert report["valid"] is True
    codes = {item["code"] for item in report["warnings"]}
    assert "THEORY_FUNCTION_MISSING" not in codes
    assert "OUTPUT_SCHEMA_MISSING" not in codes


def test_generative_process_without_closure_rationale_raises_openness_incomplete(
    tmp_path: Path,
) -> None:
    openness = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "processes": [
            {
                "id": "formulate-strategy",
                "executor": {"mode": "generative", "model_profile": "creator-model"},
                "context_policy": "private",
                "prompt_ref": "formulate",
                "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                "openness_rationale": "strategy form is the focal phenomenon",
            }
        ],
    }
    source = _write_package(tmp_path, openness=openness)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "x-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "OPENNESS_INCOMPLETE" in codes


def test_unmapped_theory_function_raises(tmp_path: Path) -> None:
    theory = {
        **VSR_THEORY,
        "process_mappings": [{"process": "missing-process", "theory_function": "variation"}],
    }
    source = _write_package(tmp_path, theory=theory)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "x-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "THEORY_FUNCTION_UNMAPPED" in codes


def test_feedback_source_must_be_declared_in_domain(tmp_path: Path) -> None:
    theory = {
        **VSR_THEORY,
        "feedback": [
            {"from": "undeclared-state", "to": "formulate-strategy", "relation": "conditions"}
        ],
    }
    source = _write_package(tmp_path, theory=theory)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "x-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "FEEDBACK_CONTEXT_MISSING" in codes


def test_mechanism_implements_undeclared_function(tmp_path: Path) -> None:
    domain = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "mechanisms": [{"id": "platform-recommendation", "implements": "obfuscation"}],
    }
    source = _write_package(tmp_path, domain=domain)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "x-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "MECHANISM_UNBOUND" in codes


def test_unmapped_generative_process_is_a_non_blocking_warning(tmp_path: Path) -> None:
    openness = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "processes": [
            {
                "id": "formulate-strategy",
                "executor": {"mode": "generative", "model_profile": "creator-model"},
                "context_policy": "private",
                "prompt_ref": "formulate",
                "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                "openness_rationale": "strategy form is focal",
                "closure_rationale": "a menu would remove the variation",
            },
            {
                "id": "observe-response",
                "executor": {"mode": "generative", "model_profile": "creator-model"},
                "context_policy": "private",
                "prompt_ref": "observe",
                "outputs": [{"artifact_type": "observation", "schema_ref": "observation-schema"}],
                "openness_rationale": "observation form is open",
                "closure_rationale": "fixed observation forms would constrain the study",
            },
            {
                "id": "platform-evaluation",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
            {
                "id": "performance-settlement",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
            {
                "id": "strategy-feedback",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
        ],
    }
    domain = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "states": [{"id": "performance-history"}],
        "artifacts": [{"id": "creator-strategy"}, {"id": "observation"}],
    }
    models = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "models": [{"id": "creator-model", "provider": "mock", "model": "mock-1"}],
    }
    package_root = tmp_path / "package"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "prompts").mkdir(exist_ok=True)
    (package_root / "prompts" / "formulate.txt").write_text("Formulate a strategy.")
    (package_root / "prompts" / "observe.txt").write_text("Observe and respond.")
    (package_root / "schemas").mkdir(exist_ok=True)
    (package_root / "schemas" / "strategy-schema.yaml").write_text("type: object\n")
    (package_root / "schemas" / "observation-schema.yaml").write_text("type: object\n")
    source = _write_package(
        tmp_path, theory=VSR_THEORY, openness=openness, domain=domain, models=models
    )
    build = StudyCompiler(source).compile(tmp_path / "with-warning-build")
    report = json.loads((build.path / "validation_report.json").read_text())
    codes = {item["code"] for item in report["warnings"]}
    assert "OPENNESS_UNJUSTIFIED" in codes
    from genesis.assistant import StudyAssistant

    response = StudyAssistant().inspect_package(source).as_dict()
    assert response["valid"] is True
    assert any(item.get("severity") == "warning" for item in response["issues"])


def test_second_theory_template_compiles_without_core_changes(tmp_path: Path) -> None:
    theory = {
        "schema_version": "1.0",
        "study_id": "enactment-study",
        "theory_family": "interpretation-enactment",
        "process_mappings": [
            {"process": "interpret-event", "theory_function": "interpretation"},
            {"process": "enact-response", "theory_function": "enactment"},
            {"process": "institutionalize-response", "theory_function": "institutionalization"},
            {"process": "community-feedback", "theory_function": "feedback"},
        ],
    }
    study = {
        "schema_version": "1.0",
        "study_id": "enactment-study",
        "title": "enactment",
    }
    openness = {
        "schema_version": "1.0",
        "study_id": "enactment-study",
        "processes": [
            {
                "id": "interpret-event",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
            {
                "id": "enact-response",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
            {
                "id": "institutionalize-response",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
            {
                "id": "community-feedback",
                "executor": {"mode": "deterministic"},
                "context_policy": "private",
            },
        ],
    }
    protocol = {
        "schema_version": "1.0",
        "study_id": "enactment-study",
        "time_model": {"type": "rounds"},
    }
    outcomes = {"schema_version": "1.0", "study_id": "enactment-study", "outcomes": []}
    models = {"schema_version": "1.0", "study_id": "enactment-study"}
    domain = {"schema_version": "1.0", "study_id": "enactment-study"}
    source = _write_package(
        tmp_path,
        theory=theory,
        study=study,
        openness=openness,
        domain=domain,
        protocol=protocol,
        outcomes=outcomes,
        models=models,
    )
    build = StudyCompiler(source).compile(tmp_path / "enactment-build")
    assert build.study_id == "enactment-study"


def test_template_registry_exposes_vsr_and_second_template() -> None:
    workflows = Path(__file__).resolve().parents[1] / "workflows"
    templates = WorkflowRegistry(workflows, checklist_ids=known_checklist_ids()).theory_templates()
    vsr = templates.get("variation-selection-retention")
    assert vsr is not None
    assert set(vsr["functions"]) == {"variation", "selection", "retention", "feedback"}
    assert templates.get("interpretation-enactment") is not None
    assert templates.get("unknown-family") is None
    assert set(templates) == {"variation-selection-retention", "interpretation-enactment"}


def test_third_theory_template_loads_without_python_changes(tmp_path: Path) -> None:
    package = tmp_path / "custom-workflow"
    (package / "stages").mkdir(parents=True)
    templates = package / "theory-templates"
    templates.mkdir()
    (package / "workflow.yaml").write_text(
        "id: custom-workflow\nversion: '1.0'\ntitle: custom\n"
        "instructions: instructions.md\nstages: [foundation]\n"
        "theory_templates: theory-templates\n"
    )
    (package / "instructions.md").write_text("instructions")
    (package / "stages" / "foundation.yaml").write_text(
        "id: foundation\ntitle: Foundation\nopening_question: q\n"
        "instructions: i\nowned_paths: ['/study']\n"
    )
    (templates / "diffusion-feedback.yaml").write_text(
        "id: diffusion-feedback\nfunctions: [exposure, adoption, feedback]\n"
        "questions: ['How are actors exposed?', 'What causes adoption?']\n"
    )
    registry = WorkflowRegistry(package, checklist_ids=known_checklist_ids())
    loaded = registry.theory_templates()
    assert loaded["diffusion-feedback"]["functions"] == [
        "exposure",
        "adoption",
        "feedback",
    ]

    theory = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "theory_family": "diffusion-feedback",
        "process_mappings": [{"process": "formulate-strategy", "theory_function": "exposure"}],
    }
    source = _write_package(tmp_path, theory=theory)
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source, theory_templates=loaded).compile(tmp_path / "custom-build")
    assert "THEORY_FUNCTION_MISSING" in {item.code for item in excinfo.value.issues}


def test_compiled_processes_carry_theory_function(tmp_path: Path) -> None:
    """Review finding 8: theory mappings surface on compiled process records."""
    import json

    source = _vsr_package(tmp_path)
    build = StudyCompiler(source).compile(tmp_path / "vsr-role-build")
    processes = json.loads((build.path / "processes.json").read_text())
    by_id = {process["id"]: process for process in processes}
    assert by_id["formulate-strategy"]["theory_function"] == "variation"
    assert by_id["platform-evaluation"]["theory_function"] == "selection"
    assert "theory_function" not in by_id.get("unmapped-process", {})


def test_generative_output_without_usable_schema_fails(tmp_path: Path) -> None:
    """Finding 8: generative outputs require a usable schema file."""
    theory = {
        **VSR_THEORY,
        "process_mappings": [{"process": "formulate-strategy", "theory_function": "variation"}],
    }
    openness = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "processes": [
            {
                "id": "formulate-strategy",
                "executor": {"mode": "generative", "model_profile": "creator-model"},
                "context_policy": "private",
                "prompt_ref": "formulate",
                "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                "openness_rationale": "strategy form is focal",
                "closure_rationale": "a menu would remove the variation",
            }
        ],
    }
    models = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "models": [{"id": "creator-model", "provider": "mock", "model": "mock-1"}],
    }
    domain = {
        "schema_version": "1.0",
        "study_id": "platform-governance",
        "states": [{"id": "performance-history"}],
        "artifacts": [{"id": "creator-strategy"}],
    }
    openness["processes"][0]["outputs"] = [
        {"artifact_type": "strategy", "schema_ref": "creator-strategy"}
    ]
    package_root = tmp_path / "package"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "prompts").mkdir(exist_ok=True)
    (package_root / "prompts" / "formulate.txt").write_text("Formulate a strategy.")
    source = _write_package(
        tmp_path, theory=theory, openness=openness, domain=domain, models=models
    )
    with pytest.raises(ValidationIssue) as excinfo:
        StudyCompiler(source).compile(tmp_path / "no-schema-build")
    codes = {issue.code for issue in excinfo.value.issues}
    assert "OUTPUT_SCHEMA_MISSING" in codes

"""Task 1: data-driven workflow-package registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.elicitation import (
    WorkflowRegistry,
    known_checklist_ids,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


@pytest.fixture
def registry() -> WorkflowRegistry:
    return WorkflowRegistry(WORKFLOWS, checklist_ids=known_checklist_ids())


def test_three_layer_workflow_is_data_driven(registry: WorkflowRegistry) -> None:
    workflow = registry.get("three-layer-study")
    assert [stage.id for stage in workflow.stages] == [
        "study-foundation",
        "openness",
        "theory",
        "domain",
        "experiment-design",
    ]
    assert workflow.stages[1].approval_required is True
    assert "/openness" in workflow.stages[1].owned_paths
    assert workflow.session_persistence == "disk"
    assert "openness.processes" in workflow.invalidation
    assert workflow.stage("openness").depends_on == ("study-foundation",)
    assert workflow.stage("domain").depends_on == ("openness", "theory")
    assert workflow.dependants_of("openness") == (
        "theory",
        "domain",
        "experiment-design",
    )


def test_workflow_contains_no_scientific_strings_in_python(registry: WorkflowRegistry) -> None:
    workflow = registry.get("three-layer-study")
    openness = workflow.stage("openness")
    # The opening question lives in the workflow file, not in the engine.
    assert "open-ended" in openness.opening_question
    assert workflow.instructions_text().startswith("# Three-Layer")


def test_stage_file_controls_questions_and_templates(registry: WorkflowRegistry) -> None:
    stage = registry.get("three-layer-study").stage("experiment-design")
    assert "/protocol" in stage.owned_paths
    assert "/outcomes" in stage.owned_paths
    assert "/models" in stage.owned_paths
    assert set(stage.templates) == {"protocol", "outcomes", "models"}
    assert stage.ambiguity_topics


def test_bundled_workflow_declares_bounded_critical_decisions(
    registry: WorkflowRegistry,
) -> None:
    workflow = registry.get("three-layer-study")
    foundation = workflow.stage("study-foundation")
    assert foundation.clarification.max_turns == 4
    assert [decision.id for decision in foundation.critical_decisions] == [
        "focal-question",
        "simulation-boundary",
        "comparison-objective",
        "owners-and-sources",
    ]
    assert all(
        decision.target_paths for decision in foundation.critical_decisions if decision.required
    )
    assert all(1 <= stage.clarification.max_turns <= 8 for stage in workflow.stages)


def test_invalid_critical_decision_configuration_is_rejected(tmp_path: Path) -> None:
    package = tmp_path / "critical-study"
    (package / "stages").mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: critical-study\nversion: '1.0'\ntitle: critical\n"
        "stages: [foundation]\ninstructions: instructions.md\n"
    )
    (package / "instructions.md").write_text("instructions")
    stage_path = package / "stages" / "foundation.yaml"
    stage_path.write_text(
        "id: foundation\ntitle: Foundation\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['/study']\nclarification: {max_turns: 0}\n"
        "critical_decisions: []\n"
    )
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())

    stage_path.write_text(
        "id: foundation\ntitle: Foundation\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['/study']\nclarification: {max_turns: 4}\n"
        "critical_decisions:\n"
        "  - id: actor-state\n"
        "    question: Which actor state matters?\n"
        "    rationale: It changes executable state.\n"
        "    target_paths: ['/domain/states']\n"
        "    required: true\n"
    )
    with pytest.raises(ValueError, match="outside owned paths"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_duplicate_stages_are_rejected(tmp_path: Path) -> None:
    package = tmp_path / "dup-study" / "stages"
    package.mkdir(parents=True)
    (tmp_path / "dup-study" / "workflow.yaml").write_text(
        "id: dup-study\nversion: '1.0'\ntitle: dup\nstages: [a, a]\n"
    )
    for name in ("a", "b"):
        (package / f"{name}.yaml").write_text(
            "id: a\ntitle: A\nopening_question: q\ninstructions: i\nowned_paths: ['/study']\n"
        )
    with pytest.raises(ValueError, match="duplicate stage"):
        WorkflowRegistry(tmp_path / "dup-study", checklist_ids=known_checklist_ids())


def test_missing_stage_and_template_files_are_rejected(tmp_path: Path) -> None:
    package = tmp_path / "bad-study"
    (package / "stages").mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: bad-study\nversion: '1.0'\ntitle: bad\nstages: [missing]\n"
        "instructions: instructions.md\n"
    )
    (package / "instructions.md").write_text("hi")
    with pytest.raises(ValueError, match="missing stage file"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())
    # Present stage but missing template file.
    (package / "stages" / "missing.yaml").write_text(
        "id: missing\ntitle: M\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['/study']\ntemplates: {study: no-such.yaml}\n"
    )
    with pytest.raises(ValueError, match="missing template"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_unknown_checklist_ids_are_rejected(tmp_path: Path) -> None:
    package = tmp_path / "check-study"
    (package / "stages").mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: check-study\nversion: '1.0'\ntitle: check\nstages: [s1]\n"
        "instructions: instructions.md\n"
    )
    (package / "instructions.md").write_text("hi")
    (package / "stages" / "s1.yaml").write_text(
        "id: s1\ntitle: S\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['/study']\nchecklist_items: [no-such-item]\n"
    )
    with pytest.raises(ValueError, match="unknown checklist"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_invalid_owned_paths_and_completion_rules_are_rejected(tmp_path: Path) -> None:
    package = tmp_path / "rule-study"
    (package / "stages").mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: rule-study\nversion: '1.0'\ntitle: rule\nstages: [s1]\ninstructions: instructions.md\n"
    )
    (package / "instructions.md").write_text("hi")
    (package / "stages" / "s1.yaml").write_text(
        "id: s1\ntitle: S\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['study']\ncompletion_rules: [{rule: run_arbitrary_code}]\n"
    )
    with pytest.raises(ValueError, match="owned path"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())
    (package / "stages" / "s1.yaml").write_text(
        "id: s1\ntitle: S\nopening_question: q\ninstructions: i\n"
        "owned_paths: ['/study']\ncompletion_rules: [{rule: run_arbitrary_code}]\n"
    )
    with pytest.raises(ValueError, match="unknown completion rule"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_invalidation_targets_must_be_declared_sections(tmp_path: Path) -> None:
    package = tmp_path / "inv-study"
    (package / "stages").mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: inv-study\nversion: '1.0'\ntitle: inv\nstages: [s1]\ninstructions: instructions.md\n"
        "invalidation:\n  openness.processes: [no.such.section]\n"
    )
    (package / "instructions.md").write_text("hi")
    (package / "stages" / "s1.yaml").write_text(
        "id: s1\ntitle: S\nopening_question: q\ninstructions: i\nowned_paths: ['/study']\n"
    )
    with pytest.raises(ValueError, match="undeclared invalidation"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_definitions_are_immutable_copies(registry: WorkflowRegistry) -> None:
    workflow = registry.get("three-layer-study")
    workflow.stages[0].owned_paths = ("/hacked",)
    again = registry.get("three-layer-study")
    assert "/study" in again.stages[0].owned_paths


def _write_dependency_workflow(tmp_path: Path, dependencies: dict[str, list[str]]) -> Path:
    package = tmp_path / "dependency-study"
    stages = package / "stages"
    stages.mkdir(parents=True)
    (package / "workflow.yaml").write_text(
        "id: dependency-study\nversion: '1.0'\ntitle: dependency\n"
        "stages: [a, b, c]\ninstructions: instructions.md\n"
    )
    (package / "instructions.md").write_text("instructions")
    for stage_id in ("a", "b", "c"):
        depends_on = dependencies.get(stage_id, [])
        (stages / f"{stage_id}.yaml").write_text(
            f"id: {stage_id}\ntitle: {stage_id}\nopening_question: q\n"
            f"instructions: i\nowned_paths: ['/study']\ndepends_on: {depends_on!r}\n"
        )
    return package


def test_unknown_stage_dependencies_are_rejected(tmp_path: Path) -> None:
    package = _write_dependency_workflow(tmp_path, {"b": ["missing"]})
    with pytest.raises(ValueError, match="unknown dependency"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())


def test_dependency_cycles_are_rejected(tmp_path: Path) -> None:
    package = _write_dependency_workflow(tmp_path, {"a": ["c"], "c": ["a"]})
    with pytest.raises(ValueError, match="dependency cycle"):
        WorkflowRegistry(package, checklist_ids=known_checklist_ids())

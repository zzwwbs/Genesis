"""Task 3: typed assistant-evaluation and specification-patch contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from genesis.elicitation import (
    AssistantEvaluation,
    SpecificationPatch,
    SpecificationPatchOperation,
    WorkflowRegistry,
    known_checklist_ids,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def _foundation_stage():
    return (
        WorkflowRegistry(WORKFLOWS, checklist_ids=known_checklist_ids())
        .get("three-layer-study")
        .stage("study-foundation")
    )


def test_clarification_requires_three_suggestions() -> None:
    with pytest.raises(ValidationError):
        AssistantEvaluation.model_validate(
            {
                "status": "needs_clarification",
                "summary": "Timing is unclear.",
                "evidence": [],
                "assumptions": [],
                "ambiguities": [
                    {
                        "id": "timing",
                        "question": "When?",
                        "reason": "Causal order",
                        "consequential": True,
                        "suggestions": [{"label": "Now", "value": "Now"}],
                    }
                ],
                "checklist_updates": [],
            }
        )


def test_clarification_requires_at_least_one_ambiguity() -> None:
    with pytest.raises(ValidationError, match="requires at least one ambiguity"):
        AssistantEvaluation.model_validate(
            {
                "status": "needs_clarification",
                "summary": "Nothing clear yet.",
                "ambiguities": [],
            }
        )


def test_ready_to_draft_forbids_consequential_ambiguity() -> None:
    with pytest.raises(ValidationError, match="consequential ambiguity"):
        AssistantEvaluation.model_validate(
            {
                "status": "ready_to_draft",
                "summary": "Ready.",
                "ambiguities": [
                    {
                        "id": "timing",
                        "question": "When?",
                        "reason": "Causal order",
                        "consequential": True,
                        "suggestions": [
                            {"label": "A", "value": "A"},
                            {"label": "B", "value": "B"},
                            {"label": "C", "value": "C"},
                        ],
                    }
                ],
            }
        )


def test_ready_to_draft_allows_documented_assumptions() -> None:
    """Assumptions on readiness are documented closures (patch path requires
    'evidence or listed assumptions'); genuine unresolved status is carried by
    ambiguities, which stay consequential-blocked."""
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "ready_to_draft",
            "summary": "Ready, with an explicitly assumed population.",
            "assumptions": ["The observed population is representative."],
            "ambiguities": [],
        }
    )
    assert evaluation.assumptions == ("The observed population is representative.",)


def test_ready_to_draft_still_forbids_consequential_ambiguity() -> None:
    with pytest.raises(ValidationError, match="consequential"):
        AssistantEvaluation.model_validate(
            {
                "status": "ready_to_draft",
                "summary": "Not actually ready.",
                "assumptions": [],
                "ambiguities": [
                    {
                        "id": "scope",
                        "question": "Which platform?",
                        "reason": "Binds the setting.",
                        "consequential": True,
                        "suggestions": [],
                        "target_paths": [],
                        "decision_id": None,
                    }
                ],
            }
        )


def test_valid_clarification_yields_one_question_and_three_suggestions() -> None:
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "needs_clarification",
            "summary": "Observation timing matters.",
            "evidence": [{"claim": "Peers observe behavior.", "source_turns": [1]}],
            "assumptions": [],
            "ambiguities": [
                {
                    "id": "information-delay",
                    "question": "When does an agent observe a neighbor's action?",
                    "reason": "Timing changes the causal sequence.",
                    "consequential": True,
                    "suggestions": [
                        {"label": "Immediate observation", "value": "Same round."},
                        {"label": "Next-round observation", "value": "Following round."},
                        {"label": "Aggregated observation", "value": "Summary after each round."},
                    ],
                }
            ],
            "checklist_updates": [
                {"item_id": "l1-informational-position", "proposed_status": "partial"}
            ],
        }
    )
    evaluation.validate_turn_references({1, 2})
    assert evaluation.next_question == "When does an agent observe a neighbor's action?"
    assert len(evaluation.next_suggestions) == 3


def test_evidence_requires_valid_turn_references() -> None:
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "ready_to_draft",
            "summary": "Ready.",
            "evidence": [{"claim": "Agents observe neighbors.", "source_turns": [9]}],
            "ambiguities": [],
        }
    )
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        evaluation.validate_turn_references({1, 2})


def test_forbidden_self_approval_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AssistantEvaluation.model_validate(
            {
                "status": "ready_to_draft",
                "summary": "Ready.",
                "approved": True,  # extra field
                "ambiguities": [],
            }
        )


def test_unsupported_operations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SpecificationPatchOperation.model_validate(
            {"op": "update", "path": "/openness/processes/-", "value": {}}
        )


def test_patch_owned_paths_and_base_version_are_validated() -> None:
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 3,
            "operations": [
                {
                    "op": "add",
                    "path": "/openness/processes/-",
                    "value": {"id": "compose"},
                }
            ],
            "evidence": [{"target": "/openness/processes/0/id", "source_turns": [2, 4]}],
            "affected_checklist_items": ["l1-openness-need"],
        }
    )
    patch.validate_owned_paths(("/openness", "/prompts"))
    patch.validate_base_version(3)
    with pytest.raises(ValueError, match="PATCH_PATH_FORBIDDEN"):
        patch.validate_owned_paths(("/theory",))
    with pytest.raises(ValueError, match="PATCH_BASE_STALE"):
        patch.validate_base_version(4)
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        patch.validate_turn_references({1})


def test_partial_extras_are_forbidden_in_patch() -> None:
    with pytest.raises(ValidationError):
        SpecificationPatch.model_validate(
            {
                "stage_id": "openness",
                "base_specification_version": 1,
                "operations": [],
                "automatically_approved": True,
            }
        )


def test_consequential_ambiguity_with_empty_suggestions_is_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly three suggestions"):
        AssistantEvaluation.model_validate(
            {
                "status": "needs_clarification",
                "summary": "Needs a decision.",
                "ambiguities": [
                    {
                        "id": "timing",
                        "question": "When?",
                        "reason": "Causal order",
                        "consequential": True,
                        "suggestions": [],
                    }
                ],
            }
        )


def test_operations_require_evidence_or_assumptions() -> None:
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 1,
            "operations": [{"op": "add", "path": "/openness/processes/-", "value": {"id": "p"}}],
            "evidence": [],
            "assumptions": [],
        }
    )
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        patch.validate_evidence()


def test_evidence_targets_must_match_operations() -> None:
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 1,
            "operations": [{"op": "add", "path": "/openness/processes/-", "value": {"id": "p"}}],
            "evidence": [{"target": "/theory/none", "source_turns": [1]}],
            "assumptions": [],
        }
    )
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        patch.validate_evidence()
    valid = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 1,
            "operations": [{"op": "add", "path": "/openness/processes/-", "value": {"id": "p"}}],
            "evidence": [{"target": "/openness/processes/-", "source_turns": [1]}],
            "assumptions": [],
        }
    )
    valid.validate_evidence()


def test_every_operation_requires_evidence_or_a_scoped_assumption() -> None:
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 1,
            "operations": [
                {"op": "replace", "path": "/openness/processes", "value": []},
                {"op": "replace", "path": "/prompts/compose", "value": "Compose."},
            ],
            "evidence": [{"target": "/openness/processes", "source_turns": [1]}],
            "assumptions": [],
        }
    )
    with pytest.raises(ValueError, match="uncovered operation.*prompts/compose"):
        patch.validate_evidence()


def test_operation_scoped_assumption_can_cover_an_operation() -> None:
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 1,
            "operations": [
                {"op": "replace", "path": "/openness/processes", "value": []},
                {"op": "replace", "path": "/prompts/compose", "value": "Compose."},
            ],
            "evidence": [{"target": "/openness/processes", "source_turns": [1]}],
            "assumptions": [
                {
                    "target": "/prompts/compose",
                    "statement": "Use a concise initial prompt pending researcher refinement.",
                }
            ],
        }
    )
    patch.validate_evidence()


def test_fenced_and_prosy_evaluation_responses_are_tolerated() -> None:
    """Transport leniency: markdown fences or prose around JSON are stripped."""
    from genesis.elicitation import ElicitationAssistant

    assistant = ElicitationAssistant()
    fenced = """```json
{"status": "ready_to_draft", "summary": "Ready.", "evidence": [], "ambiguities": []}
```"""
    evaluation = assistant.parse_evaluation(fenced, {1})
    assert evaluation.status == "ready_to_draft"
    prosy = (
        "Here is my evaluation:\n"
        '{"status": "needs_clarification", "summary": "One thing.", "evidence": [], '
        '"ambiguities": [{"id": "timing", "question": "When?", "reason": "Order", '
        '"consequential": true, "suggestions": [{"label": "A", "value": "1"}, '
        '{"label": "B", "value": "2"}, {"label": "C", "value": "3"}]}]}'
    )
    clarification = assistant.parse_evaluation(prosy, {1})
    assert clarification.status == "needs_clarification"
    with pytest.raises(ValueError, match="ASSISTANT_OUTPUT_INVALID"):
        assistant.parse_evaluation("no json at all", {1})


def test_schema_envelope_echo_is_tolerated() -> None:
    """Models that echo $schema/$id headers still pass contract validation."""
    from genesis.elicitation import ElicitationAssistant

    assistant = ElicitationAssistant()
    echoed = assistant.parse_evaluation(
        '{"$schema": "https://json-schema.org/draft/2020-12/schema", '
        '"$id": "assistant-evaluation", "status": "ready_to_draft", '
        '"summary": "Ready.", "evidence": [], "ambiguities": []}',
        {1},
    )
    assert echoed.status == "ready_to_draft"


def test_ambiguity_without_consequential_field_defaults_to_true() -> None:
    """Models omitting the consequential flag still produce valid clarifications."""
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "needs_clarification",
            "summary": "Sampling needs a decision.",
            "evidence": [],
            "ambiguities": [
                {
                    "id": "platform-sampling",
                    "question": "Which users are sampled?",
                    "reason": "Sampling changes the population.",
                    "suggestions": [
                        {"label": "A", "value": "All users"},
                        {"label": "B", "value": "Active users"},
                        {"label": "C", "value": "New users"},
                    ],
                }
            ],
        }
    )
    assert evaluation.ambiguities[0].consequential is True
    assert evaluation.next_question == "Which users are sampled?"
    assert len(evaluation.next_suggestions) == 3


def test_decision_grounded_clarification_validates_against_stage() -> None:
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "needs_clarification",
            "summary": "The executable boundary is still unclear.",
            "decision_coverage": [
                {
                    "decision_id": "focal-question",
                    "status": "covered",
                    "evidence_turns": [1],
                },
                {"decision_id": "simulation-boundary", "status": "unresolved"},
                {"decision_id": "comparison-objective", "status": "unresolved"},
            ],
            "ambiguities": [
                {
                    "id": "simulation-boundary-detail",
                    "decision_id": "simulation-boundary",
                    "target_paths": ["/study/extensions/genesis.elicitation/simulation-boundary"],
                    "question": "Which actors must the simulation represent?",
                    "reason": "Actor scope changes executable population configuration.",
                    "consequential": True,
                    "suggestions": [
                        {"label": "Creators", "value": "Creators only."},
                        {"label": "Creators and users", "value": "Creators and users."},
                        {
                            "label": "Full platform",
                            "value": "Creators, users, and a platform.",
                        },
                    ],
                }
            ],
        }
    )
    evaluation.validate_decisions(_foundation_stage(), {1})


def test_undeclared_or_optional_decision_cannot_block_clarification() -> None:
    undeclared = AssistantEvaluation.model_validate(
        {
            "status": "needs_clarification",
            "summary": "Background history is unclear.",
            "decision_coverage": [],
            "ambiguities": [
                {
                    "id": "platform-history",
                    "decision_id": "platform-history",
                    "target_paths": ["/study/description"],
                    "question": "What is the complete history of the platform?",
                    "reason": "It provides background.",
                    "consequential": True,
                    "suggestions": [
                        {"label": "Short", "value": "Short history."},
                        {"label": "Long", "value": "Long history."},
                        {"label": "None", "value": "No history."},
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="ASSISTANT_DECISION_INVALID.*platform-history"):
        undeclared.validate_decisions(_foundation_stage(), {1})

    optional = undeclared.model_copy(deep=True)
    optional.ambiguities[0].decision_id = "owners-and-sources"
    optional.ambiguities[0].target_paths = ("/study/owners",)
    with pytest.raises(ValueError, match="optional decision"):
        optional.validate_decisions(_foundation_stage(), {1})


def test_ready_evaluation_requires_coverage_for_every_required_decision() -> None:
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "ready_to_draft",
            "summary": "Ready.",
            "decision_coverage": [
                {
                    "decision_id": "focal-question",
                    "status": "covered",
                    "evidence_turns": [1],
                }
            ],
            "ambiguities": [],
        }
    )
    with pytest.raises(ValueError, match="required decisions remain unresolved"):
        evaluation.validate_decisions(_foundation_stage(), {1})


def test_evaluation_cannot_repeat_a_covered_decision_while_another_is_unresolved() -> None:
    evaluation = AssistantEvaluation.model_validate(
        {
            "status": "needs_clarification",
            "summary": "The assistant repeats a settled question.",
            "decision_coverage": [
                {
                    "decision_id": "focal-question",
                    "status": "covered",
                    "evidence_turns": [1],
                },
                {
                    "decision_id": "simulation-boundary",
                    "status": "covered",
                    "evidence_turns": [1],
                },
                {
                    "decision_id": "comparison-objective",
                    "status": "unresolved",
                },
            ],
            "ambiguities": [
                {
                    "id": "boundary-again",
                    "decision_id": "simulation-boundary",
                    "target_paths": ["/study/extensions/genesis.elicitation/simulation-boundary"],
                    "question": "Please clarify the simulation boundary again.",
                    "reason": "A repeated question should not prolong elicitation.",
                    "consequential": True,
                    "suggestions": [
                        {"label": "A", "value": "A"},
                        {"label": "B", "value": "B"},
                        {"label": "C", "value": "C"},
                    ],
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="already covered"):
        evaluation.validate_decisions(_foundation_stage(), {1})

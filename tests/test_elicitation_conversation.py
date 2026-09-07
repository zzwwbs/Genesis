"""Task 4: conversational clarification loop with a scripted assistant."""

from __future__ import annotations

import json as _json
import re
from pathlib import Path

import pytest

from genesis.elicitation import ElicitationAssistant, ElicitationTurn, WorkflowRegistry
from genesis.providers import ProviderCapabilities, ProviderResponse
from genesis.service import GenesisService

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
FOUNDATION_DECISIONS = (
    "focal-question",
    "simulation-boundary",
    "comparison-objective",
)


def _pending_turn(prompt: str) -> int:
    match = re.search(r"Pending researcher turn: (\d+)", prompt)
    return int(match.group(1)) if match else 1


def _foundation_coverage(turn_id: int, *, ready: bool) -> list[dict]:
    return [
        {
            "decision_id": decision_id,
            "status": ("covered" if ready or decision_id == "focal-question" else "unresolved"),
            "evidence_turns": ([turn_id] if ready or decision_id == "focal-question" else []),
        }
        for decision_id in FOUNDATION_DECISIONS
    ]


def test_assistant_and_draft_prompts_include_only_current_stage_turns(
    service: GenesisService,
) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    stored = service._elicitation_engine.require_session(session["session_id"])
    stored.current_stage = "openness"
    stored.turns = [
        ElicitationTurn(
            id=1,
            stage_id="study-foundation",
            question="Foundation?",
            answer="FOUNDATION-ONLY-ANSWER",
        ),
        ElicitationTurn(
            id=2,
            stage_id="openness",
            question="Openness?",
            answer="CURRENT-STAGE-ANSWER",
        ),
    ]
    workflow = WorkflowRegistry(WORKFLOWS).get("three-layer-study")
    stage = workflow.stage("openness")
    evaluation_prompt = ElicitationAssistant().assemble_evaluation_request(
        workflow,
        stage,
        stored,
        pending_answer="NEW-CURRENT-STAGE-ANSWER",
        clarification_state={
            "turns_used": 1,
            "turns_remaining": 5,
            "decision_coverage": {"open-processes": "covered"},
        },
    )
    draft_prompt = service._assemble_draft_request(stored, workflow, stage)
    for prompt in (evaluation_prompt, draft_prompt):
        assert "CURRENT-STAGE-ANSWER" in prompt
        assert "FOUNDATION-ONLY-ANSWER" not in prompt
    assert "## Simulation-critical decisions" in evaluation_prompt
    assert "open-processes [required]" in evaluation_prompt
    assert "Turns remaining: 5" in evaluation_prompt
    assert "Pending researcher turn: 3" in evaluation_prompt
    assert "turn 2 question: Openness?" in evaluation_prompt


class ScriptedProvider:
    provider = "scripted"

    def __init__(self, **_kwargs) -> None:
        self._observed_prompts: list[str] = []

    def generate(self, request) -> ProviderResponse:
        prompt = request.prompt
        self._observed_prompts.append(prompt)
        if "Evaluate the researcher's most recent answer" in prompt:
            pending_turn = _pending_turn(prompt)
            if "Messages are observed next round." in prompt:
                text = _json.dumps(
                    {
                        "status": "ready_to_draft",
                        "summary": "Observation timing is settled.",
                        "evidence": [
                            {
                                "claim": "Messages are observed next round.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _foundation_coverage(pending_turn, ready=True),
                        "ambiguities": [],
                    }
                )
            else:
                text = _json.dumps(
                    {
                        "status": "needs_clarification",
                        "summary": "Observation timing is unclear.",
                        "evidence": [
                            {
                                "claim": "Agents may observe neighbors.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _foundation_coverage(pending_turn, ready=False),
                        "ambiguities": [
                            {
                                "id": "information-delay",
                                "decision_id": "simulation-boundary",
                                "target_paths": [
                                    "/study/extensions/genesis.elicitation/simulation-boundary"
                                ],
                                "question": "When does an agent observe a neighbor's action?",
                                "reason": "Timing changes the causal sequence.",
                                "consequential": True,
                                "suggestions": [
                                    {"label": "Immediate observation", "value": "Same round."},
                                    {
                                        "label": "Next-round observation",
                                        "value": "Messages are observed next round.",
                                    },
                                    {
                                        "label": "Aggregated observation",
                                        "value": "Summary after each round.",
                                    },
                                ],
                            }
                        ],
                    }
                )
        else:
            base_version = 1
            base_match = re.search(r"Base specification version: (\d+)", prompt)
            if base_match:
                base_version = int(base_match.group(1))
            text = _json.dumps(
                {
                    "stage_id": "study-foundation",
                    "base_specification_version": base_version,
                    "operations": [
                        {
                            "op": "replace",
                            "path": "/study/title",
                            "value": "Cooperation study (revised)",
                        }
                    ],
                    "evidence": [{"target": "/study/title", "source_turns": [2]}],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "affected_checklist_items": [],
                }
            )
        return ProviderResponse(
            text, self.provider, "scripted-model", "req-1", parsed=_json.loads(text)
        )


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ScriptedProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "scripted-model",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    return genesis


def test_free_form_answer_remains_researcher_evidence(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    assert session["current_stage"] == "study-foundation"
    first = service.submit_elicitation_message(session["session_id"], "Agents choose what to say.")
    assert len(first["current_suggestions"]) == 3
    second = service.submit_elicitation_message(
        session["session_id"],
        "Messages are observed next round.",
        response_mode="free_form",
    )
    # The free-form answer is recorded verbatim as researcher evidence.
    turns = second["turns"]
    assert turns[-1]["answer"] == "Messages are observed next round."
    assert turns[-1]["response_mode"] == "free_form"
    # No canonical file mutation happened through conversation turns.
    assert service.get_specification("study")["version"] == 1
    assert second["status"] == "awaiting_approval"


def test_suggestion_selection_is_researcher_submission(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
            "base_specification_version": 1,
        }
    )
    result = service.submit_elicitation_message(session["session_id"], "Agents choose what to say.")
    suggestions = result["current_suggestions"]
    assert len(suggestions) == 3
    # Selecting the second suggestion copies its value into the answer.
    picked = service.submit_elicitation_message(
        session["session_id"],
        suggestions[1]["value"],
        suggestion_index=1,
    )
    assert picked["turns"][-1]["response_mode"] == "suggested"
    assert picked["turns"][-1]["answer"] == suggestions[1]["value"]


def test_provider_failure_leaves_session_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingProvider(ScriptedProvider):
        def generate(self, request) -> ProviderResponse:
            raise ConnectionError("network down")

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ExplodingProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        with pytest.raises(ValueError, match="ASSISTANT_UNAVAILABLE"):
            genesis.submit_elicitation_message(session["session_id"], "An answer.")
        # Still awaiting the same answer; no turn recorded and no version bump.
        again = genesis.get_elicitation(session["session_id"])
        assert again["status"] == "awaiting_answer"
        assert again["turns"] == []
        assert genesis.get_specification("study")["version"] == 1
    finally:
        genesis.close()


def test_malformed_assistant_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class GarbageProvider(ScriptedProvider):
        def generate(self, request) -> ProviderResponse:
            return ProviderResponse("not json at all", self.provider, "m", "req-1")

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", GarbageProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        with pytest.raises(ValueError, match="ASSISTANT_OUTPUT_INVALID"):
            genesis.submit_elicitation_message(session["session_id"], "An answer.")
    finally:
        genesis.close()


def _repair_test_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responses: list[str]
) -> tuple[GenesisService, object]:
    class RepairSequenceProvider:
        provider = "repair-sequence"

        def __init__(self) -> None:
            self.responses = list(responses)
            self.requests = []

        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(structured_output=True)

        def generate(self, request) -> ProviderResponse:
            self.requests.append(request)
            text = self.responses.pop(0)
            return ProviderResponse(text, self.provider, "m", f"req-{len(self.requests)}")

    provider = RepairSequenceProvider()
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", lambda **_kwargs: provider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    return genesis, provider


def _ready_evaluation() -> str:
    return _json.dumps(
        {
            "status": "ready_to_draft",
            "summary": "The foundation is sufficiently specified.",
            "evidence": [{"claim": "Researcher supplied the scope.", "source_turns": [1]}],
            "decision_coverage": _foundation_coverage(1, ready=True),
            "ambiguities": [],
        }
    )


def _decision_evaluation(
    *,
    status: str,
    focal: str = "covered",
    boundary: str = "covered",
    comparison: str = "covered",
    evidence_turns: dict[str, list[int]] | None = None,
    ambiguity_decision: str | None = None,
    question: str | None = None,
) -> str:
    evidence_turns = evidence_turns or {
        "focal-question": [1],
        "simulation-boundary": [1],
        "comparison-objective": [1],
    }
    statuses = {
        "focal-question": focal,
        "simulation-boundary": boundary,
        "comparison-objective": comparison,
    }
    ambiguities = []
    if ambiguity_decision is not None:
        targets = {
            "focal-question": ["/study/description"],
            "simulation-boundary": ["/study/extensions/genesis.elicitation/simulation-boundary"],
            "comparison-objective": ["/study/extensions/genesis.elicitation/comparison-objective"],
        }
        ambiguities = [
            {
                "id": f"{ambiguity_decision}-detail",
                "decision_id": ambiguity_decision,
                "target_paths": targets[ambiguity_decision],
                "question": question or f"Please clarify {ambiguity_decision}.",
                "reason": "It changes executable study configuration.",
                "consequential": True,
                "suggestions": [
                    {"label": "A", "value": "Option A."},
                    {"label": "B", "value": "Option B."},
                    {"label": "C", "value": "Option C."},
                ],
            }
        ]
    return _json.dumps(
        {
            "status": status,
            "summary": "Decision-grounded evaluation.",
            "evidence": [],
            "decision_coverage": [
                {
                    "decision_id": decision_id,
                    "status": decision_status,
                    "evidence_turns": (
                        evidence_turns.get(decision_id, []) if decision_status == "covered" else []
                    ),
                }
                for decision_id, decision_status in statuses.items()
            ],
            "ambiguities": ambiguities,
        }
    )


def test_required_coverage_forces_readiness_even_if_model_keeps_clarifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _decision_evaluation(
        status="needs_clarification",
        ambiguity_decision="comparison-objective",
    )
    genesis, _provider = _repair_test_service(tmp_path, monkeypatch, [response])
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        result = genesis.submit_elicitation_message(
            session["session_id"],
            "Study governance effects across actors and experimental conditions.",
        )
        assert result["status"] == "awaiting_approval"
        assert result["stages"]["study-foundation"]["status"] == "draft_ready"
        assert result["clarification"]["remaining_decisions"] == []
    finally:
        genesis.close()


def test_model_cannot_declare_readiness_with_required_decisions_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = _decision_evaluation(
        status="ready_to_draft",
        boundary="unresolved",
        comparison="unresolved",
        evidence_turns={"focal-question": [1]},
    )
    genesis, _provider = _repair_test_service(tmp_path, monkeypatch, [invalid, invalid, invalid])
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        with pytest.raises(ValueError, match="required decisions remain unresolved"):
            genesis.submit_elicitation_message(session["session_id"], "A narrow answer.")
        unchanged = genesis.get_elicitation(session["session_id"])
        assert unchanged["turns"] == []
        assert unchanged["clarification"]["turns_remaining"] == 4
    finally:
        genesis.close()


def test_question_limit_enters_fixed_resolution_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = [
        _decision_evaluation(
            status="needs_clarification",
            boundary="unresolved",
            comparison="unresolved",
            evidence_turns={"focal-question": [1]},
            ambiguity_decision="simulation-boundary",
        ),
        _decision_evaluation(
            status="needs_clarification",
            comparison="unresolved",
            evidence_turns={"focal-question": [1], "simulation-boundary": [2]},
            ambiguity_decision="comparison-objective",
        ),
        _decision_evaluation(
            status="needs_clarification",
            comparison="unresolved",
            evidence_turns={"focal-question": [1], "simulation-boundary": [2]},
            ambiguity_decision="comparison-objective",
            question="Please clarify the comparison objective (scope of conditions).",
        ),
        _json.dumps(
            {
                "stage_id": "study-foundation",
                "base_specification_version": 1,
                "operations": [
                    {"op": "replace", "path": "/study/title", "value": "Draft for review"}
                ],
                "evidence": [{"target": "/study/title", "source_turns": [4]}],
            }
        ),
    ]
    genesis, _provider = _repair_test_service(tmp_path, monkeypatch, responses)
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        for index in range(4):
            session = genesis.submit_elicitation_message(
                session["session_id"], f"Foundation answer {index + 1}."
            )
        assert session["status"] == "awaiting_approval"
        assert session["stages"]["study-foundation"]["review_mode"] is True
        assert session["pending_preview"] is not None
        assert session["clarification"]["turns_remaining"] == 0
        assert session["clarification"]["limit_reached"] is True
        assert session["clarification"]["remaining_decisions"] == ["comparison-objective"]
        assert "Review the draft" in session["current_question"]
        assert session["current_suggestions"] == []
        assert genesis.get_specification("study")["version"] == 1
    finally:
        genesis.close()


def test_truncated_evaluation_is_repaired_and_records_answer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genesis, provider = _repair_test_service(
        tmp_path, monkeypatch, ['{"status": "ready_to_draft"', _ready_evaluation()]
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        result = genesis.submit_elicitation_message(
            session["session_id"], "A bounded platform-governance study."
        )
        assert result["status"] == "awaiting_approval"
        assert len(result["turns"]) == 1
        assert [item["status"] for item in result["last_assistant_attempts"]] == [
            "invalid",
            "valid",
        ]
        assert len(provider.requests) == 2
        assert "corrected JSON object" in provider.requests[1].prompt
        assert "response_format" in provider.requests[0].parameters
    finally:
        genesis.close()


def test_schema_invalid_evaluation_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = _json.dumps({"status": "ready_to_draft", "summary": "Ready", "extra": 1})
    genesis, provider = _repair_test_service(tmp_path, monkeypatch, [invalid, _ready_evaluation()])
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        result = genesis.submit_elicitation_message(session["session_id"], "A scoped study.")
        assert len(result["turns"]) == 1
        assert len(provider.requests) == 2
        assert "extra" in provider.requests[1].prompt
    finally:
        genesis.close()


def test_exhausted_repairs_leave_answer_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genesis, provider = _repair_test_service(
        tmp_path,
        monkeypatch,
        ["not json", "still not json", '{"status":', _ready_evaluation()],
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        with pytest.raises(ValueError, match="after 3 attempts"):
            genesis.submit_elicitation_message(session["session_id"], "A scoped study.")
        failed = genesis.get_elicitation(session["session_id"])
        assert failed["status"] == "awaiting_answer"
        assert failed["turns"] == []
        assert len(failed["last_assistant_attempts"]) == 3

        recovered = genesis.submit_elicitation_message(session["session_id"], "A scoped study.")
        assert len(recovered["turns"]) == 1
    finally:
        genesis.close()


def test_truncated_patch_is_repaired_without_duplicating_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch = _json.dumps(
        {
            "stage_id": "study-foundation",
            "base_specification_version": 1,
            "operations": [{"op": "replace", "path": "/study/title", "value": "Scoped study"}],
            "evidence": [{"target": "/study/title", "source_turns": [1]}],
            "assumptions": [],
            "unresolved_questions": [],
            "affected_checklist_items": [],
        }
    )
    genesis, provider = _repair_test_service(
        tmp_path, monkeypatch, [_ready_evaluation(), '{"stage_id":', patch]
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        genesis.submit_elicitation_message(session["session_id"], "A scoped study.")
        result = genesis.draft_elicitation(session["session_id"])
        assert len(result["turns"]) == 1
        assert result["pending_preview"] is None
        assert [item["status"] for item in result["last_assistant_attempts"]] == [
            "invalid",
            "valid",
        ]
        assert len(provider.requests) == 3
    finally:
        genesis.close()


def test_draft_request_validates_patch_contract(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    service.submit_elicitation_message(session["session_id"], "Agents choose what to say.")
    session = service.get_elicitation(session["session_id"])
    service.submit_elicitation_message(
        session["session_id"], "Messages are observed next round.", response_mode="free_form"
    )
    session = service.get_elicitation(session["session_id"])
    assert session["status"] == "awaiting_approval"
    drafted = service.draft_elicitation(session["session_id"])
    assert drafted["status"] == "awaiting_approval"
    assert drafted["base_specification_version"] == 1


def test_cancel_elicitation_discards_session(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    result = service.cancel_elicitation(session["session_id"])
    assert result["status"] == "cancelled"
    with pytest.raises(KeyError, match="ELICITATION_SESSION_EXPIRED"):
        service.get_elicitation(session["session_id"])


def test_elicitation_provider_honors_saved_timeout(service: GenesisService, monkeypatch) -> None:
    service.create_model_profile(
        {
            "id": "slow-assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
            "timeout": 180,
        }
    )
    captured = {}

    def construct(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", construct)
    service._elicitation_provider("slow-assistant")
    assert captured["timeout"] == 180


def test_workflow_projection_omits_private_instructions(service: GenesisService) -> None:
    workflows = service.list_elicitation_workflows()
    assert [workflow["id"] for workflow in workflows] == ["three-layer-study"]
    projection = service.get_elicitation_workflow("three-layer-study")
    stage_ids = [stage["id"] for stage in projection["stages"]]
    assert stage_ids == [
        "study-foundation",
        "openness",
        "theory",
        "domain",
        "experiment-design",
    ]
    assert "instructions" not in projection
    assert "instructions.md" != projection["stages"][0].get("opening_question")


def test_draft_rejects_unknown_checklist_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown assistant checklist identifiers are never silently discarded."""

    class TopicEchoProvider(ScriptedProvider):
        def generate(self, request):
            import re as _re

            prompt = request.prompt
            if "Evaluate the researcher's most recent answer" in prompt:
                return super().generate(request)
            stage_id = "study-foundation"
            match = _re.search(r"Stage: .*\((.*)\)", prompt)
            if match:
                stage_id = match.group(1)
            text = _json.dumps(
                {
                    "stage_id": stage_id,
                    "base_specification_version": 1,
                    "operations": [
                        {"op": "replace", "path": "/study/title", "value": "T (revised)"}
                    ],
                    "evidence": [{"target": "/study/title", "source_turns": [1]}],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "affected_checklist_items": [
                        "unknown-foundation-item",
                        "l1-traceability",
                    ],
                }
            )
            return ProviderResponse(text, self.provider, "m", "req-1", parsed=_json.loads(text))

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", TopicEchoProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        genesis.submit_elicitation_message(session["session_id"], "A foundation answer.")
        session = genesis.get_elicitation(session["session_id"])
        genesis.submit_elicitation_message(
            session["session_id"],
            "Messages are observed next round.",
            response_mode="free_form",
        )
        with pytest.raises(ValueError, match="ASSISTANT_OUTPUT_INVALID.*checklist"):
            genesis.draft_elicitation(session["session_id"])
    finally:
        genesis.close()


def test_repeated_questions_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The assistant may not ask the same clarifying question twice."""

    class RepeatQuestionProvider(ScriptedProvider):
        def generate(self, request):

            prompt = request.prompt
            if "Evaluate the researcher's most recent answer" not in prompt:
                return super().generate(request)
            text = _json.dumps(
                {
                    "status": "needs_clarification",
                    "summary": "Still unclear.",
                    "evidence": [],
                    "ambiguities": [
                        {
                            "id": "timing",
                            "decision_id": "focal-question",
                            "target_paths": ["/study/description"],
                            "question": "Which focal question should the study answer?",
                            "reason": "Scope",
                            "consequential": True,
                            "suggestions": [
                                {"label": "A", "value": "a"},
                                {"label": "B", "value": "b"},
                                {"label": "C", "value": "c"},
                            ],
                        }
                    ],
                    "decision_coverage": [
                        {"decision_id": "focal-question", "status": "unresolved"}
                    ],
                }
            )
            return ProviderResponse(text, self.provider, "m", "req-1", parsed=_json.loads(text))

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RepeatQuestionProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        genesis.submit_elicitation_message(session["session_id"], "First answer.")
        with pytest.raises(ValueError, match="already asked"):
            genesis.submit_elicitation_message(session["session_id"], "Second answer.")
    finally:
        genesis.close()

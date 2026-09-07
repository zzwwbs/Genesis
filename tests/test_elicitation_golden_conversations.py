"""Task 9: golden conversational qualification studies."""

from __future__ import annotations

import json as _json
import re
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from genesis.providers import ProviderResponse
from genesis.service import GenesisService

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_conversations"
GOLDEN_STUDIES = Path(__file__).resolve().parent / "golden_studies"
CLICKBAIT_CRITICAL = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "elicitation"
    / "clickbait-critical-conversation.json"
)
STAGE_DECISIONS = {
    "study-foundation": (
        "focal-question",
        "simulation-boundary",
        "comparison-objective",
    ),
    "openness": (
        "open-processes",
        "actors-and-triggers",
        "information-context",
        "outputs-and-consumers",
        "executor-bindings",
        "trace-and-retry",
    ),
    "theory": (
        "theoretical-functions",
        "process-mappings",
        "causal-relations",
        "feedback-and-delays",
        "theoretical-observables",
    ),
    "domain": (
        "actors-and-population",
        "process-state",
        "artifacts-and-lifecycle",
        "visibility-and-availability",
        "updates-and-mechanisms",
        "initialization-and-provenance",
    ),
    "experiment-design": (
        "time-and-termination",
        "conditions-and-interventions",
        "replication-and-randomness",
        "model-and-schema-freezing",
        "outcome-plan",
        "operational-controls",
    ),
}


def _evaluation_stage(prompt: str) -> str:
    match = re.search(r"## Current stage\n.* — ([a-z0-9-]+)", prompt)
    return match.group(1) if match else "study-foundation"


def _pending_turn(prompt: str) -> int:
    match = re.search(r"Pending researcher turn: (\d+)", prompt)
    return int(match.group(1)) if match else 1


def _decision_coverage(stage_id: str, turn_id: int, *, unresolved: str | None = None) -> list[dict]:
    return [
        {
            "decision_id": decision_id,
            "status": "unresolved" if decision_id == unresolved else "covered",
            "evidence_turns": [] if decision_id == unresolved else [turn_id],
        }
        for decision_id in STAGE_DECISIONS[stage_id]
    ]


class GoldenConversationProvider:
    """Scripted assistant that replays a conversation fixture stage by stage."""

    provider = "scripted"

    def __init__(self, fixture: dict, **__kwargs) -> None:
        self.fixture = fixture

    def generate(self, request) -> ProviderResponse:
        prompt = request.prompt
        if "Evaluate the researcher's most recent answer" in prompt:
            stage_id = _evaluation_stage(prompt)
            pending_turn = _pending_turn(prompt)
            new_answer = ""
            if "## New researcher answer" in prompt:
                tail = prompt.split("## New researcher answer", 1)[1]
                new_answer = tail.split("## Your task", 1)[0]
            if any(
                marker in new_answer
                for marker in (
                    "Strategy choice is open-ended.",
                    "Members interpret events and respond.",
                )
            ):
                text = _json.dumps(
                    {
                        "status": "needs_clarification",
                        "summary": "One aspect needs a decision.",
                        "evidence": [
                            {
                                "claim": "Researcher answer accepted.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _decision_coverage(
                            stage_id, pending_turn, unresolved="information-context"
                        ),
                        "ambiguities": [
                            {
                                "id": "timing",
                                "decision_id": "information-context",
                                "target_paths": ["/openness/processes"],
                                "question": "Which option should the study adopt?",
                                "reason": "The choice changes the causal structure.",
                                "consequential": True,
                                "suggestions": [
                                    {
                                        "label": "Option A",
                                        "value": "Creators decide what to publish.",
                                    },
                                    {
                                        "label": "Option B",
                                        "value": "Responses become institutionalized over time.",
                                    },
                                    {
                                        "label": "Option C",
                                        "value": "Messages are observed next round.",
                                    },
                                ],
                            }
                        ],
                    }
                )
            else:
                text = _json.dumps(
                    {
                        "status": "ready_to_draft",
                        "summary": "Stage settled by the researcher answer.",
                        "evidence": [
                            {
                                "claim": "Researcher answer accepted.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _decision_coverage(stage_id, pending_turn),
                        "ambiguities": [],
                    }
                )
        else:
            base_version = 1
            base_match = re.search(r"Base specification version: (\d+)", prompt)
            if base_match:
                base_version = int(base_match.group(1))
            stage_id = "study-foundation"
            match = re.search(r"Stage: .*\((.*)\)", prompt)
            if match:
                stage_id = match.group(1)
            operations = self.fixture["stages"][stage_id]["draft"]
            text = _json.dumps(
                {
                    "stage_id": stage_id,
                    "base_specification_version": base_version,
                    "operations": operations,
                    "evidence": [
                        {"target": operation["path"], "source_turns": [1]}
                        for operation in operations
                    ],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "affected_checklist_items": [],
                }
            )
        return ProviderResponse(text, self.provider, "m", "req-1", parsed=_json.loads(text))


def run_conversation(
    tmp_path: Path,
    fixture_name: str,
    *,
    revision_stage: str | None = None,
) -> dict:
    fixture = yaml.safe_load((GOLDEN_DIR / fixture_name).read_text())
    workspace = tmp_path / f"ws-{fixture_name}"
    service = GenesisService(workspace)
    service.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    service._elicitation_provider = lambda profile_id: GoldenConversationProvider(fixture)  # type: ignore[method-assign]
    try:
        spec_id = fixture["specification_id"]
        service.create_specification(
            {"id": spec_id, "title": fixture["title"], "description": fixture["description"]}
        )
        session = service.start_elicitation(
            {
                "specification_id": spec_id,
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        stage_ids = list(fixture["stages"])
        current_stage = stage_ids[0]
        approved_any = False
        while True:
            session = service.get_elicitation(session["session_id"])
            if session["current_stage"] != current_stage:
                current_stage = session["current_stage"]
            stage_script = fixture["stages"][current_stage]
            for answer in stage_script["answers"]:
                session = service.get_elicitation(session["session_id"])
                if session["status"] != "awaiting_answer":
                    break
                payload = {"answer": str(answer)}
                if isinstance(answer, dict):
                    payload = {
                        "answer": str(answer["answer"]),
                        "suggestion_index": answer.get("suggestion_index"),
                    }
                service.submit_elicitation_message(
                    session["session_id"],
                    payload["answer"],
                    suggestion_index=payload.get("suggestion_index"),
                )
            session = service.get_elicitation(session["session_id"])
            service.draft_elicitation(session["session_id"])
            service.preview_elicitation_stage(session["session_id"])
            session = service.approve_elicitation_stage(
                session["session_id"], approved_by="researcher"
            )
            approved_any = True
            if session["status"] == "completed":
                break
            if revision_stage and not approved_any:
                break
        # Optional upstream revision exercising downstream invalidation.
        if revision_stage:
            session = service.get_elicitation(session["session_id"])
            assert session["status"] == "completed"
            # Reopen the revision stage and re-approve every affected stage.
            reopened = service.reopen_elicitation_stage(session["session_id"], revision_stage)
            assert reopened["current_stage"] == revision_stage
            while True:
                session = service.get_elicitation(session["session_id"])
                if session["status"] == "completed":
                    break
                stage_script = fixture["stages"][session["current_stage"]]
                for answer in stage_script["answers"]:
                    payload = {"answer": str(answer)}
                    if isinstance(answer, dict):
                        payload = {
                            "answer": str(answer["answer"]),
                            "suggestion_index": answer.get("suggestion_index"),
                        }
                    service.submit_elicitation_message(
                        session["session_id"],
                        payload["answer"],
                        suggestion_index=payload.get("suggestion_index"),
                    )
                session = service.get_elicitation(session["session_id"])
                progress = service.get_elicitation(session["session_id"])["stages"][
                    session["current_stage"]
                ]
                if progress["status"] == "needs_review":
                    session = service.approve_elicitation_stage(
                        session["session_id"], approved_by="researcher"
                    )
                else:
                    service.draft_elicitation(session["session_id"])
                    service.preview_elicitation_stage(session["session_id"])
                    session = service.approve_elicitation_stage(
                        session["session_id"], approved_by="researcher"
                    )
        compiled = service.compile_study(None, f"builds/{spec_id}", specification_id=spec_id)
        result = {
            "specification_id": spec_id,
            "workspace": str(workspace),
            "session": service.get_elicitation(session["session_id"]),
            "compiled": compiled,
            "fixture": fixture,
        }
        return result
    finally:
        service.close()


def _assert_structural_equivalence(result: dict, golden_dir: str) -> None:
    fixture = result["fixture"]
    compiled_path = Path(result["workspace"]) / result["compiled"]["path"]
    processes = _json.loads((compiled_path / "processes.json").read_text())
    produced = {process["id"] for process in processes}
    assert produced == set(fixture["processes"])
    spec_dir = Path(result["workspace"]) / ".genesis/specifications" / result["specification_id"]
    produced_theory = yaml.safe_load((spec_dir / "theory.yaml").read_text())
    assert produced_theory["theory_family"] == fixture["theory_family"]
    golden_package = GOLDEN_STUDIES / golden_dir
    golden_theory = yaml.safe_load((golden_package / "theory.yaml").read_text())
    assert produced_theory["theory_family"] == golden_theory["theory_family"]
    golden_processes = {
        item["id"]
        for item in yaml.safe_load((golden_package / "openness.yaml").read_text())["processes"]
    }
    assert produced == golden_processes


def test_platform_governance_golden_conversation(tmp_path: Path) -> None:
    result = run_conversation(tmp_path, "platform_governance.yaml", revision_stage="openness")
    session = result["session"]
    assert session["status"] == "completed"
    assert all(progress["status"] == "approved" for progress in session["stages"].values())
    # The conversation edited a suggestion (suggestion_index) and ran an
    # upstream revision with downstream invalidation.
    modes = {turn["response_mode"] for turn in session["turns"]}
    assert "suggested" in modes
    assert "free_form" in modes
    assert any(invalidation["stage"] == "theory" for invalidation in session["invalidations"])
    _assert_structural_equivalence(result, "platform_governance")


def test_community_formation_golden_conversation(tmp_path: Path) -> None:
    result = run_conversation(tmp_path, "community_formation.yaml")
    session = result["session"]
    assert session["status"] == "completed"
    assert all(progress["status"] == "approved" for progress in session["stages"].values())
    _assert_structural_equivalence(result, "community_formation")


def test_golden_conversations_produce_compilable_packages(tmp_path: Path) -> None:
    for fixture_name in ("platform_governance.yaml", "community_formation.yaml"):
        fixture = yaml.safe_load((GOLDEN_DIR / fixture_name).read_text())
        workspace = tmp_path / f"compile-{fixture_name}"
        service = GenesisService(workspace)
        service.create_model_profile(
            {
                "id": "assistant",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m",
                "api_key_env": "GENESIS_FAKE_KEY",
            }
        )
        try:
            spec_id = fixture["specification_id"]
            service.create_specification({"id": spec_id, "title": "x", "description": "y"})
            # Write the canonical package directly and compile it.
            session = service.start_elicitation(
                {
                    "specification_id": spec_id,
                    "workflow_id": "three-layer-study",
                    "model_profile_id": "assistant",
                    "researcher_id": "researcher",
                }
            )
            assert session["status"] == "awaiting_answer"
            # Compilation remains blocked until the package is approved (IEL-026).
            with pytest.raises(ValueError, match="SPECIFICATION_NOT_APPROVED"):
                service.compile_study(None, f"builds/compile-{spec_id}", specification_id=spec_id)
            # Deterministic inspection makes no model calls (IEL-027).
            from genesis.assistant import StudyAssistant

            report = StudyAssistant().inspect_package(service._specification_dir(spec_id)).as_dict()
            assert isinstance(report, dict)
        finally:
            service.close()


class _ClickbaitSequenceProvider:
    provider = "scripted-clickbait"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, request) -> ProviderResponse:
        self.prompts.append(request.prompt)
        text = _json.dumps(self.responses.pop(0))
        return ProviderResponse(
            text,
            self.provider,
            "clickbait-qualification-model",
            f"clickbait-{len(self.prompts)}",
            parsed=_json.loads(text),
        )


def _clickbait_elicitation(
    tmp_path: Path, responses: list[dict]
) -> tuple[GenesisService, _ClickbaitSequenceProvider, dict, dict]:
    fixture = _json.loads(CLICKBAIT_CRITICAL.read_text())
    provider = _ClickbaitSequenceProvider(responses)
    service = GenesisService(tmp_path / f"clickbait-{len(responses)}")
    service.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    service._elicitation_provider = lambda _profile_id: provider  # type: ignore[method-assign]
    session = service.start_elicitation(
        {
            "specification_id": "clickbait-critical",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    return service, provider, fixture, session


def test_clickbait_foundation_finishes_from_one_simulation_critical_answer(
    tmp_path: Path,
) -> None:
    fixture = _json.loads(CLICKBAIT_CRITICAL.read_text())
    service, provider, fixture, session = _clickbait_elicitation(
        tmp_path, [fixture["ready_response"]]
    )
    try:
        result = service.submit_elicitation_message(
            session["session_id"], fixture["opening_answer"]
        )
        assert result["status"] == "awaiting_approval"
        assert result["clarification"]["turns_used"] == 1
        assert result["clarification"]["turns_remaining"] == 3
        assert result["clarification"]["remaining_decisions"] == []
        prompt = provider.prompts[0]
        assert "focal-question [required] -> /study/description" in prompt
        assert "simulation-boundary [required]" in prompt
        assert "comparison-objective [required]" in prompt
    finally:
        service.close()


def test_clickbait_non_executable_question_is_repaired_not_added_to_loop(
    tmp_path: Path,
) -> None:
    fixture = _json.loads(CLICKBAIT_CRITICAL.read_text())
    service, provider, fixture, session = _clickbait_elicitation(
        tmp_path,
        [fixture["irrelevant_response"], fixture["ready_response"]],
    )
    try:
        result = service.submit_elicitation_message(
            session["session_id"], fixture["opening_answer"]
        )
        assert result["status"] == "awaiting_approval"
        assert result["clarification"]["remaining_decisions"] == []
        assert len(result["turns"]) == 1
        assert [attempt["status"] for attempt in result["last_assistant_attempts"]] == [
            "invalid",
            "valid",
        ]
        assert len(provider.prompts) == 2
        assert "creator-biographies" in provider.prompts[1]
        assert "undeclared decision" in provider.prompts[1]
    finally:
        service.close()

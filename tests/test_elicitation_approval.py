"""Task 6: explicit stage approval, revision, and dependency invalidation."""

from __future__ import annotations

import json as _json
import re
from pathlib import Path

import pytest

from genesis.providers import ProviderResponse
from genesis.service import GenesisService

STAGE_OPENING = {
    "study-foundation": (
        "Briefly describe the social phenomenon, research question, intended "
        "contribution, and setting you want to simulate."
    ),
    "openness": (
        "Which participant decisions, interpretations, communications, or creative "
        "actions must remain open-ended rather than represented by fixed rules?"
    ),
    "theory": (
        "What theoretical explanation should connect generated actions to subsequent "
        "social consequences and change over time?"
    ),
    "domain": (
        "In the substantive setting being modelled, who or what exists, what can they "
        "observe, and what can change?"
    ),
    "experiment-design": (
        "What comparisons, conditions, replications, and measurements are needed to "
        "answer the research question?"
    ),
}

STAGE_PATCHES = {
    "study-foundation": [
        {"op": "replace", "path": "/study/title", "value": "Stage-approval study"},
        {"op": "replace", "path": "/study/description", "value": "Approved description."},
    ],
    "openness": [
        {
            "op": "replace",
            "path": "/openness/processes",
            "value": [
                {
                    "id": "compose-message",
                    "executor": {"mode": "deterministic"},
                    "context_policy": "public",
                }
            ],
        }
    ],
    "theory": [
        {"op": "replace", "path": "/theory/theory_family", "value": "exploratory"},
    ],
    "domain": [
        {"op": "add", "path": "/domain/actors/-", "value": {"id": "member"}},
    ],
    "experiment-design": [
        {"op": "replace", "path": "/protocol/time_model/end", "value": 20},
    ],
}

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


class StageScriptedProvider:
    provider = "scripted"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        prompt = request.prompt
        if "Evaluate the researcher's most recent answer" in prompt:
            stage_id = _evaluation_stage(prompt)
            pending_turn = _pending_turn(prompt)
            text = _json.dumps(
                {
                    "status": "ready_to_draft",
                    "summary": "Stage settled.",
                    "evidence": [
                        {
                            "claim": "Researcher answer accepted.",
                            "source_turns": [pending_turn],
                        }
                    ],
                    "decision_coverage": [
                        {
                            "decision_id": decision_id,
                            "status": "covered",
                            "evidence_turns": [pending_turn],
                        }
                        for decision_id in STAGE_DECISIONS[stage_id]
                    ],
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
            operations = STAGE_PATCHES.get(stage_id, [])
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


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", StageScriptedProvider)
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
    genesis.create_specification({"id": "stage-study", "title": "Start", "description": "draft"})
    return genesis


def _answer(service: GenesisService, session_id: str, text: str = "An answer.") -> dict:
    service.submit_elicitation_message(session_id, text)
    return service.get_elicitation(session_id)


def _draft_and_approve(
    service: GenesisService, session_id: str, approved_by: str = "researcher"
) -> dict:
    service.draft_elicitation(session_id)
    service.preview_elicitation_stage(session_id)
    return service.approve_elicitation_stage(session_id, approved_by=approved_by)


def test_approval_uses_deterministic_coverage_not_model_clarification_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubbornProvider(StageScriptedProvider):
        def generate(self, request) -> ProviderResponse:
            prompt = request.prompt
            if "Evaluate the researcher's most recent answer" not in prompt:
                return super().generate(request)
            pending_turn = _pending_turn(prompt)
            text = _json.dumps(
                {
                    "status": "needs_clarification",
                    "summary": "The model keeps asking despite complete coverage.",
                    "evidence": [],
                    "decision_coverage": [
                        {
                            "decision_id": decision_id,
                            "status": "covered",
                            "evidence_turns": [pending_turn],
                        }
                        for decision_id in STAGE_DECISIONS["study-foundation"]
                    ],
                    "ambiguities": [
                        {
                            "id": "focal-question-repeat",
                            "decision_id": "focal-question",
                            "target_paths": ["/study/description"],
                            "question": "Can you restate the focal question?",
                            "reason": "The model asks for redundant detail.",
                            "consequential": True,
                            "suggestions": [
                                {"label": "A", "value": "Restatement A."},
                                {"label": "B", "value": "Restatement B."},
                                {"label": "C", "value": "Restatement C."},
                            ],
                        }
                    ],
                }
            )
            return ProviderResponse(
                text, self.provider, "m", "req-stubborn", parsed=_json.loads(text)
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", StubbornProvider)
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
    genesis.create_specification({"id": "stage-study", "title": "Start", "description": "draft"})
    try:
        session = genesis.start_elicitation(
            {
                "specification_id": "stage-study",
                "workflow_id": "three-layer-study",
                "model_profile_id": "assistant",
                "researcher_id": "researcher",
            }
        )
        answered = genesis.submit_elicitation_message(
            session["session_id"], "A complete bounded governance study."
        )
        assert answered["stages"]["study-foundation"]["status"] == "draft_ready"
        genesis.draft_elicitation(session["session_id"])
        genesis.preview_elicitation_stage(session["session_id"])
        approved = genesis.approve_elicitation_stage(
            session["session_id"], approved_by="researcher"
        )
        assert approved["current_stage"] == "openness"
    finally:
        genesis.close()


def test_approval_revalidates_persisted_schema_error(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    sid = session["session_id"]
    _answer(service, sid)
    _draft_and_approve(service, sid)
    _answer(service, sid)
    service.draft_elicitation(sid)
    stored = service._elicitation_engine.require_session(sid)
    patch = dict(stored.pending_patch)
    patch["operations"][0]["value"][0]["outputs"] = [
        {"artifact_type": "strategy", "schema_ref": "creator-strategy-schema"}
    ]
    patch["operations"][0]["value"][0]["inputs"] = ["creator-local-observation"]
    service._elicitation_store.put_pending_patch(sid, patch)
    preview = service.preview_elicitation_stage(sid)["pending_preview"]
    assert any(w["code"] == "REF_SCHEMA" for w in preview["validation"]["warnings"])
    # Simulate validation persisted by the server before schema deferral was added.
    preview["validation"]["errors"] = [
        {"code": "REF_SCHEMA", "message": "unknown output schema 'creator-strategy-schema'"}
    ]
    service._elicitation_store.put_pending_preview(sid, preview)
    resumed = GenesisService(service.workspace)
    try:
        approved = resumed.approve_elicitation_stage(sid, approved_by="researcher")
        assert approved["current_stage"] == "theory"
        workflow = resumed._workflow_registry.get("three-layer-study")
        domain_validation = resumed._patch_preview._inspect(
            resumed._specification_dir("stage-study"),
            stage=workflow.stage("domain"),
            workflow=workflow,
        )
        assert any(e["code"] == "REF_INPUT" for e in domain_validation["errors"])
        assert not (
            resumed._specification_dir("stage-study") / "schemas/creator-strategy-schema.json"
        ).exists()
    finally:
        resumed.close()


def test_first_approval_records_identity_and_bumps_version(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    advanced = _draft_and_approve(service, session["session_id"])
    assert advanced["status"] == "awaiting_answer"
    assert advanced["current_stage"] == "openness"
    assert advanced["stages"]["study-foundation"]["status"] == "approved"
    assert advanced["stages"]["study-foundation"]["accepted_revision"] == 2
    assert advanced["current_question"] == STAGE_OPENING["openness"]
    # A new immutable package version was recorded with approval metadata.
    metadata = service.get_specification("stage-study")
    assert metadata["version"] == 2
    approvals = metadata["elicitation_approvals"]
    assert approvals[-1]["stage"] == "study-foundation"
    assert approvals[-1]["approved_by"] == "researcher"
    assert approvals[-1]["revision"] == 2


def test_approval_requires_expected_flow(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    with pytest.raises(ValueError, match="STAGE_INCOMPLETE"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")
    # The assistant must not approve its own output.
    _answer(service, session["session_id"])
    with pytest.raises(ValueError, match="STAGE_APPROVAL_REQUIRED"):
        service.approve_elicitation_stage(session["session_id"], approved_by="assistant")
    with pytest.raises(ValueError, match="STAGE_APPROVAL_REQUIRED"):
        service.approve_elicitation_stage(session["session_id"], approved_by="assistant")


def test_revision_returns_to_clarification(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    session = service.draft_elicitation(session["session_id"])
    revised = service.revise_elicitation_stage(session["session_id"])
    assert revised["status"] == "awaiting_approval"
    assert revised["stages"]["study-foundation"]["review_mode"] is True
    # No package version was written by the revision.
    assert service.get_specification("stage-study")["version"] == 1


def test_upstream_revision_reopens_only_dependent_stages(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    for _stage in ("study-foundation", "openness", "theory", "domain"):
        _answer(service, session["session_id"])
        session = _draft_and_approve(service, session["session_id"])
    session = service.get_elicitation(session["session_id"])
    assert session["current_stage"] == "experiment-design"
    # Reopen Layer 1: downstream dependent stages become needs_review.
    reopened = service.reopen_elicitation_stage(session["session_id"], "openness")
    assert reopened["current_stage"] == "openness"
    session = service.get_elicitation(session["session_id"])
    assert session["stages"]["theory"]["status"] == "needs_review"
    assert session["stages"]["domain"]["status"] == "needs_review"
    assert session["stages"]["experiment-design"]["status"] == "needs_review"
    # Re-approve openness with its patch, then re-approve each affected stage.
    _answer(service, session["session_id"])
    session = _draft_and_approve(service, session["session_id"])
    session = service.get_elicitation(session["session_id"])
    assert session["current_stage"] == "theory"
    assert session["stages"]["theory"]["status"] == "needs_review"
    # Re-approving a needs_review stage requires no fresh patch.
    reapproved = service.approve_elicitation_stage(session["session_id"], approved_by="researcher")
    assert reapproved["current_stage"] == "domain"


def test_final_approval_completes_session_and_approves_package(service: GenesisService) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    for _stage in ("study-foundation", "openness", "theory", "domain", "experiment-design"):
        _answer(service, session["session_id"])
        session = _draft_and_approve(service, session["session_id"])
    session = service.get_elicitation(session["session_id"])
    assert session["status"] == "completed"
    assert all(progress["status"] == "approved" for progress in session["stages"].values())
    # The package is approved and compilable; compilation makes no model calls.
    compiled = service.compile_study(None, "builds/stage-study", specification_id="stage-study")
    assert compiled["study_id"] == "stage-study"


def test_stale_preview_cannot_overwrite_external_edits(service: GenesisService) -> None:
    """Review P1: a preview taken before an external edit must be rejected."""
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])
    # An external edit lands after the preview was pinned.
    current = service.get_specification("stage-study")
    service.update_specification(
        "stage-study",
        {"description": "externally edited description"},
        current["version"],
    )
    with pytest.raises(ValueError, match="PATCH_BASE_STALE"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")


@pytest.mark.parametrize(
    ("asset_directory", "asset_name"),
    [
        ("prompts", "external.txt"),
        ("schemas", "external.json"),
        ("data", "external.csv"),
        ("extensions", "external.yaml"),
    ],
)
def test_stale_preview_detects_external_asset_edits(
    service: GenesisService, asset_directory: str, asset_name: str
) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])

    asset_root = service._specification_dir("stage-study") / asset_directory
    asset_root.mkdir(parents=True, exist_ok=True)
    (asset_root / asset_name).write_text("external change\n")

    with pytest.raises(ValueError, match="PATCH_BASE_STALE"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")


def test_stale_preview_detects_package_version_change_without_content_change(
    service: GenesisService,
) -> None:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])

    current = service.get_specification("stage-study")
    unchanged = service._current_form_payload(
        service._specification_dir("stage-study"), "stage-study"
    )
    service.update_specification("stage-study", unchanged, current["version"])

    with pytest.raises(ValueError, match="PATCH_BASE_STALE"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")


def test_advisory_unresolved_patch_questions_approve_with_deferred_notes(
    service: GenesisService,
) -> None:
    """Advisory (non-consequential) unresolved questions no longer block approval;
    they are recorded as deferred notes so sessions cannot deadlock at the limit."""
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    pending = dict(stored.pending_patch or {})
    pending["unresolved_questions"] = ["Which population should be sampled?"]
    service._elicitation_store.put_pending_patch(session["session_id"], pending)
    service.preview_elicitation_stage(session["session_id"])
    result = service.approve_elicitation_stage(session["session_id"], approved_by="researcher")
    assert result["status"] != "failed"
    stored = service._elicitation_engine.require_session(session["session_id"])
    deferred = [q for progress in stored.stages.values() for q in progress.deferred_questions]
    assert "Which population should be sampled?" in deferred


def test_session_base_advances_after_each_accepted_stage(service: GenesisService) -> None:
    """Review P1: the session base tracks the latest immutable package version."""
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    _draft_and_approve(service, session["session_id"])
    projection = service.get_elicitation(session["session_id"])
    assert projection["base_specification_version"] == 2
    assert service.get_specification("stage-study")["version"] == 2


def _first_required_decision_id(service, session) -> str | None:
    stage = service._elicitation_engine.stage_for(session)[1]
    for decision in stage.critical_decisions:
        if decision.required:
            return decision.id
    return None


def test_limit_reached_with_unresolved_questions_approves_with_deferred_notes(
    service: GenesisService,
) -> None:
    """Deadlock regression: clarification budget exhausted, all REQUIRED decisions
    covered, assistant patch still carries unresolved questions -> approval must
    proceed and record them as deferred notes (not STAGE_INCOMPLETE)."""
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    pending = dict(stored.pending_patch or {})
    pending["unresolved_questions"] = [
        "Which social media platform, creator and audience population, and temporal scope "
        "will define the study setting?"
    ]
    service._elicitation_store.put_pending_patch(session["session_id"], pending)

    # simulate the limit state: budget exhausted, no REQUIRED decisions remaining
    def _limit_mutate(stored_session):
        progress = stored_session.stages[stored_session.current_stage]
        progress.limit_reached = True
        progress.turn_count = progress.turn_count + 5
        for decision in progress.decision_coverage:
            progress.decision_coverage[decision] = "covered"

    service._elicitation_store.update(session["session_id"], _limit_mutate)
    service.preview_elicitation_stage(session["session_id"])
    result = service.approve_elicitation_stage(session["session_id"], approved_by="researcher")
    assert result["status"] != "failed"
    stored = service._elicitation_engine.require_session(session["session_id"])
    assert stored.stages.get("study-foundation") or stored.stages.get(stored.current_stage), (
        "the limited stage must still resolve"
    )
    approved_stages = [
        progress for progress in stored.stages.values() if progress.status == "approved"
    ]
    assert approved_stages, "the limited stage must have been approved"
    deferred = [q for progress in approved_stages for q in progress.deferred_questions]
    assert len(deferred) == 1
    assert "temporal scope" in deferred[0]
    projection = service.get_elicitation(session["session_id"])
    stage_ids = list(projection["stages"].keys())
    assert any(
        len(projection["stages"][sid].get("deferred_questions", [])) == 1 for sid in stage_ids
    )


def test_unresolved_questions_with_remaining_required_decisions_still_block(
    service: GenesisService,
) -> None:
    """Consequential ambiguity still blocks approval."""
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    pending = dict(stored.pending_patch or {})
    pending["unresolved_questions"] = ["Which population should be sampled?"]
    service._elicitation_store.put_pending_patch(session["session_id"], pending)

    # leave a REQUIRED decision unresolved -> consequential -> must block
    def _unresolved_mutate(stored_session):
        progress = stored_session.stages[stored_session.current_stage]
        progress.review_mode = False
        for decision in progress.decision_coverage:
            progress.decision_coverage[decision] = "covered"
        first_required = _first_required_decision_id(service, stored_session)
        if first_required:
            progress.decision_coverage[first_required] = "unresolved"

    service._elicitation_store.update(session["session_id"], _unresolved_mutate)
    service.preview_elicitation_stage(session["session_id"])
    with pytest.raises(ValueError, match="STAGE_INCOMPLETE.*unresolved questions"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")


class LocalStageProvider(StageScriptedProvider):
    """Scripted provider that cites STAGE-LOCAL turn numbers (1..k) in the
    evaluation evidence, exercising the local->global normalization."""

    def generate(self, request) -> ProviderResponse:
        if "Evaluate the researcher's most recent answer" in request.prompt:
            stage_id = _evaluation_stage(request.prompt)
            text = _json.dumps(
                {
                    "status": "ready_to_draft",
                    "summary": "Stage settled (local numbering).",
                    "evidence": [{"claim": "Researcher answer accepted.", "source_turns": [1]}],
                    "decision_coverage": [
                        {
                            "decision_id": decision_id,
                            "status": "covered",
                            "evidence_turns": [1],
                        }
                        for decision_id in STAGE_DECISIONS[stage_id]
                    ],
                    "ambiguities": [],
                }
            )
            return ProviderResponse(text, self.provider, "m", "req-local", parsed=_json.loads(text))
        return super().generate(request)


def test_stage_local_evidence_turns_are_normalized_to_global_ids(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the assistant may number this stage's rendered conversation
    from 1 even when global turn ids continue (stage 2 starts at turn 3 here).
    Local ids must be accepted and normalized to the stage's global ids."""
    # stage 1 with the standard scripted provider
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    _draft_and_approve(service, session["session_id"])
    served = service._elicitation_engine.require_session(session["session_id"])
    assert served.current_stage != "study-foundation", "must advance past stage 1"
    # stage 2 has no turns yet; its first global id continues after stage 1
    pending_global = len(served.turns) + 1
    assert pending_global > 1

    # stage 2: local numbering provider; citing local turn 1 must be accepted
    # and normalized to the stage's first GLOBAL id
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", LocalStageProvider)
    _answer(service, session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    last_turn = stored.turns[-1]
    evaluation = last_turn.evaluation or {}
    evidence_turns = [
        turn for claim in evaluation.get("evidence", []) for turn in claim.get("source_turns", [])
    ]
    assert evidence_turns == [pending_global], (
        f"local turn 1 must be normalized to global {pending_global}, got {evidence_turns}"
    )


def test_evaluation_citing_truly_unknown_turn_is_rejected(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cite that is neither a global id nor within the stage's local range must
    still be rejected."""

    class BadStageProvider(StageScriptedProvider):
        def generate(self, request) -> ProviderResponse:
            if "Evaluate the researcher's most recent answer" in request.prompt:
                stage_id = _evaluation_stage(request.prompt)
                text = _json.dumps(
                    {
                        "status": "ready_to_draft",
                        "summary": "Bad cite.",
                        "evidence": [{"claim": "Bad cite.", "source_turns": [999]}],
                        "decision_coverage": [
                            {
                                "decision_id": decision_id,
                                "status": "covered",
                                "evidence_turns": [999],
                            }
                            for decision_id in STAGE_DECISIONS[stage_id]
                        ],
                        "ambiguities": [],
                    }
                )
                return ProviderResponse(
                    text, self.provider, "m", "req-bad", parsed=_json.loads(text)
                )
            return super().generate(request)

    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", BadStageProvider)
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID.*unknown turn 999"):
        service.submit_elicitation_message(session["session_id"], "An answer.")


def test_ready_to_draft_allows_documented_assumptions(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ready_to_draft evaluations may carry documented assumptions;
    only consequential ambiguity blocks readiness."""

    class ReadyWithAssumptionsProvider(StageScriptedProvider):
        def generate(self, request) -> ProviderResponse:
            if "Evaluate the researcher's most recent answer" in request.prompt:
                stage_id = _evaluation_stage(request.prompt)
                text = _json.dumps(
                    {
                        "status": "ready_to_draft",
                        "summary": "Ready with documented assumptions.",
                        "evidence": [{"claim": "Researcher answer accepted.", "source_turns": [1]}],
                        "assumptions": ["Owners defaulted to the researcher"],
                        "decision_coverage": [
                            {"decision_id": decision_id, "status": "covered", "evidence_turns": [1]}
                            for decision_id in STAGE_DECISIONS[stage_id]
                        ],
                        "ambiguities": [],
                    }
                )
                return ProviderResponse(
                    text, self.provider, "m", "req-assumptions", parsed=_json.loads(text)
                )
            return super().generate(request)

    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ReadyWithAssumptionsProvider)
    outcome = service.submit_elicitation_message(
        session["session_id"], "An answer that documents assumptions."
    )
    stored = service._elicitation_engine.require_session(session["session_id"])
    assert stored.turns[-1].evaluation["assumptions"] == ["Owners defaulted to the researcher"]
    assert outcome["stages"][stored.current_stage]["status"] == "draft_ready", (
        "readiness must be reached with documented assumptions"
    )


VSR_THEORY_VALUE = {
    "schema_version": "1.0",
    "study_id": "stage-study",
    "theory_family": "variation-selection-retention",
    "constructs": [
        {"id": "creator-strategy", "theory_role": "variation_unit"},
        {"id": "performance-history", "theory_role": "retention_state"},
    ],
    "process_mappings": [
        {"process": "compose-message", "theory_function": "variation"},
        {"process": "compose-message", "theory_function": "selection"},
        {"process": "compose-message", "theory_function": "retention"},
        {"process": "compose-message", "theory_function": "feedback"},
    ],
    "relations": [
        {"from": "compose-message", "to": "compose-message", "relation": "produces_variant"}
    ],
    "feedback": [],
    "observables": [{"id": "strategy-diversity", "definition": "distinct strategies"}],
}


def _theory_patch(value: dict) -> list[dict]:
    return [{"op": "replace", "path": "/theory", "value": value}]


class TheoryVSRProvider(StageScriptedProvider):
    """Draft provider selecting the REGISTERED VSR family with the given value."""

    def __init__(self, theory_value: dict) -> None:
        self.theory_value = theory_value
        super().__init__()

    def generate(self, request) -> ProviderResponse:
        if "Evaluate the researcher's most recent answer" in request.prompt:
            return super().generate(request)
        base_version = 1
        base_match = re.search(r"Base specification version: (\d+)", str(request.prompt))
        if base_match:
            base_version = int(base_match.group(1))
        text = _json.dumps(
            {
                "stage_id": "theory",
                "base_specification_version": base_version,
                "operations": _theory_patch(self.theory_value),
                "evidence": [{"target": "/theory", "source_turns": [1]}],
                "assumptions": [],
                "unresolved_questions": [],
                "affected_checklist_items": [],
            }
        )
        return ProviderResponse(text, self.provider, "m", "req-theory", parsed=_json.loads(text))


def _theory_spec_entered(service: GenesisService) -> dict:
    session = service.start_elicitation(
        {
            "specification_id": "stage-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    _answer(service, session["session_id"])
    _draft_and_approve(service, session["session_id"])  # stage 1: study-foundation
    assert (
        service._elicitation_engine.require_session(session["session_id"]).current_stage
        == "openness"
    )
    _answer(service, session["session_id"])
    _draft_and_approve(service, session["session_id"])  # stage 2: openness
    assert (
        service._elicitation_engine.require_session(session["session_id"]).current_stage == "theory"
    )
    return session


def test_theory_draft_with_unmapped_vsr_family_fails_preview_and_approval(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THEORY_FUNCTION_MISSING must surface in the PREVIEW and block approval."""
    session = _theory_spec_entered(service)
    incomplete = dict(VSR_THEORY_VALUE)
    incomplete["process_mappings"] = []
    monkeypatch.setattr(
        service, "_elicitation_provider", lambda _profile: TheoryVSRProvider(incomplete)
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    compile_errors = (stored.pending_preview or {}).get("validation", {}).get("compile_errors", [])
    codes = [entry.get("code") for entry in compile_errors]
    assert "THEORY_FUNCTION_MISSING" in codes, compile_errors
    with pytest.raises(ValueError, match="SPECIFICATION_INVALID.*mandatory functions"):
        service.approve_elicitation_stage(session["session_id"], approved_by="researcher")


def test_theory_draft_with_complete_vsr_mappings_approves(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered family with every mandatory function mapped to a Layer-1
    process approves cleanly."""
    session = _theory_spec_entered(service)
    monkeypatch.setattr(
        service,
        "_elicitation_provider",
        lambda _profile: TheoryVSRProvider(VSR_THEORY_VALUE),
    )
    _answer(service, session["session_id"])
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])
    stored = service._elicitation_engine.require_session(session["session_id"])
    compile_errors = (stored.pending_preview or {}).get("validation", {}).get("compile_errors", [])
    assert compile_errors == [], compile_errors
    result = service.approve_elicitation_stage(session["session_id"], approved_by="researcher")
    assert result["status"] != "failed"

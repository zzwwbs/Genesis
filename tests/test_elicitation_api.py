"""Task 7: elicitation HTTP API acceptance tests."""

from __future__ import annotations

import json as _json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app

STAGE_PATCHES = {
    "study-foundation": [
        {"op": "replace", "path": "/study/title", "value": "API study"},
        {"op": "replace", "path": "/study/description", "value": "API description."},
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
    "theory": [{"op": "replace", "path": "/theory/theory_family", "value": "exploratory"}],
    "domain": [{"op": "add", "path": "/domain/actors/-", "value": {"id": "member"}}],
    "experiment-design": [{"op": "replace", "path": "/protocol/time_model/end", "value": 20}],
}

STAGE_DECISIONS = {
    "study-foundation": ("focal-question", "simulation-boundary", "comparison-objective"),
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


def _coverage(stage_id: str, turn_id: int, unresolved: str | None = None) -> list[dict]:
    return [
        {
            "decision_id": decision_id,
            "status": "unresolved" if decision_id == unresolved else "covered",
            "evidence_turns": [] if decision_id == unresolved else [turn_id],
        }
        for decision_id in STAGE_DECISIONS[stage_id]
    ]


class ApiScriptedProvider:
    provider = "scripted"

    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def generate(self, request) -> _json:
        self.calls += 1
        prompt = request.prompt
        if "Evaluate the researcher's most recent answer" in prompt:
            stage_id = _evaluation_stage(prompt)
            pending_turn = _pending_turn(prompt)
            new_answer = ""
            if "## New researcher answer" in prompt:
                new_answer = prompt.split("## New researcher answer", 1)[1].split(
                    "## Your task", 1
                )[0]
            if "repair this response" in new_answer and self.calls == 1:
                text = '{"status": "ready_to_draft"'
            elif "needs a timing choice" in new_answer:
                text = _json.dumps(
                    {
                        "status": "needs_clarification",
                        "summary": "Observation timing needs a decision.",
                        "evidence": [
                            {
                                "claim": "Timing is not yet specified.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _coverage(
                            stage_id, pending_turn, unresolved="information-context"
                        ),
                        "ambiguities": [
                            {
                                "id": "observation-timing",
                                "decision_id": "information-context",
                                "target_paths": ["/openness/processes"],
                                "question": "When should members observe messages?",
                                "reason": "Timing changes the causal order.",
                                "consequential": True,
                                "suggestions": [
                                    {"label": "Immediate", "value": "In the same round."},
                                    {"label": "Next round", "value": "In the next round."},
                                    {
                                        "label": "Round summary",
                                        "value": "As a summary after each round.",
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
                        "summary": "Stage settled.",
                        "evidence": [
                            {
                                "claim": "Researcher answer accepted.",
                                "source_turns": [pending_turn],
                            }
                        ],
                        "decision_coverage": _coverage(stage_id, pending_turn),
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
        try:
            parsed = _json.loads(text)
        except _json.JSONDecodeError:
            parsed = text
        return type(
            "R",
            (),
            {"text": text, "provider": self.provider, "model": "m", "parsed": parsed},
        )()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ApiScriptedProvider)
    return TestClient(create_app(tmp_path / "workspace"))


def _profile(client: TestClient) -> None:
    response = client.post(
        "/llm/profiles",
        json={
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        },
    )
    assert response.status_code == 201
    client.post(
        "/specifications",
        json={"id": "api-study", "title": "Start", "description": "draft"},
    )


def test_session_lifecycle_via_api(client: TestClient) -> None:
    _profile(client)
    started = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    )
    assert started.status_code == 201
    session = started.json()
    session_id = session["session_id"]
    assert session["status"] == "awaiting_answer"
    assert session["current_stage"] == "study-foundation"
    assert session["clarification"] == {
        "turns_used": 0,
        "turns_remaining": 4,
        "max_turns": 4,
        "limit_reached": False,
        "deferred_questions": [],
        "required_decisions": 3,
        "covered_decisions": 0,
        "remaining_decisions": [
            "focal-question",
            "simulation-boundary",
            "comparison-objective",
        ],
    }
    # Message turn.
    turn = client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "A foundation answer.", "response_mode": "free_form"},
    )
    assert turn.status_code == 200
    assert turn.json()["status"] == "awaiting_approval"
    # Draft + preview.
    draft = client.post(f"/elicitation/sessions/{session_id}/draft")
    assert draft.status_code == 200
    preview = client.get(f"/elicitation/sessions/{session_id}/preview")
    assert preview.status_code == 200
    assert "study.yaml" in preview.json()["pending_preview"]["yaml_files"]
    # Approve advances to the next stage.
    approved = client.post(
        f"/elicitation/sessions/{session_id}/approve",
        json={"approved_by": "researcher"},
    )
    assert approved.status_code == 200
    assert approved.json()["current_stage"] == "openness"
    # Expected-version conflicts are rejected on mutating routes.
    conflict = client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "x", "expected_version": 99},
    )
    assert conflict.status_code == 409


def test_expired_session_returns_named_error(client: TestClient) -> None:
    _profile(client)
    response = client.get("/elicitation/sessions/no-such-session")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "ELICITATION_SESSION_EXPIRED"
    assert "remediation" in body["error"]


def test_workflow_projection_omits_private_instructions(client: TestClient) -> None:
    workflows = client.get("/elicitation/workflows")
    assert workflows.status_code == 200
    ids = [item["id"] for item in workflows.json()]
    assert ids == ["three-layer-study"]
    projection = client.get("/elicitation/workflows/three-layer-study").json()
    assert "instructions" not in projection
    stage_ids = [stage["id"] for stage in projection["stages"]]
    assert stage_ids == [
        "study-foundation",
        "openness",
        "theory",
        "domain",
        "experiment-design",
    ]


def test_malformed_assistant_output_uses_common_error_envelope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(client)

    class GarbageProvider(ApiScriptedProvider):
        def generate(self, request) -> _json:
            return type("R", (), {"text": "not json", "provider": "s", "model": "m"})()

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", GarbageProvider)
    started = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    response = client.post(
        f"/elicitation/sessions/{started['session_id']}/messages",
        json={"answer": "hello"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ASSISTANT_OUTPUT_INVALID"


def test_wrong_provider_rejects_session_creation(client: TestClient) -> None:
    response = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "x",
            "workflow_id": "three-layer-study",
            "model_profile_id": "missing-profile",
            "researcher_id": "researcher",
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PROFILE_NOT_FOUND"


def test_new_study_web_flow_initializes_the_draft(client: TestClient) -> None:
    """Issue 1: starting elicitation on a missing specification creates a canonical draft."""
    _profile(client)
    started = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "brand-new-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    )
    assert started.status_code == 201
    session_id = started.json()["session_id"]
    assert started.json()["base_specification_version"] == 1
    # The browser flow can now preview the first stage immediately.
    message = client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "A brand-new study foundation."},
    )
    assert message.status_code == 200
    assert message.json()["status"] == "awaiting_approval"
    draft = client.post(f"/elicitation/sessions/{session_id}/draft")
    assert draft.status_code == 200
    preview = client.get(f"/elicitation/sessions/{session_id}/preview")
    assert preview.status_code == 200
    assert "study.yaml" in preview.json()["pending_preview"]["yaml_files"]
    spec = client.get("/specifications/brand-new-study")
    assert spec.status_code == 200


def test_reopen_route_and_expected_version_conflicts(client: TestClient) -> None:
    """Issues 6/8: reopen through the API; mutating routes honour expected versions."""
    _profile(client)
    started = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    session_id = started["session_id"]
    # Stale expected version on draft.
    stale = client.post(
        f"/elicitation/sessions/{session_id}/draft",
        json={"expected_version": 99},
    )
    assert stale.status_code == 409
    # Complete the first stage, then reopen it via the route.
    client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "A foundation answer."},
    )
    draft = client.post(f"/elicitation/sessions/{session_id}/draft")
    assert draft.status_code == 200
    preview = client.get(f"/elicitation/sessions/{session_id}/preview")
    assert preview.status_code == 200
    approve = client.post(
        f"/elicitation/sessions/{session_id}/approve",
        json={"approved_by": "researcher"},
    )
    assert approve.status_code == 200
    assert approve.json()["current_stage"] == "openness"
    reopened = client.post(
        f"/elicitation/sessions/{session_id}/reopen",
        json={"stage_id": "study-foundation"},
    )
    assert reopened.status_code == 200
    assert reopened.json()["current_stage"] == "study-foundation"
    assert reopened.json()["status"] == "awaiting_answer"


def test_session_creation_is_idempotent_with_client_session_id(client: TestClient) -> None:
    """Issue 8: a client-supplied session id is honoured and idempotent."""
    _profile(client)
    first = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
            "session_id": "client-session-1",
        },
    )
    assert first.status_code == 201
    assert first.json()["session_id"] == "client-session-1"
    second = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
            "session_id": "client-session-1",
        },
    )
    assert second.status_code == 201
    assert second.json()["session_id"] == "client-session-1"


def test_all_elicitation_mutations_reject_stale_expected_version(
    client: TestClient,
) -> None:
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    session_id = session["session_id"]
    mutations = [
        ("messages", {"answer": "answer", "expected_version": 99}),
        ("draft", {"expected_version": 99}),
        ("approve", {"approved_by": "researcher", "expected_version": 99}),
        ("revise", {"expected_version": 99}),
        ("reopen", {"stage_id": "study-foundation", "expected_version": 99}),
        ("cancel", {"expected_version": 99}),
    ]
    for route, payload in mutations:
        response = client.post(f"/elicitation/sessions/{session_id}/{route}", json=payload)
        assert response.status_code == 409, route
        assert response.json()["error"]["code"] == "PATCH_BASE_STALE"


def test_elicitation_mutation_idempotency_and_key_conflict(client: TestClient) -> None:
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    session_id = session["session_id"]
    payload = {"answer": "A foundation answer.", "expected_version": 1}
    headers = {"Idempotency-Key": "message-1"}
    first = client.post(
        f"/elicitation/sessions/{session_id}/messages", json=payload, headers=headers
    )
    second = client.post(
        f"/elicitation/sessions/{session_id}/messages", json=payload, headers=headers
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert len(second.json()["turns"]) == 1

    conflict = client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "Different answer.", "expected_version": 1},
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_allowed_actions_follow_elicitation_state(client: TestClient) -> None:
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    session_id = session["session_id"]
    assert session["allowed_actions"] == ["submit_message", "cancel"]

    answered = client.post(
        f"/elicitation/sessions/{session_id}/messages",
        json={"answer": "A foundation answer.", "expected_version": 1},
    ).json()
    assert answered["allowed_actions"] == ["draft", "revise", "cancel"]

    drafted = client.post(
        f"/elicitation/sessions/{session_id}/draft", json={"expected_version": 1}
    ).json()
    assert drafted["allowed_actions"] == [
        "preview",
        "revise",
        "submit_message",
        "draft",
        "edit_draft",
        "cancel",
    ]

    previewed = client.get(f"/elicitation/sessions/{session_id}/preview").json()
    assert previewed["allowed_actions"] == [
        "approve",
        "revise",
        "submit_message",
        "draft",
        "edit_draft",
        "cancel",
    ]


def test_limit_draft_feedback_manual_edit_and_next_stage(client: TestClient) -> None:
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    sid = session["session_id"]
    service = client.app.state.service
    service._elicitation_store.update(
        sid, lambda s: setattr(s.stages[s.current_stage], "turn_count", 3)
    )
    result = client.post(
        f"/elicitation/sessions/{sid}/messages", json={"answer": "Study creator governance."}
    )
    assert result.status_code == 200, result.text
    draft = result.json()
    assert draft["pending_preview"]
    assert draft["stages"]["study-foundation"]["review_mode"]
    assert draft["clarification"]["covered_decisions"] == 0
    assert "approve" in draft["allowed_actions"]
    feedback = client.post(
        f"/elicitation/sessions/{sid}/messages", json={"answer": "Use a narrower scope."}
    )
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["pending_preview"]
    assert len(feedback.json()["turns"]) == 2
    assert feedback.json()["current_suggestions"] == []
    original = feedback.json()["pending_preview"]["yaml_files"]["study.yaml"]
    edited = client.post(
        f"/elicitation/sessions/{sid}/edit",
        json={
            "filename": "study.yaml",
            "content": original.replace("API study", "Researcher edited title"),
            "expected_version": 1,
        },
    )
    assert edited.status_code == 200, edited.text
    assert "Researcher edited title" in edited.json()["pending_preview"]["yaml_files"]["study.yaml"]
    assert service.get_specification("api-study")["version"] == 1
    forbidden = client.post(
        f"/elicitation/sessions/{sid}/edit",
        json={"filename": "theory.yaml", "content": "theory_family: exploratory"},
    )
    assert forbidden.status_code != 200
    approved = client.post(
        f"/elicitation/sessions/{sid}/approve", json={"approved_by": "researcher"}
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["current_stage"] == "openness"
    assert not approved.json()["stages"]["openness"]["review_mode"]
    assert approved.json()["clarification"]["turns_used"] == 0
    next_answer = client.post(
        f"/elicitation/sessions/{sid}/messages",
        json={"answer": "Creators generate messages using a declared model."},
    )
    assert next_answer.status_code == 200, next_answer.text


def test_failed_feedback_preserves_review_and_retry_records_answer_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    sid = session["session_id"]
    client.post(f"/elicitation/sessions/{sid}/messages", json={"answer": "Study governance."})
    client.post(f"/elicitation/sessions/{sid}/draft")
    original = client.get(f"/elicitation/sessions/{sid}/preview").json()
    generate = ApiScriptedProvider.generate

    def unavailable(self, request):
        raise RuntimeError("temporary provider failure")

    monkeypatch.setattr(ApiScriptedProvider, "generate", unavailable)
    failed = client.post(
        f"/elicitation/sessions/{sid}/messages", json={"answer": "Revise the scope."}
    )
    assert failed.status_code != 200
    restored = client.get(f"/elicitation/sessions/{sid}").json()
    assert restored["pending_preview"] == original["pending_preview"]
    assert restored["turns"] == original["turns"]
    monkeypatch.setattr(ApiScriptedProvider, "generate", generate)
    retried = client.post(
        f"/elicitation/sessions/{sid}/messages", json={"answer": "Revise the scope."}
    )
    assert retried.status_code == 200, retried.text
    assert len(retried.json()["turns"]) == len(original["turns"]) + 1
    assert retried.json()["clarification"]["turns_used"] == original["clarification"]["turns_used"]


def test_complete_five_stage_api_flow_recovers_revises_and_compiles(
    client: TestClient,
) -> None:
    """Browser-equivalent acceptance flow without direct canonical-file edits."""
    _profile(client)
    session = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    session_id = session["session_id"]
    key_number = 0

    def mutate(route: str, payload: dict) -> dict:
        nonlocal key_number
        key_number += 1
        response = client.post(
            f"/elicitation/sessions/{session_id}/{route}",
            json={**payload, "expected_version": session["base_specification_version"]},
            headers={"Idempotency-Key": f"complete-{key_number}"},
        )
        if response.status_code != 200:
            inspection = client.post("/specifications/api-study/inspect")
            pytest.fail(f"{response.text}\ninspection={inspection.text}")
        return response.json()

    def complete_current_stage(answer: str) -> None:
        nonlocal session
        session = mutate("messages", {"answer": answer, "response_mode": "free_form"})
        session = mutate("draft", {})
        preview = client.get(f"/elicitation/sessions/{session_id}/preview")
        assert preview.status_code == 200, preview.text
        session = preview.json()
        assert not session["pending_preview"]["validation"]["errors"]
        session = mutate("approve", {"approved_by": "researcher"})

    # The first answer forces one malformed model response and proves repair is
    # transparent to the browser flow.
    complete_current_stage("repair this response: a scoped foundation study")

    # The openness stage first clarifies, then records an edited suggestion.
    session = mutate("messages", {"answer": "needs a timing choice"})
    assert len(session["current_suggestions"]) == 3
    edited = session["current_suggestions"][1]["value"] + " with a one-round delay"
    session = mutate(
        "messages",
        {"answer": edited, "suggestion_index": 1, "response_mode": "free_form"},
    )
    assert session["turns"][-1]["response_mode"] == "edited"
    session = mutate("draft", {})
    session = client.get(f"/elicitation/sessions/{session_id}/preview").json()
    session = mutate("approve", {"approved_by": "researcher"})

    for answer in (
        "Use an exploratory theory family.",
        "Members are the relevant actors.",
        "Run twenty rounds and compare outcomes.",
    ):
        complete_current_stage(answer)
    assert session["status"] == "completed"

    # Reopen Layer 1 and verify the configured dependency closure before
    # reapproving all affected stages.
    session = mutate("reopen", {"stage_id": "openness"})
    assert session["stages"]["theory"]["status"] == "needs_review"
    assert session["stages"]["domain"]["status"] == "needs_review"
    assert session["stages"]["experiment-design"]["status"] == "needs_review"
    complete_current_stage("Keep messages deterministic in the first release.")
    for stage_id in ("theory", "domain", "experiment-design"):
        assert session["current_stage"] == stage_id
        assert session["stages"][stage_id]["status"] == "needs_review"
        session = mutate("approve", {"approved_by": "researcher"})
    assert session["status"] == "completed"
    assert all(item["status"] == "approved" for item in session["stages"].values())

    compiled = client.post(
        "/compile",
        json={"specification_id": "api-study", "output": "builds/api-study-complete"},
    )
    assert compiled.status_code == 200, compiled.text
    assert compiled.json()["study_id"] == "api-study"


def test_session_preview_and_idempotency_survive_restart(client: TestClient) -> None:
    _profile(client)
    started = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()
    sid = started["session_id"]
    route = f"/elicitation/sessions/{sid}"
    body = {"answer": "Study governance."}
    answered = client.post(route + "/messages", json=body, headers={"Idempotency-Key": "first"})
    assert answered.status_code == 200
    assert client.post(route + "/draft").status_code == 200
    preview = client.get(route + "/preview").json()
    workspace = client.app.state.service.workspace
    client.app.state.service.close()
    restarted = TestClient(create_app(workspace))
    try:
        restored = restarted.get(route)
        assert restored.status_code == 200
        assert restored.json() == preview
        assert sid in {
            item["session_id"] for item in restarted.get("/elicitation/sessions").json()["items"]
        }
        replay = restarted.post(
            route + "/messages", json=body, headers={"Idempotency-Key": "first"}
        )
        assert replay.json() == answered.json()
        assert len(restarted.get(route).json()["turns"]) == 1
        approved = restarted.post(route + "/approve", json={"approved_by": "researcher"})
        assert approved.status_code == 200, approved.text
        assert approved.json()["current_stage"] == "openness"
    finally:
        restarted.app.state.service.close()


def test_prose_processes_are_repaired_before_preview(client: TestClient, monkeypatch) -> None:
    _profile(client)
    sid = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()["session_id"]
    route = f"/elicitation/sessions/{sid}"
    client.post(route + "/messages", json={"answer": "Study governance."})
    client.post(route + "/draft")
    client.get(route + "/preview")
    assert client.post(route + "/approve", json={"approved_by": "researcher"}).status_code == 200
    client.post(route + "/messages", json={"answer": "Creators generate articles."})
    original = ApiScriptedProvider.generate
    prompts = []

    def generate(self, request):
        prompts.append(request.prompt)
        response = original(self, request)
        if len(prompts) == 1:
            payload = _json.loads(response.text)
            payload["operations"][0]["value"] = ["create-article — scientific rationale"]
            response.text = _json.dumps(payload)
        return response

    monkeypatch.setattr(ApiScriptedProvider, "generate", generate)
    result = client.post(route + "/draft")
    assert result.status_code == 200, result.text
    assert [a["status"] for a in result.json()["last_assistant_attempts"]] == ["invalid", "valid"]
    assert "ProcessSpec" in prompts[0]
    assert "valid dictionary" in prompts[1]
    assert client.get(route + "/preview").status_code == 200


def test_layer_one_context_reference_defers_only_until_domain(
    client: TestClient, monkeypatch
) -> None:
    _profile(client)
    sid = client.post(
        "/elicitation/sessions",
        json={
            "specification_id": "api-study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        },
    ).json()["session_id"]
    route = f"/elicitation/sessions/{sid}"
    client.post(route + "/messages", json={"answer": "Study creator governance."})
    client.post(route + "/draft")
    client.get(route + "/preview")
    assert client.post(route + "/approve", json={"approved_by": "researcher"}).status_code == 200
    monkeypatch.setitem(
        STAGE_PATCHES,
        "openness",
        [
            {
                "op": "replace",
                "path": "/openness/processes",
                "value": [
                    {
                        "id": "compose-message",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "creator-context",
                    }
                ],
            }
        ],
    )
    client.post(route + "/messages", json={"answer": "Creators observe creator-context."})
    assert client.post(route + "/draft").status_code == 200
    preview = client.get(route + "/preview").json()["pending_preview"]["validation"]
    assert not preview["errors"]
    assert any(w["code"] == "REF_CONTEXT_POLICY" for w in preview["warnings"])
    approved = client.post(route + "/approve", json={"approved_by": "researcher"})
    assert approved.status_code == 200, approved.text
    assert approved.json()["current_stage"] == "theory"
    service = client.app.state.service
    directory = service._specification_dir("api-study")
    workflow = service._workflow_registry.get("three-layer-study")
    domain_check = service._patch_preview._inspect(
        directory, stage=workflow.stage("domain"), workflow=workflow
    )
    assert any(e["code"] == "REF_CONTEXT_POLICY" for e in domain_check["errors"])
    from genesis.compiler import StudyCompiler

    compiler = StudyCompiler(directory)
    assert any(e["code"] == "REF_CONTEXT_POLICY" for e in compiler._validate(compiler._load()))

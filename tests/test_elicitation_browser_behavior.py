"""Behavior-level browser workspace contract without coupling to visual copy."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app


def _html(tmp_path: Path) -> str:
    return TestClient(create_app(tmp_path / "workspace")).get("/ui").text


def test_actions_are_rendered_from_server_allowed_actions(tmp_path: Path) -> None:
    html = _html(tmp_path)
    for marker in (
        "function renderAllowedActions(allowedActions)",
        "session.allowed_actions",
        "submit_message: 'send-answer-btn'",
        "draft: 'draft-btn'",
        "preview: 'preview-btn'",
        "approve: 'approve-btn'",
        "revise: 'revise-btn'",
        "cancel: 'cancel-btn'",
    ):
        assert marker in html
    assert "stageStatus === 'draft_ready'" not in html


def test_workspace_has_intentional_loading_error_and_completed_states(
    tmp_path: Path,
) -> None:
    html = _html(tmp_path)
    for marker in (
        "function setElicitationBusy(isBusy",
        "function showElicitationError(error",
        "retryElicitationAction()",
        'id="elicitation-loading"',
        'id="retry-action-btn"',
        "renderCompletedState(session)",
        "last_assistant_attempts",
    ):
        assert marker in html


def test_review_panel_and_mobile_drawers_are_operable(tmp_path: Path) -> None:
    html = _html(tmp_path)
    for marker in (
        "review-tab",
        "aria-selected",
        "toggleStageRail()",
        "toggleReviewPanel()",
        'id="mobile-stage-toggle"',
        'id="mobile-review-toggle"',
        "@media (max-width: 900px)",
        ":focus-visible",
    ):
        assert marker in html


def test_five_stage_rail_uses_workflow_projection(tmp_path: Path) -> None:
    html = _html(tmp_path)
    for marker in (
        "loadElicitationWorkflow()",
        "elicitationWorkflow.stages",
        "renderStageRail(session)",
        "stage-state-badge",
        "stage-progress-meter",
    ):
        assert marker in html


def test_clarification_budget_is_rendered_from_server_projection(tmp_path: Path) -> None:
    html = _html(tmp_path)
    for marker in (
        'id="elicitation-clarification-progress"',
        "function renderClarificationProgress(session)",
        "session.clarification",
        "turns_remaining",
        "remaining_decisions",
        "limit_reached",
    ):
        assert marker in html

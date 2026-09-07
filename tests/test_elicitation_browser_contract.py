"""Task 8: the chat-first workspace contract in the browser surface."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app


def test_web_ui_ships_the_chat_workspace(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    response = client.get("/ui")
    assert response.status_code == 200
    html = response.text
    for marker in (
        "startElicitation()",
        "elicitation-progress",
        "elicitation-question",
        "elicitation-suggestions",
        "suggestion-card",
        "elicitation-composer",
        "sendElicitation()",
        "approveElicitation()",
        "reviseElicitation()",
        "cancelElicitation()",
        "reopenElicitation()",
        "elicitation-error",
        "elicitation-status-hint",
        "send-answer-btn",
        "draft-btn",
        "approve-btn",
        '<select id="elicitation-profile-id"',
        "loadElicitationProfiles()",
        "requireElicitationSession()",
        "STABLE_ID",
        "elicitation-transcript",
        "elicitation-preview-panel",
        "showElicitationTab('diff')",
        "showElicitationTab('full')",
    ):
        assert marker in html, f"missing chat marker {marker}"


def test_chat_workspace_uses_separate_actions(tmp_path: Path) -> None:
    """Sending an answer must be a distinct action from approving a stage."""
    client = TestClient(create_app(tmp_path / "workspace"))
    html = client.get("/ui").text
    send_index = html.find("function sendElicitation()")
    approve_index = html.find("function approveElicitation()")
    assert send_index != -1 and approve_index != -1
    # Approval requires an explicit separate action, never an auto-approve path.
    assert "autoAppro" not in html


def test_suggestion_selection_provenance_is_submitted(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    html = client.get("/ui").text
    for marker in (
        "selectedSuggestionIndex",
        "selectedSuggestionValue",
        "suggestion_index: selectedSuggestionIndex",
        "resetSuggestionSelection()",
        "trackSuggestionEdit()",
        "expected_version: elicitationVersion",
        "Idempotency-Key",
    ):
        assert marker in html


def test_web_ui_is_tab_based_with_four_workspaces(tmp_path: Path) -> None:
    """The redesigned UI hosts four tabs: models, specification, run, outputs."""
    client = TestClient(create_app(tmp_path / "workspace"))
    html = client.get("/ui").text
    for marker in (
        "switchTab('models')",
        "switchTab('specification')",
        "switchTab('run')",
        "switchTab('outputs')",
        'id="tab-models"',
        'id="tab-specification"',
        'id="tab-run"',
        'id="tab-outputs"',
        "listModelProfiles()",
    ):
        assert marker in html, f"missing tab marker {marker}"
    # Order matters: the tab bar precedes the panels.
    assert html.index("switchTab('models')") < html.index('id="tab-models"')
    # The compile/run controls live in the Run tab, analysis in the Outputs tab.
    assert html.index('id="tab-run"') < html.index("compileStudy()")
    assert html.index('id="tab-outputs"') < html.index("runOutcomes()")


def test_elicitation_workspace_has_semantic_research_layout(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    html = client.get("/ui").text
    for marker in (
        'id="workspace-topbar"',
        'id="elicitation-stage-rail"',
        'id="elicitation-chat-main"',
        'id="elicitation-review-panel"',
        'id="elicitation-action-footer"',
        'aria-label="Study stages"',
        'aria-label="Specification review"',
        'aria-live="polite"',
        'id="elicitation-complete-state"',
        'id="elicitation-diagnostics"',
    ):
        assert marker in html

"""Acceptance tests for the browser-first specification workflow.

These tests intentionally describe the web contract before its implementation:
the assistant creates a reviewable draft, a researcher approves one exact
version, and only then may the service compile and execute it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app

CANONICAL_FILES = {
    "study.yaml",
    "openness.yaml",
    "theory.yaml",
    "domain.yaml",
    "protocol.yaml",
    "outcomes.yaml",
    "models.yaml",
}


def draft_payload(study_id: str = "web-study", *, title: str = "Web study") -> dict:
    """Return the small researcher-facing form payload used by the web flow."""

    return {
        "id": study_id,
        "title": title,
        "description": "A study created through the local specification assistant.",
        "research_question": "How do local governance rules affect cooperation?",
        "owners": ["researcher"],
    }


def create_draft(client: TestClient, study_id: str = "web-study") -> dict:
    response = client.post("/specifications", json=draft_payload(study_id))
    assert response.status_code == 201, response.text
    return response.json()


def approve_draft(client: TestClient, study_id: str = "web-study") -> dict:
    draft = create_draft(client, study_id)
    approved = client.post(
        f"/specifications/{study_id}/approve",
        headers={"If-Match": str(draft["version"])},
        json={"approved_by": "researcher", "decision": "approve"},
    )
    assert approved.status_code == 200, approved.text
    return approved.json()


def test_create_draft_writes_seven_canonical_files_and_returns_draft_status(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(workspace))

    response = client.post("/specifications", json=draft_payload())

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == "web-study"
    assert body["status"] == "draft"
    assert body["version"] == 1
    specification_dir = workspace / ".genesis" / "specifications" / "web-study"
    assert {path.name for path in specification_dir.glob("*.yaml")} == CANONICAL_FILES


def test_assistant_inspection_returns_issues_and_suggestions_without_approval(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    create_draft(client)

    response = client.post("/specifications/web-study/inspect")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "package_validation"
    assert "issues" in body
    assert "suggestions" in body
    assert body["approval_required"] is True
    assert body["status"] == "draft"
    assert client.get("/specifications/web-study").json()["status"] == "draft"

    saved = client.get("/specifications/web-study")
    assert saved.status_code == 200
    assert set(saved.json()["files"]) == CANONICAL_FILES
    assert "study_id: web-study" in saved.json()["files"]["study.yaml"]


def test_editing_increments_version_and_approval_requires_current_version(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    draft = create_draft(client)

    updated = client.put(
        "/specifications/web-study",
        headers={"If-Match": str(draft["version"])},
        json=draft_payload(title="Updated web study"),
    )

    assert updated.status_code == 200, updated.text
    current = updated.json()
    assert current["version"] == draft["version"] + 1
    assert current["title"] == "Updated web study"

    stale = client.post(
        "/specifications/web-study/approve",
        headers={"If-Match": str(draft["version"])},
        json={"approved_by": "researcher", "decision": "approve"},
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["error"]["code"] == "EXPECTED_VERSION"
    assert client.get("/specifications/web-study").json()["status"] == "draft"

    approved = client.post(
        "/specifications/web-study/approve",
        headers={"If-Match": str(current["version"])},
        json={"approved_by": "researcher", "decision": "approve"},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "approved"
    assert approved.json()["approved_version"] == current["version"]
    stored_metadata = (
        tmp_path / "workspace" / ".genesis" / "specifications" / "web-study" / "metadata.json"
    ).read_text()
    assert '"files"' not in stored_metadata


def test_compile_rejects_an_unapproved_specification(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    create_draft(client)

    response = client.post(
        "/compile",
        json={"specification_id": "web-study", "output": "builds/unapproved"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"]["code"] == "SPECIFICATION_NOT_APPROVED"
    assert not (tmp_path / "workspace" / "builds" / "unapproved").exists()


def test_approved_specification_compiles_to_a_verified_build(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(workspace))
    approve_draft(client)

    response = client.post(
        "/compile",
        json={"specification_id": "web-study", "output": "builds/web-study"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["study_id"] == "web-study"
    assert body["build_hash"]
    assert (workspace / "builds" / "web-study" / "build_manifest.json").is_file()


def test_run_execution_endpoint_completes_a_run_from_an_approved_build(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    approve_draft(client)
    compiled = client.post(
        "/compile",
        json={"specification_id": "web-study", "output": "builds/web-study"},
    )
    assert compiled.status_code == 200, compiled.text

    created = client.post(
        "/runs",
        json={
            "id": "web-run",
            "study_id": "web-study",
            "build": "builds/web-study",
        },
    )
    assert created.status_code == 201, created.text

    executed = client.post("/runs/web-run/execute")

    assert executed.status_code == 200, executed.text
    assert executed.json()["status"] == "completed"
    assert client.get("/runs/web-run").json()["status"] == "completed"


def test_draft_approval_and_run_status_survive_application_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = TestClient(create_app(workspace))
    approved = approve_draft(first)
    assert approved["status"] == "approved"
    compiled = first.post(
        "/compile",
        json={"specification_id": "web-study", "output": "builds/web-study"},
    )
    assert compiled.status_code == 200, compiled.text
    assert (
        first.post(
            "/runs", json={"id": "web-run", "study_id": "web-study", "build": "builds/web-study"}
        ).status_code
        == 201
    )
    assert first.post("/runs/web-run/execute").json()["status"] == "completed"

    restarted = TestClient(create_app(workspace))

    assert restarted.get("/specifications/web-study").json()["status"] == "approved"
    assert restarted.get("/specifications/web-study").json()["version"] == approved["version"]
    assert restarted.get("/runs/web-run").json()["status"] == "completed"


def test_specification_and_compile_paths_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(workspace))

    escaped_id = client.post("/specifications", json=draft_payload("../escape"))
    assert escaped_id.status_code == 422, escaped_id.text
    assert escaped_id.json()["error"]["code"] == "INVALID_ID"

    approve_draft(client)
    escaped_output = client.post(
        "/compile",
        json={"specification_id": "web-study", "output": "../escape-build"},
    )
    assert escaped_output.status_code == 422, escaped_output.text
    assert escaped_output.json()["error"]["code"] == "WORKSPACE_CONTAINMENT"
    assert not (tmp_path / "escape-build").exists()


def test_ui_exposes_the_approval_gated_workflow(tmp_path: Path) -> None:
    """The chat-first specification surface and approval-gated run controls ship."""
    client = TestClient(create_app(tmp_path / "workspace"))
    response = client.get("/ui")
    assert response.status_code == 200
    # Chat-first elicitation is the intended authoring path (ACC-013).
    assert "Start guided specification" in response.text
    assert "Send response" in response.text
    assert "Approve this stage" in response.text
    assert "Request revision" in response.text
    assert "Reopen a stage" in response.text
    # Imported specifications bridge saved/exported packages into the chat.
    assert "Import an existing specification" in response.text
    assert "importSpecification()" in response.text
    # Run tab raises the approval gate before compile/execute.
    assert "Compile approved draft" in response.text
    assert "Execute experiment" in response.text
    assert "Study ID" in response.text
    # Model configuration stays in the Models tab.
    assert "Model configuration" in response.text
    assert "Save model profile" in response.text
    assert "Test connection" in response.text
    # The old guided single-draft form is gone.
    assert "Guided draft form" not in response.text
    assert "Generate draft" not in response.text


def test_validation_errors_are_json_even_when_request_body_is_not_decoded(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))

    response = client.post(
        "/specifications/web-study/approve",
        headers={"If-Match": "1"},
        content=b'{"approved_by":"researcher"}',
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_web_ui_exposes_workspace_and_patch_review_controls(tmp_path: Path) -> None:
    """AW-12/AW-02: the browser surface ships dashboard, explorer, replay and patch controls."""
    client = TestClient(create_app(tmp_path / "workspace"))
    response = client.get("/ui")
    assert response.status_code == 200
    html = response.text
    for marker in (
        "dashboardView()",
        "traceExplorer()",
        "replayWorkspace()",
        "processMap()",
        "reviewPatch()",
        "listExperiments()",
        "listBuilds()",
    ):
        assert marker in html, f"missing UI control {marker}"

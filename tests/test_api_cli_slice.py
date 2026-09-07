from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.cli import build_parser


def test_studies_are_idempotent_and_use_common_error_envelope() -> None:
    client = TestClient(create_app())
    payload = {"id": "demo-study", "title": "Demo"}
    first = client.post("/studies", json=payload, headers={"Idempotency-Key": "create-1"})
    second = client.post("/studies", json=payload, headers={"Idempotency-Key": "create-1"})
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    missing = client.get("/studies/no-such-study")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_run_controls_and_expected_version() -> None:
    client = TestClient(create_app())
    created = client.post("/runs", json={"id": "run-1", "study_id": "demo-study"})
    assert created.status_code == 201
    paused = client.post("/runs/run-1/pause", headers={"If-Match": '"0"'})
    assert paused.status_code == 200
    stale = client.post("/runs/run-1/resume", headers={"If-Match": '"0"'})
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "EXPECTED_VERSION"


def test_api_groups_and_cli_surface_exist() -> None:
    client = TestClient(create_app())
    for path in (
        "/providers",
        "/extensions",
        "/runs/run-missing/events",
        "/runs/run-missing/artifacts",
    ):
        response = client.get(path)
        assert response.status_code in {200, 404}
    commands = set(build_parser()._subparsers._group_actions[0].choices)
    assert {
        "init",
        "validate",
        "compile",
        "run",
        "status",
        "pause",
        "resume",
        "cancel",
        "trace",
        "replay",
        "outcomes",
        "export",
        "import",
        "doctor",
        "integrity-check",
    } <= commands

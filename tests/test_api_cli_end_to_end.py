import json
from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.cli import main


def test_api_state_survives_application_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    first = TestClient(create_app(workspace))
    created = first.post("/studies", json={"id": "demo-study", "title": "Demo"})
    assert created.status_code == 201

    second = TestClient(create_app(workspace))
    fetched = second.get("/studies/demo-study")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Demo"


def test_api_rejects_malformed_expected_version(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    assert client.post("/runs", json={"id": "run-1"}).status_code == 201
    response = client.post("/runs/run-1/pause", headers={"If-Match": "not-an-integer"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_cli_uses_the_same_durable_workspace(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    main(["init", str(workspace)])
    assert json.loads(capsys.readouterr().out)["status"] == "initialized"
    main(["run", str(workspace), "--run-id", "run-1"])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    main(["status", str(workspace), "--run-id", "run-1"])
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_cli_compiles_across_filesystem_roots(tmp_path: Path, capsys) -> None:
    source = Path(__file__).parent / "golden_studies" / "platform_governance"
    output = tmp_path / "compiled"

    main(["compile", str(source), "--output", str(output)])

    result = json.loads(capsys.readouterr().out)
    assert result["study_id"] == "platform-governance"
    assert (output / "integrity_manifest.json").is_file()

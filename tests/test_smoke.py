"""Smoke tests for the GENESIS application foundation."""

from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.cli import main


def test_application_factory_exposes_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "genesis"}


def test_cli_version_reports_package_version(capsys) -> None:
    main(["version"])

    assert capsys.readouterr().out.strip() == "0.1.0"

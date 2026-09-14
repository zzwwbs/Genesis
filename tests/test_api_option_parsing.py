"""Bad request values are client errors, and flags must be booleans (2026-09-14 M13/M14).

`int("x")` and `ReplayMode("bogus")` raised bare ValueErrors that the error
mapper reported as HTTP 500 with a traceback; `bool("false")` is True, so a
client sending the string enabled parallel dispatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


@pytest.mark.parametrize(
    "body",
    [
        {"replications": "two"},
        {"max_events": "lots"},
        {"max_workers": 2.5},
        {"replications": True},
        {"parallel": "false"},
        {"plan": "true"},
    ],
)
def test_malformed_protocol_options_are_refused_as_client_errors(
    client: TestClient, body: dict
) -> None:
    response = client.post("/runs/missing/protocol", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("route", ["/runs/missing/replays/preview", "/runs/missing/replays"])
def test_an_unknown_replay_mode_is_a_client_error(client: TestClient, route: str) -> None:
    response = client.post(route, json={"mode": "bogus"})
    assert response.status_code == 422, response.text
    assert "replay mode must be one of" in response.json()["error"]["message"]


def test_an_import_size_limit_must_be_an_integer(client: TestClient) -> None:
    response = client.post("/runs/import", json={"source": "x", "size_limit_bytes": "big"})
    assert response.status_code == 422

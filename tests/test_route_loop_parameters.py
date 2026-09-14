"""Routes built in a loop must not publish their loop variable as a query field.

`_action` and `_field` were default arguments, which FastAPI exposes, so
`POST /runs/r/pause?_action=cancel` cancelled the run and
`POST /runs/r/events?_field=outcomes` wrote into its outcomes.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from genesis.app import create_app


def test_no_generated_route_accepts_its_loop_variable(tmp_path: Path) -> None:
    paths = TestClient(create_app(tmp_path)).get("/openapi.json").json()["paths"]
    for suffix in ("pause", "resume", "cancel", "events", "artifacts", "outcomes"):
        names = [
            p["name"] for p in paths[f"/runs/{{run_id}}/{suffix}"]["post"].get("parameters", [])
        ]
        assert names == ["run_id"], (suffix, names)

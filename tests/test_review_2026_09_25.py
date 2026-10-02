"""Four findings from the 2026-09-25 code review.

An invocation id that two different turns could spell, and a retry cursor that
then silently skipped the second; a world id taken from a file name that the
engine later refused, after the runs existed; a record of stripped model
parameters that named ones never sent; and a protocol route that could not set
a per-profile concurrency limit.
"""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from genesis.providers import OpenAICompatibleProvider, ProviderRequest
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
)
from genesis.service import _resolve_initializations


class _Ledger:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def commit_process_result(self, event, state, artifacts, fail_after=None, **_: Any) -> None:
        self.events.append(dict(event))

    def latest_json_state(self, run_id):
        return None

    def list_events(self, run_id):
        return list(self.events)

    def list_artifacts(self, run_id):
        return []


class _Runs:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def executor(self, name: str) -> Any:
        runs = self

        class Executor:
            def execute(self, invocation):
                runs.ran.append(name)
                return ProcessResult(outputs={})

        return Executor()


COLLIDING = [
    {
        "id": "vote",
        "actors": ["final-x"],
        "context_policy": "private",
        "trigger": {"phase": 1, "repeat": True},
    },
    {
        "id": "vote-final",
        "actors": ["x"],
        "context_policy": "private",
        "trigger": {"phase": 1, "repeat": True},
    },
]


@pytest.mark.parametrize("persisted", [True, False])
def test_two_turns_that_spell_one_invocation_id_are_refused(persisted: bool) -> None:
    """`vote` + `final-x` and `vote-final` + `x` both spell `r-vote-final-x-1`.

    With a ledger, the retry cursor gave the second turn the first one's
    committed attempt: it ran nothing, was counted executed, and the run
    reported completed.
    """
    runs = _Runs()
    controller = RunController(
        Scheduler([dict(process) for process in COLLIDING]),
        ExecutorRegistry(
            {"vote": runs.executor("vote"), "vote-final": runs.executor("vote-final")}
        ),
        ContextEngine({"private": {"allow": []}}),
        persistence=_Ledger() if persisted else None,
    )
    with pytest.raises(ValueError, match="INVOCATION_ID_COLLISION"):
        controller.run("r", phase_limit=2, seed=1)
    assert controller.status != "completed"
    assert "vote-final" not in runs.ran


def test_a_collision_with_a_turn_committed_before_a_pause_is_refused_on_resume() -> None:
    """Ownership is read back from the ledger, not only from this process."""
    ledger = _Ledger()
    ledger.events.append(
        {
            "event_id": "r-vote-final-x-1-attempt-1",
            "invocation_id": "r-vote-final-x-1",
            "process_id": "vote",
            "actors": ["final-x"],
            "attempt": 1,
            "kind": "process_completed",
            "phase": 1,
        }
    )
    runs = _Runs()
    controller = RunController(
        Scheduler([dict(COLLIDING[1])]),
        ExecutorRegistry({"vote-final": runs.executor("vote-final")}),
        ContextEngine({"private": {"allow": []}}),
        persistence=ledger,
    )
    with pytest.raises(ValueError, match="INVOCATION_ID_COLLISION"):
        controller.run("r", phase_limit=2, seed=1)
    assert runs.ran == []


def test_turns_that_do_not_collide_all_run() -> None:
    runs = _Runs()
    processes = [
        {
            "id": "vote",
            "actors": ["a"],
            "context_policy": "private",
            "trigger": {"phase": 1, "repeat": True},
        },
        {
            "id": "tally",
            "actors": ["a"],
            "context_policy": "private",
            "trigger": {"phase": 1, "repeat": True},
        },
    ]
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"vote": runs.executor("vote"), "tally": runs.executor("tally")}),
        ContextEngine({"private": {"allow": []}}),
        persistence=_Ledger(),
    )
    controller.run("r", phase_limit=2, seed=1)
    assert sorted(runs.ran) == ["tally", "vote"]
    assert controller.status == "completed"


def _empirical_build(tmp_path: Path, *names: str) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "initialization.json").write_text(json.dumps({"mode": "empirical"}))
    for name in names:
        (tmp_path / "data" / name).write_text("{}")
    return tmp_path


@pytest.mark.parametrize("name", ["world_c.json", "texas_2020.json", "World-C.json"])
def test_a_world_id_that_no_run_id_could_carry_is_refused_before_any_run(
    tmp_path: Path, name: str
) -> None:
    """The engine refused `exp-world_c-base-1` -- after the runs existed."""
    build = _empirical_build(tmp_path, name)
    with pytest.raises(ValueError, match="not lowercase kebab-case"):
        _resolve_initializations([f"data/{name}"], build)


def test_an_underscored_file_can_still_be_used_with_an_explicit_world_id(
    tmp_path: Path,
) -> None:
    build = _empirical_build(tmp_path, "world_c.json")
    worlds = _resolve_initializations(
        [{"id": "world-c", "data_source": "data/world_c.json"}], build
    )
    assert worlds == [{"id": "world-c", "data_source": "data/world_c.json"}]


class _RejectsMaxTokens(BaseHTTPRequestHandler):
    sent: list[list[str]] = []

    def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        keys = ("temperature", "max_tokens", "max_completion_tokens")
        type(self).sent.append(sorted(key for key in body if key in keys))
        if "max_tokens" in body:
            code, payload = (
                400,
                {
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens' is not supported with "
                        "this model",
                        "code": "unsupported_parameter",
                    }
                },
            )
        else:
            code, payload = (
                200,
                {
                    "id": "x",
                    "model": "m",
                    "choices": [{"message": {"content": '{"ok": true}'}}],
                    "usage": {},
                },
            )
        out = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.mark.parametrize(
    ("parameters", "degraded"),
    [
        ({"max_tokens": 128}, ["max_tokens"]),
        ({"max_tokens": 128, "temperature": 0.7}, ["max_tokens"]),
    ],
)
def test_the_degraded_parameter_record_names_only_what_was_removed(
    parameters: dict[str, Any], degraded: list[str]
) -> None:
    """A request with only max_tokens was recorded as having lost temperature too."""
    server = HTTPServer(("127.0.0.1", 0), _RejectsMaxTokens)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="m",
            api_key_env="GENESIS_UNUSED",
            api_key="k",
            timeout=10,
            max_retries=0,
        )
        response = provider.generate(ProviderRequest(model="m", prompt="hi", parameters=parameters))
    finally:
        server.shutdown()
    assert response.metadata.get("degraded_parameters") == degraded


def test_the_protocol_route_takes_a_per_profile_concurrency_limit() -> None:
    """Only the integer form was accepted; now the service checks either form."""
    from genesis.app import _concurrency_option

    assert _concurrency_option({"study-model": 8}) == {"study-model": 8}
    assert _concurrency_option(8) == 8
    # The scalar form keeps its old coercion, so an existing client is unaffected.
    assert _concurrency_option("8") == 8
    assert _concurrency_option(None) is None


def test_a_bad_per_profile_limit_is_refused_before_any_cell_is_created(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "genesis-workspace" / "builds" / "clickbait-2x2-v6"
    if not source.is_dir():
        pytest.skip("genesis-workspace/ is local-only")
    from fastapi.testclient import TestClient

    from genesis.app import create_app
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    service = GenesisService(workspace)
    try:
        shutil.copytree(source, workspace / "builds" / "v6")
        service.create_run({"id": "proto", "study_id": "clickbait-2x2-v6", "build": "builds/v6"})
    finally:
        service.close()

    client = TestClient(create_app(workspace))

    def post(limit: Any) -> int:
        return client.post(
            "/runs/proto/protocol",
            json={
                "plan": True,
                "only_conditions": ["governance-none-peer-visibility-low"],
                "max_concurrency": limit,
            },
        ).status_code

    assert post({"study-model": 8}) == 200
    assert post({"unknown-profile": 8}) == 422
    assert post({"study-model": 0}) == 422

    service = GenesisService(workspace)
    try:
        assert [run["id"] for run in service.list_runs()] == ["proto"]
    finally:
        service.close()

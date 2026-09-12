"""Concurrent model calls (CON-001..CON-016).

Step 1 — provider backoff and operational concurrency settings (CON-001..003).
All provider traffic goes to a local scripted server; nothing leaves the machine.
"""

from __future__ import annotations

import http.server
import json
import socket
import socketserver
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.cli import _max_concurrency, build_parser
from genesis.providers import (
    OpenAICompatibleProvider,
    ProviderExecutor,
    ProviderRequest,
    ProviderResponse,
    retry_wait,
)
from genesis.runtime import ProcessInvocation
from genesis.service import GenesisService, _check_profile_drift

KEY_ENV = "GENESIS_CONCURRENCY_TEST_KEY"
OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()


class _FastServer(http.server.ThreadingHTTPServer):
    """Skip HTTPServer.server_bind's reverse-DNS getfqdn() call."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port


@contextmanager
def scripted_server(
    script: list[tuple[int, dict[str, str]]],
) -> Iterator[tuple[str, list[float]]]:
    """Answer each POST with the next (status, headers); the last entry repeats."""
    hits: list[float] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, headers = script[min(len(hits), len(script) - 1)]
            hits.append(time.monotonic())
            stall = headers.get("X-Test-Stall")
            if stall:
                # Promise a body, then send none: the client's read must time out.
                self.send_response(status)
                self.send_header("Content-Length", "1000")
                self.end_headers()
                self.wfile.flush()
                time.sleep(float(stall))
                return
            body = OK_BODY if status == 200 else b'{"error": "unavailable"}'
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            pass

    server = _FastServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", hits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _provider(base_url: str, monkeypatch: pytest.MonkeyPatch, **options: Any):
    monkeypatch.setenv(KEY_ENV, "test-key")
    settings = {"backoff_base": 0.01, "backoff_cap": 0.05, "timeout": 5.0, **options}
    return OpenAICompatibleProvider(base_url=base_url, model="m", api_key_env=KEY_ENV, **settings)


def _request() -> ProviderRequest:
    return ProviderRequest(model="m", prompt="hello")


# --- CON-001: transient failures are retried with backoff ---------------------


def test_rate_limit_honours_retry_after_then_succeeds(monkeypatch) -> None:
    with scripted_server([(429, {"Retry-After": "1"}), (200, {})]) as (url, hits):
        response = _provider(url, monkeypatch, backoff_cap=5.0).generate(_request())
    assert response.text == "ok"
    assert len(hits) == 2
    assert hits[1] - hits[0] >= 0.9
    assert response.metadata["retries"] == [{"retry": 1, "status": 429, "wait_seconds": 1.0}]


def test_server_errors_are_retried_until_success(monkeypatch) -> None:
    with scripted_server([(503, {}), (502, {}), (200, {})]) as (url, hits):
        response = _provider(url, monkeypatch).generate(_request())
    assert len(hits) == 3
    assert [retry["status"] for retry in response.metadata["retries"]] == [503, 502]


def test_a_successful_first_call_records_no_retries(monkeypatch) -> None:
    with scripted_server([(200, {})]) as (url, hits):
        response = _provider(url, monkeypatch).generate(_request())
    assert len(hits) == 1
    assert "retries" not in response.metadata


def test_request_errors_are_not_retried(monkeypatch) -> None:
    with scripted_server([(400, {})]) as (url, hits):
        with pytest.raises(ValueError, match=r"PROVIDER_HTTP: provider returned HTTP 400: "):
            _provider(url, monkeypatch).generate(_request())
    assert len(hits) == 1


@pytest.mark.parametrize("max_retries", [0, 2])
def test_retries_are_bounded(monkeypatch, max_retries: int) -> None:
    with scripted_server([(503, {})]) as (url, hits):
        with pytest.raises(ValueError, match="PROVIDER_HTTP: provider returned HTTP 503") as err:
            _provider(url, monkeypatch, max_retries=max_retries).generate(_request())
    assert len(hits) == max_retries + 1
    assert ("after 2 retries" in str(err.value)) is (max_retries == 2)


def test_an_unreachable_provider_is_retried_then_reported_unavailable(monkeypatch) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    provider = _provider(f"http://127.0.0.1:{port}/v1", monkeypatch, max_retries=2)
    with pytest.raises(ValueError, match="PROVIDER_UNAVAILABLE: .*after 2 retries"):
        provider.generate(_request())


def test_cancelling_interrupts_a_backoff_wait(monkeypatch) -> None:
    cancel = threading.Event()
    with scripted_server([(429, {"Retry-After": "30"})]) as (url, hits):
        provider = _provider(url, monkeypatch, backoff_cap=60.0, cancel_event=cancel)
        threading.Timer(0.2, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(ValueError, match="PROVIDER_CANCELLED"):
            provider.generate(_request())
    assert time.monotonic() - started < 5
    assert len(hits) == 1


def test_a_stalled_error_body_is_retried_rather_than_escaping(monkeypatch) -> None:
    with scripted_server([(503, {"X-Test-Stall": "1.5"}), (200, {})]) as (url, hits):
        response = _provider(url, monkeypatch, timeout=0.5).generate(_request())
    assert response.text == "ok"
    assert len(hits) == 2


def test_certificate_failures_are_not_retried() -> None:
    import ssl
    import urllib.error

    from genesis.providers import _transient_connection_error

    refused = ssl.SSLCertVerificationError("certificate verify failed")
    assert not _transient_connection_error(urllib.error.URLError(refused))
    assert _transient_connection_error(urllib.error.URLError(ConnectionRefusedError()))
    assert _transient_connection_error(TimeoutError())


def test_retry_wait_prefers_retry_after_and_caps_it() -> None:
    assert retry_wait(1, "7", cap=60.0) == 7.0
    assert retry_wait(1, "120", cap=60.0) == 60.0
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
    assert 3.0 <= retry_wait(1, later, cap=60.0) <= 5.5
    earlier = format_datetime(datetime.now(UTC) - timedelta(seconds=5), usegmt=True)
    assert retry_wait(1, earlier, cap=60.0) == 0.0


def test_retry_wait_backs_off_exponentially_with_full_jitter() -> None:
    class Ceiling:
        def uniform(self, low: float, high: float) -> float:
            assert low == 0.0
            return high

    assert retry_wait(1, None, base=1.0, cap=60.0, rng=Ceiling()) == 1.0  # type: ignore[arg-type]
    assert retry_wait(4, None, base=1.0, cap=60.0, rng=Ceiling()) == 8.0  # type: ignore[arg-type]
    assert retry_wait(10, None, base=1.0, cap=60.0, rng=Ceiling()) == 60.0  # type: ignore[arg-type]
    assert retry_wait(1, "soon", base=1.0, cap=60.0, rng=Ceiling()) == 1.0  # type: ignore[arg-type]


def test_the_executor_keeps_provider_retries_on_the_attempt() -> None:
    class Provider:
        provider = "scripted"

        def __init__(self, metadata: dict[str, Any]) -> None:
            self.metadata = metadata

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse("x", self.provider, request.model, "r", metadata=self.metadata)

    invocation = ProcessInvocation("i1", "r1", "p1", context={"x": 1})
    retried = ProviderExecutor(
        Provider({"retries": [{"retry": 1, "status": 429, "wait_seconds": 0.5}]}), model="m"
    ).execute(invocation)
    # ProcessResult freezes metadata; compare plain copies.
    assert [dict(item) for item in retried.metadata["provider_attempts"][0]["retries"]] == [
        {"retry": 1, "status": 429, "wait_seconds": 0.5}
    ]
    clean = ProviderExecutor(Provider({}), model="m").execute(invocation)
    assert "retries" not in clean.metadata["provider_attempts"][0]


# --- CON-002: operational profile settings -------------------------------------

LIVE_PROFILE = {
    "id": "live",
    "provider": "openai-compatible",
    "base_url": "https://example.test/v1",
    "model": "m",
    "api_key_env": "GENESIS_CONCURRENCY_TEST_KEY",
}


def test_profiles_default_and_validate_operational_settings(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        stored = service.create_model_profile(LIVE_PROFILE)
        assert (stored["max_concurrency"], stored["max_retries"]) == (1, 3)
        tuned = service.create_model_profile(
            {**LIVE_PROFILE, "id": "tuned", "max_concurrency": 8, "max_retries": 0}
        )
        assert (tuned["max_concurrency"], tuned["max_retries"]) == (8, 0)
        for bad in (0, 65, True, "4", 2.0):
            with pytest.raises(ValueError, match="MAX_CONCURRENCY"):
                service.create_model_profile({**LIVE_PROFILE, "id": "bad", "max_concurrency": bad})
        for bad in (-1, 11, False):
            with pytest.raises(ValueError, match="MODEL_RETRIES"):
                service.create_model_profile({**LIVE_PROFILE, "id": "bad", "max_retries": bad})
        pooled = service.create_model_profile(
            {"id": "pooled", "provider": "answer-pool", "model": "m", "pool": "p.json"}
        )
        assert pooled["max_concurrency"] == 1 and "max_retries" not in pooled
    finally:
        service.close()


@pytest.mark.parametrize("field", ["max_concurrency", "max_retries"])
def test_operational_settings_are_refused_inside_parameters(tmp_path: Path, field: str) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match=f"MODEL_PARAMETERS: {field}"):
            service.create_model_profile({**LIVE_PROFILE, "parameters": {field: 4}})
    finally:
        service.close()


def test_operational_settings_reset_to_defaults_and_are_always_validated(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        tuned = service.create_model_profile(
            {**LIVE_PROFILE, "max_concurrency": 8, "max_retries": 0}
        )
        reset = service.update_model_profile(
            "live", {"max_concurrency": None, "max_retries": None}, tuned["version"]
        )
        assert (reset["max_concurrency"], reset["max_retries"]) == (1, 3)
        with pytest.raises(ValueError, match="MODEL_RETRIES"):
            service.create_model_profile(
                {
                    "id": "pooled",
                    "provider": "answer-pool",
                    "model": "m",
                    "pool": "p.json",
                    "max_retries": "garbage",
                }
            )
    finally:
        service.close()


def test_interactive_provider_paths_bound_their_retries(tmp_path: Path, monkeypatch) -> None:
    constructed: list[dict[str, Any]] = []

    class Spy:
        provider = "openai-compatible"

        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse("OK", self.provider, request.model, "r")

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", Spy)
    service = GenesisService(tmp_path / "ws")
    try:
        service.create_model_profile({**LIVE_PROFILE, "max_retries": 1})
        service.test_model_profile("live")
        service._elicitation_provider("live")
        # The connectivity check never retries; elicitation follows the profile.
        assert [kwargs["max_retries"] for kwargs in constructed] == [0, 1]
    finally:
        service.close()


def test_operational_settings_leave_identity_and_drift_untouched(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    try:
        slow = service._validate_model_profile(LIVE_PROFILE)
        fast = service._validate_model_profile(
            {**LIVE_PROFILE, "max_concurrency": 16, "max_retries": 7}
        )
        assert service._profile_identity(slow) == service._profile_identity(fast)
        compiled = {"provider": "openai-compatible", "model": "m", "parameters": {}}
        _check_profile_drift("live", fast, compiled, "0" * 64)
    finally:
        service.close()


# --- CON-003: per-run override, recorded outside the manifest -----------------

STUDY = {
    "id": "concurrency-study",
    "title": "concurrency study",
    "models": [{"id": "mp", "provider": "openai-compatible", "model": "m1", "parameters": {}}],
    "processes": [
        {
            "id": "compose",
            "openness_rationale": "content form is the phenomenon",
            "closure_rationale": "a fixed corpus would remove the variation",
            "executor": {"mode": "generative", "model_profile": "mp"},
            "context_policy": "public",
            "prompt_ref": "compose",
            "outputs": [{"artifact_type": "text", "schema_ref": "compose-out"}],
        }
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {"artifacts": [{"id": "compose-out", "artifact_type": "text"}]},
    "protocol": {"time_model": {"type": "rounds", "end": 1}},
    "outcomes": [],
    "prompts": {"compose": "Compose from {context}"},
}


class _TextProvider:
    provider = "openai-compatible"
    constructed: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).constructed.append(kwargs)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        text = json.dumps({"text": "generated"})
        metadata = {"retries": list(self.retries)} if self.retries else {}
        return ProviderResponse(
            text, self.provider, request.model, "r", parsed=json.loads(text), metadata=metadata
        )


def _compiled_service(
    tmp_path: Path, monkeypatch, study: dict[str, Any] | None = None, **profile: Any
) -> GenesisService:
    chosen = study or STUDY
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _TextProvider)
    _TextProvider.constructed.clear()
    _TextProvider.retries = []
    workspace = tmp_path / "ws"
    service = GenesisService(workspace)
    service.create_model_profile(
        {**LIVE_PROFILE, "id": "mp", "model": "m1", "api_key_env": "GENESIS_C_KEY", **profile}
    )
    draft = service.create_specification(chosen)
    schemas = workspace / ".genesis" / "specifications" / chosen["id"] / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    revised = service.update_specification(chosen["id"], {"description": "x"}, draft["version"])
    service.approve_specification(chosen["id"], revised["version"], "researcher")
    compiled = service.compile_study(None, "builds/c", specification_id=chosen["id"])
    for run_id in ("run-a", "run-b"):
        service.create_run({"id": run_id, "study_id": chosen["id"], "build": compiled["path"]})
    return service


def test_runs_record_effective_concurrency_outside_the_manifest(tmp_path, monkeypatch) -> None:
    service = _compiled_service(tmp_path, monkeypatch, max_concurrency=3)
    try:
        assert service.execute_run("run-a")["status"] == "completed"
        assert service.execute_run("run-b", max_concurrency={"mp": 5})["status"] == "completed"
        first, second = service.get_run("run-a"), service.get_run("run-b")
        assert first["executions"][-1]["model_concurrency"] == {
            "mp": {"max_concurrency": 3, "source": "profile"}
        }
        assert second["executions"][-1]["model_concurrency"] == {
            "mp": {"max_concurrency": 5, "source": "override"}
        }
        assert "max_concurrency" not in json.dumps(first["manifest"])
    finally:
        service.close()


def test_runs_pass_the_profile_retry_bound_to_the_provider(tmp_path, monkeypatch) -> None:
    service = _compiled_service(tmp_path, monkeypatch, max_retries=0)
    try:
        service.execute_run("run-a")
        assert _TextProvider.constructed
        assert {kwargs["max_retries"] for kwargs in _TextProvider.constructed} == {0}
    finally:
        service.close()


def test_provider_retries_reach_the_recorded_events(tmp_path, monkeypatch) -> None:
    service = _compiled_service(tmp_path, monkeypatch)
    _TextProvider.retries = [{"retry": 1, "status": 429, "wait_seconds": 0.25}]
    try:
        service.execute_run("run-a")
        completed = [
            event
            for event in service.trace_run("run-a")
            if event.get("kind") == "process_completed"
        ]
        assert completed
        for event in completed:
            assert event["metadata"]["provider_attempts"][0]["retries"] == [
                {"retry": 1, "status": 429, "wait_seconds": 0.25}
            ]
    finally:
        service.close()


def test_a_resumed_run_records_a_second_execution(tmp_path, monkeypatch) -> None:
    service = _compiled_service(tmp_path, monkeypatch)
    calls: list[ProviderRequest] = []
    original = _TextProvider.generate

    def pause_on_first_call(self: _TextProvider, request: ProviderRequest) -> ProviderResponse:
        if not calls:
            run = service.get_run("run-a")
            service.transition_run("run-a", "paused", run["version"])
        calls.append(request)
        return original(self, request)

    monkeypatch.setattr(_TextProvider, "generate", pause_on_first_call)
    try:
        assert service.execute_run("run-a")["status"] == "paused"
        assert service.execute_run("run-a", max_concurrency=2)["status"] == "completed"
        executions = service.get_run("run-a")["executions"]
        assert [execution["model_concurrency"]["mp"] for execution in executions] == [
            {"max_concurrency": 1, "source": "profile"},
            {"max_concurrency": 2, "source": "override"},
        ]
    finally:
        service.close()


@pytest.mark.parametrize("override", [0, {"unknown-profile": 2}, {"mp": True}, "4"])
def test_a_bad_override_is_refused_before_the_run_starts(tmp_path, monkeypatch, override) -> None:
    service = _compiled_service(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="MAX_CONCURRENCY"):
            service.execute_run("run-a", max_concurrency=override)
        assert service.get_run("run-a")["status"] == "created"
    finally:
        service.close()


def test_cli_parses_max_concurrency() -> None:
    assert _max_concurrency(None) is None
    assert _max_concurrency(["4"]) == 4
    assert _max_concurrency(["mp=2", "other=3"]) == {"mp": 2, "other": 3}
    for bad in (["x"], ["4", "5"], ["4", "mp=2"], ["mp=2", "mp=3"], ["4_0"], [" 4"], ["-1"]):
        with pytest.raises(SystemExit):
            _max_concurrency(bad)
    args = build_parser().parse_args(["run", ".", "--max-concurrency", "3"])
    assert args.max_concurrency == ["3"]


@pytest.mark.parametrize("value", ["x", "4_0", " 4", "0", "65", "mp=4", "a=1=2", "=5"])
def test_cli_refuses_a_bad_max_concurrency_before_creating_the_run(
    tmp_path: Path, value: str
) -> None:
    from genesis.cli import main

    workspace = tmp_path / "ws"
    with pytest.raises(SystemExit):
        main(["run", str(workspace), "--run-id", "stray", "--max-concurrency", value])
    service = GenesisService(workspace)
    try:
        with pytest.raises(KeyError):
            service.get_run("stray")
    finally:
        service.close()


def test_api_execute_accepts_only_known_options(tmp_path: Path) -> None:
    service = GenesisService(tmp_path / "ws")
    seen: list[Any] = []

    def execute_run(run_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.append((run_id, kwargs))
        return {"id": run_id, "status": "completed"}

    service.execute_run = execute_run  # type: ignore[method-assign]
    try:
        client = TestClient(create_app(service=service))
        assert client.post("/runs/r1/execute").status_code == 200
        assert client.post("/runs/r1/execute", json={"max_concurrency": 4}).status_code == 200
        refused = client.post("/runs/r1/execute", json={"parallel": True})
        assert refused.status_code >= 400
        assert refused.json()["error"]["code"] == "INVALID_FIELD"
        assert seen == [("r1", {"max_concurrency": None}), ("r1", {"max_concurrency": 4})]
    finally:
        service.close()


# --- Step 2: one actor turn is prepare / execute / commit (CON-004) ------------


def _single_process_controller(tmp_path: Path, executor: Any) -> tuple[Any, Any]:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import ContextEngine, ExecutorRegistry, RunController, Scheduler

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    return controller, persistence


def test_an_executor_that_raises_is_recorded_as_an_executor_exception(tmp_path: Path) -> None:
    class Raising:
        def execute(self, invocation: Any) -> Any:
            raise ValueError("boom")

    controller, persistence = _single_process_controller(tmp_path, Raising())
    try:
        with pytest.raises(ValueError, match="boom"):
            controller.run("run-1")
        assert persistence.list_events("run-1")[-1]["classification"] == "executor_exception"
    finally:
        persistence.close()


def test_an_executor_returning_an_exception_object_is_an_invalid_output(tmp_path: Path) -> None:
    class Returning:
        def execute(self, invocation: Any) -> Any:
            return ValueError("not a result")

    controller, persistence = _single_process_controller(tmp_path, Returning())
    try:
        with pytest.raises(AttributeError):
            controller.run("run-1")
        assert persistence.list_events("run-1")[-1]["classification"] == "invalid_output"
    finally:
        persistence.close()


# --- Step 3: information timing (CON-005..CON-010, CON-015) --------------------

from genesis.information_timing import (  # noqa: E402
    batch_dependencies,
    is_batched,
    timing_diagnostics,
    whole_field_writes,
)

DEMO_PACKAGE = Path(__file__).resolve().parents[1] / "demos" / "full-chain-package"


def _proc(**fields: Any) -> dict[str, Any]:
    return {
        "id": "p",
        "actors": ["a", "b", "c"],
        "executor": {"mode": "computational"},
        "context_policy": "ctx",
        **fields,
    }


def test_single_invocation_processes_have_no_batch() -> None:
    assert not is_batched(_proc(actors=None))
    assert not is_batched(_proc(actors=["a"]))
    assert not is_batched(_proc(actors={"ids": ["a", "b"], "fan_out": False}))
    assert is_batched(_proc(actors={"source": "state.people"}))
    writes = [{"field": "notes", "op": "append"}]
    assert batch_dependencies(_proc(actors=None, state_effects=writes), {"allow": ["notes"]}) == []


@pytest.mark.parametrize("allow", ["notes", "state.notes", "state.notes.a", "state"])
def test_reading_a_field_the_batch_writes_is_a_dependency(allow: str) -> None:
    process = _proc(state_effects=[{"field": "notes", "op": "append"}])
    reasons = batch_dependencies(process, {"allow": [allow]})
    assert reasons and "notes" in reasons[0]
    assert batch_dependencies(process, {"allow": ["other", "state.other"]}) == []


def test_a_scoped_read_is_still_a_dependency() -> None:
    # A field scoped to actor.ids may name the recipient, not the writer: a sibling
    # can address a record to this actor within the same batch.
    process = _proc(state_effects=[{"field": "inbox", "op": "append"}])
    own = {"allow": ["inbox"], "scope": {"inbox": {"field": "to", "in": "actor.ids"}}}
    assert batch_dependencies(process, own)
    events = {
        "allow": ["events"],
        "scope": {"events": {"field": "recipient_id", "in": "actor.ids"}},
    }
    assert batch_dependencies(_proc(), events)


def test_a_simultaneous_model_call_may_only_write_state_that_composes() -> None:
    """A model call's declared operation is now applied, so composable ops are allowed.

    Its outputs were previously written whole whatever the declaration said, so
    any state write conflicted under simultaneous timing. Only a write that
    still replaces the field whole does.
    """
    composing = _proc(
        executor={"mode": "generative"},
        state_effects=[{"field": "notes", "op": "append"}],
        information_timing={"mode": "simultaneous"},
    )
    assert whole_field_writes(composing) == []
    errors, _ = timing_diagnostics([composing], {"ctx": {"allow": []}})
    assert [e["code"] for e in errors] == []
    setting = {**composing, "state_effects": [{"field": "notes", "op": "set"}]}
    assert whole_field_writes(setting) == ["its model call, whose effects set 'notes' whole"]
    errors, _ = timing_diagnostics([setting], {"ctx": {"allow": []}})
    assert [e["code"] for e in errors] == ["SIMULTANEOUS_WRITE_CONFLICT"]
    silent = {**composing, "state_effects": []}
    assert whole_field_writes(silent) == []


def test_compiling_requires_timing_only_for_a_batch_dependent_process(tmp_path: Path) -> None:
    def study(study_id: str, timing: dict[str, Any] | None) -> dict[str, Any]:
        process: dict[str, Any] = {
            "id": "note",
            "actors": ["a", "b"],
            "executor": {"mode": "deterministic"},
            "context_policy": "see-notes",
            "state_effects": [{"field": "notes", "op": "append"}],
        }
        if timing is not None:
            process["information_timing"] = timing
        return {
            "id": study_id,
            "title": study_id,
            "models": [],
            "processes": [process],
            "theory": {"theory_family": "exploratory"},
            "domain": {
                "states": [{"id": "notes", "value_type": "array", "initial": []}],
                "visibility": [{"id": "see-notes", "allow": ["notes"]}],
            },
            "protocol": {"time_model": {"type": "rounds", "end": 1}},
            "outcomes": [],
            "prompts": {},
        }

    service = GenesisService(tmp_path / "ws")
    try:
        from genesis.compiler import StudyCompiler

        draft = service.create_specification(study("undeclared-timing", None))
        with pytest.raises(ValueError, match="INFORMATION_TIMING_REQUIRED.*'notes'"):
            StudyCompiler(service._specification_dir("undeclared-timing")).compile(tmp_path / "u")
        with pytest.raises(ValueError, match="SPECIFICATION_INVALID"):
            service.approve_specification("undeclared-timing", draft["version"], "researcher")
        draft = service.create_specification(study("declared-timing", {"mode": "simultaneous"}))
        service.approve_specification("declared-timing", draft["version"], "researcher")
        compiled = service.compile_study(None, "builds/d", specification_id="declared-timing")
        processes = json.loads((Path(compiled["path"]) / "processes.json").read_text())
        assert processes[0]["information_timing"] == {"mode": "simultaneous", "order": "listed"}
    finally:
        service.close()


def test_scope_selectors_and_trigger_conditions_are_channels() -> None:
    process = _proc(state_effects=[{"field": "follows", "op": "append"}])
    scoped = {
        "allow": ["posts"],
        "scope": {"posts": {"field": "a", "in": ["state.follows.${actor}"]}},
    }
    assert batch_dependencies(process, scoped)
    triggered = _proc(
        state_effects=[{"field": "count", "op": "increment"}],
        trigger={
            "type": "condition",
            "predicate": {"all": [{"path": "count", "op": "lt", "value": 3}]},
        },
    )
    assert batch_dependencies(triggered, {"allow": []})
    static = _proc(
        state_effects=[{"field": "count", "op": "increment"}],
        trigger={
            "type": "condition",
            "predicate": {"path": "condition.arm", "op": "eq", "value": 1},
        },
    )
    assert batch_dependencies(static, {"allow": []}) == []


def test_events_matter_only_for_executors_that_emit_them() -> None:
    computational = _proc()
    model_call = _proc(executor={"mode": "generative"})
    for policy in (
        {"allow": ["events"]},
        {"allow": ["x"], "available_when": {"x": {"event": "posted"}}},
        {"allow": ["x"], "available_when": {"predicate": {"path": "events", "op": "truthy"}}},
    ):
        assert batch_dependencies(computational, policy), policy
        assert batch_dependencies(model_call, policy) == [], policy
    assert batch_dependencies(_proc(trigger={"type": "event", "event": "posted"}), {})
    assert (
        batch_dependencies(
            _proc(executor={"mode": "generative"}, trigger={"type": "event", "event": "posted"}), {}
        )
        == []
    )


def test_inputs_the_batch_produces_are_a_dependency() -> None:
    outputs = [{"artifact_type": "note", "schema_ref": "note"}]
    assert batch_dependencies(_proc(inputs=["note"], outputs=outputs), {})
    assert batch_dependencies(_proc(inputs=["p"]), {})
    assert batch_dependencies(_proc(inputs=["other"], outputs=outputs), {}) == []


def test_timing_diagnostics_require_only_where_timing_matters() -> None:
    policies = {"ctx": {"allow": ["notes"]}}
    writes = [{"field": "notes", "op": "append"}]
    errors, warnings = timing_diagnostics([_proc(state_effects=writes)], policies)
    assert [e["code"] for e in errors] == ["INFORMATION_TIMING_REQUIRED"] and not warnings
    assert "'notes'" in errors[0]["message"]
    for mode in ("sequential", "simultaneous"):
        declared = _proc(state_effects=writes, information_timing={"mode": mode})
        assert timing_diagnostics([declared], policies) == ([], [])
    unused = _proc(information_timing={"mode": "simultaneous"})
    errors, warnings = timing_diagnostics([unused], {"ctx": {"allow": []}})
    assert not errors and [w["code"] for w in warnings] == ["INFORMATION_TIMING_UNUSED"]


def test_simultaneous_batches_refuse_whole_field_writes() -> None:
    assert whole_field_writes(_proc(state_effects=[{"field": "n", "op": "append"}])) == []
    assert whole_field_writes(_proc(state_effects=[{"field": "n", "op": "set"}]))
    assert whole_field_writes(_proc(state_effects=[{"field": "n"}]))
    assert whole_field_writes(_proc(state_effects=["n"]))
    assert whole_field_writes(_proc(executor={"mode": "state-transition"}))
    conflicting = _proc(
        state_effects=[{"field": "n", "op": "set"}], information_timing={"mode": "simultaneous"}
    )
    errors, _ = timing_diagnostics([conflicting], {"ctx": {"allow": []}})
    assert "SIMULTANEOUS_WRITE_CONFLICT" in [e["code"] for e in errors]
    sequential = {**conflicting, "information_timing": {"mode": "sequential"}}
    errors, _ = timing_diagnostics([sequential], {"ctx": {"allow": []}})
    assert "SIMULTANEOUS_WRITE_CONFLICT" not in [e["code"] for e in errors]


def test_undeclared_timing_leaves_package_serialization_unchanged() -> None:
    from genesis.specification.models import ProcessSpec

    base = {"id": "p", "executor": {"mode": "deterministic"}, "context_policy": "private"}
    assert "information_timing" not in ProcessSpec.model_validate(base).model_dump(mode="json")
    declared = ProcessSpec.model_validate({**base, "information_timing": {"mode": "simultaneous"}})
    assert declared.model_dump(mode="json")["information_timing"] == {
        "mode": "simultaneous",
        "order": "listed",
    }
    with pytest.raises(ValueError):
        ProcessSpec.model_validate({**base, "information_timing": {"mode": "whenever"}})


@pytest.mark.skipif(not DEMO_PACKAGE.is_dir(), reason="demo package is local-only")
def test_demo_clickbait_package_needs_no_further_timing_declaration() -> None:
    import yaml

    from genesis.compiler import _resolve_context_policies
    from genesis.specification.models import DomainSpec, OpennessSpec

    openness = OpennessSpec.model_validate(
        yaml.safe_load((DEMO_PACKAGE / "openness.yaml").read_text())
    )
    domain = DomainSpec.model_validate(yaml.safe_load((DEMO_PACKAGE / "domain.yaml").read_text()))
    policies: dict[str, Any] = {name: {"allow": []} for name in ("private", "public", "none")}
    policies.update({str(policy["id"]): policy for policy in _resolve_context_policies(domain)})
    processes = {p.id: p.model_dump(mode="json") for p in openness.processes}
    errors, _ = timing_diagnostics(processes.values(), policies)
    assert errors == []
    # The two model-call processes can run concurrently without any declaration.
    for model_call in ("create-article", "user-interpret"):
        process = processes[model_call]
        assert is_batched(process)
        assert batch_dependencies(process, policies[process["context_policy"]]) == []


# Runtime semantics, driven through RunController directly.


class _NoteTaker:
    """Records how many notes each actor could see, then appends its own."""

    def __init__(self, *, whole_field: bool = False) -> None:
        self.seen: list[tuple[int | float, str, int]] = []
        self.whole_field = whole_field

    def execute(self, invocation: Any) -> Any:
        from genesis.runtime import ProcessResult

        notes = list(invocation.context.data.get("notes", ()))
        actor = invocation.actor_ids[0]
        self.seen.append((invocation.phase, actor, len(notes)))
        if self.whole_field:
            return ProcessResult(state_effects={"notes": [*notes, actor]})
        return ProcessResult(state_effects=[{"field": "notes", "op": "append", "value": actor}])


def _timing_controller(
    tmp_path: Path,
    executor: Any,
    *,
    timing: dict[str, Any] | None = None,
    actors: list[str] | None = None,
    name: str = "db",
    extra: dict[str, Any] | None = None,
    status_provider: Any = None,
    initial_notes: list[str] | None = None,
) -> tuple[Any, Any]:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )

    process: dict[str, Any] = {
        "id": "p",
        "actors": actors or ["a", "b", "c"],
        "context_policy": "see-notes",
        "state_effects": [{"field": "notes", "op": "append"}],
        **(extra or {}),
    }
    if timing is not None:
        process["information_timing"] = timing
    persistence = PersistenceCoordinator(tmp_path / f"{name}.db", tmp_path / f"{name}-objects")
    controller = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"see-notes": {"allow": ["notes"]}, "see-nothing": {"allow": []}}),
        persistence=persistence,
        state_store=StateStore(
            {"notes": list, "count": int}, {"notes": list(initial_notes or []), "count": 0}
        ),
        status_provider=status_provider,
    )
    return controller, persistence


def _completed(persistence: Any, run_id: str) -> list[dict[str, Any]]:
    return [e for e in persistence.list_events(run_id) if e.get("kind") == "process_completed"]


def test_sequential_actors_see_earlier_siblings_and_simultaneous_actors_do_not(tmp_path) -> None:
    sequential, simultaneous = _NoteTaker(), _NoteTaker()
    seq_controller, seq_db = _timing_controller(
        tmp_path, sequential, timing={"mode": "sequential"}, name="seq"
    )
    sim_controller, sim_db = _timing_controller(
        tmp_path, simultaneous, timing={"mode": "simultaneous"}, name="sim"
    )
    try:
        seq_controller.run("r", phase_limit=1)
        sim_controller.run("r", phase_limit=1)
        assert [seen for _, _, seen in sequential.seen] == [0, 1, 2]
        assert [seen for _, _, seen in simultaneous.seen] == [0, 0, 0]
        assert sim_controller.state_store.snapshot()["notes"] == ["a", "b", "c"]
        sim_events = _completed(sim_db, "r")
        assert {e["view_state_version"] for e in sim_events} == {0}
        assert [e["view_state_version"] for e in _completed(seq_db, "r")] == [0, 1, 2]
        assert sim_events[0]["information_timing"] == {"mode": "simultaneous", "order": "listed"}
    finally:
        seq_db.close()
        sim_db.close()


def test_undeclared_timing_is_not_recorded_as_a_choice(tmp_path) -> None:
    controller, db = _timing_controller(tmp_path, _NoteTaker())
    try:
        controller.run("r", phase_limit=1)
        events = _completed(db, "r")
        assert all("information_timing" not in e for e in events)
        assert [e["view_state_version"] for e in events] == [0, 1, 2]
    finally:
        db.close()


def test_a_simultaneous_batch_refuses_whole_field_effects_at_run_time(tmp_path) -> None:
    controller, db = _timing_controller(
        tmp_path, _NoteTaker(whole_field=True), timing={"mode": "simultaneous"}
    )
    try:
        with pytest.raises(ValueError, match="SIMULTANEOUS_WRITE_CONFLICT"):
            controller.run("r", phase_limit=1)
    finally:
        db.close()
    sequential, db = _timing_controller(
        tmp_path, _NoteTaker(whole_field=True), timing={"mode": "sequential"}, name="seq"
    )
    try:
        sequential.run("r", phase_limit=1)
        assert sequential.state_store.snapshot()["notes"] == ["a", "b", "c"]
    finally:
        db.close()


def test_a_simultaneous_trigger_is_evaluated_once_as_the_batch_begins(tmp_path) -> None:
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, invocation: Any) -> Any:
            from genesis.runtime import ProcessResult

            self.calls += 1
            return ProcessResult(state_effects=[{"field": "count", "op": "increment", "value": 1}])

    extra = {
        "state_effects": [{"field": "count", "op": "increment"}],
        "trigger": {"type": "condition", "predicate": {"path": "count", "op": "lt", "value": 1}},
    }
    results = {}
    for mode in ("sequential", "simultaneous"):
        counter = Counter()
        controller, db = _timing_controller(
            tmp_path, counter, timing={"mode": mode}, name=mode, extra=extra
        )
        try:
            controller.run("r", phase_limit=1)
            results[mode] = counter.calls
        finally:
            db.close()
    assert results["simultaneous"] == 3
    assert results["sequential"] < 3


ACTORS = ["a", "b", "c", "d", "e", "f"]


def _orders(tmp_path: Path, name: str, **run: Any) -> list[list[str]]:
    taker = _NoteTaker()
    controller, db = _timing_controller(
        tmp_path,
        taker,
        timing={"mode": "sequential", "order": "shuffled"},
        actors=ACTORS,
        name=name,
        extra={"trigger": {"repeat": True}},
    )
    try:
        controller.run(run.pop("run_id", "r"), phase_limit=3, **run)
    finally:
        db.close()
    by_phase: dict[Any, list[str]] = {}
    for phase, actor, _ in taker.seen:
        by_phase.setdefault(phase, []).append(actor)
    return [by_phase[phase] for phase in sorted(by_phase)]


def test_shuffled_order_is_seeded_per_phase_and_reproducible(tmp_path) -> None:
    first = _orders(tmp_path, "one", seed=7)
    again = _orders(tmp_path, "two", seed=7)
    other = _orders(tmp_path, "three", seed=8)
    assert len(first) == 3 and all(sorted(order) == ACTORS for order in first)
    assert first == again
    assert first != other
    assert len({tuple(order) for order in first}) > 1
    assert any(order != ACTORS for order in first)


def test_shuffled_order_is_shared_across_matched_conditions(tmp_path) -> None:
    shared = {"enabled": True, "shared_streams": ["activation-order"]}
    left = _orders(tmp_path, "l", seed=3, condition_id="control", matching=shared)
    right = _orders(tmp_path, "r", seed=3, condition_id="treated", matching=shared)
    assert left == right
    unshared_left = _orders(tmp_path, "ul", seed=3, condition_id="control")
    unshared_right = _orders(tmp_path, "ur", seed=3, condition_id="treated")
    assert unshared_left != unshared_right


def test_resuming_mid_batch_rebuilds_the_simultaneous_view(tmp_path) -> None:
    uninterrupted = _NoteTaker()
    controller, db = _timing_controller(
        tmp_path, uninterrupted, timing={"mode": "simultaneous"}, name="whole"
    )
    try:
        controller.run("r", phase_limit=1)
        whole = [(e["actors"], e["view_state_version"]) for e in _completed(db, "r")]
    finally:
        db.close()

    first_half = _NoteTaker()
    polls: list[int] = []

    def pause_after_first_commit() -> str:
        polls.append(len(first_half.seen))
        return "paused" if first_half.seen else "running"

    paused, db = _timing_controller(
        tmp_path,
        first_half,
        timing={"mode": "simultaneous"},
        name="split",
        status_provider=pause_after_first_commit,
    )
    paused.run("r", phase_limit=1)
    assert paused.status == "paused" and len(first_half.seen) == 1
    db.close()

    second_half = _NoteTaker()
    resumed, db = _timing_controller(
        tmp_path, second_half, timing={"mode": "simultaneous"}, name="split"
    )
    try:
        resumed.run("r", phase_limit=1)
        assert [seen for _, _, seen in first_half.seen + second_half.seen] == [0, 0, 0]
        split = [(e["actors"], e["view_state_version"]) for e in _completed(db, "r")]
        assert split == whole
        assert resumed.state_store.snapshot()["notes"] == ["a", "b", "c"]
    finally:
        db.close()


def test_resuming_mid_batch_does_not_re_evaluate_a_simultaneous_trigger(tmp_path) -> None:
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, invocation: Any) -> Any:
            from genesis.runtime import ProcessResult

            self.calls += 1
            return ProcessResult(state_effects=[{"field": "count", "op": "increment", "value": 1}])

    extra = {
        "state_effects": [{"field": "count", "op": "increment"}],
        "trigger": {"type": "condition", "predicate": {"path": "count", "op": "lt", "value": 1}},
    }
    first = Counter()
    paused, db = _timing_controller(
        tmp_path,
        first,
        timing={"mode": "simultaneous"},
        extra=extra,
        status_provider=lambda: "paused" if first.calls else "running",
    )
    paused.run("r", phase_limit=1)
    assert paused.status == "paused" and first.calls == 1
    db.close()
    second = Counter()
    resumed, db = _timing_controller(tmp_path, second, timing={"mode": "simultaneous"}, extra=extra)
    try:
        resumed.run("r", phase_limit=1)
        assert second.calls == 2
        assert len(_completed(db, "r")) == 3
    finally:
        db.close()


class _Appender:
    """Appends a new, id-safe record per actor to the field actors are drawn from."""

    def __init__(self) -> None:
        self.order: list[str] = []

    def execute(self, invocation: Any) -> Any:
        from genesis.runtime import ProcessResult

        actor = invocation.actor_ids[0]
        self.order.append(actor)
        return ProcessResult(
            state_effects=[{"field": "notes", "op": "append", "value": f"z{actor}"}]
        )


def test_resuming_keeps_the_batch_actors_and_their_shuffled_order(tmp_path) -> None:
    timing = {"mode": "simultaneous", "order": "shuffled"}
    sourced = {"actors": {"source": "notes"}}
    notes = ["a", "b", "c", "d"]
    whole_run = _Appender()
    controller, db = _timing_controller(
        tmp_path, whole_run, timing=timing, extra=sourced, name="whole", initial_notes=notes
    )
    try:
        controller.run("r", phase_limit=1)
        whole_state = controller.state_store.snapshot()["notes"]
    finally:
        db.close()

    first = _Appender()
    paused, db = _timing_controller(
        tmp_path,
        first,
        timing=timing,
        extra=sourced,
        name="split",
        initial_notes=notes,
        status_provider=lambda: "paused" if first.order else "running",
    )
    paused.run("r", phase_limit=1)
    db.close()
    second = _Appender()
    resumed, db = _timing_controller(
        tmp_path, second, timing=timing, extra=sourced, name="split", initial_notes=notes
    )
    try:
        resumed.run("r", phase_limit=1)
        assert first.order + second.order == whole_run.order
        assert resumed.state_store.snapshot()["notes"] == whole_state
    finally:
        db.close()


def test_an_undeclared_independent_batch_reads_from_its_view_even_through_raw_events(
    tmp_path,
) -> None:
    class EventReader:
        def __init__(self) -> None:
            self.seen: list[int] = []

        def execute(self, invocation: Any) -> Any:
            from genesis.runtime import ProcessResult

            # A callable can read the raw event history, which no policy governs.
            self.seen.append(len(invocation.event_history))
            return ProcessResult(events=[{"type": "noted"}])

    results = {}
    for timing in (None, {"mode": "sequential"}):
        reader = EventReader()
        controller, db = _timing_controller(
            tmp_path,
            reader,
            timing=timing,
            name=f"events-{timing is None}",
            extra={"context_policy": "see-nothing", "state_effects": []},
        )
        try:
            controller.run("r", phase_limit=1)
            results[timing is None] = reader.seen
        finally:
            db.close()
    assert results[True] == [0, 0, 0]
    assert results[False] == [0, 1, 2]


def test_replayed_results_may_commit_recorded_whole_field_deltas(tmp_path) -> None:
    class Replayed:
        def execute(self, invocation: Any) -> Any:
            from genesis.runtime import ProcessResult

            actor = invocation.actor_ids[0]
            return ProcessResult(state_effects={"notes": [actor]}, metadata={"recorded": True})

    controller, db = _timing_controller(tmp_path, Replayed(), timing={"mode": "simultaneous"})
    try:
        controller.run("r", phase_limit=1)
        assert len(_completed(db, "r")) == 3
    finally:
        db.close()


# --- Step 4: concurrent execution (CON-011..CON-015) ---------------------------


class _PooledCall:
    """A concurrency-safe stand-in for a model call.

    Its answer depends only on the invocation (seed and context), never on when
    or alongside what it runs; each call sleeps a random, unseeded moment so
    completion order varies between runs.
    """

    concurrent_safe = True

    def __init__(self, *, fail_once: set[str] | None = None, fail: set[str] | None = None) -> None:
        import random
        import threading

        self._jitter = random.Random()
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[str] = []
        self.fail_once = set(fail_once or ())
        self.fail = set(fail or ())

    def execute(self, invocation: Any) -> Any:
        from genesis.runtime import ProcessResult

        actor = invocation.actor_ids[0]
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls.append(actor)
            fail_now = actor in self.fail or actor in self.fail_once
            self.fail_once.discard(actor)
        try:
            time.sleep(self._jitter.uniform(0.0, 0.02))
            if fail_now:
                raise ValueError(f"provider refused {actor}")
            notes = len(invocation.context.data.get("notes", ()))
            return ProcessResult(
                outputs={"text": f"{actor}:{invocation.seed % 1000}:{notes}"},
                state_effects=[
                    {
                        "field": "drafts",
                        "op": "append",
                        "value": f"{actor}:{invocation.seed % 1000}",
                    }
                ],
                metadata={"mode": "generative", "usage": {"prompt_tokens": 3}},
            )
        finally:
            with self._lock:
                self.in_flight -= 1


WRITERS = [f"w{index}" for index in range(1, 9)]


def _pooled_controller(
    tmp_path: Path,
    executor: Any,
    *,
    limit: int,
    name: str,
    policy_allow: list[str] | None = None,
    timing: dict[str, Any] | None = None,
    retry: int = 1,
    status_provider: Any = None,
    cancel_event: Any = None,
) -> tuple[Any, Any]:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )

    process: dict[str, Any] = {
        "id": "write",
        "actors": WRITERS,
        "executor": {"mode": "generative"},
        "context_policy": "writer",
        "state_effects": [{"field": "drafts", "op": "append"}],
        "retry_policy": {"max_attempts": retry},
    }
    if timing is not None:
        process["information_timing"] = timing
    persistence = PersistenceCoordinator(tmp_path / f"{name}.db", tmp_path / f"{name}-objects")
    controller = RunController(
        Scheduler([process]),
        ExecutorRegistry({"write": executor}),
        ContextEngine(
            {"writer": {"allow": policy_allow if policy_allow is not None else ["notes"]}}
        ),
        persistence=persistence,
        state_store=StateStore(
            {"notes": list, "drafts": list}, {"notes": ["n1", "n2"], "drafts": []}
        ),
        status_provider=status_provider,
        max_concurrency={"write": limit},
        cancel_event=cancel_event,
    )
    return controller, persistence


_VOLATILE_EVENT_KEYS = {"latency_ms", "recorded_at", "created_at", "timestamp", "event_hash"}


def _record(controller: Any, persistence: Any, run_id: str = "r") -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in _VOLATILE_EVENT_KEYS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {
        "events": clean(persistence.list_events(run_id)),
        "state": controller.state_store.snapshot() if controller.state_store else None,
        "dispatch": clean(controller.dispatch_log),
        "commit": clean(controller.commit_log),
    }


@pytest.mark.parametrize("timing", [None, {"mode": "simultaneous"}])
def test_concurrency_never_changes_the_recorded_run(tmp_path: Path, timing: Any) -> None:
    records = {}
    overlaps = {}
    for limit in (1, 2, 8):
        executor = _PooledCall()
        controller, db = _pooled_controller(
            tmp_path, executor, limit=limit, name=f"c{limit}", timing=timing
        )
        try:
            controller.run("r", phase_limit=1)
            records[limit] = _record(controller, db)
            overlaps[limit] = executor.max_in_flight
            decision = controller.execution_decisions["write"]
        finally:
            db.close()
        assert decision["concurrent"] is (limit > 1)
    assert records[1] == records[2] == records[8]
    assert overlaps[1] == 1 and overlaps[8] > 1
    assert len(records[8]["state"]["drafts"]) == len(WRITERS)


def test_a_batch_dependent_sequential_process_stays_one_at_a_time(tmp_path: Path) -> None:
    executor = _PooledCall()
    controller, db = _pooled_controller(
        tmp_path,
        executor,
        limit=8,
        name="dependent",
        policy_allow=["drafts"],
        timing={"mode": "sequential"},
    )
    try:
        controller.run("r", phase_limit=1)
        assert executor.max_in_flight == 1
        decision = controller.execution_decisions["write"]
        assert decision["concurrent"] is False and "same batch" in decision["reason"]
    finally:
        db.close()


def test_a_terminal_failure_commits_exactly_the_serial_prefix(tmp_path: Path) -> None:
    records = {}
    for limit in (1, 8):
        executor = _PooledCall(fail={"w4"})
        controller, db = _pooled_controller(tmp_path, executor, limit=limit, name=f"f{limit}")
        try:
            with pytest.raises(ValueError, match="provider refused w4"):
                controller.run("r", phase_limit=1)
            records[limit] = _record(controller, db)
            if limit == 8:
                discarded = controller.discarded_calls
        finally:
            db.close()
    assert records[1] == records[8]
    committed = [(e["kind"], e["actors"]) for e in records[8]["events"]]
    assert committed == [
        ("process_completed", ["w1"]),
        ("process_completed", ["w2"]),
        ("process_completed", ["w3"]),
        ("process_failed", ["w4"]),
    ]
    # All eight calls started at once; w5..w8 ran but could not be committed.
    assert discarded["calls"] == 4


def test_retries_keep_the_serial_dispatch_order(tmp_path: Path) -> None:
    records = {}
    for limit in (1, 8):
        controller, db = _pooled_controller(
            tmp_path, _PooledCall(fail_once={"w3"}), limit=limit, name=f"retry{limit}", retry=2
        )
        try:
            controller.run("r", phase_limit=1)
            records[limit] = _record(controller, db)
        finally:
            db.close()
    assert records[1] == records[8]
    assert [(d["invocation_id"].split("-")[-2], d["attempt"]) for d in records[8]["dispatch"]][
        :4
    ] == [
        ("w1", 1),
        ("w2", 1),
        ("w3", 1),
        ("w3", 2),
    ]


def test_pausing_mid_batch_then_resuming_equals_an_uninterrupted_run(tmp_path: Path) -> None:
    whole_controller, whole_db = _pooled_controller(tmp_path, _PooledCall(), limit=1, name="whole")
    try:
        whole_controller.run("r", phase_limit=1)
        whole = _record(whole_controller, whole_db)
    finally:
        whole_db.close()

    holder: dict[str, Any] = {}

    def pause_after_three() -> str:
        controller = holder.get("controller")
        return "paused" if controller is not None and len(controller.commit_log) >= 3 else "running"

    paused, db = _pooled_controller(
        tmp_path, _PooledCall(), limit=8, name="split", status_provider=pause_after_three
    )
    holder["controller"] = paused
    paused.run("r", phase_limit=1)
    assert paused.status == "paused"
    assert len(db.list_events("r")) == 3
    db.close()

    resumed, db = _pooled_controller(tmp_path, _PooledCall(), limit=8, name="split")
    try:
        resumed.run("r", phase_limit=1)
        split = _record(resumed, db)
    finally:
        db.close()
    assert split["events"] == whole["events"]
    assert split["state"] == whole["state"]


def test_the_event_budget_is_never_over_dispatched(tmp_path: Path) -> None:
    executor = _PooledCall()
    controller, db = _pooled_controller(tmp_path, executor, limit=8, name="budget")
    try:
        controller.run("r", phase_limit=1, max_events=3)
        assert len(executor.calls) == 3
        assert len(db.list_events("r")) == 3
    finally:
        db.close()


def test_a_service_run_calls_the_model_concurrently_and_records_why(tmp_path, monkeypatch) -> None:
    import threading

    lock = threading.Lock()
    flight = {"now": 0, "max": 0}

    class SlowProvider(_TextProvider):
        def generate(self, request: ProviderRequest) -> ProviderResponse:
            with lock:
                flight["now"] += 1
                flight["max"] = max(flight["max"], flight["now"])
            try:
                time.sleep(0.05)
                return super().generate(request)
            finally:
                with lock:
                    flight["now"] -= 1

    study = json.loads(json.dumps(STUDY))
    study["processes"][0]["actors"] = ["a1", "a2", "a3", "a4"]
    service = _compiled_service(tmp_path, monkeypatch, study=study)
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", SlowProvider)
    try:
        assert service.execute_run("run-a", max_concurrency=4)["status"] == "completed"
        assert flight["max"] > 1
        execution = service.get_run("run-a")["executions"][-1]
        assert execution["processes"]["compose"] == {
            "max_concurrency": 4,
            "concurrent": True,
            "reason": "batch-independent",
        }
    finally:
        service.close()


# --- Re-review fixes (information timing) --------------------------------------


def _interleave_controller(
    tmp_path: Path, reader: Any, *, name: str, limit: int = 1
) -> tuple[Any, Any]:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
    )

    class Flagger:
        def execute(self, invocation: Any) -> Any:
            return ProcessResult(state_effects=[{"field": "flag", "op": "increment", "value": 1}])

    processes = [
        {
            # Sorts before 'p' and becomes ready once 'p' has committed once.
            "id": "a",
            "context_policy": "none",
            "state_effects": [{"field": "flag", "op": "increment"}],
            "trigger": {
                "type": "condition",
                "predicate": {"path": "count", "op": "gte", "value": 1},
            },
        },
        {
            "id": "p",
            "actors": ["x", "y", "z"],
            "executor": {"mode": "generative"},
            "context_policy": "see-flag",
            "state_effects": [{"field": "count", "op": "increment"}],
        },
    ]
    persistence = PersistenceCoordinator(tmp_path / f"{name}.db", tmp_path / f"{name}-objects")
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"a": Flagger(), "p": reader}),
        ContextEngine({"none": {"allow": []}, "see-flag": {"allow": ["flag"]}}),
        persistence=persistence,
        state_store=StateStore({"count": int, "flag": int}, {"count": 0, "flag": 0}),
        max_concurrency={"p": limit},
    )
    return controller, persistence


class _FlagReader:
    concurrent_safe = True

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def execute(self, invocation: Any) -> Any:
        from genesis.runtime import ProcessResult

        self.seen.append((invocation.actor_ids[0], invocation.context.data.get("flag", 0)))
        return ProcessResult(state_effects=[{"field": "count", "op": "increment", "value": 1}])


def test_an_undeclared_batch_still_sees_a_process_that_interleaves(tmp_path: Path) -> None:
    reader = _FlagReader()
    controller, db = _interleave_controller(tmp_path, reader, name="serial")
    try:
        controller.run("r", phase_limit=1)
        assert reader.seen == [("x", 0), ("y", 1), ("z", 1)]
        order = [(e["process_id"], e.get("actors")) for e in _completed(db, "r")]
        assert order == [("p", ["x"]), ("a", []), ("p", ["y"]), ("p", ["z"])]
    finally:
        db.close()


def test_a_batch_that_can_ready_another_process_is_not_run_concurrently(tmp_path: Path) -> None:
    reader = _FlagReader()
    controller, db = _interleave_controller(tmp_path, reader, name="guarded", limit=4)
    try:
        controller.run("r", phase_limit=1)
        assert reader.seen == [("x", 0), ("y", 1), ("z", 1)]
        decision = controller.execution_decisions["p"]
        assert decision["concurrent"] is False and "another process ready" in decision["reason"]
    finally:
        db.close()


def test_resuming_past_a_skipped_actor_does_not_run_it_again(tmp_path: Path) -> None:
    class SkipFirst:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def execute(self, invocation: Any) -> Any:
            from genesis.runtime import ProcessResult

            actor = invocation.actor_ids[0]
            self.calls.append(actor)
            if actor == "a":
                return ProcessResult(status="failed", metadata={"code": "REFUSED"})
            return ProcessResult()

    extra = {
        "context_policy": "see-nothing",
        "state_effects": [],
        "retry_policy": {"failure_policy": "skip_with_event"},
    }
    first = SkipFirst()
    paused, db = _timing_controller(
        tmp_path, first, extra=extra, status_provider=lambda: "paused" if first.calls else "running"
    )
    paused.run("r", phase_limit=1)
    db.close()
    second = SkipFirst()
    resumed, db = _timing_controller(tmp_path, second, extra=extra)
    try:
        resumed.run("r", phase_limit=1)
        assert second.calls == ["b", "c"]
        kinds = [e["kind"] for e in db.list_events("r")]
        assert kinds == ["process_skipped", "process_completed", "process_completed"]
    finally:
        db.close()


def test_the_actor_order_is_recorded_only_when_the_declaration_cannot_reproduce_it(
    tmp_path: Path,
) -> None:
    listed, db = _timing_controller(
        tmp_path, _NoteTaker(), timing={"mode": "simultaneous"}, name="listed"
    )
    try:
        listed.run("r", phase_limit=1)
        assert all("batch_actors" not in e for e in db.list_events("r"))
    finally:
        db.close()
    shuffled, db = _timing_controller(
        tmp_path,
        _NoteTaker(),
        timing={"mode": "simultaneous", "order": "shuffled"},
        name="shuffled",
    )
    try:
        shuffled.run("r", phase_limit=1)
        carrying = [e for e in db.list_events("r") if "batch_actors" in e]
        assert len(carrying) == 1
        assert sorted(tuple(group) for group in carrying[0]["batch_actors"]) == [
            ("a",),
            ("b",),
            ("c",),
        ]
    finally:
        db.close()


# --- Step 4 review fixes -------------------------------------------------------


def test_a_concurrent_batch_consumes_every_scheduled_entry(tmp_path: Path) -> None:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
        StateStore,
    )

    class Scheduling:
        def execute(self, invocation: Any) -> Any:
            return ProcessResult(
                scheduling_effects=[{"type": "schedule", "process_id": "write", "phase": 1}]
            )

    records = {}
    for limit in (1, 8):
        processes = [
            {"id": "sched", "actors": ["s1", "s2", "s3"], "context_policy": "none"},
            {
                "id": "write",
                "actors": WRITERS,
                "executor": {"mode": "generative"},
                "context_policy": "writer",
                "state_effects": [{"field": "drafts", "op": "append"}],
                "trigger": {"type": "phase", "phase": 1},
            },
        ]
        db = PersistenceCoordinator(tmp_path / f"s{limit}.db", tmp_path / f"s{limit}-objects")
        controller = RunController(
            Scheduler(processes),
            ExecutorRegistry({"sched": Scheduling(), "write": _PooledCall()}),
            ContextEngine({"none": {"allow": []}, "writer": {"allow": ["notes"]}}),
            persistence=db,
            state_store=StateStore({"notes": list, "drafts": list}, {"notes": [], "drafts": []}),
            max_concurrency={"write": limit},
        )
        try:
            controller.run("r", phase_limit=3)
            records[limit] = _record(controller, db)
        finally:
            db.close()
    assert records[1] == records[8]
    writes = [e for e in records[8]["events"] if e.get("process_id") == "write"]
    assert len(writes) == len(WRITERS)


def test_a_call_that_fails_to_prepare_still_commits_the_earlier_actors(tmp_path: Path) -> None:
    records = {}
    for limit in (1, 8):
        controller, db = _pooled_controller(
            tmp_path, _PooledCall(), limit=limit, name=f"prep{limit}"
        )
        original = controller.context_engine.build

        def build(policy_id: str, invocation: Any, state: Any, _original: Any = original) -> Any:
            if tuple(invocation.actor_ids) == ("w5",):
                raise ValueError("context unavailable for w5")
            return _original(policy_id, invocation, state)

        controller.context_engine.build = build
        try:
            with pytest.raises(ValueError, match="context unavailable for w5"):
                controller.run("r", phase_limit=1)
            records[limit] = _record(controller, db)
        finally:
            db.close()
    assert records[1] == records[8]
    assert [e["actors"] for e in records[8]["events"]] == [["w1"], ["w2"], ["w3"], ["w4"]]


def test_a_failing_batch_cancels_the_calls_still_in_flight(tmp_path: Path) -> None:
    import threading

    from genesis.runtime import ProcessResult

    cancel = threading.Event()

    class Waiting:
        concurrent_safe = True

        def __init__(self) -> None:
            self.cancelled: list[str] = []
            self.lock = threading.Lock()

        def execute(self, invocation: Any) -> Any:
            actor = invocation.actor_ids[0]
            if actor == "w1":
                raise ValueError("provider refused w1")
            if cancel.wait(5):
                with self.lock:
                    self.cancelled.append(actor)
            return ProcessResult(metadata={"usage": {"prompt_tokens": 1}})

    executor = Waiting()
    controller, db = _pooled_controller(
        tmp_path, executor, limit=8, name="cancel", cancel_event=cancel
    )
    try:
        started = time.monotonic()
        with pytest.raises(ValueError, match="provider refused w1"):
            controller.run("r", phase_limit=1)
        assert time.monotonic() - started < 2
        assert cancel.is_set()
        deadline = time.monotonic() + 2
        while len(executor.cancelled) < len(WRITERS) - 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sorted(executor.cancelled) == WRITERS[1:]
    finally:
        db.close()


# --- Step 4 re-review fixes ----------------------------------------------------


def _scheduled_controller(
    tmp_path: Path,
    processes: list[dict[str, Any]],
    executors: dict[str, Any],
    *,
    name: str,
    limit: int,
) -> tuple[Any, Any]:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )

    db = PersistenceCoordinator(tmp_path / f"{name}.db", tmp_path / f"{name}-objects")
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry(executors),
        ContextEngine({"none": {"allow": []}, "writer": {"allow": ["notes"]}}),
        persistence=db,
        state_store=StateStore(
            {"notes": list, "drafts": list, "flag": int}, {"notes": [], "drafts": [], "flag": 0}
        ),
        max_concurrency={"write": limit},
    )
    return controller, db


def _writer_process(**trigger: Any) -> dict[str, Any]:
    return {
        "id": "write",
        "actors": WRITERS,
        "executor": {"mode": "generative"},
        "context_policy": "writer",
        "state_effects": [{"field": "drafts", "op": "append"}],
        "trigger": trigger,
    }


def test_a_batch_kept_ready_by_a_scheduled_entry_stays_serial(tmp_path: Path) -> None:
    from genesis.runtime import ProcessResult

    class ScheduleOnce:
        def execute(self, invocation: Any) -> Any:
            return ProcessResult(
                scheduling_effects=[{"type": "schedule", "process_id": "write", "phase": 1}]
            )

    records = {}
    for limit in (1, 8):
        processes = [
            {"id": "sched", "context_policy": "none"},
            _writer_process(type="phase", phase=2),
        ]
        controller, db = _scheduled_controller(
            tmp_path,
            processes,
            {"sched": ScheduleOnce(), "write": _PooledCall()},
            name=f"gate{limit}",
            limit=limit,
        )
        try:
            controller.run("r", phase_limit=3)
            records[limit] = _record(controller, db)
        finally:
            db.close()
    assert records[1] == records[8]
    phase_one = [
        e for e in records[8]["events"] if e.get("process_id") == "write" and e.get("phase") == 1
    ]
    assert [e["actors"] for e in phase_one] == [["w1"]]


def test_a_process_that_overtakes_a_scheduled_batch_still_runs_between_its_actors(
    tmp_path: Path,
) -> None:
    from genesis.runtime import ProcessResult

    class Flagger:
        def execute(self, invocation: Any) -> Any:
            return ProcessResult(state_effects=[{"field": "flag", "op": "increment", "value": 1}])

    records = {}
    for limit in (1, 8):
        processes = [
            _writer_process(type="phase", phase=1, repeat=True),
            {
                "id": "a",
                "context_policy": "none",
                "state_effects": [{"field": "flag", "op": "increment"}],
                "trigger": {"type": "phase", "phase": 1},
            },
        ]
        controller, db = _scheduled_controller(
            tmp_path,
            processes,
            {"a": Flagger(), "write": _PooledCall()},
            name=f"overtake{limit}",
            limit=limit,
        )
        controller.scheduler.schedule("write", 0)
        controller.scheduler.schedule("write", 0)
        try:
            controller.run("r", phase_start=1, phase_limit=2)
            records[limit] = _record(controller, db)
        finally:
            db.close()
    assert records[1] == records[8]
    order = [e["process_id"] for e in records[8]["events"]]
    assert order[:3] == ["write", "write", "a"]


def test_pausing_after_a_failed_prepare_keeps_that_actor_queued(tmp_path: Path) -> None:
    holder: dict[str, Any] = {}
    failures = {"w5": 1}

    def pause_after_two() -> str:
        controller = holder.get("controller")
        if controller is None or holder.get("resumed"):
            return "running"
        return "paused" if len(controller.commit_log) >= 2 else "running"

    controller, db = _pooled_controller(
        tmp_path, _PooledCall(), limit=8, name="prep-pause", status_provider=pause_after_two
    )
    holder["controller"] = controller
    original = controller.context_engine.build

    def build(policy_id: str, invocation: Any, state: Any) -> Any:
        if tuple(invocation.actor_ids) == ("w5",) and failures["w5"]:
            failures["w5"] -= 1
            raise ValueError("context briefly unavailable for w5")
        return original(policy_id, invocation, state)

    controller.context_engine.build = build
    try:
        controller.run("r", phase_limit=1)
        assert controller.status == "paused"
        holder["resumed"] = True
        controller.resume()
        controller.run("r", phase_limit=1)
        committed = [e["actors"][0] for e in _completed(db, "r")]
        assert committed == WRITERS
    finally:
        db.close()


# --- External review fixes ------------------------------------------------------


def test_an_override_for_a_profile_no_process_uses_is_refused(tmp_path, monkeypatch) -> None:
    study = json.loads(json.dumps(STUDY))
    study["models"].append(
        {"id": "spare", "provider": "openai-compatible", "model": "m2", "parameters": {}}
    )
    service = _compiled_service(tmp_path, monkeypatch, study=study)
    try:
        with pytest.raises(ValueError, match="MAX_CONCURRENCY: .*no process.*spare"):
            service.execute_run("run-a", max_concurrency={"spare": 2})
        assert service.get_run("run-a")["status"] == "created"
    finally:
        service.close()


def test_a_completed_run_still_refuses_a_malformed_override(tmp_path, monkeypatch) -> None:
    service = _compiled_service(tmp_path, monkeypatch)
    try:
        assert service.execute_run("run-a")["status"] == "completed"
        with pytest.raises(ValueError, match="MAX_CONCURRENCY"):
            service.execute_run("run-a", max_concurrency=0)
    finally:
        service.close()


def _opening_once(provider: Any, first: BaseException) -> dict[str, int]:
    calls = {"n": 0}

    def opener(_request: Any) -> tuple[bytes, int]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise first
        return OK_BODY, 200

    provider._open_cancellable = opener
    return calls


def _bare_provider(**options: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://provider.invalid/v1",
        model="m",
        api_key_env="E",
        api_key="k",
        backoff_base=0.001,
        backoff_cap=0.001,
        **options,
    )


@pytest.mark.parametrize("kind", ["IncompleteRead", "BadStatusLine"])
def test_a_response_broken_mid_flight_is_retried(kind: str) -> None:
    import http.client

    broken = {
        "IncompleteRead": http.client.IncompleteRead(b"partial"),
        "BadStatusLine": http.client.BadStatusLine("garbage"),
    }[kind]
    provider = _bare_provider()
    calls = _opening_once(provider, broken)
    response = provider.generate(_request())
    assert response.text == "ok" and calls["n"] == 2
    assert response.metadata["retries"][0]["status"] == "unavailable"


def test_a_malformed_url_is_not_retried() -> None:
    import http.client

    provider = _bare_provider()
    calls = _opening_once(provider, http.client.InvalidURL("bad url"))
    with pytest.raises(ValueError, match="PROVIDER_UNAVAILABLE"):
        provider.generate(_request())
    assert calls["n"] == 1


@pytest.mark.parametrize("status", [425, 520, 524, 529])
def test_gateway_and_overload_statuses_are_retried(monkeypatch, status: int) -> None:
    with scripted_server([(status, {}), (200, {})]) as (url, hits):
        response = _provider(url, monkeypatch).generate(_request())
    assert response.text == "ok" and len(hits) == 2


@pytest.mark.parametrize("header", ["nan", "inf", "-inf"])
def test_a_non_finite_retry_after_falls_back_to_backoff(header: str) -> None:
    class Ceiling:
        def uniform(self, low: float, high: float) -> float:
            return high

    assert retry_wait(3, header, base=1.0, cap=60.0, rng=Ceiling()) == 4.0  # type: ignore[arg-type]


def test_an_unset_retry_bound_means_the_default() -> None:
    assert _bare_provider(max_retries=None).max_retries == 3

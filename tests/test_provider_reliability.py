"""AW-18: provider reliability — schema validation, bounded repair, cost, fallback."""

from __future__ import annotations

from pathlib import Path

from genesis.providers import (
    ProviderExecutor,
    ProviderRequest,
    ProviderResponse,
    validate_schema,
)
from genesis.runtime import ProcessInvocation, ProcessResult
from genesis.service import GenesisService

SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "integer"}},
}


class ScriptedProvider:
    provider = "scripted"

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        value = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ProviderResponse(
            str(value), self.provider, request.model, f"req-{self.calls}", parsed=value
        )


def _invocation() -> ProcessInvocation:
    return ProcessInvocation("i1", "r1", "p1", context={"x": 1})


def test_schema_validator_reports_paths() -> None:
    errors = validate_schema(SCHEMA, {"answer": "not-an-int"})
    assert errors and "answer" in errors[0]
    assert validate_schema(SCHEMA, {"answer": 4}) == []
    assert validate_schema({"type": "array", "items": {"type": "integer"}}, [1, "x"]) == [
        "root[1]: expected integer, got str"
    ]


def test_schema_valid_output_passes_and_records_cost() -> None:
    provider = ScriptedProvider([{"answer": 42}])
    executor = ProviderExecutor(
        provider,
        model="m",
        output_schema=SCHEMA,
        parameters={"price_per_1k_input": 1.0, "price_per_1k_output": 2.0},
    )
    result = executor.execute(_invocation())
    assert result.status == "succeeded"
    assert result.outputs == {"response": {"answer": 42}}
    assert result.metadata["schema_valid"] is True
    assert result.metadata["repair_count"] == 0
    assert isinstance(result.metadata["estimated_cost"], float)
    assert provider.calls == 1


def test_bounded_repair_recovers_from_invalid_output() -> None:
    provider = ScriptedProvider([{"answer": "wrong"}, {"answer": 7}])
    executor = ProviderExecutor(provider, model="m", output_schema=SCHEMA)
    result = executor.execute(_invocation())
    assert result.status == "succeeded"
    assert result.outputs == {"response": {"answer": 7}}
    assert result.metadata["repair_count"] == 1
    assert provider.calls == 2


def test_unrecoverable_output_fails_with_validation_code() -> None:
    provider = ScriptedProvider([{"answer": "wrong"}])
    executor = ProviderExecutor(provider, model="m", output_schema=SCHEMA, max_repairs=0)
    result = executor.execute(_invocation())
    assert result.status == "failed"
    assert result.metadata["code"] == "OUTPUT_VALIDATION_FAILED"
    assert result.metadata["schema_valid"] is False


def test_declared_fallback_executes_only_on_failure(tmp_path: Path) -> None:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import ContextEngine, ExecutorRegistry, RunController, Scheduler

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    persistence.create_run({"id": "run-1", "build": "b"})
    processes = [
        {
            "id": "p",
            "executor": {},
            "context_policy": "public",
            "retry_policy": {
                "max_attempts": 1,
                "failure_policy": "use_declared_fallback",
                "fallback_outputs": {"counter": 9},
            },
            "state_effects": [{"field": "counter", "op": "set"}],
        }
    ]

    class FailingExecutor:
        def execute(self, _invocation: ProcessInvocation) -> ProcessResult:
            return ProcessResult(status="failed", metadata={"code": "OUTPUT_VALIDATION_FAILED"})

    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"p": FailingExecutor()}),
        ContextEngine({"public": {"allow": []}}),
        persistence=persistence,
        state_store=__import__("genesis.runtime", fromlist=["StateStore"]).StateStore(
            {"counter": int}, {"counter": 0}
        ),
    )
    controller.run("run-1", phase_limit=2)
    persisted = persistence.list_events("run-1")
    completed = [event for event in persisted if event["kind"] == "process_completed"]
    assert len(completed) == 1
    assert completed[0]["metadata"]["fallback"] is True
    assert completed[0]["metadata"]["original_code"] == "OUTPUT_VALIDATION_FAILED"
    import json

    artifact_payload = json.loads(persistence.list_artifacts("run-1")[0]["payload"])
    assert artifact_payload["outputs"]["counter"] == 9
    persistence.close()


def test_skip_with_event_failure_policy(tmp_path: Path) -> None:
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        ProcessResult,
        RunController,
        Scheduler,
    )

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    persistence.create_run({"id": "run-2", "build": "b"})
    processes = [
        {
            "id": "p",
            "executor": {},
            "context_policy": "public",
            "retry_policy": {
                "max_attempts": 1,
                "failure_policy": "skip_with_event",
                "fallback_outputs": {},
            },
        }
    ]

    class FailingExecutor:
        def execute(self, _invocation: ProcessInvocation) -> ProcessResult:
            return ProcessResult(status="failed", metadata={"code": "executor_defect"})

    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"p": FailingExecutor()}),
        ContextEngine({"public": {"allow": []}}),
        persistence=persistence,
    )
    controller.run("run-2", phase_limit=2)
    assert controller.status == "completed"
    events = persistence.list_events("run-2")
    kinds = {event["kind"] for event in events}
    assert "process_skipped" in kinds
    skipped = [event for event in events if event["kind"] == "process_skipped"]
    assert skipped[0]["metadata"]["skipped_fallback"] is True
    persistence.close()


def test_openai_adapter_decodes_json_completion_text() -> None:
    """Review finding 6: the OpenAI-compatible adapter decodes JSON completions."""
    import http.server
    import json as _json
    import os as _os
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            body = _json.dumps(
                {"choices": [{"message": {"content": '{"text": "ok"}'}}], "usage": {}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    import socketserver

    class FastServer(http.server.ThreadingHTTPServer):
        """Skip HTTPServer.server_bind's reverse-DNS getfqdn() call."""

        def server_bind(self) -> None:
            socketserver.TCPServer.server_bind(self)
            host, port = self.server_address[:2]
            self.server_name = str(host)
            self.server_port = port

    server = FastServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _os.environ["GENESIS_JSON_TEST_KEY"] = "test-key"
    try:
        from genesis.providers import OpenAICompatibleProvider, ProviderRequest

        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="m",
            api_key_env="GENESIS_JSON_TEST_KEY",
            timeout=5.0,
        )
        response = provider.generate(ProviderRequest(model="m", prompt="hi"))
        assert response.parsed == {"text": "ok"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_service_path_validates_generative_outputs_against_schema(
    tmp_path: Path, monkeypatch
) -> None:
    """Review finding 6: the real service path repairs/fails on schema-invalid output."""

    from genesis.providers import ProviderRequest as PR

    class SchemaProvider:
        provider = "openai-compatible"

        def __init__(self, **_kwargs) -> None:
            pass

        def generate(self, request: PR) -> ProviderResponse:
            return ProviderResponse(
                '{"answer": 42}',
                self.provider,
                request.model,
                "req-schema",
                parsed={"answer": 42},
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", SchemaProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_SCHEMA_KEY",
            }
        )
        draft = service.create_specification(
            {
                "id": "schema-study",
                "title": "schema study",
                "models": [{"id": "mp", "provider": "openai-compatible", "model": "m1"}],
                "processes": [
                    {
                        "id": "compose",
                        "openness_rationale": "openness",
                        "closure_rationale": "closure",
                        "executor": {"mode": "generative", "model_profile": "mp"},
                        "context_policy": "public",
                        "prompt_ref": "compose",
                        "outputs": [{"artifact_type": "response", "schema_ref": "answer-schema"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"artifacts": [{"id": "answer-out", "artifact_type": "text"}]},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [],
                "prompts": {"compose": "Answer from {context}"},
            }
        )
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "schema-study" / "schemas"
        )
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "answer-schema.yaml").write_text(
            "type: object\nproperties:\n  answer: {type: integer}\nrequired: [answer]\n"
        )
        # Editing re-records the draft digest (which now covers the schema asset).
        revised = service.update_specification(
            "schema-study", {"description": "with schema"}, draft["version"]
        )
        service.approve_specification("schema-study", revised["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/schema-study", specification_id="schema-study"
        )
        service.create_run(
            {"id": "schema-run", "study_id": "schema-study", "build": compiled["path"]}
        )
        result = service.execute_run("schema-run")
        assert result["status"] == "completed"
        events = service.trace_run("schema-run")
        completed = [e for e in events if e["kind"] == "process_completed"]
        assert completed[0]["metadata"]["schema_valid"] is True
    finally:
        service.close()


def test_cancellation_aborts_in_flight_provider_requests() -> None:
    """AW-18: cancelling during a request raises PROVIDER_CANCELLED promptly."""
    import http.server
    import json as _json
    import os as _os
    import socketserver
    import threading as _threading
    import time as _time

    from genesis.providers import OpenAICompatibleProvider

    class SlowHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            _time.sleep(3.0)
            body = _json.dumps(
                {"choices": [{"message": {"content": "late"}}], "usage": {}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class FastServer(http.server.ThreadingHTTPServer):
        def server_bind(self) -> None:
            socketserver.TCPServer.server_bind(self)
            host, port = self.server_address[:2]
            self.server_name = str(host)
            self.server_port = port

    server = FastServer(("127.0.0.1", 0), SlowHandler)
    server.daemon_threads = True
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _os.environ["GENESIS_CANCEL_KEY"] = "fake-key"
        cancel_event = _threading.Event()
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="slow-model",
            api_key_env="GENESIS_CANCEL_KEY",
            timeout=30.0,
            cancel_event=cancel_event,
        )
        assert provider.capabilities().cancellation is True
        from genesis.runtime import ProcessInvocation
        from genesis.service import ProviderExecutor

        executor = ProviderExecutor(provider, model="slow-model", cancel_event=cancel_event)
        invocation = ProcessInvocation("i-1", "r-1", "p", context=None)
        started = _time.perf_counter()
        cancel_event.set()
        import pytest

        with pytest.raises(ValueError, match="PROVIDER_CANCELLED"):
            executor.execute(invocation)
        assert _time.perf_counter() - started < 2.0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _os.environ.pop("GENESIS_CANCEL_KEY", None)


def test_pasted_api_key_is_sent_instead_of_environment(tmp_path: Path, monkeypatch) -> None:
    """Pasted profile keys are used directly; no environment variable is required."""
    import http.server
    import json as _json
    import socketserver
    import threading as _threading

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    captured: dict = {}

    class KeyHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            captured["authorization"] = self.headers.get("Authorization", "")
            body = _json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class FastServer(http.server.ThreadingHTTPServer):
        def server_bind(self) -> None:
            socketserver.TCPServer.server_bind(self)
            host, port = self.server_address[:2]
            self.server_name = str(host)
            self.server_port = port

    server = FastServer(("127.0.0.1", 0), KeyHandler)
    server.daemon_threads = True
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.delenv("GENESIS_PASTED_KEY", raising=False)
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="m",
            api_key_env="GENESIS_PASTED_KEY",
            api_key="sk-pasted-directly",
        )
        provider.generate(ProviderRequest(model="m", prompt="hello", parameters={}))
        assert captured["authorization"] == "Bearer sk-pasted-directly"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unsupported_parameters_are_auto_removed_on_retry(tmp_path: Path) -> None:
    """Modern models: max_tokens is translated to max_completion_tokens on a 400."""

    import http.server
    import json as _json
    import os as _os
    import socketserver
    import threading as _threading

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    seen_bodies: list[dict] = []
    calls = {"count": 0}

    class DegradeHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            seen_bodies.append(_json.loads(raw))
            calls["count"] += 1
            if calls["count"] == 1:
                body = _json.dumps(
                    {
                        "error": {
                            "message": "Unsupported parameter: 'max_tokens' is not supported "
                            "with this model. Use 'max_completion_tokens' instead.",
                            "type": "invalid_request_error",
                            "param": "max_tokens",
                            "code": "unsupported_parameter",
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = _json.dumps(
                {"choices": [{"message": {"content": "ok"}}], "usage": {}, "id": "r-1"}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class FastServer(http.server.ThreadingHTTPServer):
        def server_bind(self) -> None:
            socketserver.TCPServer.server_bind(self)
            host, port = self.server_address[:2]
            self.server_name = str(host)
            self.server_port = port

    server = FastServer(("127.0.0.1", 0), DegradeHandler)
    server.daemon_threads = True
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _os.environ["GENESIS_DEGRADE_KEY"] = "k"
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="m",
            api_key_env="GENESIS_DEGRADE_KEY",
        )
        response = provider.generate(
            ProviderRequest(
                model="m",
                prompt="hello",
                parameters={"temperature": 0.2, "max_tokens": 256},
            )
        )
        assert calls["count"] == 2
        assert "max_tokens" not in seen_bodies[1]
        assert seen_bodies[1]["max_completion_tokens"] == 256
        # Only the rejected parameter is degraded; temperature is preserved.
        assert seen_bodies[1]["temperature"] == 0.2
        assert response.metadata.get("degraded_parameters") == ["max_tokens"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _os.environ.pop("GENESIS_DEGRADE_KEY", None)


def test_unsupported_structured_output_parameter_can_degrade_safely() -> None:
    from genesis.providers import OpenAICompatibleProvider

    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema"},
    }
    fixed = OpenAICompatibleProvider._unsupported_parameter_fix(
        payload,
        "unsupported_parameter: 'response_format' is not supported with this model",
    )
    assert fixed is not None
    assert "response_format" not in fixed
    assert fixed["messages"] == payload["messages"]


def test_persistent_parameter_errors_still_fail_loudly(tmp_path: Path, monkeypatch) -> None:
    """A second 400 after degradation propagates the provider error."""

    import http.server
    import json as _json
    import os as _os
    import socketserver
    import threading as _threading

    import pytest as _pytest

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    class Always400Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            body = _json.dumps(
                {
                    "error": {
                        "message": "Unsupported parameter: 'temperature' is not supported "
                        "with this model",
                        "param": "temperature",
                        "code": "unsupported_parameter",
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    class FastServer(http.server.ThreadingHTTPServer):
        def server_bind(self) -> None:
            socketserver.TCPServer.server_bind(self)
            host, port = self.server_address[:2]
            self.server_name = str(host)
            self.server_port = port

    server = FastServer(("127.0.0.1", 0), Always400Handler)
    server.daemon_threads = True
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _os.environ["GENESIS_ALWAYS400_KEY"] = "k"
        provider = OpenAICompatibleProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="m",
            api_key_env="GENESIS_ALWAYS400_KEY",
        )
        with _pytest.raises(ValueError, match="PROVIDER_HTTP"):
            provider.generate(ProviderRequest(model="m", prompt="hello", parameters={}))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _os.environ.pop("GENESIS_ALWAYS400_KEY", None)

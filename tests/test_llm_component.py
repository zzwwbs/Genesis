"""Tests for configurable OpenAI-compatible model profiles and provider calls."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest
from fastapi.testclient import TestClient

from genesis.app import create_app
from genesis.providers import OpenAICompatibleProvider, ProviderRequest, ProviderResponse
from genesis.runtime import ProcessInvocation
from genesis.service import GenesisService


class FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def test_openai_compatible_provider_formats_request_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data or b"{}")
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "id": "chatcmpl-test",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )

    monkeypatch.setenv("GENESIS_TEST_API_KEY", "secret-value")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1/",
        model="test-model",
        api_key_env="GENESIS_TEST_API_KEY",
        timeout=7,
    )

    response = provider.generate(
        ProviderRequest(
            model="test-model",
            prompt="Say hello",
            parameters={"temperature": 0},
        )
    )

    assert captured == {
        "url": "https://example.test/v1/chat/completions",
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer secret-value",
            "Content-type": "application/json",
        },
        "body": {
            "messages": [{"content": "Say hello", "role": "user"}],
            "model": "test-model",
            "temperature": 0,
        },
        "timeout": 7,
    }
    assert response.text == "hello"
    assert response.model == "test-model"
    assert response.request_id == "chatcmpl-test"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 1}


def test_openai_compatible_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENESIS_MISSING_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="GENESIS_MISSING_API_KEY",
    )

    with pytest.raises(ValueError, match="credential unavailable"):
        provider.generate(ProviderRequest(model="test-model", prompt="hello"))


def test_provider_executor_records_latency_and_context_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENESIS_TEST_API_KEY", "secret-value")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, timeout: FakeResponse(
            {"id": "request-1", "choices": [{"message": {"content": "OK"}}]}
        ),
    )
    provider = OpenAICompatibleProvider(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="GENESIS_TEST_API_KEY",
    )
    from genesis.providers import ProviderExecutor

    result = ProviderExecutor(provider, model="test-model").execute(
        ProcessInvocation("inv-1", "run-1", "process-1", context={"state": {"x": 1}})
    )

    assert result.metadata["context_hash"] is None
    assert result.metadata["latency_ms"] >= 0


def test_model_profile_api_persists_non_secret_configuration(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    response = client.post(
        "/llm/profiles",
        json={
            "id": "openai-default",
            "provider": "openai-compatible",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "parameters": {"temperature": 0.2, "max_tokens": 100},
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == "openai-default"
    assert body["model"] == "gpt-4o-mini"
    assert body["api_key_env"] == "OPENAI_API_KEY"
    assert "api_key" not in body
    assert (
        "OPENAI_API_KEY"
        in (tmp_path / "workspace" / ".genesis" / "model-profiles.json").read_text()
    )


def test_model_profile_status_reports_key_presence_without_revealing_value(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    client.post(
        "/llm/profiles",
        json={
            "id": "openai-default",
            "provider": "openai-compatible",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")

    response = client.get("/llm/profiles/openai-default/status")

    assert response.status_code == 200
    assert response.json() == {"id": "openai-default", "credential_present": True}
    assert "secret-value" not in response.text


def test_model_profile_connection_test_is_explicit_and_returns_normalized_metadata(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))
    client.post(
        "/llm/profiles",
        json={
            "id": "openai-default",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "gpt-test",
            "api_key_env": "OPENAI_API_KEY",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, timeout: FakeResponse(
            {"id": "test-request", "choices": [{"message": {"content": "OK"}}]}
        ),
    )

    response = client.post("/llm/profiles/openai-default/test", json={"prompt": "Say OK"})

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "OK"
    assert response.json()["request_id"] == "test-request"
    assert "secret-value" not in response.text


def test_model_profile_rejects_non_http_endpoint(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "workspace"))

    response = client.post(
        "/llm/profiles",
        json={
            "id": "bad-profile",
            "provider": "openai-compatible",
            "base_url": "file:///tmp/model",
            "model": "gpt-test",
            "api_key_env": "OPENAI_API_KEY",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MODEL_URL"


def test_approved_generative_process_uses_selected_model_profile(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    service.create_model_profile(
        {
            "id": "test-model",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "test-model-v1",
            "api_key_env": "GENESIS_TEST_API_KEY",
            "parameters": {"temperature": 0},
        }
    )
    draft = service.create_specification(
        {
            "id": "generative-study",
            "title": "Generative study",
            "models": [
                {
                    "id": "test-model",
                    "provider": "openai-compatible",
                    "model": "test-model-v1",
                    "parameters": {"temperature": 0},
                }
            ],
            "processes": [
                {
                    "id": "respond",
                    "openness_rationale": "Use the model to produce a bounded response.",
                    "closure_rationale": "the response form is part of the phenomenon studied",
                    "executor": {"mode": "generative", "model_profile": "test-model"},
                    "context_policy": "private",
                    "prompt_ref": "respond",
                    "outputs": [{"artifact_type": "response", "schema_ref": "answer"}],
                }
            ],
            "artifacts": [{"id": "answer", "artifact_type": "text"}],
            "prompts": {"respond": "Answer briefly using this context: {context}"},
        }
    )
    schema_dir = workspace / ".genesis/specifications/generative-study/schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "answer.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    revised = service.update_specification(
        "generative-study", {"description": "with schema"}, draft["version"]
    )
    approved = service.approve_specification("generative-study", revised["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(
        None, "builds/generative-study", specification_id="generative-study"
    )
    assert (workspace / "builds/generative-study/model_profiles.json").is_file()
    assert (workspace / "builds/generative-study/prompt_templates.json").is_file()

    service.create_run(
        {"id": "generative-run", "study_id": "generative-study", "build": compiled["path"]}
    )

    class FakeProvider:
        def __init__(self, **_kwargs):
            pass

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            import json as _json

            text = _json.dumps({"text": "generated"})
            return ProviderResponse(
                text,
                "openai-compatible",
                request.model,
                "req-1",
                parsed=_json.loads(text),
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeProvider)
    result = service.execute_run("generative-run")

    assert result["status"] == "completed"
    events = service.trace_run("generative-run")
    assert any(event.get("metadata", {}).get("provider") == "openai-compatible" for event in events)

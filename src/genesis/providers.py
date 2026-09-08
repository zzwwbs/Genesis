"""Provider-neutral local adapters.

The port intentionally has no network or credential concerns; external adapters can
implement the same request/response dataclasses later.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from genesis.runtime import ProcessInvocation, ProcessResult
from genesis.schema_validation import SchemaDiagnostic
from genesis.schema_validation import validate_schema as _authoritative_validate_schema


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    prompt: str
    parameters: dict[str, Any] = field(default_factory=dict)
    context_hash: str | None = None
    artifact_id: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    request_id: str
    parsed: Any = None
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderCapabilities:
    structured_output: bool = True
    cancellation: bool = False
    token_estimation: bool = True
    streaming: bool = False


class ModelProvider(Protocol):
    def capabilities(self) -> ProviderCapabilities: ...
    def generate(self, request: ProviderRequest) -> ProviderResponse: ...
    def estimate_tokens(self, request: ProviderRequest) -> int: ...


class ProviderExecutor:
    """Adapt a provider response to the runtime result contract with trace metadata."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        prompt_template: str = "{context}",
        parameters: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        output_schema_validator: Callable[[Any], list[SchemaDiagnostic]] | None = None,
        output_key: str = "response",
        mode: str = "generative",
        max_repairs: int = 1,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template
        self.parameters = dict(parameters or {})
        self.output_schema = output_schema
        self.output_schema_validator = output_schema_validator
        self.output_key = output_key
        self.mode = mode
        self.max_repairs = max(0, int(max_repairs))
        self.cancel_event = cancel_event

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ValueError("PROVIDER_CANCELLED: provider call cancelled before execution")

    def _schema_messages(self, value: Any) -> list[str]:
        """Human-readable validation messages for repair prompts and metadata.

        Uses the authoritative catalog-backed validator when one is bound
        (SCH-002: the same resolver and dialect as compilation), otherwise the
        standalone Draft 2020-12 validator for a directly supplied schema.
        """
        if self.output_schema_validator is not None:
            return [
                f"{diagnostic.instance_pointer or 'root'}: {diagnostic.message}"
                for diagnostic in self.output_schema_validator(value)
            ]
        return validate_schema(self.output_schema or {}, value)

    def _estimated_cost(self, usage: dict[str, int]) -> float | None:
        """Estimate cost from usage and configured per-1k-token prices (AW-18)."""
        price_in = self.parameters.get("price_per_1k_input")
        price_out = self.parameters.get("price_per_1k_output")
        if not isinstance(price_in, int | float) or not isinstance(price_out, int | float):
            return None
        return int(usage.get("prompt_tokens", 0)) / 1000 * float(price_in) + int(
            usage.get("completion_tokens", 0)
        ) / 1000 * float(price_out)

    def _outputs(self, value: Any) -> dict[str, Any]:
        outputs = {self.output_key: value}
        if self.mode == "generative" and self.output_key != "response":
            outputs["response"] = value
        return outputs

    def execute(self, invocation: ProcessInvocation) -> ProcessResult:
        context_value = getattr(invocation.context, "data", invocation.context)
        context = json.dumps(context_value, sort_keys=True, default=str)
        prompt = self.prompt_template.replace("{context}", context)
        prompt = prompt.replace("{actor_ids}", ", ".join(invocation.actor_ids)).replace(
            "{phase}", str(invocation.phase)
        )
        request = ProviderRequest(
            model=self.model,
            prompt=prompt,
            parameters=self.parameters,
            context_hash=getattr(invocation.context, "content_hash", None),
        )
        self._raise_if_cancelled()
        started = time.perf_counter()
        response = self.provider.generate(request)
        provider_attempts = [
            {
                "request_id": response.request_id,
                "raw_response": response.text,
                "parsed_response": response.parsed,
            }
        ]
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        value = response.parsed if response.parsed is not None else response.text
        if self.output_schema is not None:
            errors = self._schema_messages(value)
            repairs = 0
            while errors and repairs < self.max_repairs:
                self._raise_if_cancelled()
                repairs += 1
                request = ProviderRequest(
                    model=self.model,
                    prompt=prompt
                    + "\n\nValidation errors in your previous response: "
                    + "; ".join(errors)
                    + "\nRespond again with a corrected JSON object.",
                    parameters=self.parameters,
                    context_hash=getattr(invocation.context, "content_hash", None),
                )
                response = self.provider.generate(request)
                provider_attempts.append(
                    {
                        "request_id": response.request_id,
                        "raw_response": response.text,
                        "parsed_response": response.parsed,
                    }
                )
                value = response.parsed if response.parsed is not None else response.text
                errors = self._schema_messages(value)
            metadata: dict[str, Any] = {
                "mode": self.mode,
                "provider": response.provider,
                "model": response.model,
                "request_id": response.request_id,
                "prompt_hash": prompt_hash,
                "context_hash": request.context_hash,
                "parameters": dict(request.parameters),
                "usage": dict(response.usage),
                "provider_metadata": dict(response.metadata),
                "latency_ms": latency_ms,
                "repair_count": repairs,
                "schema_valid": not errors,
                "raw_response": response.text,
                "parsed_response": response.parsed,
                "provider_attempts": provider_attempts,
            }
            estimated = self._estimated_cost(response.usage)
            if estimated is not None:
                metadata["estimated_cost"] = estimated
            if errors:
                metadata["code"] = "OUTPUT_VALIDATION_FAILED"
                metadata["validation_errors"] = errors
                return ProcessResult(
                    status="failed",
                    outputs=self._outputs(value),
                    metadata=metadata,
                )
            return ProcessResult(outputs=self._outputs(value), metadata=metadata)
        return ProcessResult(
            outputs=self._outputs(
                response.parsed if response.parsed is not None else response.text
            ),
            metadata={
                "mode": self.mode,
                "provider": response.provider,
                "model": response.model,
                "request_id": response.request_id,
                "prompt_hash": prompt_hash,
                "context_hash": request.context_hash,
                "parameters": dict(request.parameters),
                "usage": dict(response.usage),
                "provider_metadata": dict(response.metadata),
                "latency_ms": latency_ms,
                "raw_response": response.text,
                "parsed_response": response.parsed,
                "provider_attempts": provider_attempts,
            },
        )


class DeterministicMockProvider:
    provider = "deterministic-mock"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def estimate_tokens(self, request: ProviderRequest) -> int:
        return max(1, len(request.prompt.split()))

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        canonical = json.dumps(
            {"model": request.model, "prompt": request.prompt, "parameters": request.parameters},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        text = f"mock:{digest[:16]}"
        return ProviderResponse(
            text,
            self.provider,
            request.model,
            digest,
            parsed=text,
            usage={"prompt_tokens": self.estimate_tokens(request), "completion_tokens": 1},
        )


class OpenAICompatibleProvider:
    """Small adapter for providers implementing OpenAI Chat Completions."""

    provider = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = api_key
        self.timeout = timeout
        self.cancel_event = cancel_event

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_output=True,
            token_estimation=True,
            cancellation=self.cancel_event is not None,
        )

    @staticmethod
    def _unsupported_parameter_fix(payload: dict[str, Any], detail: str) -> dict[str, Any] | None:
        """Strip parameters a model rejects (e.g. temperature, max_tokens).

        Newer OpenAI models reject ``max_tokens`` (use ``max_completion_tokens``)
        and some reject ``temperature``. When the provider answers with an
        unsupported-parameter error, rebuild the payload once without the
        offending parameters before surfacing the failure.
        """
        lowered = detail.lower()
        marker = (
            "unsupported_parameter" in lowered
            or "unsupported parameter" in lowered
            or "not supported with this model" in lowered
        )
        if not marker:
            return None
        fixed = dict(payload)
        changed = False
        if "'max_tokens'" in lowered or '"max_tokens"' in lowered:
            max_tokens = fixed.pop("max_tokens", None)
            if max_tokens is None:
                fixed.pop("max_completion_tokens", None)
            else:
                fixed["max_completion_tokens"] = max_tokens
            changed = True
        if "temperature" in lowered:
            fixed.pop("temperature", None)
            changed = True
        if "response_format" in lowered:
            fixed.pop("response_format", None)
            changed = True
        return fixed if changed else None

    def estimate_tokens(self, request: ProviderRequest) -> int:
        return max(1, len(request.prompt.split()))

    def _open_cancellable(self, http_request: Any) -> tuple[bytes, int]:
        """Open the request, aborting the wait when the researcher cancels.

        The socket read runs on a daemon thread; the caller waits on a
        completion event and a cancellation event. On cancellation the
        request is treated as aborted immediately (the daemon thread is
        bounded by ``timeout`` and discards its result).
        """
        done = threading.Event()
        box: list[tuple[bytes, int] | Exception] = []

        def _open() -> None:
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                    box.append((response.read(), getattr(response, "status", 200)))
            except Exception as exc:  # surfaced by the caller
                box.append(exc)
            finally:
                done.set()

        threading.Thread(target=_open, daemon=True).start()
        while not done.is_set():
            if self.cancel_event is not None and self.cancel_event.is_set():
                raise ValueError("PROVIDER_CANCELLED: provider request aborted by researcher")
            done.wait(timeout=0.05)
        result = box[0]
        if isinstance(result, Exception):
            raise result
        return result

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        api_key = self.api_key or os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"PROVIDER_CREDENTIAL: credential unavailable; set {self.api_key_env} "
                "or store an api_key on the model profile"
            )
        payload = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            **request.parameters,
        }
        degraded_parameters: list[str] = []
        attempts = 0
        while True:
            body = json.dumps(payload, separators=(",", ":")).encode()
            http_request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                method="POST",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                raw, status = self._open_cancellable(http_request)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:2000]
                if attempts == 0:
                    fixed = self._unsupported_parameter_fix(payload, detail)
                    if fixed is not None:
                        payload = fixed
                        degraded_parameters = [
                            key
                            for key in ("temperature", "max_tokens", "max_completion_tokens")
                            if key not in payload
                        ]
                        attempts += 1
                        continue
                raise ValueError(
                    f"PROVIDER_HTTP: provider returned HTTP {exc.code}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ValueError(f"PROVIDER_UNAVAILABLE: provider request failed: {exc}") from exc
            if not 200 <= status < 300:
                raise ValueError(f"PROVIDER_HTTP: provider returned HTTP {status}")
            break
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("PROVIDER_RESPONSE: provider returned invalid JSON") from exc
        try:
            choice = result["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("PROVIDER_RESPONSE: response has no chat completion choice") from exc
        if not isinstance(text, str):
            raise ValueError("PROVIDER_RESPONSE: completion content must be text")
        parsed: Any = text
        if text.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text  # plain-text completion; leave as-is
        usage = result.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        return ProviderResponse(
            text=text,
            provider=self.provider,
            model=str(result.get("model", request.model or self.model)),
            request_id=str(result.get("id", "")),
            parsed=parsed,
            usage={key: int(value) for key, value in usage.items() if isinstance(value, int)},
            metadata={
                "base_url": self.base_url,
                **({"degraded_parameters": degraded_parameters} if degraded_parameters else {}),
            },
        )


class RecordedArtifactProvider:
    provider = "recorded-artifact"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(structured_output=True, token_estimation=False)

    def __init__(self, artifacts: dict[str, Any]):
        self._artifacts = dict(artifacts)

    def estimate_tokens(self, request: ProviderRequest) -> int:
        return 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not request.artifact_id:
            raise ValueError("recorded provider requires artifact_id")
        if request.artifact_id not in self._artifacts:
            raise KeyError(request.artifact_id)
        recorded = self._artifacts[request.artifact_id]
        if not isinstance(recorded, dict) or "payload" not in recorded or "hash" not in recorded:
            raise ValueError("recorded artifact requires payload and hash")
        value = recorded["payload"]
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if digest != recorded["hash"]:
            raise ValueError(f"artifact integrity failure: {request.artifact_id}")
        return ProviderResponse(
            encoded,
            self.provider,
            request.model,
            digest,
            parsed=value,
            metadata={"artifact_id": request.artifact_id, "recorded": True},
        )


def validate_schema(schema: dict[str, Any], value: Any, path: str = "root") -> list[str]:
    """Validate a value under Draft 2020-12, returning human-readable messages.

    Delegates to the authoritative package validator (SCH-001/002), so the same
    dialect and semantics back repair prompts, runtime enforcement and
    compile-time checks. ``path`` is accepted for API compatibility; messages
    carry full instance paths derived from the validator.
    """
    return _authoritative_validate_schema(schema, value)


class ProviderRegistry:
    def __init__(self, providers: dict[str, ModelProvider] | None = None):
        self._providers = dict(providers or {})

    def register(self, name: str, provider: ModelProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> ModelProvider:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"provider unavailable: {name}") from None

    def capabilities(self) -> dict[str, ProviderCapabilities]:
        return {name: provider.capabilities() for name, provider in self._providers.items()}

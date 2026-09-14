"""A package's model runs on the profile it names, and drift means what is sent.

`endpoint_ref` is documented as a name "resolved from local configuration, not
inlined", but nothing resolved it: the run used the profile named after the
model id, and the drift check compared the name to that profile's base URL --
a name and a URL, which can never match. A package that bound its model to the
`deepseek` profile was refused MODEL_PROFILE_DRIFT: endpoint on every run.

The parameter check demanded the profile's parameters equal the package's, but
a run sends the two merged with the package's taking precedence. A profile that
simply omits a package-supplied temperature changes nothing that is sent, and
was refused; a profile adding a key the package does not set changes the
request, and must be.

Both refusals also fired after the run was marked running, leaving it failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from genesis.providers import ProviderResponse
from genesis.service import GenesisService
from tests.test_profile_unification import PAYLOAD

SENT: list[dict[str, Any]] = []


class RecordingProvider:
    provider = "openai-compatible"

    def __init__(self, **kwargs: Any) -> None:
        self.base_url = kwargs.get("base_url")

    def generate(self, request: Any) -> ProviderResponse:
        import json

        SENT.append({"base_url": self.base_url, "parameters": dict(request.parameters or {})})
        text = json.dumps({"text": "generated"})
        return ProviderResponse(text, self.provider, request.model, "req", parsed=json.loads(text))


def _profile(pid: str, base_url: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": pid,
        "provider": "openai-compatible",
        "base_url": base_url,
        "model": "m1",
        "api_key_env": "GENESIS_ENDPOINT_KEY",
        **extra,
    }


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    endpoint_ref: str | None,
    package_parameters: dict[str, Any],
    profiles: list[dict[str, Any]],
) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RecordingProvider)
    monkeypatch.setenv("GENESIS_ENDPOINT_KEY", "test-only")
    SENT.clear()
    service = GenesisService(tmp_path / "workspace")
    for profile in profiles:
        service.create_model_profile(profile)
    payload = {
        **PAYLOAD,
        "models": [
            {
                "id": "mp",
                "provider": "openai-compatible",
                "model": "m1",
                "parameters": package_parameters,
                **({"endpoint_ref": endpoint_ref} if endpoint_ref else {}),
            }
        ],
    }
    draft = service.create_specification(payload)
    schemas = tmp_path / "workspace" / ".genesis" / "specifications" / "drift-study" / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    (schemas / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    revised = service.update_specification("drift-study", {"description": "d"}, draft["version"])
    service.approve_specification("drift-study", revised["version"], "researcher")
    compiled = service.compile_study(None, "builds/drift-study", specification_id="drift-study")
    service.create_run({"id": "r", "study_id": "drift-study", "build": compiled["path"]})
    return service


def test_a_model_runs_on_the_profile_its_endpoint_ref_names(tmp_path, monkeypatch) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        endpoint_ref="named",
        package_parameters={},
        profiles=[
            _profile("mp", "https://by-model-id.test"),
            _profile("named", "https://named.test"),
        ],
    )
    try:
        assert service.execute_run("r")["status"] == "completed"
        assert {call["base_url"] for call in SENT} == {"https://named.test"}
    finally:
        service.close()


def test_without_an_endpoint_ref_the_model_id_names_the_profile(tmp_path, monkeypatch) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        endpoint_ref=None,
        package_parameters={},
        profiles=[_profile("mp", "https://by-model-id.test")],
    )
    try:
        assert service.execute_run("r")["status"] == "completed"
        assert {call["base_url"] for call in SENT} == {"https://by-model-id.test"}
    finally:
        service.close()


def test_an_endpoint_ref_naming_no_profile_is_refused_and_leaves_the_run(
    tmp_path, monkeypatch
) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        endpoint_ref="absent",
        package_parameters={},
        profiles=[_profile("mp", "https://by-model-id.test")],
    )
    try:
        with pytest.raises(ValueError, match="absent"):
            service.execute_run("r")
        assert service.get_run("r")["status"] == "created"
        assert SENT == []
    finally:
        service.close()


def test_a_profile_omitting_the_package_s_parameters_changes_nothing_sent(
    tmp_path, monkeypatch
) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        endpoint_ref="named",
        package_parameters={"temperature": 0.7},
        profiles=[_profile("named", "https://named.test")],
    )
    try:
        assert service.execute_run("r")["status"] == "completed"
        assert all(call["parameters"].get("temperature") == 0.7 for call in SENT)
    finally:
        service.close()


def test_a_profile_adding_a_parameter_the_package_does_not_set_is_drift(
    tmp_path, monkeypatch
) -> None:
    service = _service(
        tmp_path,
        monkeypatch,
        endpoint_ref="named",
        package_parameters={"temperature": 0.7},
        profiles=[_profile("named", "https://named.test", parameters={"top_p": 0.5})],
    )
    try:
        with pytest.raises(ValueError, match="MODEL_PROFILE_DRIFT.*parameters"):
            service.execute_run("r")
        assert service.get_run("r")["status"] == "created"
        assert SENT == []
    finally:
        service.close()

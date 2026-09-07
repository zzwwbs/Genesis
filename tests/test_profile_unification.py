"""AW-10: model profile unification — run-time drift detection against compiled packages."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.providers import ProviderResponse
from genesis.service import GenesisService

PAYLOAD = {
    "id": "drift-study",
    "title": "drift study",
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
    "protocol": {"time_model": {"type": "rounds", "end": 2}},
    "outcomes": [],
    "prompts": {"compose": "Compose from {context}"},
}


class FakeProvider:
    provider = "openai-compatible"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        import json as _json

        text = _json.dumps({"text": "generated"})
        return ProviderResponse(
            text, self.provider, request.model, "req-1", parsed=_json.loads(text)
        )


def _setup(tmp_path: Path, monkeypatch, *, model: str = "m1") -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeProvider)
    service = GenesisService(tmp_path / "workspace")
    service.create_model_profile(
        {
            "id": "mp",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": model,
            "api_key_env": "GENESIS_DRIFT_KEY",
        }
    )
    draft = service.create_specification(PAYLOAD)
    schema_dir = tmp_path / "workspace" / ".genesis" / "specifications" / "drift-study" / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
    )
    revised = service.update_specification(
        "drift-study", {"description": "with schema"}, draft["version"]
    )
    approved = service.approve_specification("drift-study", revised["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(None, "builds/drift-study", specification_id="drift-study")
    service.create_run({"id": "drun", "study_id": "drift-study", "build": compiled["path"]})
    return service


def test_matching_profile_executes_and_records_version(tmp_path: Path, monkeypatch) -> None:
    service = _setup(tmp_path, monkeypatch, model="m1")
    try:
        result = service.execute_run("drun")
        assert result["status"] == "completed"
        manifest = service.get_run("drun")["manifest"]
        assert manifest["model_versions"] == {"mp": "m1"}
    finally:
        service.close()


def test_drifted_runtime_model_fails_loudly(tmp_path: Path, monkeypatch) -> None:
    service = _setup(tmp_path, monkeypatch, model="m1")
    try:
        current = service.get_model_profile("mp")
        service.update_model_profile("mp", {"model": "m2-different"}, current["version"])
        with pytest.raises(ValueError, match="MODEL_PROFILE_DRIFT.*model"):
            service.execute_run("drun")
    finally:
        service.close()


def test_missing_profile_still_fails_clearly(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(PAYLOAD)
        schema_dir = (
            tmp_path / "workspace" / ".genesis" / "specifications" / "drift-study" / "schemas"
        )
        schema_dir.mkdir(parents=True, exist_ok=True)
        (schema_dir / "compose-out.yaml").write_text(
            "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
        )
        revised = service.update_specification(
            "drift-study", {"description": "with schema"}, draft["version"]
        )
        service.approve_specification("drift-study", revised["version"], "researcher")
        compiled = service.compile_study(None, "builds/drift-study", specification_id="drift-study")
        service.create_run({"id": "drun", "study_id": "drift-study", "build": compiled["path"]})
        with pytest.raises(KeyError):
            service.execute_run("drun")
    finally:
        service.close()


def test_pasted_profile_key_reports_credential_without_environment(tmp_path: Path) -> None:
    """A pasted api_key satisfies credential checks without any env variable."""
    import os

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        os.environ.pop("GENESIS_FAKE_KEY", None)
        service.create_model_profile(
            {
                "id": "pasted-key",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m",
                "api_key_env": "GENESIS_FAKE_KEY",
                "api_key": "sk-local-pasted",
            }
        )
        status = service.model_profile_status("pasted-key")
        assert status["credential_present"] is True
        # Read endpoints never expose the pasted key.
        public = service.get_model_profile("pasted-key")
        assert "api_key" not in public
        # Updating without a key keeps it (merge semantics); a new pasted key replaces it.
        service.update_model_profile(
            "pasted-key",
            {
                "id": "pasted-key",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m2",
                "api_key_env": "GENESIS_FAKE_KEY",
            },
            1,
        )
        assert service._full_model_profile("pasted-key").get("api_key") == "sk-local-pasted"
        service.update_model_profile(
            "pasted-key",
            {
                "id": "pasted-key",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m3",
                "api_key_env": "GENESIS_FAKE_KEY",
                "api_key": "sk-replaced",
            },
            2,
        )
        assert service._full_model_profile("pasted-key").get("api_key") == "sk-replaced"
    finally:
        service.close()

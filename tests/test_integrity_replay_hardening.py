import hashlib
import json
import shutil
from pathlib import Path

import pytest

from genesis.assistant import StudyAssistant
from genesis.compiler import StudyCompiler
from genesis.extensions import ExtensionManifest, ExtensionRegistry
from genesis.providers import ProviderExecutor, ProviderResponse, RecordedArtifactProvider
from genesis.replay import ReplayManager, ReplayMode, ReplayRequest
from genesis.runtime import ContextEnvelope, ProcessInvocation


def artifact(value):
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"payload": value, "hash": digest}


def test_replay_requires_hashes_and_deeply_freezes_payloads() -> None:
    manager = ReplayManager()
    request = ReplayRequest("run-1", ReplayMode.ARTIFACT, artifact_ids=("a",))
    with pytest.raises(ValueError, match="hash"):
        manager.replay(request, {"a": {"value": 1}})
    result = manager.replay(request, {"a": artifact({"nested": {"value": 1}})})
    with pytest.raises(TypeError):
        result.artifacts["a"]["nested"]["value"] = 2


def test_replay_modes_validate_required_and_forbidden_fields() -> None:
    manager = ReplayManager()
    with pytest.raises(ValueError, match="artifact_ids"):
        manager.replay(ReplayRequest("run-1", ReplayMode.ARTIFACT), {})
    with pytest.raises(ValueError, match="boundary"):
        manager.replay(ReplayRequest("run-1", ReplayMode.PARTIAL), {})
    with pytest.raises(ValueError, match="justification"):
        manager.replay(ReplayRequest("run-1", ReplayMode.BRANCH, boundary="phase-1"), {})


def test_provider_executor_serializes_context_envelope_as_structured_json() -> None:
    class CapturingProvider:
        request = None

        def generate(self, request):
            self.request = request
            return ProviderResponse("ok", "capture", request.model, "request-1")

    provider = CapturingProvider()
    envelope = ContextEnvelope("private", "inv-1", {"state": {"score": 2}}, "hash-1")
    ProviderExecutor(provider, model="mock").execute(
        ProcessInvocation("inv-1", "run-1", "process-1", context=envelope)
    )
    assert json.loads(provider.request.prompt) == {"state": {"score": 2}}
    assert provider.request.context_hash == "hash-1"


def test_recorded_provider_requires_and_verifies_artifact_hash() -> None:
    good = artifact({"answer": 42})
    provider = RecordedArtifactProvider({"a": good})
    assert provider.generate(_provider_request("a")).parsed == {"answer": 42}
    with pytest.raises(ValueError, match="integrity"):
        RecordedArtifactProvider({"a": {**good, "hash": "0" * 64}}).generate(_provider_request("a"))


def _provider_request(artifact_id: str):
    from genesis.providers import ProviderRequest

    return ProviderRequest(model="recorded", prompt="ignored", artifact_id=artifact_id)


def test_assistant_rejects_tampered_compiled_build(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(Path(__file__).parent / "fixtures" / "specification", source)
    build = tmp_path / "build"
    StudyCompiler(source).compile(build)
    processes = build / "processes.json"
    processes.chmod(0o644)
    processes.write_text("[{}]\n")
    response = StudyAssistant().inspect_build(build)
    assert response.valid is False
    assert response.issues[0]["code"] == "BUILD_INTEGRITY"


def test_extension_registration_verifies_declared_digest() -> None:
    material = b"trusted extension payload"
    manifest = ExtensionManifest(
        id="demo-extension",
        version="1.0.0",
        genesis_range=">=0.1,<0.2",
        schema_range=">=1.0,<2.0",
        capabilities=("executor",),
        entry_point="demo:factory",
        integrity_hash=hashlib.sha256(material).hexdigest(),
    )
    registry = ExtensionRegistry(genesis_version="0.1.0", schema_version="1.0")
    with pytest.raises(ValueError, match="digest"):
        registry.register(manifest, lambda: None, enabled=True, integrity_material=b"tampered")
    registry.register(manifest, lambda: "ok", enabled=True, integrity_material=material)
    assert registry.get("demo-extension")() == "ok"

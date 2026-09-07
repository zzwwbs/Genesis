"""AW-21: golden studies and release qualification (ACC-011)."""

from __future__ import annotations

import shutil
from pathlib import Path

from genesis.analysis import AnalysisExporter
from genesis.providers import ProviderResponse
from genesis.service import GenesisService

GOLDEN = Path(__file__).parent / "golden_studies"


class GoldenProvider:
    provider = "openai-compatible"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        import json as _json

        text = _json.dumps({"text": f"generated-{request.model}", "body": "gen", "meaning": "made"})
        return ProviderResponse(
            text, self.provider, request.model, "req-golden", parsed=_json.loads(text)
        )


class RaisingProvider:
    """Compilation must never construct a provider (AST-006)."""

    def __init__(self, **_kwargs) -> None:
        raise AssertionError("provider must not be constructed during compilation")


def _workspace_for(tmp_path: Path, name: str) -> GenesisService:
    spec_id = name.replace("_", "-")
    workspace = tmp_path / f"ws-{name}"
    service = GenesisService(workspace)
    for profile_id, model in (
        ("creator-model", "creator-gpt"),
        ("member-model", "member-gpt"),
    ):
        parameters = {"temperature": 0.7} if profile_id == "creator-model" else {"temperature": 0.5}
        service.create_model_profile(
            {
                "id": profile_id,
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": model,
                "api_key_env": "GENESIS_GOLDEN_KEY",
                "parameters": parameters,
            }
        )
    source = workspace / "imports" / name
    shutil.copytree(GOLDEN / name, source)
    imported = service.import_package(source, specification_id=spec_id)
    assert imported["status"] == "draft"
    approved = service.approve_specification(spec_id, 1, "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(None, f"builds/{name}", specification_id=spec_id)
    service.create_run({"id": f"{spec_id}-run", "study_id": spec_id, "build": compiled["path"]})
    return service


def test_golden_studies_compile_without_provider_calls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RaisingProvider)
    for name in ("platform_governance", "community_formation"):
        service = _workspace_for(tmp_path, name)
        service.close()


def test_governance_golden_runs_protocol_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", GoldenProvider)
    service = _workspace_for(tmp_path, "platform_governance")
    try:
        result = service.execute_protocol("platform-governance-run")
        assert result["status"] == "completed"
        assert len(result["runs"]) == 4  # 2 conditions x 2 replications
        for trial_id in result["runs"]:
            run = service.get_run(trial_id)
            assert run["status"] == "completed"
            manifest = run["manifest"]
            assert manifest["model_versions"] == {"creator-model": "creator-gpt"}
            assert manifest["condition_id"] in {"transparency-visible", "transparency-hidden"}
        # Events carry generated strategy artifacts with provider metadata.
        trace = service.trace_run(result["runs"][0])
        generative = [
            event
            for event in trace
            if event.get("process_id") == "formulate-strategy"
            and event.get("kind") == "process_completed"
        ]
        assert generative
        assert generative[0]["metadata"]["provider"] == "openai-compatible"
        outcomes = service.evaluate_outcomes(result["runs"][0])
        assert any(row["outcome_id"] == "strategy-diversity" for row in outcomes)
    finally:
        service.close()


def test_community_golden_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", GoldenProvider)
    service = _workspace_for(tmp_path, "community_formation")
    try:
        result = service.execute_run("community-formation-run")
        assert result["status"] == "completed"
        trace = service.trace_run("community-formation-run")
        assert any(
            event.get("process_id") == "interpret-event"
            and event.get("kind") == "process_completed"
            for event in trace
        )
        service.export_run("community-formation-run", "exports/community")
        assert AnalysisExporter.verify_bundle(service.workspace / "exports/community")
    finally:
        service.close()


def test_golden_studies_are_structurally_different(tmp_path: Path, monkeypatch) -> None:
    """SYS-003: both golden studies run without domain-specific core branches."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", GoldenProvider)
    import yaml

    gov = yaml.safe_load((GOLDEN / "platform_governance" / "theory.yaml").read_text())
    com = yaml.safe_load((GOLDEN / "community_formation" / "theory.yaml").read_text())
    assert gov["theory_family"] == "variation-selection-retention"
    assert com["theory_family"] == "interpretation-enactment"
    gov_processes = {
        p["id"]
        for p in yaml.safe_load((GOLDEN / "platform_governance" / "openness.yaml").read_text())[
            "processes"
        ]
    }
    com_processes = {
        p["id"]
        for p in yaml.safe_load((GOLDEN / "community_formation" / "openness.yaml").read_text())[
            "processes"
        ]
    }
    assert not (gov_processes & com_processes)
    # Both execute to completion through the same service, proving no core branches.
    from genesis.compiler import StudyCompiler

    for name in ("platform_governance", "community_formation"):
        StudyCompiler(GOLDEN / name).compile(tmp_path / f"{name}-build")

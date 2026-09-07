from __future__ import annotations

import json

import yaml

from genesis.assistant import StudyAssistant
from genesis.service import GenesisService


def test_inspect_package_returns_structured_validation_and_guidance(tmp_path) -> None:
    (tmp_path / "study.yaml").write_text(
        "schema_version: '1.0'\nstudy_id: demo-study\ntitle: Demo\n"
    )
    assistant = StudyAssistant()

    response = assistant.inspect_package(tmp_path)

    assert response.kind == "package_validation"
    assert response.valid is False
    assert response.issues[0]["code"] == "MISSING_ARTIFACT"
    assert response.suggestions
    assert all(
        item["provenance"]["origin"] == "assistant_proposed" for item in response.suggestions
    )
    assert all("source_refs" in item["provenance"] for item in response.suggestions)


def test_inspect_build_is_read_only_and_proposes_preflight_guidance(tmp_path) -> None:
    manifest = {"study_id": "demo-study", "build_hash": "abc123", "files": []}
    build = tmp_path / "build_manifest.json"
    build.write_text(json.dumps(manifest))
    before = build.read_bytes()

    response = StudyAssistant().inspect_build(build)

    assert response.kind == "build_preflight"
    assert response.valid is True
    assert response.summary["study_id"] == "demo-study"
    assert any(item["code"] == "PREFLIGHT_REVIEW" for item in response.guidance)
    assert build.read_bytes() == before


def test_suggestions_never_claim_empirical_provenance_or_approval(tmp_path) -> None:
    response = StudyAssistant().inspect_package(tmp_path)

    assert response.approval_required is True
    serialized = repr(response)
    assert "empirical" not in serialized.lower()
    assert "confirmed" not in serialized.lower()


def test_compiler_validation_errors_use_the_stable_error_envelope(tmp_path) -> None:
    service = GenesisService(tmp_path / "workspace")
    service.create_specification(
        {"id": "duplicate-study", "title": "Duplicate study", "description": "draft"}
    )
    package = service._specification_dir("duplicate-study")
    openness_path = package / "openness.yaml"
    openness = yaml.safe_load(openness_path.read_text())
    process = {
        "id": "compose-message",
        "executor": {"mode": "deterministic"},
        "context_policy": "public",
    }
    openness["processes"] = [process, process]
    openness_path.write_text(yaml.safe_dump(openness, sort_keys=False))

    response = StudyAssistant().inspect_package(package)

    duplicate = next(item for item in response.issues if item["code"] == "DUPLICATE_ID")
    assert response.valid is False
    assert duplicate == {
        "code": "DUPLICATE_ID",
        "severity": "error",
        "source_file": "openness.yaml",
        "json_pointer": "/openness/processes/compose-message",
        "message": "duplicate process id 'compose-message'",
    }

"""Task 5: deterministic field-level patch previews with YAML diffs."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from genesis.elicitation import (
    SpecificationPatch,
    SpecificationPatchOperation,
    apply_operations,
)
from genesis.providers import ProviderResponse
from genesis.service import GenesisService


class ScriptedDraftProvider:
    provider = "scripted"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        text = _json.dumps(
            {
                "stage_id": "study-foundation",
                "base_specification_version": 1,
                "operations": [
                    {
                        "op": "replace",
                        "path": "/study/title",
                        "value": "Cooperation study (revised)",
                    },
                    {
                        "op": "replace",
                        "path": "/study/description",
                        "value": "Evaluated description.",
                    },
                ],
                "evidence": [
                    {"target": "/study/title", "source_turns": [1]},
                    {"target": "/study/description", "source_turns": [1]},
                ],
                "assumptions": [
                    {
                        "target": "/study/description",
                        "statement": "single assumption",
                    }
                ],
                "unresolved_questions": ["calibration?"],
                "affected_checklist_items": [],
            }
        )
        return ProviderResponse(text, self.provider, "m", "req-1", parsed=_json.loads(text))


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ScriptedDraftProvider)
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "assistant",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    genesis.create_specification(
        {"id": "study", "title": "Cooperation study", "description": "draft"}
    )
    return genesis


class _ReadyProvider(ScriptedDraftProvider):
    def generate(self, request) -> ProviderResponse:
        if "Evaluate the researcher's most recent answer" in request.prompt:
            text = _json.dumps(
                {
                    "status": "ready_to_draft",
                    "summary": "Foundation is settled.",
                    "evidence": [{"claim": "Foundation summary.", "source_turns": [1]}],
                    "decision_coverage": [
                        {
                            "decision_id": decision_id,
                            "status": "covered",
                            "evidence_turns": [1],
                        }
                        for decision_id in (
                            "focal-question",
                            "simulation-boundary",
                            "comparison-objective",
                        )
                    ],
                    "ambiguities": [],
                }
            )
        else:
            return super().generate(request)
        return ProviderResponse(text, self.provider, "m", "req-1", parsed=_json.loads(text))


def _prepare_session(service: GenesisService) -> dict:
    session = service.start_elicitation(
        {
            "specification_id": "study",
            "workflow_id": "three-layer-study",
            "model_profile_id": "assistant",
            "researcher_id": "researcher",
        }
    )
    session = service.submit_elicitation_message(
        session["session_id"], "A brief study foundation answer."
    )
    return service.get_elicitation(session["session_id"])


def test_prompt_removal_survives_preview_and_approval_payload(service: GenesisService) -> None:
    form = service.get_specification("study")["form"]
    form["prompts"] = {"old": "Original prompt"}
    service.update_specification("study", form, 1)
    workflow = service._workflow_registry.get("three-layer-study")
    directory = service._specification_dir("study")
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "openness",
            "base_specification_version": 2,
            "operations": [{"op": "replace", "path": "/prompts", "value": {}}],
            "evidence": [{"target": "/prompts", "source_turns": [1]}],
        }
    )
    preview = service._patch_preview.preview(
        patch=patch,
        stage=workflow.stage("openness"),
        workflow=workflow,
        current_form=form,
        source_turn_ids={1},
        package_hash=service._package_content_hash(directory),
        live_directory=directory,
    )
    assert preview["candidate_form"]["prompts"] == {}
    assert (directory / "prompts/old.txt").exists()
    service.update_specification("study", preview["candidate_form"], 2)
    assert not (directory / "prompts/old.txt").exists()


def test_output_schema_can_be_deferred_then_authored_and_persisted(service: GenesisService) -> None:
    from genesis.compiler import StudyCompiler

    form = service.get_specification("study")["form"]
    form["processes"] = [
        {
            "id": "formulate-strategy",
            "executor": {"mode": "deterministic"},
            "context_policy": "public",
            "outputs": [{"artifact_type": "strategy", "schema_ref": "creator-strategy-schema"}],
        }
    ]
    service.update_specification("study", form, 1)
    directory = service._specification_dir("study")
    workflow = service._workflow_registry.get("three-layer-study")
    early = service._patch_preview._inspect(
        directory, stage=workflow.stage("openness"), workflow=workflow
    )
    assert any(w["code"] == "REF_SCHEMA" for w in early["warnings"])
    final = service._patch_preview._inspect(
        directory, stage=workflow.stage("experiment-design"), workflow=workflow
    )
    assert any(e["code"] == "REF_SCHEMA" for e in final["errors"])
    schema = {
        "type": "object",
        "properties": {"strategy": {"type": "string"}},
        "required": ["strategy"],
    }
    patch = SpecificationPatch.model_validate(
        {
            "stage_id": "experiment-design",
            "base_specification_version": 2,
            "operations": [
                {"op": "add", "path": "/schemas/creator-strategy-schema", "value": schema}
            ],
            "evidence": [{"target": "/schemas/creator-strategy-schema", "source_turns": [1]}],
        }
    )
    preview = service._patch_preview.preview(
        patch=patch,
        stage=workflow.stage("experiment-design"),
        workflow=workflow,
        current_form=form,
        source_turn_ids={1},
        package_hash=service._package_content_hash(directory),
        live_directory=directory,
    )
    assert not any(e["code"] == "REF_SCHEMA" for e in preview["validation"]["errors"])
    assert not (directory / "schemas/creator-strategy-schema.json").exists()
    service.update_specification("study", preview["candidate_form"], 2)
    assert (directory / "schemas/creator-strategy-schema.json").exists()
    compiler = StudyCompiler(directory)
    assert not any(e["code"] == "REF_SCHEMA" for e in compiler._validate(compiler._load()))
    revised = service.get_specification("study")["form"]
    revised["schemas"] = {}
    service.update_specification("study", revised, 3)
    assert not (directory / "schemas/creator-strategy-schema.json").exists()
    assert any(e["code"] == "REF_SCHEMA" for e in compiler._validate(compiler._load()))


def test_add_replace_remove_operations(service: GenesisService) -> None:
    current = {"study": {"title": "T", "owners": ["a"]}, "openness": {"processes": []}}
    result = apply_operations(
        current,
        (
            SpecificationPatchOperation(op="replace", path="/study/title", value="T2"),
            SpecificationPatchOperation(op="add", path="/study/owners/-", value="b"),
            SpecificationPatchOperation(op="add", path="/openness/processes/-", value={"id": "p"}),
            SpecificationPatchOperation(op="remove", path="/study/owners/-"),
        ),
    )
    assert result["study"]["title"] == "T2"
    assert result["study"]["owners"] == ["a"]  # removed the appended 'b'
    assert result["openness"]["processes"] == [{"id": "p"}]


def test_path_restrictions_are_enforced(service: GenesisService) -> None:
    patch = SpecificationPatch(
        stage_id="study-foundation",
        base_specification_version=1,
        operations=(SpecificationPatchOperation(op="replace", path="/theory/x", value=1),),
    )
    with pytest.raises(ValueError, match="PATCH_PATH_FORBIDDEN"):
        patch.validate_owned_paths(("/study",))


def test_unknown_targets_and_invalid_ids_are_rejected(service: GenesisService) -> None:
    with pytest.raises(ValueError, match="PATCH_PATH_FORBIDDEN"):
        apply_operations(
            {"study": {"title": "T"}},
            (SpecificationPatchOperation(op="replace", path="/study/missing/deep", value=1),),
        )
    with pytest.raises(ValueError, match="PATCH_PATH_FORBIDDEN"):
        apply_operations(
            {"study": {"title": "T"}},
            (SpecificationPatchOperation(op="remove", path="/study/missing", value=None),),
        )


def test_stale_base_versions_are_rejected(service: GenesisService) -> None:
    patch = SpecificationPatch(
        stage_id="study-foundation",
        base_specification_version=7,
        operations=(SpecificationPatchOperation(op="replace", path="/study/title", value="X"),),
    )
    with pytest.raises(ValueError, match="PATCH_BASE_STALE"):
        patch.validate_base_version(3)


def test_evidence_references_are_validated(service: GenesisService) -> None:
    patch = SpecificationPatch(
        stage_id="study-foundation",
        base_specification_version=1,
        operations=(SpecificationPatchOperation(op="replace", path="/study/title", value="X"),),
        evidence=({"target": "/study/title", "source_turns": (99,)},),
    )
    with pytest.raises(ValueError, match="ASSISTANT_EVIDENCE_INVALID"):
        patch.validate_turn_references({1})


def test_preview_shows_yaml_diff_and_validation(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _ReadyProvider)
    session = _prepare_session(service)
    assert session["status"] == "awaiting_approval"
    service.draft_elicitation(session["session_id"])
    session = service.preview_elicitation_stage(session["session_id"])
    preview = session["pending_preview"]
    assert preview is not None
    assert "study.yaml" in preview["yaml_files"]
    assert "study.yaml" in preview["diffs"]
    diff_text = preview["diffs"]["study.yaml"]
    assert "Cooperation study (revised)" in "\n".join(preview["yaml_files"].values())
    assert "+" in diff_text and "-" in diff_text
    assert isinstance(preview["validation"], dict)
    assert "errors" in preview["validation"]


def test_preview_makes_no_writes(service: GenesisService, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _ReadyProvider)
    session = _prepare_session(service)
    service.draft_elicitation(session["session_id"])
    service.preview_elicitation_stage(session["session_id"])
    # The specification is untouched and no durable build was produced.
    assert service.get_specification("study")["version"] == 1
    directory = service._specification_dir("study")
    assert not (directory / "build_manifest.json").exists()
    assert (directory / "study.yaml").read_text().startswith("schema_version")


def test_preview_sanitises_invented_study_fields(tmp_path) -> None:
    """Assistants inventing study fields or study_id no longer break canonical YAML."""
    from genesis.elicitation import PatchPreviewService, SpecificationPatch

    class _Stage:
        id = "study-foundation"
        owned_paths = ("/study",)
        checklist_items = ()

    service = GenesisService(tmp_path / "workspace")
    try:
        service.create_specification({"id": "clean-study", "title": "T", "description": "d"})
        previewer = PatchPreviewService(service)
        current = service._current_form_payload(
            service._specification_dir("clean-study"), "clean-study"
        )
        patch = SpecificationPatch(
            stage_id="study-foundation",
            base_specification_version=1,
            operations=(
                {
                    "op": "replace",
                    "path": "/study",
                    "value": {
                        "study_id": "social_media_governance_clickbait",
                        "focal_phenomenon": "The emergence of clickbait",
                        "population": "Creators",
                        "title": "T",
                    },
                },
            ),
            evidence=({"target": "/study", "source_turns": (1,)},),
            assumptions=(),
        )
        result = previewer.preview(
            patch=patch,
            stage=_Stage(),
            workflow=service._workflow_registry.get("three-layer-study"),
            current_form=current,
            source_turn_ids={1},
            package_hash="",
            live_directory=service._specification_dir("clean-study"),
        )
        study_yaml = result["yaml_files"]["study.yaml"]
        assert "focal_phenomenon" not in study_yaml
        assert "social_media_governance_clickbait" not in study_yaml
        assert "study_id: clean-study" in study_yaml
    finally:
        service.close()

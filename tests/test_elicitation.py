"""AW-02: elicitation checklists, LLM-mediated drafting, and approval gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genesis.providers import ProviderResponse
from genesis.service import GenesisService

STUDY = {
    "id": "elicit-study",
    "title": "elicitation study",
    "processes": [
        {
            "id": "tick",
            "executor": {},
            "context_policy": "public",
            "state_effects": [{"field": "counter", "op": "set"}],
        }
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]},
    "protocol": {"time_model": {"type": "rounds", "end": 3}},
    "outcomes": [],
    "models": [],
}


class FakeAssistantProvider:
    provider = "openai-compatible"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        proposal = {
            "id": "elicit-study",
            "title": "model drafted title",
            "processes": STUDY["processes"],
            "theory": {
                "theory_family": "variation-selection-retention",
                "process_mappings": [{"process": "tick", "theory_function": "variation"}],
                "observables": [{"id": "strategy-diversity", "definition": "distinct"}],
            },
            "domain": STUDY["domain"],
            "protocol": STUDY["protocol"],
            "outcomes": [],
            "models": [],
        }
        return ProviderResponse(json.dumps(proposal), self.provider, request.model, "req-elicit")


def _service(tmp_path: Path) -> GenesisService:
    return GenesisService(tmp_path / "workspace")


def test_checklist_is_created_and_auto_evaluated(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        service.create_specification(STUDY)
        checklist = service.get_checklist("elicit-study")
        items = {item["id"]: item["status"] for item in checklist["items"]}
        # Deterministic study: Layer 1 openness items are not applicable.
        assert items["l1-openness-need"] == "not_applicable"
        assert items["l1-selective-closure"] == "not_applicable"
        # Context policy declared -> complete.
        assert items["l1-informational-position"] == "complete"
        assert items["l2-theory-logic"] == "not_applicable"  # exploratory family
    finally:
        service.close()


def test_manual_unresolved_required_item_blocks_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        draft = service.create_specification(STUDY)
        service.update_checklist_item("elicit-study", "l1-traceability", status="unresolved")
        with pytest.raises(ValueError, match="SPECIFICATION_INCOMPLETE.*l1-traceability"):
            service.approve_specification("elicit-study", draft["version"], "researcher")
        service.update_checklist_item(
            "elicit-study", "l1-traceability", status="complete", confirmed_by="researcher"
        )
        approved = service.approve_specification("elicit-study", draft["version"], "researcher")
        assert approved["status"] == "approved"
    finally:
        service.close()


def test_update_reopens_items_until_refreshed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        service.create_specification(STUDY)
        service.update_checklist_item("elicit-study", "l1-traceability", status="unresolved")
        # Editing the specification keeps manual statuses...
        draft = service.get_specification("elicit-study")
        service.update_specification("elicit-study", {"title": "renamed"}, draft["version"])
        items = {i["id"]: i["status"] for i in service.get_checklist("elicit-study")["items"]}
        assert items["l1-traceability"] == "unresolved"
        # ...until the researcher refreshes, re-evaluating rules deterministically.
        refreshed = {i["id"]: i["status"] for i in service.refresh_checklist("elicit-study")}
        assert refreshed["l1-traceability"] == "not_applicable"
    finally:
        service.close()


def test_draft_from_model_proposes_without_writing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeAssistantProvider)
    service = _service(tmp_path)
    try:
        service.create_model_profile(
            {
                "id": "assistant",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "assistant-v1",
                "api_key_env": "GENESIS_ASSISTANT_KEY",
            }
        )
        service.create_specification(STUDY)
        result = service.draft_from_model("elicit-study", instruction="Draft a study.")
        assert result["proposal"]["title"] == "model drafted title"
        assert "theory" in result["preview"]
        # Nothing was written: the draft is unchanged.
        metadata = service.get_specification("elicit-study")
        assert metadata["form"]["title"] == "elicitation study"
    finally:
        service.close()


def test_accept_draft_applies_proposal_and_refreshes_checklist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeAssistantProvider)
    service = _service(tmp_path)
    try:
        service.create_specification(STUDY)
        service.create_model_profile(
            {
                "id": "assistant",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "assistant-v1",
                "api_key_env": "GENESIS_ASSISTANT_KEY",
            }
        )
        proposal = service.draft_from_model("elicit-study")["proposal"]
        updated = service.accept_draft("elicit-study", proposal, confirmed_by="researcher")
        assert updated["form"]["title"] == "model drafted title"
        items = {i["id"]: i["status"] for i in service.get_checklist("elicit-study")["items"]}
        assert items["l2-theory-logic"] == "complete"  # mappings now present
    finally:
        service.close()


def test_draft_from_model_rejects_invalid_proposal(tmp_path: Path, monkeypatch) -> None:
    class BadProvider(FakeAssistantProvider):
        def generate(self, request) -> ProviderResponse:
            return ProviderResponse('{"id": "no-title"}', self.provider, request.model, "req-bad")

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", BadProvider)
    service = _service(tmp_path)
    try:
        service.create_model_profile(
            {
                "id": "assistant",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "assistant-v1",
                "api_key_env": "GENESIS_ASSISTANT_KEY",
            }
        )
        with pytest.raises(ValueError, match="PROPOSAL_INVALID"):
            service.draft_from_model(None, instruction="draft from scratch")
    finally:
        service.close()


def test_accept_draft_rejects_unknown_fields(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        with pytest.raises(ValueError, match="INVALID_FIELD"):
            service.accept_draft("brand-new", {"id": "brand-new", "title": "x", "bogus": 1})
    finally:
        service.close()


def test_specification_patch_is_typed_and_deterministically_validated() -> None:
    """Finding 8: patches are model-validated; bad paths and empty ops fail."""
    import pytest

    from genesis.service import GenesisService, SpecificationPatch

    with pytest.raises(ValueError, match="PATCH_PATH"):
        SpecificationPatch(operations=[{"path": "unsupported-section", "op": "set", "value": {}}])
    patch = SpecificationPatch(
        operations=[{"path": "specification", "op": "set", "value": {"id": "x"}}],
        evidence=[{"type": "review", "source": "verifier-1"}],
        assumptions=["parameters are point values"],
        unresolved_questions=["calibration?"],
        affected_checklist_items=["l1-traceability"],
    )
    patch.validate_deterministic()
    payload = patch.model_dump(mode="json")
    assert payload["operations"][0]["path"] == "specification"
    assert payload["evidence"][0]["source"] == "verifier-1"

    service = GenesisService(_workspace_tmp())
    try:
        with pytest.raises(ValueError, match="PATCH_PATH"):
            service.accept_draft(
                "no-study",
                {"id": "no-study", "title": "x"},
                confirmed_by="researcher",
                patch={"operations": [{"path": "bogus", "op": "set", "value": {}}]},
            )
    finally:
        service.close()


def _workspace_tmp() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp())


def test_field_level_patch_application_and_history(tmp_path) -> None:
    """AW-02: patch ops apply field-by-field, evidence and history are kept."""
    from genesis.service import GenesisService, _apply_field_operations

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        service.create_specification(
            {
                "id": "field-patch-study",
                "title": "original title",
                "processes": [
                    {
                        "id": "tick",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {},
                "protocol": {"time_model": {"type": "rounds", "end": 2}},
                "outcomes": [],
                "models": [],
            }
        )
        payload = service._current_form_payload(
            service._specification_dir("field-patch-study"), "field-patch-study"
        )
        applied = _apply_field_operations(
            payload,
            [
                {"path": "specification.title", "op": "set", "value": "revised title"},
                {
                    "path": "specification.processes[0].executor.mode",
                    "op": "set",
                    "value": "stochastic",
                },
            ],
        )
        assert applied["title"] == "revised title"
        assert applied["processes"][0]["executor"]["mode"] == "stochastic"
        applied["id"] = "field-patch-study"
        accepted = service.accept_draft(
            "field-patch-study",
            {},
            confirmed_by="researcher",
            patch={
                "operations": [
                    {"path": "specification.title", "op": "set", "value": "final title"}
                ],
                "evidence": [{"type": "review-note", "source": "verifier-2"}],
                "assumptions": ["single section change"],
                "unresolved_questions": [],
                "affected_checklist_items": ["l1-traceability"],
            },
        )
        assert accepted["title"] == "final title"
        assert accepted["assistant_confirmed_by"] == "researcher"
        history = accepted["patch_history"]
        assert len(history) == 1
        assert history[0]["operations"][0]["path"] == "specification.title"
        assert history[0]["evidence"][0]["source"] == "verifier-2"
        # Unrelated fields were untouched by the field-level patch.
        metadata = service.get_specification("field-patch-study")
        assert metadata["id"] == "field-patch-study"
    finally:
        service.close()


def test_patch_application_rejects_unknown_targets() -> None:
    """AW-02: deterministic application rejects unknown paths."""
    import pytest

    from genesis.service import _apply_field_operations

    with pytest.raises(ValueError, match="PATCH_PATH"):
        _apply_field_operations(
            {"a": {"b": 1}},
            [{"path": "specification.a.missing.deep", "op": "set", "value": 2}],
        )
    # A missing leaf is created deterministically (upsert semantics).
    assert _apply_field_operations(
        {"a": {"b": 1}},
        [{"path": "specification.a.missing", "op": "set", "value": 2}],
    ) == {"a": {"b": 1, "missing": 2}}

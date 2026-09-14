"""Gap A: the drafted stage is checked against what the researcher actually said.

Cross-layer validation and compilation both ask decidable questions. Whether a
declaration follows from the researcher's own words is not decidable, so it is
asked by a reader at draft time and answered as advice: contradictions must be
acknowledged before approval, never silently passed and never hard-blocked.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any

import pytest

from genesis.intent_check import (
    assemble_intent_request,
    blocking_findings,
    finding_id,
    parse_intent_findings,
)
from genesis.providers import ProviderResponse
from genesis.service import GenesisService
from tests.test_elicitation_approval import StageScriptedProvider, _answer

CONTRADICTION = {
    "verdict": "contradicted",
    "declaration": "/openness/processes/2/context_policy",
    "researcher_said": "the detector sees the title and body and nothing else",
    "draft_says": "context_policy: platform-context",
    "consequence": "the detector can read the outcomes it is measuring",
}
UNSUPPORTED = {
    "verdict": "unsupported",
    "declaration": "/openness/processes/0/retry_policy",
    "researcher_said": None,
    "draft_says": "repair_once_then_fail",
    "consequence": "a failed call is recorded rather than retried further",
}


# --- the finding vocabulary -------------------------------------------------------


def test_findings_are_parsed_and_contradictions_are_listed_first() -> None:
    text = _json.dumps({"findings": [UNSUPPORTED, CONTRADICTION]})
    findings = parse_intent_findings(text)
    assert [item["verdict"] for item in findings] == ["contradicted", "unsupported"]
    assert findings[0]["researcher_said"] == CONTRADICTION["researcher_said"]


@pytest.mark.parametrize(
    "entry",
    [
        {"verdict": "looks-fine", "declaration": "/a"},  # not a reported verdict
        {"verdict": "contradicted"},  # no declaration to act on
        "not an object",
    ],
)
def test_malformed_findings_are_dropped(entry: Any) -> None:
    findings = parse_intent_findings(_json.dumps({"findings": [entry, CONTRADICTION]}))
    assert len(findings) == 1
    assert findings[0]["declaration"] == CONTRADICTION["declaration"]


@pytest.mark.parametrize("text", ["not json at all", _json.dumps({"summary": "fine"})])
def test_a_response_without_findings_is_refused_so_the_caller_can_record_it(text: str) -> None:
    with pytest.raises(ValueError):
        parse_intent_findings(text)


def test_a_finding_id_is_stable_across_previews() -> None:
    """An acknowledgement has to survive regenerating the preview."""
    first = parse_intent_findings(_json.dumps({"findings": [CONTRADICTION]}))[0]
    second = parse_intent_findings(_json.dumps({"findings": [CONTRADICTION]}))[0]
    assert first["id"] == second["id"] == finding_id(CONTRADICTION)
    moved = {**CONTRADICTION, "draft_says": "context_policy: detector-context"}
    assert finding_id(moved) != first["id"]


def test_only_unacknowledged_contradictions_block() -> None:
    findings = parse_intent_findings(_json.dumps({"findings": [CONTRADICTION, UNSUPPORTED]}))
    check = {"findings": findings, "acknowledged": []}
    assert [item["declaration"] for item in blocking_findings(check)] == [
        CONTRADICTION["declaration"]
    ]
    check["acknowledged"] = [findings[0]["id"]]
    assert blocking_findings(check) == []
    assert blocking_findings(None) == []


def test_the_request_carries_every_turn_not_just_this_stage() -> None:
    """The sentence stating the intent is often in a different layer's turn."""
    request = assemble_intent_request(
        [("openness", "the detector sees title and body only"), ("domain", "sixteen creators")],
        "domain",
        {"domain.yaml": "actors: []"},
        {"openness.yaml": "processes: []"},
    )
    assert "the detector sees title and body only" in request
    assert "sixteen creators" in request
    assert "Declarations under review (stage: domain)" in request
    assert "actors: []" in request
    assert "processes: []" in request


# --- the gate ---------------------------------------------------------------------


class IntentProvider(StageScriptedProvider):
    """Scripted drafting, plus a scripted intent check."""

    findings: list[dict[str, Any]] = [CONTRADICTION]

    def generate(self, request) -> ProviderResponse:
        if "What the researcher said, in their own words" not in request.prompt:
            return super().generate(request)
        text = _json.dumps({"findings": type(self).findings})
        return ProviderResponse(text, "scripted", "m", "req-intent")


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setenv("GENESIS_INTENT_CHECK", "1")
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", IntentProvider)
    IntentProvider.findings = [CONTRADICTION]
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
    genesis.create_specification({"id": "intent-study", "title": "Start", "description": "draft"})
    return genesis


def _drafted(service: GenesisService) -> str:
    started = service.start_elicitation(
        {"specification_id": "intent-study", "model_profile_id": "assistant"}
    )
    session_id = str(started["session_id"])
    _answer(service, session_id)
    service.draft_elicitation(session_id)
    service.preview_elicitation_stage(session_id)
    return session_id


def _check(service: GenesisService, session_id: str) -> dict[str, Any]:
    preview = service.get_elicitation(session_id)["pending_preview"]
    return dict(preview["intent_check"])


def test_the_preview_carries_the_check_and_pins_what_produced_it(service: GenesisService) -> None:
    check = _check(service, _drafted(service))
    assert check["status"] == "ok"
    assert check["model_profile"] == "assistant"
    assert check["request_digest"]
    assert [item["verdict"] for item in check["findings"]] == ["contradicted"]


def test_approval_is_refused_while_a_contradiction_is_unacknowledged(
    service: GenesisService,
) -> None:
    session_id = _drafted(service)
    with pytest.raises(ValueError) as raised:
        service.approve_elicitation_stage(session_id, approved_by="researcher")
    message = str(raised.value)
    assert message.startswith("INTENT_UNACKNOWLEDGED")
    # The refusal has to say what was said, what was declared, and how to proceed.
    assert CONTRADICTION["researcher_said"] in message
    assert CONTRADICTION["draft_says"] in message
    assert _check(service, session_id)["findings"][0]["id"] in message


def test_acknowledging_the_finding_lets_approval_proceed(service: GenesisService) -> None:
    session_id = _drafted(service)
    finding = _check(service, session_id)["findings"][0]
    result = service.approve_elicitation_stage(
        session_id, approved_by="researcher", acknowledged_findings=[finding["id"]]
    )
    assert result["stages"]["study-foundation"]["status"] == "approved"


def test_acknowledging_an_id_this_preview_does_not_carry_is_refused(
    service: GenesisService,
) -> None:
    """Otherwise the researcher believes they cleared something they did not."""
    session_id = _drafted(service)
    with pytest.raises(ValueError, match="INTENT_ACK_UNKNOWN"):
        service.approve_elicitation_stage(
            session_id, approved_by="researcher", acknowledged_findings=["deadbeef1234"]
        )


def test_an_unsupported_finding_does_not_block(service: GenesisService) -> None:
    IntentProvider.findings = [UNSUPPORTED]
    session_id = _drafted(service)
    assert [item["verdict"] for item in _check(service, session_id)["findings"]] == ["unsupported"]
    service.approve_elicitation_stage(session_id, approved_by="researcher")


def test_a_check_that_cannot_run_records_itself_and_never_blocks(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Advisory means advisory: a broken reader must not stop the researcher."""

    def _explode(self, request):  # noqa: ANN001, ANN202
        if "What the researcher said, in their own words" in request.prompt:
            raise RuntimeError("reader offline")
        return StageScriptedProvider.generate(self, request)

    monkeypatch.setattr(IntentProvider, "generate", _explode)
    session_id = _drafted(service)
    check = _check(service, session_id)
    assert check["status"] == "unavailable"
    assert "reader offline" in check["reason"]
    service.approve_elicitation_stage(session_id, approved_by="researcher")


def test_the_check_can_be_turned_off(
    service: GenesisService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GENESIS_INTENT_CHECK", "0")
    session_id = _drafted(service)
    check = _check(service, session_id)
    assert check["status"] == "disabled"
    assert check["findings"] == []
    service.approve_elicitation_stage(session_id, approved_by="researcher")


def test_an_identical_draft_is_not_read_twice(service: GenesisService) -> None:
    """Approval re-previews; re-reading would renumber findings under the researcher."""
    calls: list[str] = []
    original = IntentProvider.generate

    def _counting(self, request):  # noqa: ANN001, ANN202
        if "What the researcher said, in their own words" in request.prompt:
            calls.append(request.prompt)
        return original(self, request)

    IntentProvider.generate = _counting  # type: ignore[method-assign]
    try:
        session_id = _drafted(service)
        first = _check(service, session_id)["findings"][0]["id"]
        service.preview_elicitation_stage(session_id)
        second = _check(service, session_id)["findings"][0]["id"]
    finally:
        IntentProvider.generate = original  # type: ignore[method-assign]
    assert len(calls) == 1, "the same question was asked twice"
    assert first == second


def test_an_acknowledgement_survives_the_repreview_approval_performs(
    service: GenesisService,
) -> None:
    session_id = _drafted(service)
    finding_ref = _check(service, session_id)["findings"][0]["id"]
    result = service.approve_elicitation_stage(
        session_id, approved_by="researcher", acknowledged_findings=[finding_ref]
    )
    assert result["stages"]["study-foundation"]["status"] == "approved"


# --- gap C: compile errors from settled layers are shown early ---------------------


def _three_layer_workflow():
    from genesis.elicitation import WorkflowRegistry
    from genesis.service import _workflows_root

    return WorkflowRegistry(_workflows_root()).get("three-layer-study")


def test_a_compile_error_in_an_elicited_layer_is_surfaced_at_that_stage() -> None:
    """Otherwise it waits for the final gate, three approvals from the fix."""
    from genesis.elicitation import _settled_compile_errors

    workflow = _three_layer_workflow()
    domain_stage = workflow.stage("domain")
    errors = [
        {"code": "A", "source_file": "openness.yaml", "message": "already elicited"},
        {"code": "B", "source_file": "domain.yaml", "message": "this stage"},
        {"code": "C", "source_file": "protocol.yaml", "message": "not elicited yet"},
        {"code": "D", "source_file": "package", "message": "unattributable"},
    ]
    settled = _settled_compile_errors(errors, workflow, domain_stage)
    assert [item["code"] for item in settled] == ["A", "B"]
    assert settled[0]["owned_by"] == "openness"
    assert settled[1]["owned_by"] == "domain"


def test_at_the_first_stage_only_its_own_layer_is_settled() -> None:
    from genesis.elicitation import _settled_compile_errors

    workflow = _three_layer_workflow()
    first = workflow.stages[0]
    errors = [
        {"code": "A", "source_file": "study.yaml", "message": "this stage"},
        {"code": "B", "source_file": "domain.yaml", "message": "not elicited yet"},
    ]
    assert [item["code"] for item in _settled_compile_errors(errors, workflow, first)] == ["A"]


def test_the_preview_carries_the_early_compile_warnings(service: GenesisService) -> None:
    session_id = _drafted(service)
    preview = service.get_elicitation(session_id)["pending_preview"]
    assert "compile_warnings" in preview["validation"]


def test_a_finding_re_grounded_in_another_quote_is_a_different_finding() -> None:
    """An acknowledgement is of a specific reading. Keying the id on the
    declaration alone let a finding that now rests on a different sentence
    inherit the acknowledgement given to the old one."""
    from genesis.intent_check import finding_id

    base = {
        "verdict": "contradicted",
        "declaration": "/protocol/conditions",
        "draft_says": "the leaderboard is public",
    }
    first = finding_id({**base, "researcher_said": "creators never see rank"})
    second = finding_id({**base, "researcher_said": "only the top three see rank"})
    assert first != second
    assert first == finding_id({**base, "researcher_said": "creators never see rank"})

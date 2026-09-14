"""Gap D: a compiled build read back as prose, blind to the study's intent.

Compilation says a package is well formed and the intent check says a declaration
follows from what the researcher said. This says what the thing about to be run
actually does, to a reader who was never told what it was for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from genesis.providers import ProviderResponse
from genesis.readback import BUILD_PARTS, assemble_readback_request, readback_record
from genesis.service import GenesisService
from tests.test_elicitation_approval import StageScriptedProvider

READING = "Under high visibility creators see the three least-clicked articles."


class ReadbackProvider(StageScriptedProvider):
    seen: list[str] = []

    def generate(self, request) -> ProviderResponse:
        if "Compiled package declarations" not in request.prompt:
            return super().generate(request)
        type(self).seen.append(request.prompt)
        return ProviderResponse(READING, "scripted", "m", "req-readback")


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ReadbackProvider)
    ReadbackProvider.seen = []
    genesis = GenesisService(tmp_path / "workspace")
    genesis.create_model_profile(
        {
            "id": "reader",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    return genesis


def _build(service: GenesisService, **parts: Any) -> str:
    build = service.workspace / "builds" / "b1"
    build.mkdir(parents=True)
    (build / "build_manifest.json").write_text(json.dumps({"build_hash": "abc123"}))
    for name, value in parts.items():
        (build / name.replace("__", ".")).write_text(json.dumps(value))
    return "builds/b1"


def test_the_reading_is_recorded_beside_the_build_and_pinned_to_it(
    service: GenesisService,
) -> None:
    ref = _build(service, processes__json=[{"id": "publish"}])
    record = service.readback_build(ref, model_profile_id="reader")
    assert record["text"] == READING
    assert record["build_hash"] == "abc123"
    assert record["model_profile"] == "reader"
    assert record["request_digest"]
    stored = json.loads((service.workspace / "builds" / "b1" / "readback.json").read_text())
    assert stored == record


def test_the_reader_is_shown_the_declarations_and_nothing_about_the_intent(
    service: GenesisService,
) -> None:
    """Shown the intent it would restate the intent; that is the whole point."""
    ref = _build(
        service,
        processes__json=[{"id": "publish", "openness_rationale": "SHOULD NOT MATTER"}],
        context_policies__json=[{"id": "ctx", "allow": ["items"]}],
    )
    service.readback_build(ref, model_profile_id="reader")
    prompt = ReadbackProvider.seen[0]
    assert "context_policies.json" in prompt
    assert "What the researcher said" not in prompt
    assert "research design" not in prompt.lower()
    # The rationales are the design stated in prose, and processes.json carries
    # them. This test named one SHOULD NOT MATTER and never checked it was gone,
    # so the reader had been shown the intent it is meant to be blind to.
    assert "SHOULD NOT MATTER" not in prompt
    assert "openness_rationale" not in prompt


def test_every_stated_rationale_is_stripped_wherever_it_sits(service: GenesisService) -> None:
    ref = _build(
        service,
        processes__json=[
            {
                "id": "publish",
                "openness_rationale": "WHY-OPEN",
                "closure_rationale": "WHY-CLOSED",
                "origin": "WHY-THERE",
                "outputs": [{"artifact_type": "article", "closure_rationale": "WHY-NESTED"}],
            }
        ],
    )
    service.readback_build(ref, model_profile_id="reader")
    prompt = ReadbackProvider.seen[0]
    for stated in ("WHY-OPEN", "WHY-CLOSED", "WHY-THERE", "WHY-NESTED"):
        assert stated not in prompt, stated
    # What the package does is still there.
    assert "publish" in prompt and "article" in prompt


def test_the_reader_is_not_asked_what_it_cannot_see(service: GenesisService) -> None:
    """It was asked what is measured while the outcome plan was withheld, so the
    answer was a guess -- stored, pinned, as evidence."""
    request = assemble_readback_request({"processes.json": "[]"})
    assert "Which processes are measurements" in request
    assert "the analysis plan is not" in request


def test_only_the_declaration_files_are_sent(service: GenesisService) -> None:
    ref = _build(service, processes__json=[{"id": "p"}], outcome_plan__json={"outcomes": []})
    service.readback_build(ref, model_profile_id="reader")
    prompt = ReadbackProvider.seen[0]
    assert "processes.json" in prompt
    # Not in BUILD_PARTS: the reading is about what actors receive.
    assert "outcome_plan.json" not in prompt
    assert "outcome_plan.json" not in BUILD_PARTS


def test_a_build_without_a_manifest_is_refused(service: GenesisService) -> None:
    (service.workspace / "builds" / "empty").mkdir(parents=True)
    with pytest.raises(ValueError, match="BUILD_NOT_FOUND"):
        service.readback_build("builds/empty", model_profile_id="reader")


def test_the_runtime_note_keeps_the_reader_from_reporting_phase_as_a_defect() -> None:
    """Without it the reader flags every 'by: phase' cap, which is correct grammar."""
    request = assemble_readback_request({"state_model.json": "{}"})
    assert "annotated with the phase" in request
    assert "under `inputs`" in request


def test_a_reading_records_what_produced_it() -> None:
    record = readback_record(
        "text", build_ref="builds/b", build_hash="h", model_profile="m", request_digest="d"
    )
    assert record["kind"] == "build_readback"
    assert (record["build"], record["build_hash"]) == ("builds/b", "h")


# --- reachable from the page a researcher actually uses ----------------------------


def test_the_reading_can_be_asked_for_over_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service method existed with no route, so the page could not ask for it."""
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ReadbackProvider)
    ReadbackProvider.seen = []
    workspace = tmp_path / "workspace"
    genesis = GenesisService(workspace)
    genesis.create_model_profile(
        {
            "id": "reader",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    ref = _build(genesis, processes__json=[{"id": "publish"}])
    genesis.close()

    client = TestClient(create_app(workspace))
    response = client.post(f"/builds/{ref}/readback", json={"model_profile_id": "reader"})
    assert response.status_code == 200
    assert response.json()["text"] == READING
    assert response.json()["build_hash"] == "abc123"


def test_the_readback_route_refuses_what_it_does_not_support(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from genesis.app import create_app

    client = TestClient(create_app(tmp_path / "workspace"))
    unknown = client.post("/builds/builds/b1/readback", json={"model": "gpt"})
    assert unknown.status_code >= 400
    assert "model" in unknown.json()["error"]["message"]
    missing = client.post("/builds/builds/b1/readback", json={})
    assert missing.status_code >= 400
    assert "model_profile_id" in missing.json()["error"]["message"]


def test_an_accepted_reading_travels_with_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is pinned to a build hash and a model profile so it can be cited, and
    it reached no bundle -- so an export carried no sign anyone had read the
    package back."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ReadbackProvider)
    ReadbackProvider.seen = []
    genesis = GenesisService(tmp_path / "workspace")
    try:
        genesis.create_model_profile(
            {
                "id": "reader",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m",
                "api_key_env": "GENESIS_FAKE_KEY",
            }
        )
        genesis.create_specification(
            {
                "id": "read-study",
                "title": "readable",
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
                "protocol": {
                    "time_model": {"type": "rounds", "end": 1},
                    "conditions": [{"id": "base"}],
                },
                "outcomes": [],
                "models": [],
            }
        )
        version = genesis.get_specification("read-study")["version"]
        genesis.approve_specification("read-study", version, "researcher")
        compiled = genesis.compile_study(None, "builds/read", specification_id="read-study")
        genesis.readback_build(compiled["path"], model_profile_id="reader")
        genesis.create_run({"id": "rr", "study_id": "read-study", "build": compiled["path"]})
        genesis.execute_run("rr")
        genesis.export_run("rr", "exports/read")
        bundle = genesis.workspace / "exports" / "read"
        assert READING in (bundle / "readback.json").read_text()
    finally:
        genesis.close()


def test_branching_says_why_it_is_unavailable_rather_than_implying_bad_luck() -> None:
    """It is unavailable for every export, because no bundle carries a
    checkpoint payload -- not because this particular one happened to lack it."""
    from genesis.evidence import evaluate_capabilities

    capabilities = evaluate_capabilities(
        has_build=True,
        has_closure=True,
        has_recorded_outputs=True,
        has_checkpoint_evidence=False,
        has_outcomes=True,
        has_executor_code=True,
    )
    branch = next(item for item in capabilities if item["capability"] == "branch_at_checkpoint")
    assert branch["available"] is False
    assert "not offered by any export" in branch["reason"]


def test_a_reading_survives_the_export_and_import_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundle carried it and the import dropped it, which loses it just as
    completely as never exporting it."""
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", ReadbackProvider)
    ReadbackProvider.seen = []
    genesis = GenesisService(tmp_path / "workspace")
    try:
        genesis.create_model_profile(
            {
                "id": "reader",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m",
                "api_key_env": "GENESIS_FAKE_KEY",
            }
        )
        genesis.create_specification(
            {
                "id": "trip-study",
                "title": "round trip",
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
                "protocol": {
                    "time_model": {"type": "rounds", "end": 1},
                    "conditions": [{"id": "base"}],
                },
                "outcomes": [],
                "models": [],
            }
        )
        version = genesis.get_specification("trip-study")["version"]
        genesis.approve_specification("trip-study", version, "researcher")
        compiled = genesis.compile_study(None, "builds/trip", specification_id="trip-study")
        genesis.readback_build(compiled["path"], model_profile_id="reader")
        genesis.create_run({"id": "tr", "study_id": "trip-study", "build": compiled["path"]})
        genesis.execute_run("tr")
        genesis.export_run("tr", "exports/trip", mode="reproducibility")
    finally:
        genesis.close()

    # Imported into a fresh workspace, which is what carrying evidence means.
    elsewhere = GenesisService(tmp_path / "elsewhere")
    try:
        bundle = tmp_path / "workspace" / "exports" / "trip"
        shutil.copytree(bundle, elsewhere.workspace / "imports" / "trip")
        imported = elsewhere.import_run("imports/trip")
        build = elsewhere.get_run(imported["run_id"])["build"]
        assert READING in (elsewhere.resolve_path(build) / "readback.json").read_text()
    finally:
        elsewhere.close()

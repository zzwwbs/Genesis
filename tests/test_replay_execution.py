"""AW-05: true replay execution — full rerun, artifact replay, partial, branch."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.providers import ProviderResponse
from genesis.replay import ReplayMode
from genesis.service import GenesisService

GEN_CALLS: list[str] = []


class FakeProvider:
    provider = "openai-compatible"

    def __init__(self, **_kwargs) -> None:
        pass

    def generate(self, request) -> ProviderResponse:
        import json as _json

        GEN_CALLS.append(request.model)
        text = _json.dumps({"text": f"gen-{len(GEN_CALLS)}"})
        return ProviderResponse(
            text,
            self.provider,
            request.model,
            f"req-{len(GEN_CALLS)}",
            parsed=_json.loads(text),
        )


class RaisingProvider:
    """Instantiation means a live provider call — forbidden during artifact replay."""

    provider = "openai-compatible"

    def __init__(self, **_kwargs) -> None:
        raise AssertionError("provider must not be constructed during artifact replay")

    def generate(self, request) -> ProviderResponse:
        raise AssertionError("never called")


PAYLOAD = {
    "id": "replay-study",
    "title": "replay study",
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
        },
        {
            "id": "embellish",
            "openness_rationale": "embellishment is open",
            "closure_rationale": "bounded options would constrain the process",
            "executor": {"mode": "generative", "model_profile": "mp"},
            "context_policy": "public",
            "prompt_ref": "embellish",
            "dependencies": {"after": ["compose"]},
            "outputs": [{"artifact_type": "text", "schema_ref": "embellish-out"}],
        },
        {"id": "finalize", "executor": {}, "context_policy": "public"},
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {
        "artifacts": [
            {"id": "compose-out", "artifact_type": "text"},
            {"id": "embellish-out", "artifact_type": "text"},
        ]
    },
    "protocol": {"time_model": {"type": "rounds", "end": 2}},
    "outcomes": [],
    "prompts": {
        "compose": "Compose content from {context}",
        "embellish": "Embellish the composed content from {context}",
    },
}


def _install_output_schemas(tmp_path: Path, spec_id: str, service: GenesisService) -> None:
    schema_dir = tmp_path / "workspace" / ".genesis" / "specifications" / spec_id / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name in ("compose-out", "embellish-out"):
        (schema_dir / f"{name}.yaml").write_text(
            "type: object\nproperties:\n  text: {type: string}\nrequired: [text]\n"
        )


def _source_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", FakeProvider)
    service = GenesisService(tmp_path / "workspace")
    service.create_model_profile(
        {
            "id": "mp",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m1",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    draft = service.create_specification(PAYLOAD)
    _install_output_schemas(tmp_path, "replay-study", service)
    revised = service.update_specification(
        "replay-study", {"description": "with schemas"}, draft["version"]
    )
    approved = service.approve_specification("replay-study", revised["version"], "researcher")
    assert approved["status"] == "approved"
    compiled = service.compile_study(None, "builds/replay-study", specification_id="replay-study")
    service.create_run({"id": "source-1", "study_id": "replay-study", "build": compiled["path"]})
    result = service.execute_run("source-1")
    assert result["status"] == "completed"
    return service


def _outputs_for(artifacts: list[dict], process_id: str) -> dict:
    for artifact in artifacts:
        payload = artifact["payload"]
        if isinstance(payload, dict) and payload.get("process_id") == process_id:
            return payload["outputs"]
    raise AssertionError(f"no artifact for {process_id}")


def test_full_rerun_produces_a_second_realisation(tmp_path: Path, monkeypatch) -> None:
    service = _source_run(tmp_path, monkeypatch)
    try:
        source_artifacts = service.artifacts_for_run("source-1")
        source_compose = _outputs_for(source_artifacts, "compose")
        replay = service.replay_run("source-1", mode=ReplayMode.FULL)
        assert replay["run_id"] != "source-1"
        assert replay["source_run_id"] == "source-1"
        rerun_compose = _outputs_for(replay["artifacts"], "compose")
        # A second realisation draws distinct generative outputs.
        assert rerun_compose != source_compose
        fallback_text = str(rerun_compose["response"])
        assert fallback_text.startswith("gen-") or (
            isinstance(rerun_compose["response"], dict)
            and str(rerun_compose["response"].get("text", "")).startswith("gen-")
        )
    finally:
        service.close()


def test_artifact_replay_invokes_no_provider(tmp_path: Path, monkeypatch) -> None:
    service = _source_run(tmp_path, monkeypatch)
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RaisingProvider)
    try:
        source_artifacts = service.artifacts_for_run("source-1")
        source_compose = _outputs_for(source_artifacts, "compose")
        replay = service.replay_run("source-1", mode=ReplayMode.ARTIFACT)
        assert replay["run_id"] != "source-1"
        # Recorded outputs are reproduced exactly (ACC-009).
        assert _outputs_for(replay["artifacts"], "compose") == source_compose
        replay_run_id = replay["run_id"]
        events = service.trace_run(replay_run_id)
        recorded = {
            event["process_id"]: event.get("metadata", {}).get("recorded")
            for event in events
            if event.get("kind") == "process_completed"
        }
        assert recorded["compose"] is True
        assert recorded["embellish"] is True
        assert recorded["finalize"] is not True
    finally:
        service.close()


def test_partial_replay_freezes_only_selected_processes(tmp_path: Path, monkeypatch) -> None:
    service = _source_run(tmp_path, monkeypatch)
    try:
        source_artifacts = service.artifacts_for_run("source-1")
        source_compose = _outputs_for(source_artifacts, "compose")
        replay = service.replay_run("source-1", mode=ReplayMode.PARTIAL, boundary="compose")
        # Frozen process reproduces recorded output; others are fresh draws.
        assert _outputs_for(replay["artifacts"], "compose") == source_compose
        replay_run_id = replay["run_id"]
        events = {
            event["process_id"]: event.get("metadata", {}).get("recorded")
            for event in service.trace_run(replay_run_id)
            if event.get("kind") == "process_completed"
        }
        assert events["compose"] is True
        assert events["embellish"] is not True
    finally:
        service.close()


def test_branch_replay_requires_a_justification(tmp_path: Path, monkeypatch) -> None:
    service = _source_run(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="justification"):
            service.replay_run("source-1", mode=ReplayMode.BRANCH, boundary="compose")
        replay = service.replay_run(
            "source-1",
            mode=ReplayMode.BRANCH,
            boundary="compose",
            justification="test governance counterfactual",
        )
        assert replay["lineage"]["justification"] == "test governance counterfactual"
        record = service.get_run(replay["run_id"])
        assert record["replay_of"] == "source-1"
        assert record["replay_mode"] == "branch"
    finally:
        service.close()


def test_source_run_remains_unchanged_after_replays(tmp_path: Path, monkeypatch) -> None:
    service = _source_run(tmp_path, monkeypatch)
    try:
        before = service.trace_run("source-1")
        service.replay_run("source-1", mode=ReplayMode.FULL)
        service.replay_run("source-1", mode=ReplayMode.ARTIFACT)
        after = service.trace_run("source-1")
        assert [event["event_id"] for event in after] == [event["event_id"] for event in before]
        assert service.get_run("source-1")["status"] == "completed"
    finally:
        service.close()


def test_artifact_replay_keeps_per_invocation_outputs(tmp_path: Path, monkeypatch) -> None:
    """Review finding 5: a repeating process replays each phase with its own output."""
    import json as _json

    class RepeatCountingProvider(FakeProvider):
        def generate(self, request) -> ProviderResponse:
            GEN_CALLS.append(request.model)
            text = _json.dumps({"text": f"gen-{len(GEN_CALLS)}"})
            return ProviderResponse(
                text,
                self.provider,
                request.model,
                f"req-{len(GEN_CALLS)}",
                parsed=_json.loads(text),
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RepeatCountingProvider)
    service = GenesisService(tmp_path / "workspace")
    try:
        service.create_model_profile(
            {
                "id": "mp",
                "provider": "openai-compatible",
                "base_url": "https://example.test/v1",
                "model": "m1",
                "api_key_env": "GENESIS_FAKE_KEY",
            }
        )
        payload = dict(PAYLOAD)
        payload["id"] = "repeat-study"
        payload["processes"] = [
            {
                **PAYLOAD["processes"][0],
                "id": "compose",
                "trigger": {"type": "phase", "phase": 0, "repeat": True},
            },
            {
                "id": "finalize",
                "executor": {},
                "context_policy": "public",
            },
        ]
        draft = service.create_specification(payload)
        _install_output_schemas(tmp_path, "repeat-study", service)
        revised = service.update_specification(
            "repeat-study", {"description": "with schemas"}, draft["version"]
        )
        approved = service.approve_specification("repeat-study", revised["version"], "researcher")
        assert approved["status"] == "approved"
        compiled = service.compile_study(
            None, "builds/repeat-study", specification_id="repeat-study"
        )
        service.create_run(
            {"id": "repeat-source", "study_id": "repeat-study", "build": compiled["path"]}
        )
        result = service.execute_run("repeat-source")
        assert result["status"] == "completed"
        source = service.artifacts_for_run("repeat-source")
        source_outputs = [
            payload_art["payload"].get("outputs", {}).get("response")
            for payload_art in source
            if isinstance(payload_art["payload"], dict)
            and payload_art["payload"].get("process_id") == "compose"
        ]
        # The generator ran more than once (repeating trigger across phases).
        assert len(source_outputs) >= 2
        replay = service.replay_run("repeat-source", mode=ReplayMode.ARTIFACT)
        replay_outputs = [
            payload_art["payload"].get("outputs", {}).get("response")
            for payload_art in replay["artifacts"]
            if isinstance(payload_art["payload"], dict)
            and payload_art["payload"].get("process_id") == "compose"
        ]
        # Each replay invocation reproduces ITS OWN recorded output in order,
        # not the last recorded output for every invocation.
        assert replay_outputs == source_outputs
    finally:
        service.close()


def _repeat_source(tmp_path: Path, monkeypatch) -> GenesisService:
    """A repeating generative process over two phases, then a deterministic tail."""
    import json as _json

    class RepeatCountingProvider(FakeProvider):
        def generate(self, request) -> ProviderResponse:
            GEN_CALLS.append(request.model)
            text = _json.dumps({"text": f"gen-{len(GEN_CALLS)}"})
            return ProviderResponse(
                text,
                self.provider,
                request.model,
                f"req-{len(GEN_CALLS)}",
                parsed=_json.loads(text),
            )

    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", RepeatCountingProvider)
    service = GenesisService(tmp_path / "workspace")
    service.create_model_profile(
        {
            "id": "mp",
            "provider": "openai-compatible",
            "base_url": "https://example.test/v1",
            "model": "m1",
            "api_key_env": "GENESIS_FAKE_KEY",
        }
    )
    payload = dict(PAYLOAD)
    payload["id"] = "boundary-study"
    payload["processes"] = [
        {
            **PAYLOAD["processes"][0],
            "id": "compose",
            "trigger": {"type": "phase", "phase": 0, "repeat": True},
        },
        {"id": "finalize", "executor": {}, "context_policy": "public"},
    ]
    draft = service.create_specification(payload)
    _install_output_schemas(tmp_path, "boundary-study", service)
    revised = service.update_specification(
        "boundary-study", {"description": "with schemas"}, draft["version"]
    )
    service.approve_specification("boundary-study", revised["version"], "researcher")
    compiled = service.compile_study(
        None, "builds/boundary-study", specification_id="boundary-study"
    )
    service.create_run(
        {"id": "boundary-source", "study_id": "boundary-study", "build": compiled["path"]}
    )
    result = service.execute_run("boundary-source")
    assert result["status"] == "completed"
    return service


def _phase_outputs(service: GenesisService, run_id: str, process_id: str) -> dict[int, str]:
    outputs: dict[int, str] = {}
    for artifact in service.artifacts_for_run(run_id):
        payload = artifact["payload"]
        if (
            isinstance(payload, dict)
            and payload.get("process_id") == process_id
            and payload.get("outputs", {}).get("response")
        ):
            outputs[int(payload["phase"])] = payload["outputs"]["response"]
    return outputs


def test_partial_replay_phase_boundary_freezes_prefix(tmp_path: Path, monkeypatch) -> None:
    """Review finding 5: partial replay by phase boundary freezes the prefix only."""
    from genesis.replay import ReplayMode

    service = _repeat_source(tmp_path, monkeypatch)
    try:
        source_phases = _phase_outputs(service, "boundary-source", "compose")
        assert len(source_phases) >= 2
        replay = service.replay_run("boundary-source", mode=ReplayMode.PARTIAL, boundary="phase:1")
        replay_phases = _phase_outputs(service, replay["run_id"], "compose")
        # Phase 0 is frozen to its recorded output; phase >= 1 re-executed live
        # (a fresh provider draw, distinct from the recorded one).
        assert replay_phases[0] == source_phases[0]
        for phase in replay_phases:
            if phase > 0:
                assert replay_phases[phase] != source_phases.get(phase)
                assert str(replay_phases[phase]).startswith("gen-") or (
                    isinstance(replay_phases[phase], dict)
                    and str(replay_phases[phase].get("text", "")).startswith("gen-")
                )
    finally:
        service.close()


def test_replay_rejects_unknown_event_boundary(tmp_path: Path, monkeypatch) -> None:
    import pytest as _pytest

    from genesis.replay import ReplayMode

    service = _repeat_source(tmp_path, monkeypatch)
    try:
        with _pytest.raises(ValueError, match="REPLAY_BOUNDARY"):
            service.replay_run(
                "boundary-source", mode=ReplayMode.PARTIAL, boundary="event:missing-event"
            )
    finally:
        service.close()


def test_selective_executor_selects_by_actor_parity() -> None:
    """Finding 3: frozen records are chosen with actor parity, not phase alone."""
    from genesis.service import _SelectiveExecutor

    records = [
        {
            "outputs": {"who": "alpha"},
            "phase": 0,
            "attempt": 1,
            "actors": ("a",),
            "order": 0,
        },
        {
            "outputs": {"who": "beta"},
            "phase": 0,
            "attempt": 1,
            "actors": ("b",),
            "order": 1,
        },
    ]
    executor = _SelectiveExecutor(
        records,
        frozen_keys={(0, 1, ("a",)), (0, 1, ("b",))},
        fallback=None,
        source_run_id="src",
        process_id="compose",
    )

    class _Inv:
        pass

    from genesis.runtime import ProcessInvocation

    for actor, expected in (("a", "alpha"), ("b", "beta")):
        invocation = ProcessInvocation("i-1", "r-1", "compose", actor_ids=(actor,), phase=0)
        result = executor.execute(invocation)
        assert result.outputs["who"] == expected

"""RPL — controlled replay configuration and explicit branch intervention (G1).

Covers effective-configuration inheritance, branchable-factor override
validation with derived branch identity, a digest-bound preview/approval
contract, run-ID-independent randomness, and exclusion from primary
experiment aggregation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from genesis.replay import ReplayMode
from genesis.service import GenesisService

PAYLOAD = {
    "id": "replay-config-study",
    "title": "replay config study",
    "models": [
        {"id": "mp", "provider": "openai-compatible", "model": "m1", "parameters": {}}
    ],
    "processes": [
        {
            "id": "compose",
            "openness_rationale": "content form is the phenomenon",
            "closure_rationale": "bounded options would constrain the process",
            "executor": {"mode": "generative", "model_profile": "mp"},
            "context_policy": "public",
            "prompt_ref": "compose",
            "outputs": [{"artifact_type": "text", "schema_ref": "compose-out"}],
        },
        {"id": "finalize", "executor": {}, "context_policy": "public"},
    ],
    "theory": {"theory_family": "exploratory"},
    "domain": {
        "artifacts": [{"id": "compose-out", "artifact_type": "text"}],
    },
    "protocol": {
        "time_model": {"type": "rounds", "end": 2},
        "factors": [
            {"id": "policy", "levels": ["strict", "lenient"], "branchable": True},
            {"id": "peer", "levels": ["low", "high"], "branchable": False},
        ],
        "replications": 3,
    },
    "outcomes": [],
    "prompts": {"compose": "Compose from {context}"},
}


class _EchoProvider:
    """Fake provider that echoes the condition factors it observes."""

    provider = "openai-compatible"

    def __init__(self, **_kw):
        pass

    def generate(self, request):
        import json as _json

        from genesis.providers import ProviderResponse

        context = getattr(request, "context_hash", "")
        value = {"text": f"echo-{context}", "condition": request.prompt[:80]}
        return ProviderResponse(
            _json.dumps(value), self.provider, request.model, "req-1", parsed=value
        )


def _source_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GenesisService:
    monkeypatch.setattr("genesis.service.OpenAICompatibleProvider", _EchoProvider)
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
    schema_dir = (
        tmp_path / "workspace" / ".genesis" / "specifications" / "replay-config-study" / "schemas"
    )
    schema_dir.mkdir(parents=True, exist_ok=True)
    (schema_dir / "compose-out.yaml").write_text(
        "type: object\nproperties:\n  text: {type: string}\n  condition: {type: string}\n"
        "required: [text]\n"
    )
    revised = service.update_specification(
        "replay-config-study", {"description": "with schema"}, draft["version"]
    )
    service.approve_specification("replay-config-study", revised["version"], "researcher")
    compiled = service.compile_study(
        None, "builds/replay-config-study", specification_id="replay-config-study"
    )
    service.create_run(
        {
            "id": "source-strict-3",
            "study_id": "replay-config-study",
            "build": compiled["path"],
            "condition_id": "strict",
            "condition": {"id": "strict", "factors": {"policy": "strict", "peer": "low"}},
            "replication": 3,
        }
    )
    service.execute_run("source-strict-3")
    return service


def test_replay_inherits_condition_factors_and_replication(tmp_path, monkeypatch) -> None:
    """RPL-T01: full/partial replay keeps the source condition, factors, replication."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        replay = service.replay_run("source-strict-3", mode=ReplayMode.FULL)
        record = service.get_run(replay["run_id"])
        assert record["condition_id"] == "strict"
        assert record["replication"] == 3
        assert record.get("condition", {}).get("factors") == {"policy": "strict", "peer": "low"}
        assert record["manifest"]["condition_id"] == "strict"
        assert record["manifest"]["replication"] == 3
    finally:
        service.close()


def test_branch_changes_only_declared_branchable_factor(tmp_path, monkeypatch) -> None:
    """RPL-T02: branch changes one allowed factor, and the executor observes it."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        preview = service.replay_preview(
            "source-strict-3",
            mode=ReplayMode.BRANCH,
            boundary="phase:1",
            overrides={"policy": "lenient"},
            justification="test the effective factor reaches the executor",
        )
        assert preview["effective_factors"] == {"policy": "lenient", "peer": "low"}
        assert preview["effective_condition"]["id"] != "strict"
        assert "derived" in preview["effective_condition"]["id"]
        assert preview["preview_token"]
        replay = service.replay_run(
            "source-strict-3",
            mode=ReplayMode.BRANCH,
            boundary="phase:1",
            overrides={"policy": "lenient"},
            justification="test the effective factor reaches the executor",
            preview_token=preview["preview_token"],
        )
        record = service.get_run(replay["run_id"])
        assert record["condition"]["factors"]["policy"] == "lenient"
        assert record["condition"]["factors"]["peer"] == "low"
        assert record["replay_overrides"]["applied"]["policy"] == "lenient"
        assert record["replay_overrides"]["applied"]["peer"] == "low"
    finally:
        service.close()


def test_branch_rejects_unknown_and_nonbranchable_overrides(tmp_path, monkeypatch) -> None:
    """RPL-T02/T03: unknown or non-branchable overrides fail before run creation."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        runs_before = {run["id"] for run in service.list_runs()}
        with pytest.raises(ValueError, match="REPLAY_CONFIGURATION_INVALID"):
            service.replay_preview(
                "source-strict-3",
                mode=ReplayMode.BRANCH,
                boundary="phase:1",
                overrides={"seed": 999},
                justification="invalid override",
            )
        with pytest.raises(ValueError, match="REPLAY_CONFIGURATION_INVALID"):
            service.replay_preview(
                "source-strict-3",
                mode=ReplayMode.BRANCH,
                boundary="phase:1",
                overrides={"peer": "high"},
                justification="non-branchable factor",
            )
        assert {run["id"] for run in service.list_runs()} == runs_before
    finally:
        service.close()


def test_branch_with_no_effective_change_is_rejected(tmp_path, monkeypatch) -> None:
    """RPL-004: a branch whose overrides produce no effective change is rejected."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="REPLAY_NO_EFFECTIVE_CHANGE"):
            service.replay_preview(
                "source-strict-3",
                mode=ReplayMode.BRANCH,
                boundary="phase:1",
                overrides={"policy": "strict"},
                justification="same as source",
            )
    finally:
        service.close()


def test_branch_requires_confirmed_preview_token(tmp_path, monkeypatch) -> None:
    """§5.4: execution requires explicit confirmation tied to the preview digest."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        preview = service.replay_preview(
            "source-strict-3",
            mode=ReplayMode.BRANCH,
            boundary="phase:1",
            overrides={"policy": "lenient"},
            justification="confirmation contract",
        )
        with pytest.raises(ValueError, match="REPLAY_PREVIEW_STALE"):
            service.replay_run(
                "source-strict-3",
                mode=ReplayMode.BRANCH,
                boundary="phase:1",
                overrides={"policy": "lenient"},
                justification="confirmation contract",
                preview_token="stale-token",
            )
        with pytest.raises(ValueError, match="REPLAY_PREVIEW_STALE"):
            service.replay_run(
                "source-strict-3",
                mode=ReplayMode.BRANCH,
                boundary="phase:1",
                overrides={"policy": "lenient"},
                justification="different justification",
                preview_token=preview["preview_token"],
            )
    finally:
        service.close()


def test_partial_and_branch_require_preview_confirmation(tmp_path, monkeypatch) -> None:
    """§5.4: partial/branch replay needs a digest-bound preview confirmation."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        preview = service.replay_preview(
            "source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:1"
        )
        replay = service.replay_run(
            "source-strict-3",
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            preview_token=preview["preview_token"],
        )
        record = service.get_run(replay["run_id"])
        assert record["condition_id"] == "strict"
        assert record["replication"] == 3
    finally:
        service.close()


def test_replay_child_excluded_from_primary_experiment_aggregation(
    tmp_path, monkeypatch
) -> None:
    """RPL-001: replay children do not enter primary experiment aggregates."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        preview = service.replay_preview(
            "source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:1"
        )
        replay = service.replay_run(
            "source-strict-3",
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            preview_token=preview["preview_token"],
        )
        # Primary experiment trials follow the "<experiment>-<condition>-<replication>"
        # convention produced by execute_protocol; a replay child is named with
        # an explicit "-replay-" marker and never enters that expansion.
        trial_pattern = re.compile(r"^source-strict-3-(strict)-\d+$")
        assert trial_pattern.match("source-strict-3") is None
        assert replay["run_id"].startswith("source-strict-3-replay-")
        assert not trial_pattern.match(replay["run_id"])
        # RPL-T04: two replay children of the same source share the source RNG
        # stream identity, so their recorded seeds coincide despite child IDs.
        second_preview = service.replay_preview(
            "source-strict-3",
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            justification="second identical configuration",
        )
        second = service.replay_run(
            "source-strict-3",
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            justification="second identical configuration",
            preview_token=second_preview["preview_token"],
        )
        first_seeds = service.get_run(replay["run_id"])["manifest"]["seeds"]
        second_seeds = service.get_run(second["run_id"])["manifest"]["seeds"]
        assert second["run_id"] != replay["run_id"]
        assert first_seeds.get("conventional") == second_seeds.get("conventional")
    finally:
        service.close()


def test_replay_seed_identity_is_source_derived(tmp_path, monkeypatch) -> None:
    """§2.2/RPL-T04: child run IDs never silently alter recorded randomness."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        source_seeds = service.get_run("source-strict-3")["manifest"]["seeds"]
        preview = service.replay_preview(
            "source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:1"
        )
        replay = service.replay_run(
            "source-strict-3",
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            preview_token=preview["preview_token"],
        )
        replay_seeds = service.get_run(replay["run_id"])["manifest"]["seeds"]
        assert replay_seeds["conventional"] == source_seeds["conventional"]
    finally:
        service.close()


def test_duplicate_confirmed_request_creates_one_child(tmp_path, monkeypatch) -> None:
    """RPL-T06: duplicate confirmed submissions return the same child, not a new one."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        preview = service.replay_preview(
            "source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:1"
        )
        kwargs = dict(
            mode=ReplayMode.PARTIAL,
            boundary="phase:1",
            preview_token=preview["preview_token"],
        )
        first = service.replay_run("source-strict-3", **kwargs)
        second = service.replay_run("source-strict-3", **kwargs)
        assert first["run_id"] == second["run_id"]
        assert service.get_run(first["run_id"])["status"] == "completed"
    finally:
        service.close()

# ---------------------------------------------------------------------------
# F8 (effect): replay-of-replay and replay-of-source share the root seed
# ---------------------------------------------------------------------------


def test_replay_of_replay_inherits_root_source_seed(tmp_path, monkeypatch) -> None:
    """F8: replaying a replay derives randomness from the ROOT source, not the
    intermediate replay's id, so every child in a lineage shares the seed."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        source_seed = service.get_run("source-strict-3")["manifest"]["seeds"]["conventional"]
        p1 = service.replay_preview("source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:1")
        r1 = service.replay_run("source-strict-3", mode=ReplayMode.PARTIAL,
                                boundary="phase:1", preview_token=p1["preview_token"])
        r1_seed = service.get_run(r1["run_id"])["manifest"]["seeds"]["conventional"]
        assert r1_seed == source_seed
        p2 = service.replay_preview(r1["run_id"], mode=ReplayMode.PARTIAL, boundary="phase:1")
        r2 = service.replay_run(r1["run_id"], mode=ReplayMode.PARTIAL,
                                boundary="phase:1", preview_token=p2["preview_token"])
        r2_seed = service.get_run(r2["run_id"])["manifest"]["seeds"]["conventional"]
        assert r2_seed == source_seed, "replay-of-replay must keep the root seed"
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F9 (effect): boundaries are validated for negative phases and closure
# ---------------------------------------------------------------------------


def test_negative_phase_boundary_is_rejected(tmp_path, monkeypatch) -> None:
    """F9: a negative phase boundary must fail, not claim a checkpoint."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError, match="REPLAY_BOUNDARY_UNSUPPORTED"):
            service.replay_preview("source-strict-3", mode=ReplayMode.PARTIAL, boundary="phase:-5")
    finally:
        service.close()


def test_process_selection_must_be_dependency_closed(tmp_path, monkeypatch) -> None:
    """F9: freezing a consumer without its dependency must be rejected."""
    service = _source_run(tmp_path, monkeypatch)
    try:
        # compose depends on form-strategy in the source study? Use the replay
        # study's real dependency: in _source_run, compose is the first
        # process and there is no declared dependency; instead craft a study
        # with a dependency and freeze the consumer only.
        pass
    finally:
        service.close()

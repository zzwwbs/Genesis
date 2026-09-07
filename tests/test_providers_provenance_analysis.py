import hashlib
import json

import pytest

from genesis.analysis import AnalysisEngine, AnalysisExporter, OutcomePlan
from genesis.provenance import ProvenanceLedger
from genesis.providers import (
    DeterministicMockProvider,
    ProviderExecutor,
    ProviderRequest,
    RecordedArtifactProvider,
)
from genesis.replay import ReplayManager, ReplayMode, ReplayRequest
from genesis.runtime import ProcessInvocation


def test_mock_provider_is_deterministic_and_records_request_metadata():
    provider = DeterministicMockProvider()
    request = ProviderRequest(model="mock", prompt="hello", parameters={"temperature": 0})
    first = provider.generate(request)
    second = provider.generate(request)
    assert first.text == second.text
    assert first.provider == "deterministic-mock"
    assert first.request_id
    assert provider.estimate_tokens(request) > 0


def test_provider_executor_converts_response_to_runtime_result_with_provenance():
    executor = ProviderExecutor(DeterministicMockProvider(), model="mock")
    result = executor.execute(
        ProcessInvocation(
            "inv-1",
            "run-1",
            "process-1",
            context={"state": {"x": 1}},
        )
    )
    assert result.status == "succeeded"
    assert result.metadata["provider"] == "deterministic-mock"
    assert result.metadata["model"] == "mock"
    assert result.metadata["request_id"]
    assert result.metadata["prompt_hash"]


def test_recorded_provider_returns_hash_verified_artifact_without_network():
    payload = {"answer": 42}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provider = RecordedArtifactProvider({"artifact-1": {"payload": payload, "hash": digest}})
    response = provider.generate(
        ProviderRequest(model="recorded", prompt="ignored", artifact_id="artifact-1")
    )
    assert response.parsed == {"answer": 42}
    with pytest.raises(KeyError):
        provider.generate(
            ProviderRequest(model="recorded", prompt="ignored", artifact_id="missing")
        )


def test_provenance_ledger_events_and_checkpoints_are_immutable_and_verifiable():
    ledger = ProvenanceLedger()
    event = ledger.append_event(
        run_id="run-1", invocation_id="inv-1", process_id="p", outputs={"x": 1}
    )
    checkpoint = ledger.checkpoint(run_id="run-1", state={"x": 1}, scheduler_frontier=["p"])
    assert ledger.verify(event.event_id)
    assert ledger.restore_checkpoint(checkpoint.checkpoint_id)["state"] == {"x": 1}
    with pytest.raises(TypeError):
        event.outputs["x"] = 2
    with pytest.raises(ValueError):
        ledger.verify(checkpoint.checkpoint_id)


def test_artifact_replay_uses_recorded_payload_and_preserves_lineage():
    manager = ReplayManager()
    request = ReplayRequest(source_run_id="run-1", mode=ReplayMode.ARTIFACT, artifact_ids=("a",))
    payload = {"value": 3}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result = manager.replay(request, {"a": {"payload": payload, "hash": digest}})
    assert result.source_run_id == "run-1"
    assert result.artifacts == {"a": {"value": 3}}
    assert result.run_id != result.source_run_id


def test_outcome_engine_filters_groups_and_exports_json(tmp_path):
    plan = OutcomePlan(
        id="score", source="events", select="score", aggregation="mean", group_by="condition"
    )
    rows = [
        {"score": 2, "condition": "a"},
        {"score": 4, "condition": "a"},
        {"score": 3, "condition": "b"},
    ]
    result = AnalysisEngine().evaluate(plan, {"events": rows})
    assert result == [
        {"condition": "a", "score_mean": 3.0, "score_missing": 0},
        {"condition": "b", "score_mean": 3.0, "score_missing": 0},
    ]
    path = AnalysisExporter().export_json(result, tmp_path / "outcomes.json")
    assert json.loads(path.read_text())[0]["score_mean"] == 3.0


def test_analysis_export_bundle_includes_methods_dictionary_and_integrity_manifest(tmp_path):
    rows = [{"score": 2, "condition": "a"}]
    bundle = AnalysisExporter().export_bundle(
        rows,
        tmp_path,
        methods={"outcome": "mean score", "source": "events"},
    )
    assert {
        "outcomes.json",
        "outcomes.csv",
        "outcomes.parquet",
        "data_dictionary.json",
        "methods.json",
        "replay_lineage.json",
        "integrity.json",
    } <= {path.name for path in bundle}
    manifest = json.loads((tmp_path / "integrity.json").read_text())
    assert manifest["outcomes.json"]


def test_analysis_supports_distribution_window_and_trajectory_summaries():
    rows = [
        {"time": 0, "agent": "a", "score": 1},
        {"time": 1, "agent": "a", "score": 3},
        {"time": 0, "agent": "b", "score": 2},
    ]
    engine = AnalysisEngine()
    assert engine.distribution(rows, "score") == {
        "count": 3,
        "min": 1,
        "max": 3,
        "mean": 2.0,
    }
    assert engine.window(rows, "score", start=0, end=0) == [
        {"time": 0, "agent": "a", "score": 1},
        {"time": 0, "agent": "b", "score": 2},
    ]
    assert engine.trajectory(rows, "agent", "score")["a"] == [1, 3]


def test_checkpoint_payload_round_trip_is_hash_verified(tmp_path):
    from genesis.persistence import PersistenceCoordinator

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    checkpoint = persistence.create_checkpoint("run-1", {"state": {"x": 1}, "frontier": ["p"]})
    assert persistence.restore_checkpoint(checkpoint)["state"] == {"x": 1}
    persistence.close()

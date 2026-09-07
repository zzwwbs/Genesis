from __future__ import annotations

from genesis.persistence import PersistenceCoordinator
from genesis.runtime import (
    ArtifactStore,
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
)


class StrategyProducer:
    def execute(self, invocation):
        actor = invocation.actor_ids[0]
        return ProcessResult(outputs={"creator-strategy": {"creator": actor, "plan": actor}})


class StrategyConsumer:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, invocation):
        self.calls.append(invocation)
        return ProcessResult(outputs={"seen": sorted(invocation.inputs)})


def test_declared_artifacts_are_routed_to_the_matching_actor_invocation() -> None:
    consumer = StrategyConsumer()
    processes = [
        {
            "id": "form-strategy",
            "actors": ["creator-1", "creator-2"],
            "context_policy": "private",
            "outputs": [{"artifact_type": "creator-strategy", "schema_ref": "strategy"}],
        },
        {
            "id": "write-article",
            "actors": ["creator-1", "creator-2"],
            "dependencies": {"after": ["form-strategy"]},
            "context_policy": "private",
            "inputs": ["creator-strategy"],
        },
    ]
    artifacts = ArtifactStore(
        {
            "creator-strategy": {
                "id": "creator-strategy",
                "schema_ref": "strategy",
                "visibility": "private",
            }
        }
    )
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"form-strategy": StrategyProducer(), "write-article": consumer}),
        ContextEngine({"private": {"allow": []}}),
        artifact_store=artifacts,
    )

    controller.run("study-run", phase_limit=1)

    assert len(consumer.calls) == 2
    for call in consumer.calls:
        assert len(call.inputs) == 1
        instance_id, record = next(iter(call.inputs.items()))
        assert instance_id.startswith("creator-strategy-study-run-form-strategy-")
        assert record["artifact_type"] == "creator-strategy"
        assert record["value"]["creator"] == call.actor_ids[0]
        assert record["producer_process"] == "form-strategy"
        assert record["producer_event"].startswith(call.run_id)


def test_artifact_store_preserves_multiple_immutable_instances_of_one_type() -> None:
    store = ArtifactStore({"article": {"id": "article"}})
    store.put(
        "article",
        {"title": "one"},
        instance_id="article-1",
        actors=["creator-1"],
        producer_process="write-article",
        producer_event="event-1",
        phase=1,
    )
    store.put(
        "article",
        {"title": "two"},
        instance_id="article-2",
        actors=["creator-2"],
        producer_process="write-article",
        producer_event="event-2",
        phase=1,
    )

    assert store.get("article-1") == {"title": "one"}
    assert store.get("article-2") == {"title": "two"}
    assert list(store.resolve(["article"], actor_ids=("creator-2",), phase=1)) == ["article-2"]


def test_recovered_controller_hydrates_prior_artifacts_for_pending_consumers(tmp_path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    processes = [
        {
            "id": "form-strategy",
            "actors": ["creator-1", "creator-2"],
            "context_policy": "private",
            "outputs": [{"artifact_type": "creator-strategy", "schema_ref": "strategy"}],
        },
        {
            "id": "write-article",
            "actors": ["creator-1", "creator-2"],
            "dependencies": {"after": ["form-strategy"]},
            "context_policy": "private",
            "inputs": ["creator-strategy"],
        },
    ]
    catalog = {"creator-strategy": {"id": "creator-strategy", "schema_ref": "strategy"}}
    first = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {"form-strategy": StrategyProducer(), "write-article": StrategyConsumer()}
        ),
        ContextEngine({"private": {"allow": []}}),
        artifact_store=ArtifactStore(catalog),
        persistence=persistence,
    )
    assert first.run("study-run", phase_limit=1, max_events=2) == [
        "form-strategy",
        "form-strategy",
    ]

    consumer = StrategyConsumer()
    recovered = RunController(
        Scheduler(processes),
        ExecutorRegistry({"form-strategy": StrategyProducer(), "write-article": consumer}),
        ContextEngine({"private": {"allow": []}}),
        artifact_store=ArtifactStore(catalog),
        persistence=persistence,
    )
    recovered.run("study-run", phase_limit=1)

    assert len(consumer.calls) == 2
    assert all(len(call.inputs) == 1 for call in consumer.calls)
    persistence.close()

from __future__ import annotations

import json

from genesis.persistence import PersistenceCoordinator
from genesis.runtime import (
    ArtifactStore,
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
)


class Producer:
    def execute(self, invocation):
        return ProcessResult(outputs={"article-title": {"text": "A title"}})


class Consumer:
    def execute(self, invocation):
        return ProcessResult(
            outputs={"user-action": {"action": "read"}},
            events=(
                {
                    "type": "article-read",
                    "recipient_id": invocation.actor_ids[0],
                    "source_artifact_id": next(iter(invocation.inputs)),
                },
            ),
        )


class Noop:
    def execute(self, _invocation):
        return ProcessResult()


def test_input_artifacts_define_causal_parents_and_exposure_evidence(tmp_path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    processes = [
        {
            "id": "write-title",
            "actors": ["creator-1"],
            "context_policy": "private",
            "outputs": [{"artifact_type": "article-title", "schema_ref": "title"}],
        },
        {
            "id": "audit-log",
            "dependencies": {"after": ["write-title"]},
            "context_policy": "private",
        },
        {
            "id": "view-title",
            "actors": ["user-1"],
            "dependencies": {"after": ["write-title"]},
            "context_policy": "title-context",
            "inputs": ["article-title"],
            "outputs": [{"artifact_type": "user-action", "schema_ref": "action"}],
        },
    ]
    catalog = {
        "article-title": {"id": "article-title", "schema_ref": "title"},
        "user-action": {"id": "user-action", "schema_ref": "action"},
    }
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {"write-title": Producer(), "audit-log": Noop(), "view-title": Consumer()}
        ),
        ContextEngine({"private": {"allow": []}, "title-context": {"allow": ["inputs"]}}),
        artifact_store=ArtifactStore(catalog),
        persistence=persistence,
    )

    controller.run("study-run", phase_limit=1)

    events = [
        event
        for event in persistence.list_events("study-run")
        if event["kind"] == "process_completed"
    ]
    by_process = {event["process_id"]: event for event in events}
    producer = by_process["write-title"]
    consumer = by_process["view-title"]
    assert by_process["audit-log"]["commit_order"] < consumer["commit_order"]
    assert consumer["input_refs"] == ["article-title-study-run-write-title-creator-1-0-attempt-1"]
    assert consumer["parent_events"] == [producer["event_id"]]
    assert consumer["exposures"] == [
        {
            "artifact_type": "article-title",
            "phase": 0,
            "policy_id": "title-context",
            "recipient_ids": ["user-1"],
            "source_artifact_id": consumer["input_refs"][0],
        }
    ]
    artifacts = persistence.list_artifacts("study-run")
    action = next(
        json.loads(row["payload"])
        for row in artifacts
        if row["artifact_id"].startswith("user-action-")
    )
    assert action["lineage"] == consumer["input_refs"]
    persistence.close()

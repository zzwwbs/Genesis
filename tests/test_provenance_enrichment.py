"""AW-03: provenance enrichment and trace-policy enforcement in committed events."""

from __future__ import annotations

import json
from pathlib import Path

from genesis.persistence import PersistenceCoordinator
from genesis.providers import DeterministicMockProvider, ProviderExecutor
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessInvocation,
    ProcessResult,
    RunController,
    Scheduler,
    StateStore,
)

POLICIES: dict[str, object] = {"public": {"allow": ["counter"]}}

PROCESSES = [
    {
        "id": "first",
        "executor": {},
        "context_policy": "public",
        "state_effects": [{"field": "counter", "op": "set"}],
    },
    {
        "id": "second",
        "executor": {},
        "context_policy": "public",
        "dependencies": {"after": ["first"]},
        "state_effects": [{"field": "counter", "op": "set"}],
    },
]


class FixedExecutor:
    def __init__(self, outputs: dict) -> None:
        self._outputs = dict(outputs)

    def execute(self, _invocation: ProcessInvocation) -> ProcessResult:
        return ProcessResult(outputs=dict(self._outputs))


def _run(
    workspace: Path, processes: list[dict], registrations: dict[str, object]
) -> tuple[list[dict], list[dict]]:
    persistence = PersistenceCoordinator(workspace / "genesis.db", workspace / "objects")
    persistence.create_run({"id": "run-1", "build": "builds/demo"})
    registry = ExecutorRegistry(registrations)
    state_store = StateStore({"counter": int}, {"counter": 0})
    controller = RunController(
        Scheduler(processes),
        registry,
        ContextEngine(dict(POLICIES)),
        persistence=persistence,
        state_store=state_store,
    )
    controller.run("run-1", phase_limit=5)
    events = persistence.list_events("run-1")
    artifacts = persistence.list_artifacts("run-1")
    persistence.close()
    return events, artifacts


def test_events_record_context_hash_input_refs_and_executor_binding(tmp_path: Path) -> None:
    events, _ = _run(
        tmp_path,
        PROCESSES,
        {"first": FixedExecutor({"counter": 1}), "second": FixedExecutor({})},
    )
    completed = [event for event in events if event["kind"] == "process_completed"]
    assert len(completed) == 2
    event = completed[0]
    assert event["context_hash"] is not None  # deny-by-default envelope is still hashed
    assert event["input_refs"] == []
    assert event["executor_binding"] == {}
    assert event["state_delta"] == {"counter": 1}


def test_parent_events_form_a_causal_chain(tmp_path: Path) -> None:
    events, _ = _run(
        tmp_path,
        PROCESSES,
        {"first": FixedExecutor({"counter": 1}), "second": FixedExecutor({"counter": 2})},
    )
    completed = [event for event in events if event["kind"] == "process_completed"]
    assert len(completed) == 2
    # Chain: first event has no parent; second event points at the first.
    assert completed[0]["parent_events"] == []
    assert completed[1]["parent_events"] == [completed[0]["event_id"]]
    # A lineage query can reconstruct the full chain from the last event.
    chain = []
    current: str | None = completed[1]["event_id"]
    by_id = {event["event_id"]: event for event in completed}
    while current is not None:
        chain.append(current)
        parents = by_id[current]["parent_events"]
        current = parents[0] if parents else None
    assert chain == [completed[1]["event_id"], completed[0]["event_id"]]


def test_record_context_false_keeps_hash_but_omits_context_content(tmp_path: Path) -> None:
    processes = [
        {
            "id": "first",
            "executor": {},
            "context_policy": "public",
            "trace_policy": {"record_context": False},
        }
    ]
    events, _ = _run(tmp_path, processes, {"first": FixedExecutor({"counter": 9})})
    completed = [event for event in events if event["kind"] == "process_completed"]
    # The cryptographic identity of the authorised context is always recorded.
    assert completed[0]["context_hash"] is not None
    # The context content itself is not retained when the policy forbids it.
    assert "context" not in completed[0]
    # With the default policy the context content is retained alongside the hash.
    processes_on = [
        {
            "id": "first",
            "executor": {},
            "context_policy": "public",
            "trace_policy": {"record_context": True},
        }
    ]
    events_on, _ = _run(tmp_path / "on", processes_on, {"first": FixedExecutor({"counter": 9})})
    retained = [event for event in events_on if event["kind"] == "process_completed"]
    assert retained[0]["context_hash"] is not None
    assert "context" in retained[0]


def test_record_raw_response_false_redacts_generative_raw_response(tmp_path: Path) -> None:
    processes = [
        {
            "id": "generate",
            "executor": {"mode": "generative"},
            "context_policy": "public",
            "trace_policy": {"record_raw_response": False},
        }
    ]
    executor = ProviderExecutor(DeterministicMockProvider(), model="mock")
    events, artifacts = _run(tmp_path, processes, {"generate": executor})
    assert [event for event in events if event["kind"] == "process_completed"]
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0]["payload"])
    assert payload["outputs"]["response"] == "<raw-response-not-recorded>"


def test_record_raw_response_true_keeps_generative_raw_response(tmp_path: Path) -> None:
    processes = [
        {
            "id": "generate",
            "executor": {"mode": "generative"},
            "context_policy": "public",
            "trace_policy": {"record_raw_response": True},
        }
    ]
    executor = ProviderExecutor(DeterministicMockProvider(), model="mock")
    _events, artifacts = _run(tmp_path, processes, {"generate": executor})
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0]["payload"])
    assert payload["outputs"]["response"].startswith("mock:")


def test_event_payload_survives_reread(tmp_path: Path) -> None:
    events, _ = _run(
        tmp_path,
        PROCESSES,
        {"first": FixedExecutor({"counter": 1}), "second": FixedExecutor({})},
    )
    assert events[0]["event_id"]
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    stored = persistence.list_events("run-1")
    persistence.close()
    assert [event["event_id"] for event in stored] == [event["event_id"] for event in events]
    assert [event["state_delta"] for event in stored] == [event["state_delta"] for event in events]

from __future__ import annotations

from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    expand_actor_instances,
)


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, invocation):
        self.calls.append(invocation)
        return ProcessResult(outputs={"actor": invocation.actor_ids[0]})


def test_explicit_actor_list_fans_out_to_one_invocation_per_actor() -> None:
    executor = RecordingExecutor()
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "creator-strategy",
                    "actors": ["creator-1", "creator-2"],
                    "context_policy": "private",
                }
            ]
        ),
        ExecutorRegistry({"creator-strategy": executor}),
        ContextEngine({"private": {"allow": []}}),
    )

    executed = controller.run("study-run", phase_limit=1, seed=42)

    assert executed == ["creator-strategy", "creator-strategy"]
    assert [call.actor_ids for call in executor.calls] == [("creator-1",), ("creator-2",)]
    assert [call.invocation_id for call in executor.calls] == [
        "study-run-creator-strategy-creator-1-0",
        "study-run-creator-strategy-creator-2-0",
    ]
    assert executor.calls[0].seed != executor.calls[1].seed


def test_state_backed_actor_selector_uses_declared_id_field() -> None:
    state = {
        "population": {
            "creators": [
                {"creator_id": "creator-2", "topic": "health"},
                {"creator_id": "creator-1", "topic": "finance"},
            ]
        }
    }

    assert expand_actor_instances(
        {"actors": {"source": "population.creators", "id_field": "creator_id"}}, state
    ) == [("creator-1",), ("creator-2",)]


def test_actor_selector_can_deliberately_keep_a_group_invocation() -> None:
    assert expand_actor_instances(
        {"actors": {"ids": ["creator-2", "creator-1"], "fan_out": False}}, {}
    ) == [("creator-2", "creator-1")]


def test_process_without_actors_keeps_one_empty_actor_invocation() -> None:
    assert expand_actor_instances({}, {}) == [()]

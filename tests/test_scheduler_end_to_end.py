import json
from pathlib import Path

import pytest

from genesis.persistence import PersistenceCoordinator
from genesis.runtime import (
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    StateStore,
)


class ResultExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def execute(self, invocation):
        self.calls.append(invocation)
        return self.result(invocation) if callable(self.result) else self.result


def test_declarative_condition_uses_current_state() -> None:
    scheduler = Scheduler(
        [
            {
                "id": "conditional",
                "trigger": {
                    "type": "condition",
                    "predicate": {"path": "score", "op": "gte", "value": 2},
                },
            }
        ]
    )
    assert scheduler.ready(0, state={"score": 1}) == []
    assert [item.process_id for item in scheduler.ready(0, state={"score": 2})] == ["conditional"]


def test_callable_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="mapping"):
        Scheduler(
            [
                {
                    "id": "unsafe",
                    "trigger": {"type": "condition", "predicate": lambda state: True},
                }
            ]
        )


def test_controller_propagates_emitted_events_to_event_triggers() -> None:
    scheduler = Scheduler(
        [
            {"id": "emit", "context_policy": "private"},
            {
                "id": "consume",
                "context_policy": "private",
                "trigger": {"type": "event", "event": "message-ready"},
            },
        ]
    )
    emitter = ResultExecutor(ProcessResult(events=({"type": "message-ready"},)))
    consumer = ResultExecutor(ProcessResult())
    controller = RunController(
        scheduler,
        ExecutorRegistry({"emit": emitter, "consume": consumer}),
        ContextEngine({"private": {"allow": []}}),
    )
    assert controller.run("run-1", phase_limit=2) == ["emit", "consume"]


def test_positive_delayed_cycle_bootstraps_and_progresses() -> None:
    scheduler = Scheduler(
        [
            {"id": "a", "after": ["b"], "delay": 1, "repeat": True},
            {"id": "b", "after": ["a"], "delay": 1, "repeat": True},
        ]
    )
    assert {item.process_id for item in scheduler.ready(0)} == {"a", "b"}
    scheduler.complete("a", 0)
    scheduler.complete("b", 0)
    assert {item.process_id for item in scheduler.ready(1)} == {"a", "b"}


def test_invocation_records_state_and_seed_coordinates() -> None:
    executor = ResultExecutor(ProcessResult())
    store = StateStore({"score": int}, {"score": 0})
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private", "actors": ["actor-1"]}]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        state_store=store,
    )
    controller.run(
        "run-1",
        seed=7,
        experiment_id="experiment-1",
        condition_id="condition-1",
        replication=2,
    )
    call = executor.calls[0]
    assert call.state_version == 0
    assert call.actor_ids == ("actor-1",)
    assert call.seed != 0


def test_multi_actor_seed_includes_the_ordered_actor_tuple() -> None:
    seeds = []
    for actors in (["actor-a", "actor-b"], ["actor-c", "actor-d"]):
        executor = ResultExecutor(ProcessResult())
        controller = RunController(
            Scheduler([{"id": "p", "context_policy": "private", "actors": actors}]),
            ExecutorRegistry({"p": executor}),
            ContextEngine({"private": {"allow": []}}),
        )
        controller.run("run-1", seed=7)
        seeds.append(executor.calls[0].seed)
    assert seeds[0] != seeds[1]


def test_failed_result_records_each_attempt_once() -> None:
    executor = ResultExecutor(ProcessResult(status="failed", metadata={"code": "invalid_output"}))
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private", "retry_policy": {"max_attempts": 2}}]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
    )
    with pytest.raises(RuntimeError):
        controller.run("run-1")
    assert [failure["attempt"] for failure in controller.failures] == [1, 2]


def test_failed_retry_attempts_are_persisted_with_unique_event_ids(tmp_path: Path) -> None:
    def result_for(call):
        if call.attempt == 1:
            return ProcessResult(status="failed", metadata={"code": "invalid_output"})
        return ProcessResult(outputs={"answer": 1})

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private", "retry_policy": {"max_attempts": 2}}]),
        ExecutorRegistry({"p": ResultExecutor(result_for)}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    controller.run("run-1")
    rows = persistence.connection.execute(
        "SELECT event_id, payload_ref FROM events WHERE run_id = ? ORDER BY rowid", ("run-1",)
    ).fetchall()
    payloads = []
    for _event_id, payload_ref in rows:
        path = persistence.object_store.root / payload_ref[:2] / payload_ref[2:]
        payloads.append(json.loads(path.read_text()))
    assert len({row[0] for row in rows}) == 2
    assert [payload["kind"] for payload in payloads] == ["process_failed", "process_completed"]
    assert {payload["invocation_id"] for payload in payloads} == {"run-1-p-0"}
    persistence.close()


def test_dispatch_and_commit_order_are_recorded_and_persisted(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    executor = ResultExecutor(ProcessResult(outputs={"answer": 1}))
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    controller.run("run-1")
    assert controller.dispatch_log[0]["order"] == 1
    assert controller.commit_log[0]["order"] == 1
    payload_ref = persistence.connection.execute(
        "SELECT payload_ref FROM events WHERE run_id = ?", ("run-1",)
    ).fetchone()[0]
    event_path = persistence.object_store.root / payload_ref[:2] / payload_ref[2:]
    event = json.loads(event_path.read_text())
    assert event["dispatch_order"] == 1
    assert event["commit_order"] == 1
    persistence.close()


def test_schedule_effect_reactivates_completed_process() -> None:
    first = ResultExecutor(ProcessResult())
    scheduling = ResultExecutor(
        ProcessResult(scheduling_effects=({"type": "schedule", "process_id": "first", "phase": 1},))
    )
    controller = RunController(
        Scheduler(
            [
                {"id": "first", "context_policy": "private"},
                {"id": "schedule", "context_policy": "private", "after": ["first"]},
            ]
        ),
        ExecutorRegistry({"first": first, "schedule": scheduling}),
        ContextEngine({"private": {"allow": []}}),
    )
    assert controller.run("run-1", phase_limit=2) == ["first", "schedule", "first"]


def test_persistence_failure_restores_state_and_stops_dispatch(tmp_path: Path) -> None:
    class FailOncePersistence:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = 0

        def commit_process_result(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise OSError("disk busy")
            return self.delegate.commit_process_result(*args, **kwargs)

    store = StateStore({"score": int}, {"score": 0})
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "context_policy": "private",
                    "state_effects": ["score"],
                    "retry_policy": {"max_attempts": 2},
                }
            ]
        ),
        ExecutorRegistry({"p": ResultExecutor(ProcessResult(outputs={"score": 1}))}),
        ContextEngine({"private": {"allow": []}}),
        state_store=store,
        persistence=FailOncePersistence(persistence),
    )
    with pytest.raises(OSError, match="disk busy"):
        controller.run("run-1")
    assert store.snapshot() == {"score": 0}
    assert store.version == 0
    assert controller.results == []
    persistence.close()


def test_failed_result_is_not_double_recorded_when_failure_persistence_fails() -> None:
    class BrokenPersistence:
        def commit_process_result(self, *args, **kwargs):
            raise OSError("disk unavailable")

    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry(
            {"p": ResultExecutor(ProcessResult(status="failed", metadata={"error": "bad"}))}
        ),
        ContextEngine({"private": {"allow": []}}),
        persistence=BrokenPersistence(),
    )
    with pytest.raises(OSError):
        controller.run("run-1")
    assert len(controller.failures) == 1
    assert controller.failures[0]["error"] == "bad"


def test_executor_exception_is_persisted_as_failed_attempt(tmp_path: Path) -> None:
    class RaisingExecutor:
        def execute(self, invocation):
            raise RuntimeError("executor exploded")

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry({"p": RaisingExecutor()}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    with pytest.raises(RuntimeError, match="executor exploded"):
        controller.run("run-1")
    payload_ref = persistence.connection.execute(
        "SELECT payload_ref FROM events WHERE run_id = ?", ("run-1",)
    ).fetchone()[0]
    path = persistence.object_store.root / payload_ref[:2] / payload_ref[2:]
    event = json.loads(path.read_text())
    assert event["kind"] == "process_failed"
    assert event["classification"] == "executor_exception"
    persistence.close()


def test_invalid_scheduling_effect_does_not_commit_success(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry(
            {
                "p": ResultExecutor(
                    ProcessResult(
                        scheduling_effects=(
                            {"type": "schedule", "process_id": "missing", "phase": 1},
                        )
                    )
                )
            }
        ),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    with pytest.raises(ValueError, match="unknown scheduled process"):
        controller.run("run-1")
    assert persistence.count("events") == 1
    assert persistence.list_events("run-1")[0]["kind"] == "process_failed"
    assert persistence.list_events("run-1")[0]["classification"] == "invalid_output"
    persistence.close()


def test_persistence_failure_does_not_redispatch_executor(tmp_path: Path) -> None:
    calls = []

    class BrokenPersistence:
        def commit_process_result(self, *args, **kwargs):
            raise OSError("disk unavailable")

    executor = ResultExecutor(
        lambda invocation: calls.append(invocation.attempt) or ProcessResult()
    )
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "context_policy": "private",
                    "retry_policy": {"max_attempts": 2},
                }
            ]
        ),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=BrokenPersistence(),
    )
    with pytest.raises(OSError, match="disk unavailable"):
        controller.run("run-1")
    assert calls == [1]
    assert controller.results == []


def test_acyclic_delayed_dependency_does_not_bootstrap() -> None:
    scheduler = Scheduler(
        [
            {"id": "a", "phase": 5},
            {"id": "b", "after": ["a"], "delay": 1},
        ]
    )
    assert scheduler.ready(0) == []


def test_duplicate_process_ids_and_invalid_retry_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate process id"):
        Scheduler([{"id": "p"}, {"id": "p"}])
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_attempts"):
            Scheduler([{"id": "p", "retry_policy": {"max_attempts": invalid}}])


def test_repeated_event_occurrences_are_consumed_one_at_a_time() -> None:
    scheduler = Scheduler(
        [
            {
                "id": "consume",
                "repeat": True,
                "trigger": {"type": "event", "event": "message"},
            }
        ]
    )
    scheduler.signal_event("message")
    scheduler.signal_event("message")
    assert [item.process_id for item in scheduler.ready(0)] == ["consume"]
    scheduler.complete("consume", 0)
    assert [item.process_id for item in scheduler.ready(1)] == ["consume"]
    scheduler.complete("consume", 1)
    assert scheduler.ready(2) == []


def test_failed_controller_is_terminal() -> None:
    class RaisingExecutor:
        def execute(self, invocation):
            raise RuntimeError("boom")

    controller = RunController(
        Scheduler([{"id": "p", "context_policy": "private"}]),
        ExecutorRegistry({"p": RaisingExecutor()}),
        ContextEngine({"private": {"allow": []}}),
    )
    with pytest.raises(RuntimeError):
        controller.run("run-1")
    assert controller.run("run-1") == []


def test_new_controller_restores_persisted_frontier_and_state(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    process = {
        "id": "p",
        "repeat": True,
        "context_policy": "private",
        "state_effects": ["score"],
    }
    first = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": ResultExecutor(ProcessResult(outputs={"score": 1}))}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({"score": int}, {"score": 0}),
        persistence=persistence,
    )
    assert first.run("run-1", phase_limit=1) == ["p"]

    second_store = StateStore({"score": int}, {"score": 0})
    second = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": ResultExecutor(ProcessResult(outputs={"score": 2}))}),
        ContextEngine({"private": {"allow": []}}),
        state_store=second_store,
        persistence=persistence,
    )
    assert second.run("run-1", phase_limit=2) == ["p"]
    assert second.dispatch_log[-1]["phase"] == 1
    assert second.dispatch_log[-1]["order"] == 2
    assert second_store.snapshot() == {"score": 2}
    assert [event["phase"] for event in persistence.list_events("run-1")] == [0, 1]
    persistence.close()


def test_recovery_finishes_uncommitted_work_in_last_phase(tmp_path: Path) -> None:
    class FailSecondCommit:
        def __init__(self, delegate):
            self.delegate = delegate
            self.calls = 0

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def commit_process_result(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise OSError("disk unavailable")
            return self.delegate.commit_process_result(*args, **kwargs)

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    processes = [
        {"id": "a", "context_policy": "private"},
        {"id": "b", "context_policy": "private"},
    ]
    first = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {"a": ResultExecutor(ProcessResult()), "b": ResultExecutor(ProcessResult())}
        ),
        ContextEngine({"private": {"allow": []}}),
        persistence=FailSecondCommit(persistence),
    )
    with pytest.raises(OSError):
        first.run("run-1", phase_limit=1)

    a, b = ResultExecutor(ProcessResult()), ResultExecutor(ProcessResult())
    recovered = RunController(
        Scheduler(processes),
        ExecutorRegistry({"a": a, "b": b}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    assert recovered.run("run-1", phase_limit=1) == ["b"]
    assert a.calls == []
    assert len(b.calls) == 1
    persistence.close()


def test_skipped_result_is_durable_and_not_redispatched(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    process = {"id": "p", "context_policy": "private"}
    first = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": ResultExecutor(ProcessResult(status="skipped"))}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    assert first.run("run-1", phase_limit=1) == ["p"]
    executor = ResultExecutor(ProcessResult())
    recovered = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    assert recovered.run("run-1", phase_limit=1) == []
    assert executor.calls == []
    assert persistence.list_events("run-1")[0]["kind"] == "process_skipped"
    persistence.close()


def test_invalid_output_obeys_retry_policy_and_persists_each_failure(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    executor = ResultExecutor(
        ProcessResult(
            scheduling_effects=({"type": "schedule", "process_id": "missing", "phase": 1},)
        )
    )
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "p",
                    "context_policy": "private",
                    "retry_policy": {"max_attempts": 2},
                }
            ]
        ),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    with pytest.raises(ValueError):
        controller.run("run-1")
    assert len(executor.calls) == 2
    assert [event["attempt"] for event in persistence.list_events("run-1")] == [1, 2]
    persistence.close()


def test_recovery_checks_existing_event_budget_before_dispatch(tmp_path: Path) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    process = {"id": "p", "repeat": True, "context_policy": "private"}
    first = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": ResultExecutor(ProcessResult())}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    first.run("run-1", phase_limit=1)
    executor = ResultExecutor(ProcessResult())
    recovered = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": executor}),
        ContextEngine({"private": {"allow": []}}),
        persistence=persistence,
    )
    assert recovered.run("run-1", max_events=1) == []
    assert executor.calls == []
    assert persistence.count("events") == 1
    persistence.close()


def test_state_version_coordinate_matches_after_failed_attempt_and_recovery(
    tmp_path: Path,
) -> None:
    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    seen: list[int] = []

    def result(invocation):
        seen.append(invocation.state_version)
        if invocation.attempt == 1:
            return ProcessResult(status="failed", metadata={"error": "retry"})
        return ProcessResult()

    process = {
        "id": "p",
        "context_policy": "private",
        "retry_policy": {"max_attempts": 2},
    }
    controller = RunController(
        Scheduler([process]),
        ExecutorRegistry({"p": ResultExecutor(result)}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({}, {}),
        persistence=persistence,
    )
    controller.run("run-1", phase_limit=1)
    assert seen == [0, 1]
    assert controller.state_store.version == 2

    recovered = RunController(
        Scheduler([{"id": "q", "context_policy": "private"}]),
        ExecutorRegistry({"q": ResultExecutor(ProcessResult())}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({}, {}),
        persistence=persistence,
    )
    recovered.run("run-1", phase_limit=2)
    assert recovered.dispatch_log[-1]["process_id"] == "q"
    assert recovered.state_store.version == 3
    persistence.close()

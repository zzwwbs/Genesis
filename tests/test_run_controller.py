import pytest

from genesis.runtime import (
    ContextEngine,
    DeterministicExecutor,
    ExecutorRegistry,
    RunController,
    Scheduler,
    StateStore,
)


def controller(processes, fn, **kwargs):
    return RunController(
        Scheduler(processes),
        ExecutorRegistry({"p": DeterministicExecutor(fn)}),
        ContextEngine({"private": {"allow": []}}),
        **kwargs,
    )


def test_run_controller_lifecycle_captures_results_and_applies_effects():
    store = StateStore({"score": int}, {"score": 0})
    c = controller(
        [{"id": "p", "context_policy": "private", "state_effects": ["score"]}],
        lambda call: {"score": 4},
        state_store=store,
    )
    assert c.status == "created"
    assert c.run("run-1") == ["p"]
    assert c.status == "completed"
    assert c.results[0].outputs == {"score": 4}


def test_conditions_and_replications_expand_deterministically():
    c = controller([], lambda call: {})
    expanded = c.expand_runs("exp", [{"id": "base"}, {"id": "alt"}], 2)
    assert [x["run_id"] for x in expanded] == ["exp-base-1", "exp-base-2", "exp-alt-1", "exp-alt-2"]


def test_retry_policy_records_failures_and_bounded_attempts():
    attempts = []

    def fail(call):
        attempts.append(call.attempt)
        raise RuntimeError("boom")

    c = controller(
        [{"id": "p", "context_policy": "private", "retry_policy": {"max_attempts": 2}}], fail
    )
    with pytest.raises(RuntimeError):
        c.run("run-1")
    assert c.status == "failed"
    assert attempts == [1, 2]
    assert len(c.failures) == 2


def test_failed_process_results_follow_retry_policy() -> None:
    attempts: list[int] = []

    def fail_result(call):
        attempts.append(call.attempt)
        from genesis.runtime import ProcessResult

        return ProcessResult(status="failed", metadata={"code": "transient_provider"})

    class FailingExecutor:
        def execute(self, call):
            return fail_result(call)

    c = RunController(
        Scheduler([{"id": "p", "context_policy": "private", "retry_policy": {"max_attempts": 2}}]),
        ExecutorRegistry({"p": FailingExecutor()}),
        ContextEngine({"private": {"allow": []}}),
    )

    with pytest.raises(RuntimeError, match="process p failed"):
        c.run("run-1")

    assert attempts == [1, 2]
    assert c.status == "failed"


def test_pause_resume_cancel_hooks_and_limits():
    c = controller([{"id": "p", "context_policy": "private"}], lambda call: {})
    c.pause()
    assert c.status == "paused"
    assert c.run("run-1") == []
    c.resume()
    assert c.status == "running"
    c.cancel()
    assert c.status == "cancelled"
    assert c.run("run-1") == []
    limited = controller(
        [{"id": "p", "context_policy": "private", "repeat": True}], lambda call: {}
    )
    assert limited.run("run-2", phase_limit=10, max_events=3) == ["p", "p", "p"]


def test_optional_persistence_receives_committed_process_result(tmp_path):
    from genesis.persistence import PersistenceCoordinator

    persistence = PersistenceCoordinator(tmp_path / "g.db", tmp_path / "objects")
    store = StateStore({"score": int}, {"score": 0})
    c = controller(
        [{"id": "p", "context_policy": "private"}],
        lambda call: {"x": 1},
        state_store=store,
        persistence=persistence,
    )
    c.run("run-1")
    assert persistence.count("events") == 1


def test_run_controller_accepts_untyped_initial_state() -> None:
    seen: list[dict] = []

    def execute(call):
        seen.append(dict(call.context.data))
        return {"counter": 1}

    c = RunController(
        Scheduler([{"id": "p", "context_policy": "private", "state_effects": ["counter"]}]),
        ExecutorRegistry({"p": DeterministicExecutor(execute)}),
        ContextEngine({"private": {"allow": ["counter"]}}),
    )

    c.run("run-1", state={"counter": 0})

    assert seen == [{"counter": 0}]


def test_context_uses_committed_state_for_later_processes() -> None:
    seen: list[int] = []

    def first(call):
        return {"counter": 2}

    def second(call):
        seen.append(call.context.data["counter"])
        return {}

    scheduler = Scheduler(
        [
            {"id": "first", "context_policy": "private", "state_effects": ["counter"]},
            {"id": "second", "context_policy": "private", "dependencies": {"after": ["first"]}},
        ]
    )
    registry = ExecutorRegistry(
        {"first": DeterministicExecutor(first), "second": DeterministicExecutor(second)}
    )
    c = RunController(scheduler, registry, ContextEngine({"private": {"allow": ["counter"]}}))

    c.run("run-1", state={"counter": 0})

    assert seen == [2]


def test_preflight_reports_estimates_limits_and_dependency_availability() -> None:
    c = controller(
        [
            {"id": "p", "context_policy": "private", "executor": {"mode": "generative"}},
            {"id": "q", "context_policy": "private", "dependencies": {"after": ["p"]}},
        ],
        lambda call: {},
    )

    report = c.preflight(phase_limit=5, replications=3, conditions=2, max_events=10)

    assert report["phase_limit"] == 5
    assert report["estimated_runs"] == 6
    assert report["estimated_process_invocations"] == 12
    assert report["estimated_generative_calls"] == 3
    assert report["limits"]["max_events"] == 10
    assert report["dependencies_available"] is True


def test_mid_run_cancellation_stops_dispatch(tmp_path) -> None:
    """AW-18: cancelling a running controller halts further dispatch."""
    import threading
    import time

    from genesis.persistence import PersistenceCoordinator

    persistence = PersistenceCoordinator(tmp_path / "genesis.db", tmp_path / "objects")
    persistence.create_run({"id": "run-c", "build": "b"})
    processes = [
        {
            "id": "tick",
            "executor": {},
            "context_policy": "public",
            "trigger": {"type": "phase", "phase": 0, "repeat": True},
        }
    ]
    from genesis.runtime import ContextEngine, ExecutorRegistry, RunController, Scheduler

    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {
                "tick": type(
                    "E",
                    (),
                    {
                        "execute": lambda self, inv: __import__(
                            "genesis.runtime", fromlist=["ProcessResult"]
                        ).ProcessResult(outputs={"n": 1}),
                    },
                )()
            }
        ),
        ContextEngine({"public": {"allow": []}}),
        persistence=persistence,
    )

    def drive() -> None:
        try:
            controller.run("run-c", phase_limit=100)
        except Exception:
            pass

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    # Cancel after the first commit lands.
    deadline = time.time() + 10
    while len(persistence.list_events("run-c")) < 1 and time.time() < deadline:
        time.sleep(0.01)
    controller.cancel()
    thread.join(timeout=10)
    assert controller.status == "cancelled"
    events_before = len(persistence.list_events("run-c"))
    time.sleep(0.2)
    assert len(persistence.list_events("run-c")) == events_before
    persistence.close()


def test_service_level_cancellation_stops_active_run(tmp_path) -> None:
    """Finding 2: cancelling through the service/API stops dispatch mid-run."""
    import threading
    import time

    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        draft = service.create_specification(
            {
                "id": "cancel-study",
                "title": "cancel",
                "processes": [
                    {
                        "id": "tick",
                        "executor": {},
                        "context_policy": "public",
                        "state_effects": [{"field": "counter", "op": "set"}],
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {"states": [{"id": "counter", "value_type": "integer", "initial": 0}]},
                "protocol": {"time_model": {"type": "rounds", "end": 60}},
                "outcomes": [],
                "models": [],
            }
        )
        service.approve_specification("cancel-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/cancel-study", specification_id="cancel-study"
        )
        service.create_run(
            {"id": "cancel-run", "study_id": "cancel-study", "build": compiled["path"]}
        )
        outcome: dict[str, object] = {}

        def drive() -> None:
            try:
                outcome["result"] = service.execute_run("cancel-run")
            except Exception as exc:  # pragma: no cover - failure path
                outcome["error"] = str(exc)

        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        deadline = time.time() + 15
        while len(service.trace_run("cancel-run")) < 2 and time.time() < deadline:
            time.sleep(0.01)
        run = service.get_run("cancel-run")
        # Cancel through the same transition the API uses.
        cancelled = service.transition_run("cancel-run", "cancelled", run["version"])
        assert cancelled["status"] == "cancelled"
        thread.join(timeout=15)
        assert not thread.is_alive()
        events_before = len(service.trace_run("cancel-run"))
        time.sleep(0.2)
        assert len(service.trace_run("cancel-run")) == events_before
        final = service.get_run("cancel-run")
        assert final["status"] == "cancelled", final
    finally:
        service.close()

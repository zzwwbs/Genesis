import pytest

from genesis.runtime import (
    ContextEngine,
    DeterministicExecutor,
    ExecutorRegistry,
    GenerativeExecutor,
    ProcessInvocation,
    RecordedArtifactExecutor,
    RunController,
    Scheduler,
    StateStore,
    StochasticExecutor,
    derive_seed,
)


def invocation(**kwargs):
    values = dict(
        invocation_id="i1",
        run_id="r1",
        process_id="p1",
        actor_ids=("a1",),
        phase=0,
        time=0,
        state_version=0,
        inputs={},
        context=None,
        seed=7,
    )
    values.update(kwargs)
    return ProcessInvocation(**values)


def test_process_contracts_are_immutable_and_validate_ids():
    call = invocation()
    assert call.invocation_id == "i1"
    with pytest.raises((AttributeError, TypeError)):
        call.run_id = "other"
    with pytest.raises(ValueError):
        invocation(invocation_id="Bad ID")


def test_context_engine_is_deny_by_default_and_returns_hashed_immutable_envelope():
    engine = ContextEngine({"private": {"allow": ("actors.a1.profile",)}})
    envelope = engine.build(
        "private", invocation(), {"actors": {"a1": {"profile": {"x": 1}, "secret": 2}}}
    )
    assert envelope.data == {"actors": {"a1": {"profile": {"x": 1}}}}
    assert envelope.content_hash
    with pytest.raises(PermissionError):
        engine.build("missing", invocation(), {})
    with pytest.raises(TypeError):
        envelope.data["x"] = 1


def test_executor_registry_supports_modes_and_recorded_outputs():
    registry = ExecutorRegistry()
    registry.register("det", DeterministicExecutor(lambda call: {"value": call.seed}))
    registry.register("stoch", StochasticExecutor(lambda call, rng: {"value": rng.randrange(100)}))
    registry.register("gen", GenerativeExecutor(lambda call: {"text": "hello"}))
    registry.register("rec", RecordedArtifactExecutor({"text": "saved"}))
    assert registry.execute("det", invocation()).outputs == {"value": 7}
    first = registry.execute("stoch", invocation(seed=11)).outputs
    second = registry.execute("stoch", invocation(seed=11)).outputs
    assert first == second
    assert registry.execute("gen", invocation()).outputs == {"text": "hello"}
    assert registry.execute("rec", invocation()).outputs == {"text": "saved"}


def test_state_store_rejects_undeclared_effects_and_applies_declared_effects_atomically():
    store = StateStore({"score": int, "status": str})
    assert store.apply({"score": 2}, declared={"score"}) == 1
    with pytest.raises(PermissionError):
        store.apply({"secret": 1}, declared={"secret"})
    with pytest.raises(TypeError):
        store.apply({"score": "bad"}, declared={"score"})
    assert store.snapshot() == {"score": 2}


def test_scheduler_orders_phases_and_honors_delays_deterministically():
    scheduler = Scheduler(
        [
            {"id": "late", "phase": 1, "after": ["early"], "delay": 1},
            {"id": "early", "phase": 0},
        ]
    )
    assert [x.process_id for x in scheduler.ready(0)] == ["early"]
    scheduler.complete("early", 0)
    assert scheduler.ready(0) == []
    assert [x.process_id for x in scheduler.ready(1)] == ["late"]


def test_named_seed_derivation_is_stable_and_run_controller_executes():
    assert derive_seed(42, "run-1", "p", "a1") == derive_seed(42, "run-1", "p", "a1")
    calls = []
    controller = RunController(
        Scheduler([{"id": "p", "phase": 0}]),
        ExecutorRegistry(
            {"p": DeterministicExecutor(lambda call: calls.append(call.process_id) or {"ok": True})}
        ),
        ContextEngine({"private": {"allow": ()}}),
    )
    assert controller.run("run-1", phase_limit=1) == ["p"]
    assert calls == ["p"]

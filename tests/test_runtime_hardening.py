import pytest

from genesis.runtime import (
    ArtifactStore,
    ContextEngine,
    ExecutorRegistry,
    GenerativeExecutor,
    ProcessInvocation,
    ProcessResult,
    Scheduler,
    StateStore,
    derive_seed,
)


def call():
    return ProcessInvocation(
        "i1",
        "run-1",
        "proc",
        context={"nested": {"x": 1}},
        executor_binding={"model": {"temperature": 1}},
    )


def test_contracts_deep_freeze_context_binding_events_and_schedule():
    invocation = call()
    with pytest.raises(TypeError):
        invocation.context["nested"]["x"] = 2
    with pytest.raises(TypeError):
        invocation.executor_binding["model"]["temperature"] = 0
    result = ProcessResult(
        events=({"payload": {"x": 1}},), scheduling_effects=({"delay": {"rounds": 1}},)
    )
    with pytest.raises(TypeError):
        result.events[0]["payload"]["x"] = 2
    with pytest.raises(TypeError):
        result.scheduling_effects[0]["delay"]["rounds"] = 2


def test_context_policy_projects_visibility_availability_and_redaction():
    engine = ContextEngine(
        {
            "p": {
                "allow": ["actors.a.profile", "secret"],
                "visibility": {"secret": "private"},
                "availability": {"actors.a.profile": False},
                "redact": ["secret"],
            }
        }
    )
    envelope = engine.build("p", call(), {"actors": {"a": {"profile": 1}}, "secret": "s"})
    assert envelope.data == {}
    with pytest.raises(ValueError):
        ContextEngine({"bad": {"allow": ["../secret"]}})
    with pytest.raises(ValueError):
        ContextEngine({"bad": {"allow": [123]}})


def test_state_store_validates_initial_and_versioned_set_increment_append_effects():
    with pytest.raises(TypeError):
        StateStore({"score": int}, {"score": "bad"})
    store = StateStore({"score": int, "tags": list}, {"score": 1, "tags": []})
    assert (
        store.apply(
            [{"op": "increment", "field": "score", "value": 2}], {"score"}, expected_version=0
        )
        == 1
    )
    assert (
        store.apply([{"op": "append", "field": "tags", "value": "x"}], {"tags"}, expected_version=1)
        == 2
    )
    with pytest.raises(ValueError, match="expected"):
        store.apply({"score": 4}, {"score"}, expected_version=0)
    assert store.snapshot() == {"score": 3, "tags": ["x"]}


def test_state_store_supports_remove_and_declared_custom_reducers():
    store = StateStore(
        {"score": int, "tags": list, "active": bool},
        {"score": 2, "tags": ["a"], "active": True},
        reducers={"double": lambda current, value: current * value},
    )
    store.apply(
        [
            {"field": "tags", "op": "remove", "value": "a"},
            {"field": "score", "op": "custom", "reducer": "double", "value": 3},
            {"field": "active", "op": "set", "value": False, "expected_version": 0},
        ],
        {"score", "tags", "active"},
    )
    assert store.snapshot() == {"score": 6, "tags": [], "active": False}


def test_artifact_store_enforces_metadata_schema_owner_visibility_lineage_and_immutability():
    store = ArtifactStore(
        {
            "a": {
                "schema_ref": "s",
                "owner": "actor",
                "visibility": "private",
                "lifecycle_scope": "run",
            }
        }
    )
    store.put("a", {"x": 1}, owner="actor", schema_ref="s")
    with pytest.raises(ValueError, match="immutable"):
        store.put("a", {"x": 2}, owner="actor", schema_ref="s")
    with pytest.raises(PermissionError):
        store.put("b", {"x": 1})
    assert store.metadata("a")["lineage"] == []


def test_additional_executor_modes_and_generative_metadata_hook():
    seen = []
    registry = ExecutorRegistry()
    registry.register("comp", lambda c: {"x": 1}, mode="computational")
    registry.register("ext", lambda c: {"x": 2}, mode="extension")
    registry.register(
        "gen", GenerativeExecutor(lambda c: {"x": 3}, metadata_hook=lambda m: seen.append(m))
    )
    assert registry.execute("comp", call()).outputs == {"x": 1}
    assert registry.execute("ext", call()).outputs == {"x": 2}
    registry.execute("gen", call())
    assert seen


def test_seed_includes_experiment_condition_replication_and_scheduler_repeats():
    assert derive_seed(
        1, "run", "p", experiment_id="e1", condition_id="c1", replication=1
    ) != derive_seed(1, "run", "p", experiment_id="e1", condition_id="c1", replication=2)
    scheduler = Scheduler(
        [{"id": "tick", "trigger": {"type": "phase", "phase": 0, "repeat": True}}]
    )
    assert [x.process_id for x in scheduler.ready(0)] == ["tick"]
    scheduler.complete("tick", 0)
    assert [x.process_id for x in scheduler.ready(1)] == ["tick"]
    with pytest.raises(ValueError):
        Scheduler([{"id": "a", "after": ["missing"]}])

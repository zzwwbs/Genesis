"""Regressions for the September code-review findings.

Each test pins one behaviour that was previously wrong in a way no existing
test detected. They are grouped by the layer they protect: scheduling and
theory semantics, context authorization, state contracts, replay integrity,
and cross-realization reporting.
"""

from __future__ import annotations

import pytest

from genesis.runtime import (
    ArtifactStore,
    CallableExecutor,
    ContextEngine,
    ExecutorRegistry,
    ProcessInvocation,
    RunController,
    Scheduler,
    StateStore,
    edge_delays,
)

# ---------------------------------------------------------------------------
# Scheduling: within-round ordering and per-edge delay
# ---------------------------------------------------------------------------


def _chain_controller(processes, executors, policies=None, state=None, catalog=None):
    return RunController(
        Scheduler(processes),
        ExecutorRegistry(executors),
        ContextEngine(policies or {"n": {"id": "n", "allow": []}}),
        state_store=StateStore(*(state or ({}, {}))),
        artifact_store=ArtifactStore(catalog or {}),
    )


def test_repeating_chain_keeps_declared_order_in_every_round() -> None:
    """A repeating ``after`` chain must run in order each round, not only the first.

    Previously a dependency counted as satisfied once it had completed in ANY
    earlier phase, so from round 2 the whole chain became ready at once and
    executed in process-id order — reverse order here — with each consumer
    reading the previous round's artifacts.
    """
    order: list[tuple[int, str]] = []

    def make(name):
        def run(invocation):
            order.append((int(invocation.phase), name))
            return {}

        return run

    ids = ["zeta-write", "beta-detect", "alpha-read"]
    processes = [
        {
            "id": "zeta-write",
            "context_policy": "n",
            "executor": {"mode": "computational"},
            "trigger": {"phase": 0, "repeat": True},
        },
        {
            "id": "beta-detect",
            "context_policy": "n",
            "executor": {"mode": "computational"},
            "trigger": {"phase": 0, "repeat": True},
            "dependencies": {"after": ["zeta-write"]},
        },
        {
            "id": "alpha-read",
            "context_policy": "n",
            "executor": {"mode": "computational"},
            "trigger": {"phase": 0, "repeat": True},
            "dependencies": {"after": ["beta-detect"]},
        },
    ]
    controller = _chain_controller(
        processes, {pid: CallableExecutor(make(pid), "computational") for pid in ids}
    )
    controller.run("ordering-run", phase_limit=4)

    for phase in range(4):
        assert [name for p, name in order if p == phase] == ids


def test_delay_applies_per_edge_not_per_process() -> None:
    """A lag on one dependency must not delay a process's other dependencies."""
    resolved = edge_delays(
        {"after": ["fast", "slow"], "delay": {"per_dependency": {"slow": 3}}},
        ["fast", "slow"],
    )
    assert resolved == {"fast": 0, "slow": 3}

    # The legacy process-wide form still applies to every edge.
    assert edge_delays({"after": ["a", "b"], "delay": {"rounds": 2}}, ["a", "b"]) == {
        "a": 2,
        "b": 2,
    }


def test_zero_lag_cycle_is_rejected_even_when_another_edge_is_delayed() -> None:
    """A delay on one edge must not exempt the process's immediate edges.

    ``b`` follows ``c`` with a lag and ``a`` immediately; ``a`` follows ``b``.
    The a<->b cycle is immediate and must be rejected.
    """
    with pytest.raises(ValueError, match="immediate dependency cycle"):
        Scheduler(
            [
                {"id": "c"},
                {"id": "a", "dependencies": {"after": ["b"]}},
                {
                    "id": "b",
                    "dependencies": {
                        "after": ["a", "c"],
                        "delay": {"per_dependency": {"c": 1, "a": 0}},
                    },
                },
            ]
        )


def test_dependency_delay_rejects_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="names non-dependencies"):
        Scheduler(
            [
                {"id": "a"},
                {
                    "id": "b",
                    "dependencies": {"after": ["a"], "delay": {"per_dependency": {"ghost": 1}}},
                },
            ]
        )


# ---------------------------------------------------------------------------
# Theory layer: lag floor and feedback delivery
# ---------------------------------------------------------------------------


def _feedback_controller(probe, policy_allow=("feedback.last",)):
    processes = [
        {
            "id": "c",
            "context_policy": "p",
            "executor": {"mode": "computational"},
            "theory_feedback": [
                {
                    "source": "counter",
                    "context_slot": "last",
                    "lag_rounds": 1,
                    "initial": {"policy": "skip_consumer"},
                }
            ],
        }
    ]
    return RunController(
        Scheduler(processes),
        ExecutorRegistry({"c": CallableExecutor(probe, "computational")}),
        ContextEngine({"p": {"id": "p", "allow": list(policy_allow)}}),
        state_store=StateStore({"counter": int}, {"counter": 0}),
    )


@pytest.mark.parametrize("phase_start", [0, 1])
def test_lagged_feedback_waits_for_a_round_that_actually_ran(phase_start: int) -> None:
    """The lag floor is the protocol's first phase, not absolute phase 0.

    With ``time_model.start: 1`` there is no round 0, so a lag-1 consumer must
    not fire in the first round. It previously did, serving the pre-run
    snapshot as if a round had completed.
    """
    seen: list[int] = []

    def probe(invocation):
        seen.append(int(invocation.phase))
        return {}

    _feedback_controller(probe).run("lag-run", phase_start=phase_start, phase_end=phase_start + 2)
    assert seen == [phase_start + 1]


def test_feedback_slots_reach_the_executor() -> None:
    """The invocation handed to the executor carries its feedback slots.

    The context envelope always did; the invocation itself lost them in the
    final rebuild, so any executor reading ``invocation.feedback_slots``
    silently saw nothing.
    """
    captured: list[dict] = []

    def probe(invocation):
        captured.append(dict(invocation.feedback_slots or {}))
        return {}

    processes = [
        {
            "id": "c",
            "context_policy": "p",
            "executor": {"mode": "computational"},
            "theory_feedback": [
                {
                    "source": "counter",
                    "context_slot": "last",
                    "lag_rounds": 1,
                    "initial": {"policy": "declared_default", "value": {"seed": True}},
                }
            ],
        }
    ]
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry({"c": CallableExecutor(probe, "computational")}),
        ContextEngine({"p": {"id": "p", "allow": ["feedback.last"]}}),
        state_store=StateStore({"counter": int}, {"counter": 0}),
    )
    controller.run("feedback-run", phase_limit=1)

    assert captured and captured[0] == {"last": {"seed": True}}


# ---------------------------------------------------------------------------
# Context authorization
# ---------------------------------------------------------------------------


def test_bounded_executors_are_bound_by_the_context_policy() -> None:
    """A rule/state-transition process sees only policy-authorized inputs.

    Declarative executors read the invocation namespace; that namespace used
    the raw inputs, so a bounded process could read artifacts its policy never
    admitted.
    """
    seen: list[list[str]] = []

    def produce(_invocation):
        return {"secret": {"v": 1}}

    def peek(invocation):
        from genesis.runtime import _invocation_namespace

        seen.append(sorted(_invocation_namespace(invocation)["artifacts"]))
        return {}

    processes = [
        {
            "id": "p",
            "context_policy": "open",
            "executor": {"mode": "computational"},
            "outputs": [{"artifact_type": "secret"}],
        },
        {
            "id": "q",
            "context_policy": "blind",
            "executor": {"mode": "rule"},
            "inputs": ["secret"],
            "dependencies": {"after": ["p"]},
        },
    ]
    controller = RunController(
        Scheduler(processes),
        ExecutorRegistry(
            {
                "p": CallableExecutor(produce, "computational"),
                "q": CallableExecutor(peek, "computational"),
            }
        ),
        ContextEngine(
            {
                "open": {"id": "open", "allow": ["inputs"]},
                "blind": {"id": "blind", "allow": []},
            }
        ),
        state_store=StateStore({}, {}),
        artifact_store=ArtifactStore({"secret": {"id": "secret"}}),
    )
    controller.run("policy-run", phase_limit=1)

    assert seen == [[]]


def test_round_scoped_artifacts_do_not_accumulate_across_rounds() -> None:
    """``lifecycle_scope`` bounds how long an instance stays resolvable.

    Without it a round-scoped artifact accumulated every prior instance, so by
    round N a consumer's authorized context carried all N-1 earlier rounds.
    """
    store = ArtifactStore({"art": {"id": "art", "lifecycle_scope": "round"}})
    for phase in range(3):
        store.put("art", {"round": phase}, instance_id=f"art-{phase}", phase=phase)

    assert len(store.resolve(["art"], phase=2)) == 1
    assert list(store.resolve(["art"], phase=2).values())[0]["value"] == {"round": 2}
    # A run-scoped artifact still persists for the whole run.
    run_store = ArtifactStore({"art": {"id": "art", "lifecycle_scope": "run"}})
    for phase in range(3):
        run_store.put("art", {"round": phase}, instance_id=f"art-{phase}", phase=phase)
    assert len(run_store.resolve(["art"], phase=2)) == 3


# ---------------------------------------------------------------------------
# State contract
# ---------------------------------------------------------------------------


def test_number_state_accepts_integers_and_rejects_booleans() -> None:
    """JSON has one numeric type; ``number`` must admit ``0``.

    A field declared ``number`` with an integer initial value previously failed
    at run start. ``bool`` subclasses ``int``, so it must still be rejected.
    """
    store = StateStore({"revenue": (int, float)}, {"revenue": 0})
    assert store.apply({"revenue": 5}, {"revenue"}) == 1
    with pytest.raises(TypeError):
        store.apply({"revenue": True}, {"revenue"})


def test_unknown_state_value_type_is_rejected_at_compilation(tmp_path) -> None:
    """An unrecognized ``value_type`` silently disabled the type contract."""
    from genesis.compiler import StudyCompiler, ValidationIssue

    source = tmp_path / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "bad-type-study"}
    files = {
        "study": {**base, "title": "bad type"},
        "openness": {**base, "processes": []},
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "states": [{"id": "counter", "value_type": "numeric", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "end": 1}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    import yaml

    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises((ValidationIssue, ValueError), match="STATE_VALUE_TYPE|value_type"):
        StudyCompiler(source).compile(tmp_path / "build")


# ---------------------------------------------------------------------------
# Replay integrity
# ---------------------------------------------------------------------------


def test_recorded_executor_refuses_to_substitute_an_unrelated_recording() -> None:
    """An invocation the source never performed has no faithful recording.

    It previously received the LAST recorded output — another actor's, from
    another round — stamped ``recorded: True``.
    """
    from genesis.service import _RecordedExecutor

    records = [
        {"phase": 0, "attempt": 1, "actors": ("w1",), "outputs": {"a": "r0"}, "order": 0},
        {"phase": 1, "attempt": 1, "actors": ("w2",), "outputs": {"a": "r1"}, "order": 1},
    ]
    executor = _RecordedExecutor(records, source_run_id="src", process_id="write")

    assert executor.execute(
        ProcessInvocation("i", "r", "write", actor_ids=("w1",), phase=0)
    ).outputs == {"a": "r0"}

    with pytest.raises(ValueError, match="REPLAY_RECORD_MISSING"):
        executor.execute(ProcessInvocation("i", "r", "write", actor_ids=("w9",), phase=7))


def test_selective_executor_refuses_a_live_call_inside_the_frozen_prefix() -> None:
    """A divergence inside the frozen prefix is an error, not a fresh generation."""
    from genesis.service import _SelectiveExecutor

    class _Live:
        def execute(self, _invocation):
            raise AssertionError("the frozen prefix must not reach a live executor")

    executor = _SelectiveExecutor(
        [{"phase": 0, "attempt": 1, "actors": ("w1",), "outputs": {"a": "r0"}, "order": 0}],
        frozen_keys={(0, 1, ("w1",))},
        fallback=_Live(),
        source_run_id="src",
        process_id="write",
        phase_boundary=2,
    )

    assert executor.execute(
        ProcessInvocation("i", "r", "write", actor_ids=("w1",), phase=0)
    ).outputs == {"a": "r0"}

    # Phase 1 is inside the prefix (phase < 2) but was never recorded.
    with pytest.raises(ValueError, match="REPLAY_PREFIX_DIVERGED"):
        executor.execute(ProcessInvocation("i", "r", "write", actor_ids=("w1",), phase=1))


def test_manifest_seeds_must_derive_from_the_recorded_identity() -> None:
    """A rekeyed manifest cannot claim seeds it cannot re-derive."""
    from genesis.service import GenesisService

    randomness = {
        "run_id": "source-run",
        "experiment_id": "",
        "condition_id": "base",
        "replication": 1,
    }
    streams = [{"id": "demo", "seed": 55001}]
    seeds = GenesisService.derive_manifest_seeds(randomness, streams)

    consistent = {"randomness_inputs": randomness, "random_streams": streams, "seeds": seeds}
    GenesisService.verify_manifest_seeds(consistent)  # does not raise

    rekeyed = {**consistent, "randomness_inputs": {**randomness, "run_id": "renamed-run"}}
    with pytest.raises(ValueError, match="MANIFEST_SEED_MISMATCH"):
        GenesisService.verify_manifest_seeds(rekeyed)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_multi_key_grouping_labels_rows_with_declared_field_names() -> None:
    """Positional ``group_N`` labels alone hid which condition a value came from."""
    from genesis.analysis import AnalysisEngine, OutcomePlan

    plan = OutcomePlan(
        id="rate",
        source="rows",
        select="value",
        aggregation="mean",
        group_by=("condition_id", "phase"),
    )
    rows = [
        {"condition_id": "a", "phase": 1, "value": 1.0},
        {"condition_id": "b", "phase": 1, "value": 3.0},
    ]
    result = AnalysisEngine().evaluate(plan, {"rows": rows})

    by_condition = {row["condition_id"]: row for row in result}
    assert set(by_condition) == {"a", "b"}
    assert by_condition["a"]["phase"] == 1
    assert by_condition["a"]["value_mean"] == 1.0
    # Positional labels are retained for existing consumers.
    assert by_condition["b"]["group_0"] == "b"


# ---------------------------------------------------------------------------
# Second review round: engine parity, provider accounting, API error class
# ---------------------------------------------------------------------------


def _duck_service():
    from genesis.service import GenesisService

    service = GenesisService.__new__(GenesisService)
    service.last_outcome_engine = "python"
    return service


@pytest.mark.parametrize(
    "plan_kwargs",
    [
        {"group_by": ("condition_id", "phase")},
        {"group_by": "condition_id", "missingness": "zero"},
        {"group_by": "condition_id", "filters": ({"condition_id": "a"},)},
    ],
    ids=["multi-key-grouping", "missingness-zero", "declared-filters"],
)
def test_duckdb_path_declines_plans_it_cannot_evaluate_faithfully(monkeypatch, plan_kwargs) -> None:
    """The SQL engine must never silently change a declared measurement.

    It groups by a single column, ignores the missingness policy and never saw
    the declared filters, so a plan using any of those produced a different
    number depending only on ``GENESIS_USE_DUCKDB``.
    """
    from genesis.analysis import OutcomePlan

    monkeypatch.setenv("GENESIS_USE_DUCKDB", "1")
    rows = [
        {"condition_id": "a", "phase": 1, "v": 1.0},
        {"condition_id": "b", "phase": 1, "v": 9.0},
    ]
    plan = OutcomePlan(id="o", source="rows", select="v", aggregation="mean", **plan_kwargs)

    assert _duck_service()._duckdb_outcome_rows(plan, rows) is None


def test_duckdb_grouping_key_keeps_its_native_type() -> None:
    """``phase: 1`` must not become ``phase: "1"`` because DuckDB ran."""
    from genesis.analysis import duckdb_aggregate

    rows = [{"phase": 1, "v": 2.0}, {"phase": 1, "v": 4.0}]
    result = duckdb_aggregate(rows, select="v", op="mean", group_by="phase")

    assert result == [{"phase": 1, "v_mean": 3.0, "v_missing": 0}]


def test_provider_usage_and_cost_cover_every_attempt() -> None:
    """A repaired invocation really spent the failed attempts' tokens."""
    from genesis.providers import ProviderExecutor, ProviderResponse

    class _Flaky:
        provider = "test"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            parsed = {"ok": True} if self.calls >= 3 else {"wrong": 1}
            return ProviderResponse(
                text=str(parsed),
                parsed=parsed,
                provider="test",
                model=request.model,
                request_id=f"req-{self.calls}",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                metadata={},
            )

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    executor = ProviderExecutor(
        _Flaky(),
        model="m",
        prompt_template="P:{context}",
        parameters={"price_per_1k_input": 1.0, "price_per_1k_output": 2.0},
        output_schema=schema,
        output_key="thing",
        max_repairs=3,
    )
    metadata = executor.execute(ProcessInvocation("i", "r", "p", context={"a": 1})).metadata

    assert metadata["repair_count"] == 2
    assert metadata["usage"] == {"prompt_tokens": 300, "completion_tokens": 150}
    assert metadata["final_usage"] == {"prompt_tokens": 100, "completion_tokens": 50}
    assert metadata["estimated_cost"] == pytest.approx(0.6)
    # Every attempt records the prompt that produced it; a repair sends a
    # different prompt, so one prompt hash cannot describe the exchange.
    assert all("prompt_hash" in attempt for attempt in metadata["provider_attempts"])
    assert (
        metadata["provider_attempts"][0]["prompt_hash"]
        != (metadata["provider_attempts"][1]["prompt_hash"])
    )


def test_internal_faults_are_not_reported_as_client_validation_errors() -> None:
    """An internal error is a 500, not a 422 echoing the raw exception text."""
    import json as _json

    from genesis.app import _service_error

    response = _service_error(OSError("No such file or directory: /Users/someone/ws/genesis.db"))
    body = _json.loads(response.body)

    assert response.status_code == 500
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "/Users/someone" not in body["error"]["message"]

    # A declared domain error still maps to its own code and status.
    domain = _service_error(ValueError("ALREADY_EXISTS: run 'r' already imported"))
    assert domain.status_code == 409
    assert _json.loads(domain.body)["error"]["code"] == "ALREADY_EXISTS"


def test_manifest_seed_check_accepts_manifests_without_a_matching_block() -> None:
    """Verification must not reject evidence it simply cannot reconstruct.

    Manifests written before the matching block was retained cannot say whether
    their streams were matched, and matched streams derive differently.
    """
    from genesis.service import GenesisService

    randomness = {
        "run_id": "r1",
        "experiment_id": "e",
        "condition_id": "c",
        "replication": 1,
    }
    streams = [{"id": "initial-world", "seed": 1101}]
    matching = {"enabled": True, "shared_streams": ["initial-world", "conventional"]}

    for seeds in (
        GenesisService.derive_manifest_seeds(randomness, streams, None),
        GenesisService.derive_manifest_seeds(randomness, streams, matching),
    ):
        GenesisService.verify_manifest_seeds(
            {"randomness_inputs": randomness, "random_streams": streams, "seeds": seeds}
        )


def test_unenforced_protocol_budgets_are_reported_at_compilation(tmp_path) -> None:
    """A budget the runtime ignores must not read as an applied constraint."""
    import json as _json

    import yaml

    from genesis.compiler import StudyCompiler

    source = tmp_path / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "budget-study"}
    files = {
        "study": {**base, "title": "b"},
        "openness": {**base, "processes": []},
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {**base},
        "protocol": {
            **base,
            "time_model": {"type": "rounds", "end": 1},
            "budgets": {"max_events": 10, "max_reads_per_user_round": 1},
        },
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    build = StudyCompiler(source).compile(tmp_path / "build")
    report = _json.loads((build.path / "validation_report.json").read_text())
    codes = {warning["code"] for warning in report.get("warnings", [])}
    paths = {warning["path"] for warning in report.get("warnings", [])}

    assert "BUDGET_NOT_ENFORCED" in codes
    assert "protocol.budgets/max_reads_per_user_round" in paths
    # The enforced budget is not flagged.
    assert "protocol.budgets/max_events" not in paths


# ---------------------------------------------------------------------------
# Third review round: temporal lag, actor parity, prefix coverage, sum parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_lagged_edge_is_satisfied_by_producer_history_not_latest_completion(lag: int) -> None:
    """A repeating producer must not starve a consumer that waits on a lag.

    Readiness compared the producer's LATEST completion against the lag, but a
    repeating producer advances that completion every round, so ``last + lag <=
    phase`` could never hold and any lag of two or more never fired at all.
    """
    seen: list[tuple[int, str]] = []

    def make(name):
        def run(invocation):
            seen.append((int(invocation.phase), name))
            return {}

        return run

    processes = [
        {
            "id": "prod",
            "context_policy": "n",
            "executor": {"mode": "computational"},
            "trigger": {"phase": 0, "repeat": True},
        },
        {
            "id": "cons",
            "context_policy": "n",
            "executor": {"mode": "computational"},
            "trigger": {"phase": 0, "repeat": True},
            "dependencies": {"after": ["prod"], "delay": {"rounds": lag}},
        },
    ]
    controller = _chain_controller(
        processes,
        {pid: CallableExecutor(make(pid), "computational") for pid in ("prod", "cons")},
    )
    controller.run("lag-edge-run", phase_limit=6)

    consumer_phases = [phase for phase, name in seen if name == "cons"]
    assert consumer_phases == list(range(lag, 6))


def test_recorded_replay_refuses_another_actors_recording_in_the_same_phase() -> None:
    """Actor parity is required; only actor-less legacy evidence may match on phase.

    The phase/attempt fallback ignored actor identity entirely, so an actor the
    source never ran received a different actor's output as ``recorded``.
    """
    from genesis.service import _RecordedExecutor

    records = [
        {"phase": 0, "attempt": 1, "actors": ("alice",), "outputs": {"a": "ALICE"}, "order": 0},
        {"phase": 0, "attempt": 1, "actors": ("bob",), "outputs": {"a": "BOB"}, "order": 1},
    ]
    executor = _RecordedExecutor(records, source_run_id="src", process_id="write")

    assert executor.execute(
        ProcessInvocation("i", "r", "write", actor_ids=("bob",), phase=0)
    ).outputs == {"a": "BOB"}

    with pytest.raises(ValueError, match="REPLAY_RECORD_MISSING"):
        executor.execute(ProcessInvocation("i", "r", "write", actor_ids=("carol",), phase=0))

    # A legacy recording that carries no actor identity still matches on phase.
    legacy = _RecordedExecutor(
        [{"phase": 0, "attempt": 1, "actors": (), "outputs": {"a": "L"}, "order": 0}],
        source_run_id="src",
        process_id="write",
    )
    assert legacy.execute(
        ProcessInvocation("i", "r", "write", actor_ids=("carol",), phase=0)
    ).outputs == {"a": "L"}


def test_frozen_prefix_guard_covers_generative_processes_without_recordings() -> None:
    """A process with no frozen record still must not generate inside the prefix.

    Such a process received no substitute at all, so its live executor stayed
    installed: a branch that changes a condition could make a previously idle
    generative process fire inside the supposedly frozen prefix.
    """
    from genesis.service import _FrozenPrefixGuard

    class _Live:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _invocation):
            self.calls += 1
            return "generated"

    live = _Live()
    guard = _FrozenPrefixGuard(live, source_run_id="src", process_id="late", phase_boundary=2)

    with pytest.raises(ValueError, match="REPLAY_PREFIX_DIVERGED"):
        guard.execute(ProcessInvocation("i", "r", "late", phase=1))
    assert live.calls == 0

    # Outside the prefix the live executor runs normally.
    assert guard.execute(ProcessInvocation("i", "r", "late", phase=2)) == "generated"
    assert live.calls == 1


def test_duckdb_and_python_agree_on_an_all_missing_sum() -> None:
    """SQL ``sum`` over an all-NULL group is NULL; ``sum([])`` is 0."""
    from genesis.analysis import AnalysisEngine, OutcomePlan, duckdb_aggregate

    rows = [{"g": "a", "v": None}, {"g": "a", "v": None}]
    plan = OutcomePlan(id="o", source="rows", select="v", aggregation="sum", group_by="g")

    assert duckdb_aggregate(rows, select="v", op="sum", group_by="g") == AnalysisEngine().evaluate(
        plan, {"rows": rows}
    )
    ungrouped = OutcomePlan(id="o", source="rows", select="v", aggregation="sum")
    assert duckdb_aggregate(rows, select="v", op="sum") == AnalysisEngine().evaluate(
        ungrouped, {"rows": rows}
    )


# ---------------------------------------------------------------------------
# Fourth review round: event-boundary prefix coverage and key identity
# ---------------------------------------------------------------------------


def test_frozen_keys_carry_process_identity(tmp_path) -> None:
    """Invocation coordinates are shared across processes; keys must not be pooled.

    Two processes running in the same phase for the same actors share
    ``(phase, attempt, actors)``. With that as the whole key, a boundary that
    froze one process also froze the other's recording, replaying a process
    that belonged to the live suffix.
    """
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "ws")
    try:
        recorded = {
            "compose": [{"phase": 0, "attempt": 1, "actors": (), "outputs": {}, "order": 0}],
            "zcompose": [{"phase": 0, "attempt": 1, "actors": (), "outputs": {}, "order": 0}],
        }
        frozen = service._frozen_invocation_keys(
            "any-run", recorded, phase_boundary=1, event_boundary=None
        )
        assert frozen == {
            ("compose", 0, 1, ()),
            ("zcompose", 0, 1, ()),
        }
        # Each key names exactly one process, so no key of one can match another.
        assert all(len(key) == 4 and isinstance(key[0], str) for key in frozen)
    finally:
        service.close()


def test_prefix_guard_treats_an_event_boundary_phase_as_inside() -> None:
    """An event boundary sits inside a phase and cannot order a new invocation.

    A never-recorded generative invocation has no source counterpart to order
    against the boundary event, so the whole boundary phase is refused rather
    than allowing fresh generation inside a frozen prefix.
    """
    from genesis.service import _FrozenPrefixGuard

    class _Live:
        def execute(self, _invocation):
            return "generated"

    inclusive = _FrozenPrefixGuard(
        _Live(), source_run_id="src", process_id="compose", phase_boundary=0, inclusive=True
    )
    with pytest.raises(ValueError, match="REPLAY_PREFIX_DIVERGED"):
        inclusive.execute(ProcessInvocation("i", "r", "compose", phase=0))
    assert inclusive.execute(ProcessInvocation("i", "r", "compose", phase=1)) == "generated"

    # A phase boundary is exclusive: phase N itself is the re-executed suffix.
    exclusive = _FrozenPrefixGuard(
        _Live(), source_run_id="src", process_id="compose", phase_boundary=0
    )
    assert exclusive.execute(ProcessInvocation("i", "r", "compose", phase=0)) == "generated"


# ---------------------------------------------------------------------------
# Fifth review round: the boundary binds every replay executor path
# ---------------------------------------------------------------------------


def _selective(live, *, frozen, known, boundary, inclusive, records=()):
    from genesis.service import _SelectiveExecutor

    return _SelectiveExecutor(
        list(records),
        frozen_keys=frozen,
        known_keys=known,
        fallback=live,
        source_run_id="src",
        process_id="compose",
        phase_boundary=boundary,
        inclusive=inclusive,
    )


class _LiveSpy:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _invocation):
        self.calls += 1
        return "LIVE"


def test_selective_executor_honours_an_event_boundary() -> None:
    """An event-boundary replay passed no boundary to the selective executor.

    ``phase_boundary`` was None for event boundaries, so every unmatched
    invocation fell through to the live executor — including one inside the
    prefix.
    """
    live = _LiveSpy()
    executor = _selective(
        live,
        frozen={(1, 1, ())},
        known={(1, 1, ())},
        boundary=1,
        inclusive=True,
        records=[{"phase": 1, "attempt": 1, "actors": (), "outputs": {"a": "rec"}, "order": 0}],
    )

    assert executor.execute(ProcessInvocation("i", "r", "compose", phase=1)).outputs == {"a": "rec"}
    with pytest.raises(ValueError, match="REPLAY_PREFIX_DIVERGED"):
        executor.execute(ProcessInvocation("i", "r", "compose", phase=0))
    assert live.calls == 0


def test_recorded_suffix_invocation_still_re_executes_but_a_new_one_does_not() -> None:
    """Source position does not survive a branch, so the boundary still binds.

    An invocation the source also performed after the boundary is a legitimate
    suffix invocation and runs live; one the source never performed is a
    divergence when it lands inside the boundary phase.
    """
    live = _LiveSpy()
    executor = _selective(live, frozen=set(), known={(0, 1, ())}, boundary=0, inclusive=True)

    # The source made this invocation (after the boundary): re-execute it.
    assert executor.execute(ProcessInvocation("i", "r", "compose", phase=0)) == "LIVE"
    assert live.calls == 1

    # The source never made this one, and it lands inside the boundary phase.
    with pytest.raises(ValueError, match="REPLAY_PREFIX_DIVERGED"):
        executor.execute(ProcessInvocation("i", "r", "compose", phase=0, actor_ids=("newbie",)))
    assert live.calls == 1

    # Anything after the boundary phase is unambiguously suffix.
    assert executor.execute(ProcessInvocation("i", "r", "compose", phase=1)) == "LIVE"
    assert live.calls == 2


# ---------------------------------------------------------------------------
# Sixth review round: replayed effects, resume snapshot, inspectable graph
# ---------------------------------------------------------------------------


def _counter_package(root, end=3):
    import yaml

    source = root / "pkg"
    source.mkdir(parents=True)
    base = {"schema_version": "1.0", "study_id": "st-study"}
    files = {
        "study": {**base, "title": "s"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "bump",
                    "executor": {
                        "mode": "state-transition",
                        "parameters": {
                            "operations": [{"op": "increment", "state": "counter", "value": 1}]
                        },
                    },
                    "context_policy": "c",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "increment"}],
                }
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "c", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": end}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


def test_partial_replay_preserves_state_changes_of_the_frozen_prefix(tmp_path) -> None:
    """A frozen invocation replays its consequences, not only its outputs.

    Every invocation is recorded, including one whose value lives entirely in
    its state effects. Replaying outputs alone made those invocations no-ops,
    so a partial replay with no intervention silently ended in a different
    state than its source while reporting success.
    """
    from genesis.replay import ReplayMode
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source_dir = _counter_package(workspace)
    service = GenesisService(workspace)
    try:
        build = service.compile_study(source_dir, "builds/s")
        service.create_run({"id": "s", "study_id": "st-study", "build": build["path"]})
        service.execute_run("s")

        def final_counter(run_id):
            history = list(service.persistence.list_state_history(run_id))
            return history[-1][1].get("counter") if history else None

        assert final_counter("s") == 4

        preview = service.replay_preview("s", mode=ReplayMode.PARTIAL, boundary="phase:2")
        replay = service.replay_run(
            "s",
            mode=ReplayMode.PARTIAL,
            boundary="phase:2",
            preview_token=preview["preview_token"],
        )
        assert service.get_run(replay["run_id"])["status"] == "completed"
        assert final_counter(replay["run_id"]) == 4
    finally:
        service.close()


def _feedback_package(root):
    """Two writers per round plus a lag-1 reader, so a round's PARTIAL state
    differs from its final state — the only shape that can detect a stale
    reconstructed snapshot."""
    import yaml

    source = root / "pkg"
    source.mkdir(parents=True)
    base = {"schema_version": "1.0", "study_id": "ring-study"}
    bump = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    read = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:read"},
    }
    files = {
        "study": {**base, "title": "r"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "awrite",
                    "executor": bump,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "mwrite",
                    "executor": bump,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "dependencies": {"after": ["awrite"]},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "zread",
                    "executor": read,
                    "context_policy": "f",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "dependencies": {"after": ["mwrite"]},
                },
            ],
        },
        "theory": {
            **base,
            "theory_family": "exploratory",
            "feedback": [
                {
                    "id": "counter-fb",
                    "from": "counter",
                    "to": "zread",
                    "relation": "reader sees last round's counter",
                    "execution": {
                        "kind": "feedback_context",
                        "source": {"kind": "state", "id": "counter"},
                        "consumer_process": "zread",
                        "context_slot": "last",
                        "lag_rounds": 1,
                        "initial": {"policy": "declared_default", "value": {}},
                    },
                }
            ],
        },
        "domain": {
            **base,
            "visibility": [
                {"id": "s", "allow": ["counter"]},
                {"id": "f", "allow": ["feedback.last"]},
            ],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 3}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


def test_resume_keeps_the_round_start_snapshot_for_lagged_feedback(tmp_path) -> None:
    """Pausing mid-round must not change what any later reader sees.

    Two defects meet here. Re-entering the interrupted round overwrote its
    round-start snapshot with partly-updated state. Fixing that by preserving
    the reconstructed entry then exposed a second: the reconstruction also
    derives an entry for the round AFTER the interrupted one from that round's
    PARTIAL state, and preserving it carried the stale value forward.

    Two writers per round are required — with one writer the partial state
    equals the round's final state and neither defect is observable — and the
    resume must rebuild the controller from persistence, as the service does.
    """
    from genesis.service import GenesisService
    from tests import resume_executors

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _feedback_package(workspace)

    bootstrap = GenesisService(workspace)
    try:
        build_path = bootstrap.compile_study(source, "builds/ring")["path"]
    finally:
        bootstrap.close()

    def run_to_completion(run_id, cut_at_phase=None):
        resume_executors.SEEN.clear()
        service = GenesisService(workspace)
        try:
            service.create_run({"id": run_id, "study_id": "ring-study", "build": build_path})
            if cut_at_phase is None:
                service.execute_run(run_id)
                return list(resume_executors.SEEN)
            original = resume_executors.bump

            def cutting(invocation):
                result = original(invocation)
                if invocation.process_id == "awrite" and int(invocation.phase) == cut_at_phase:
                    service.transition_run(run_id, "paused", service.get_run(run_id)["version"])
                return result

            resume_executors.bump = cutting
            try:
                service.execute_run(run_id)
            finally:
                resume_executors.bump = original
            assert service.get_run(run_id)["status"] == "paused"
        finally:
            service.close()
        # A fresh service rebuilds the round-state ring from persistence.
        resumed = GenesisService(workspace)
        try:
            resumed.transition_run(run_id, "running", resumed.get_run(run_id)["version"])
            resumed.execute_run(run_id)
        finally:
            resumed.close()
        return list(resume_executors.SEEN)

    uninterrupted = run_to_completion("clean")
    # Two writers per round: the reader lags one round behind an even counter.
    assert uninterrupted == [(0, None), (1, 2), (2, 4), (3, 6)]

    interrupted = run_to_completion("cut", cut_at_phase=1)
    assert interrupted == uninterrupted


def test_process_graph_reports_theory_lag_edges(tmp_path) -> None:
    """The inspectable graph must match the dependencies the scheduler reads.

    It was built from the pre-merge declarations, so a positive-lag theory edge
    appeared in processes.json but left the graph reporting both processes as
    independent.
    """
    import yaml

    from genesis.compiler import StudyCompiler
    from genesis.elicitation import WorkflowRegistry
    from genesis.service import _workflows_root

    source = tmp_path / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "lag-study"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "demos.demo_executors:formulate_strategy"},
    }
    files = {
        "study": {**base, "title": "l"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": pid,
                    "executor": executor,
                    "context_policy": "n",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                }
                for pid in ("producer", "consumer")
            ],
        },
        "theory": {
            **base,
            "theory_family": "exploratory",
            "relations": [
                {
                    "id": "p-before-c",
                    "from": "producer",
                    "to": "consumer",
                    "relation": "consumer waits one round after producer",
                    "execution": {
                        "kind": "precedence",
                        "producer_process": "producer",
                        "consumer_process": "consumer",
                        "lag_rounds": 1,
                    },
                }
            ],
        },
        "domain": {**base, "visibility": [{"id": "n", "allow": []}]},
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    build = StudyCompiler(
        source, theory_templates=WorkflowRegistry(_workflows_root()).theory_templates()
    ).compile(tmp_path / "build")

    import json as _json

    graph = _json.loads((build.path / "process_graph.json").read_text())
    assert graph["consumer"] == [
        {"dependency": "producer", "delayed": True, "delay": {"rounds": 1}}
    ]


# ---------------------------------------------------------------------------
# Seventh review round: trace applicability, provider echo, credential file
# ---------------------------------------------------------------------------


def test_trace_traversal_is_study_agnostic(tmp_path) -> None:
    """A trace is built from provenance every run records, not from study names.

    The service used to recognise one study's processes and state fields, so on
    any other study it returned an empty illustration for user "" at a sentinel
    phase while naming a rule as applied. Traversal now follows causal parents,
    which exist for every study, and a package declares only where to start.
    """
    import yaml

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "nt"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    files = {
        "study": {**base, "title": "n"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "first",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "second",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "dependencies": {"after": ["first"]},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "s", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/n")
        service.create_run({"id": "n1", "study_id": "nt", "build": build["path"]})
        service.execute_run("n1")

        # Nothing is declared, so the caller must say where to start.
        assert service.list_traces("n1") == []
        with pytest.raises(ValueError, match="TRACE_SELECTION_REQUIRED"):
            service.natural_trace("n1")

        # But any recorded event is a valid starting point, with no declaration.
        seed = str(service.trace_run("n1")[-1]["event_id"])
        result = service.natural_trace("n1", event=seed)
        steps = result["illustration"]["steps"]
        assert result["seed_event"] == seed
        assert [step["relation"] for step in steps].count("seed") == 1
        # `second` follows `first`, so the chain recovers that ordering.
        assert {step["process"] for step in steps} == {"first", "second"}
        assert result["coverage"]["returned"] == len(steps)

        with pytest.raises(ValueError, match="ACTOR_TRACE_NOT_FOUND"):
            service.natural_trace("n1", actor="nobody")
        with pytest.raises(ValueError, match="TRACE_EVENT_NOT_FOUND"):
            service.natural_trace("n1", event="no-such-event")
    finally:
        service.close()


def test_declared_trace_seeds_from_a_declared_dataset(tmp_path) -> None:
    """A study says where a chain starts; the core does not recognise its names."""
    import yaml

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "dt"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    files = {
        "study": {**base, "title": "d"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "first",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "second",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "dependencies": {"after": ["first"]},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "s", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {
            **base,
            "datasets": [
                {
                    "id": "late-writes",
                    "source": {"kind": "events"},
                    "where": [{"field": "process_id", "op": "eq", "value": "second"}],
                }
            ],
            "traces": [
                {
                    "id": "second-write",
                    "title": "The first second-write and its chain",
                    "seed": {"dataset": "late-writes", "order_by": ["phase"]},
                    "labels": {"first": "opening", "second": "closing"},
                }
            ],
            "outcomes": [],
        },
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/d")
        service.create_run({"id": "d1", "study_id": "dt", "build": build["path"]})
        service.execute_run("d1")

        assert [trace["id"] for trace in service.list_traces("d1")] == ["second-write"]
        # A single declared trace is the default; naming it is equivalent.
        default = service.natural_trace("d1")
        named = service.natural_trace("d1", trace="second-write")
        assert default["seed_event"] == named["seed_event"]
        assert "second-write" in default["selection_rule"]

        steps = default["illustration"]["steps"]
        seed_step = next(step for step in steps if step["relation"] == "seed")
        # The seed is the EARLIEST `second` invocation, per the declared order.
        assert seed_step["process"] == "second"
        assert seed_step["phase"] == 0
        # Declared labels rename the steps; undeclared processes keep their id.
        assert seed_step["step"] == "closing"
        assert {step["step"] for step in steps} <= {"opening", "closing"}

        with pytest.raises(ValueError, match="TRACE_NOT_DECLARED"):
            service.natural_trace("d1", trace="nope")
    finally:
        service.close()


def test_compiler_rejects_a_trace_seeded_from_an_undeclared_dataset(tmp_path) -> None:
    """A trace's seed and labels must name things the package actually declares."""
    import yaml

    from genesis.compiler import StudyCompiler, ValidationIssue

    source = tmp_path / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "bad"}
    files = {
        "study": {**base, "title": "b"},
        "openness": {**base, "processes": []},
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {**base},
        "protocol": {**base, "time_model": {"type": "rounds", "end": 1}},
        "outcomes": {
            **base,
            "outcomes": [],
            "traces": [
                {
                    "id": "ghost",
                    "seed": {"dataset": "missing-dataset"},
                    "labels": {"no-such-process": "x"},
                }
            ],
        },
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    with pytest.raises((ValidationIssue, ValueError)) as caught:
        StudyCompiler(source).compile(tmp_path / "build")
    message = str(caught.value)
    assert "TRACE_SEED_UNKNOWN" in message
    assert "TRACE_LABEL_UNKNOWN" in message


def test_provider_http_error_does_not_carry_the_whole_response_body() -> None:
    """A provider body echoes the request, i.e. the authorized context.

    The full body reached event records and API callers, where the trace policy
    does not govern it.
    """
    import io
    import urllib.error

    from genesis.providers import OpenAICompatibleProvider, ProviderRequest

    provider = OpenAICompatibleProvider(
        base_url="http://x", model="m", api_key_env="E", api_key="k"
    )

    def raising(_request):
        raise urllib.error.HTTPError(
            "http://x",
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error":{"message":"echoed prompt ' + b"A" * 3000 + b'"}}'),
        )

    provider._open_cancellable = raising  # type: ignore[method-assign]

    with pytest.raises(ValueError) as caught:
        provider.generate(ProviderRequest(model="m", prompt="sensitive context"))

    message = str(caught.value)
    assert message.startswith("PROVIDER_HTTP: provider returned HTTP 400")
    assert len(message) < 400


def test_model_profile_file_is_not_world_readable(tmp_path) -> None:
    """A pasted API key must not be written with the ambient umask."""
    import stat as _stat

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = GenesisService(workspace)
    try:
        service.create_model_profile(
            {
                "id": "p1",
                "provider": "openai-compatible",
                "model": "m",
                "base_url": "https://example.invalid",
                "api_key_env": "NOPE",
                "api_key": "sk-secret",
            }
        )
        path = workspace / ".genesis" / "model-profiles.json"
        assert _stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        service.close()


def test_trace_step_limit_is_caller_controlled_and_bounded() -> None:
    """A truncated chain can be widened, but not into a whole-run dump.

    Truncation keeps the steps nearest the seed, so the limit changes how much
    of the neighbourhood is returned without changing which chain it is.
    """
    from genesis.tracing import MAX_STEPS_LIMIT, build_chain

    # One seed with a wide fan-out, so the step limit binds rather than depth.
    events = [{"event_id": "seed", "process_id": "p", "phase": 0, "commit_order": 0}]
    events += [
        {
            "event_id": f"c{index}",
            "process_id": "p",
            "phase": 1,
            "commit_order": index,
            "parent_events": ["seed"],
        }
        for index in range(2000)
    ]

    narrow, narrow_coverage = build_chain(events, [], "seed", max_steps=5)
    assert narrow_coverage["reachable"] == 2001
    assert narrow_coverage["returned"] == 5
    assert narrow_coverage["truncated"] is True
    assert narrow_coverage["truncated_by"] == ["steps"], "the depth bound was not what cut it"
    assert any(step["relation"] == "seed" for step in narrow), "the seed is always kept"

    _, wider = build_chain(events, [], "seed", max_steps=500)
    assert wider["returned"] == 500

    # A caller cannot ask for the entire run.
    _, unbounded = build_chain(events, [], "seed", max_steps=10**9)
    assert unbounded["returned"] == MAX_STEPS_LIMIT
    assert unbounded["truncated"] is True

    # A nonsensical limit still returns a usable chain.
    _, zero = build_chain(events, [], "seed", max_steps=0)
    assert zero["returned"] == 1


def test_natural_trace_step_limit_prefers_request_then_declaration(tmp_path) -> None:
    """An explicit limit wins; otherwise the declared trace's own limit applies."""
    import yaml

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "lim"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    files = {
        "study": {**base, "title": "l"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": pid,
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                    **({"dependencies": {"after": ["a"]}} if pid != "a" else {}),
                }
                for pid in ("a", "b", "c")
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "s", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 3}},
        "outcomes": {
            **base,
            "datasets": [
                {
                    "id": "a-writes",
                    "source": {"kind": "events"},
                    "where": [{"field": "process_id", "op": "eq", "value": "a"}],
                }
            ],
            "traces": [
                {
                    "id": "narrow",
                    "seed": {"dataset": "a-writes", "order_by": ["phase"]},
                    "max_steps": 2,
                }
            ],
            "outcomes": [],
        },
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/l")
        service.create_run({"id": "l1", "study_id": "lim", "build": build["path"]})
        service.execute_run("l1")

        declared = service.natural_trace("l1")
        assert declared["coverage"]["returned"] == 2, "the declaration's own limit applies"
        assert declared["coverage"]["truncated"] is True

        widened = service.natural_trace("l1", max_steps=50)
        assert widened["coverage"]["returned"] > 2, "an explicit limit overrides it"
        assert widened["seed_event"] == declared["seed_event"], "the same chain, seen wider"
    finally:
        service.close()


# ---------------------------------------------------------------------------
# Eighth review round: shared state preparation, feedback provenance, depth
# ---------------------------------------------------------------------------


def _rounds_package(root, *, with_trace: bool):
    import yaml

    source = root / "pkg"
    source.mkdir(parents=True)
    base = {"schema_version": "1.0", "study_id": "cmp"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    outcomes = {
        **base,
        "datasets": [
            {"id": "rounds", "source": {"kind": "state", "snapshot": "each_completed_round"}}
        ],
        "outcomes": [
            {
                "id": "n",
                "source": "rounds",
                "grouping": [],
                "aggregation": {"op": "count", "field": "counter"},
            }
        ],
    }
    if with_trace:
        outcomes["traces"] = [{"id": "t", "seed": {"dataset": "rounds"}}]
    files = {
        "study": {**base, "title": "c"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "a",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "b",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "dependencies": {"after": ["a"]},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "s", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": outcomes,
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


def test_trace_and_outcome_datasets_see_the_same_state_evidence(tmp_path) -> None:
    """One preparation of state evidence, shared by both consumers.

    Trace seeding prepared state snapshots separately from outcome evaluation,
    without the round annotation or the incomplete-round exclusion, so an
    ``each_completed_round`` dataset yielded one row per state COMMIT on the
    trace path and one row per completed ROUND on the outcome path.
    """
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _rounds_package(workspace, with_trace=True)
    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/c")
        service.create_run({"id": "c1", "study_id": "cmp", "build": build["path"]})
        service.execute_run("c1")

        counted = service.evaluate_outcomes("c1")[0]["counter_count"]
        seeded = len(service._dataset_rows("c1", "rounds"))
        # Three rounds, two state-writing processes each: the round-scoped
        # dataset must not report six.
        assert counted == 3
        assert seeded == counted
    finally:
        service.close()


def test_lagged_feedback_records_the_event_that_produced_the_value(tmp_path) -> None:
    """A reader of prior-round state must be traceable to the write it read.

    Causal parents came only from consumed artifacts and declared `after`
    dependencies, so a process influenced through a feedback binding recorded no
    parent at all and its trace was a single isolated step.
    """
    import yaml

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "fb"}
    write = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    read = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:read"},
    }
    files = {
        "study": {**base, "title": "f"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "awrite",
                    "executor": write,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                },
                {
                    "id": "zread",
                    "executor": read,
                    "context_policy": "f",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                },
            ],
        },
        "theory": {
            **base,
            "theory_family": "exploratory",
            "feedback": [
                {
                    "id": "fb",
                    "from": "counter",
                    "to": "zread",
                    "relation": "reader sees the prior round's counter",
                    "execution": {
                        "kind": "feedback_context",
                        "source": {"kind": "state", "id": "counter"},
                        "consumer_process": "zread",
                        "context_slot": "last",
                        "lag_rounds": 1,
                        "initial": {"policy": "declared_default", "value": {}},
                    },
                }
            ],
        },
        "domain": {
            **base,
            "visibility": [
                {"id": "s", "allow": ["counter"]},
                {"id": "f", "allow": ["feedback.last"]},
            ],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/f")
        service.create_run({"id": "f1", "study_id": "fb", "build": build["path"]})
        service.execute_run("f1")

        events = {(e["process_id"], e["phase"]): e for e in service.trace_run("f1")}
        # Lag 1: the phase-2 reader read the value the phase-1 writer committed.
        assert events[("zread", 2)]["parent_events"] == [events[("awrite", 1)]["event_id"]]
        # The first round used the declared initial value, so it depends on nothing.
        assert events[("zread", 0)]["parent_events"] == []

        chain = service.natural_trace("f1", event=events[("zread", 2)]["event_id"])
        steps = {(step["process"], step["phase"]) for step in chain["illustration"]["steps"]}
        assert ("awrite", 1) in steps, "the trace must explain where the value came from"
    finally:
        service.close()


def test_coverage_distinguishes_depth_truncation_from_step_truncation() -> None:
    """A depth-limited chain is partial even when every step found was returned."""
    from genesis.tracing import build_chain

    events = [
        {
            "event_id": f"e{index}",
            "process_id": "p",
            "phase": index,
            "commit_order": index,
            "parent_events": ([f"e{index - 1}"] if index else []),
        }
        for index in range(4)
    ]

    _, shallow = build_chain(events, [], "e3", depth=1)
    assert shallow["returned"] == 2
    assert shallow["truncated"] is True
    assert shallow["truncated_by"] == ["depth"]
    assert shallow["depth"] == 1

    _, complete = build_chain(events, [], "e3", depth=10)
    assert complete["returned"] == 4
    assert complete["truncated"] is False
    assert complete["truncated_by"] == []


# ---------------------------------------------------------------------------
# Ninth review round: retained evidence stays readable
# ---------------------------------------------------------------------------


def _legacy_catalog_package(root, *, output_schema: bool):
    import json as _json

    import yaml

    source = root / "pkg"
    source.mkdir(parents=True)
    (source / "schemas").mkdir()
    (source / "schemas" / "row.json").write_text(_json.dumps({"type": "object"}))
    base = {"schema_version": "1.0", "study_id": "lg"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    outcome = {
        "id": "n",
        "source": "events",
        "grouping": [],
        "aggregation": {"op": "count", "field": "phase"},
    }
    if output_schema:
        outcome["output_schema"] = "row"
    files = {
        "study": {**base, "title": "l"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "a",
                    "executor": executor,
                    "context_policy": "s",
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [{"field": "counter", "op": "set"}],
                }
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [{"id": "s", "allow": ["counter"]}],
            "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 1}},
        "outcomes": {**base, "outcomes": [outcome]},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


def _degrade_pinned_catalog(service, build_ref):
    """Rewrite a pinned build's catalog to a pre-tightening dialect."""
    import json as _json
    import os
    import stat as _stat

    path = service.resolve_path(build_ref) / "schemas.json"
    os.chmod(path, _stat.S_IRUSR | _stat.S_IWUSR)
    legacy = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}
    path.write_text(_json.dumps({"row": legacy}))


def test_a_legacy_schema_catalog_does_not_make_recorded_evidence_unreadable(tmp_path) -> None:
    """Tightening the package dialect must not lock out runs already on disk.

    Outcome evaluation built the catalog unguarded while every other site
    guarded it, so a build compiled before the dialect changed — authentic,
    integrity-verified evidence — could no longer be evaluated or exported.
    """
    from genesis.evidence import ExportMode
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _legacy_catalog_package(workspace, output_schema=False)
    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/l")
        service.create_run({"id": "l1", "study_id": "lg", "build": build["path"]})
        service.execute_run("l1")
        _degrade_pinned_catalog(service, build["path"])

        assert service.evaluate_outcomes("l1"), "outcomes must still be computable"
        assert service.export_run("l1", "exports/legacy", mode=ExportMode.EXPLORATION)
    finally:
        service.close()


def test_an_outcome_that_asks_for_validation_is_refused_by_name(tmp_path) -> None:
    """Tolerating a legacy catalog must not silently skip requested validation."""
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = _legacy_catalog_package(workspace, output_schema=True)
    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/l")
        service.create_run({"id": "l1", "study_id": "lg", "build": build["path"]})
        service.execute_run("l1")
        # With a bindable catalog the declared schema is enforced as before.
        assert service.evaluate_outcomes("l1")

        _degrade_pinned_catalog(service, build["path"])
        with pytest.raises(ValueError, match="OUTCOME_SCHEMA_UNAVAILABLE") as caught:
            service.evaluate_outcomes("l1")
        # The refusal names the outcome and the schema it wanted.
        assert "'row'" in str(caught.value)
    finally:
        service.close()


def test_compilation_still_refuses_a_legacy_dialect(tmp_path) -> None:
    """Enforcement belongs at compile time, not when reading retained evidence."""
    import json as _json

    from genesis.compiler import StudyCompiler, ValidationIssue

    source = _legacy_catalog_package(tmp_path, output_schema=False)
    (source / "schemas" / "row.json").write_text(
        _json.dumps({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"})
    )
    with pytest.raises((ValidationIssue, ValueError), match="SCHEMA_DIALECT_UNSUPPORTED"):
        StudyCompiler(source).compile(tmp_path / "build")

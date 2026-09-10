"""Regressions for the September code-review findings.

Each test pins one behaviour that was previously wrong in a way no existing
test detected. They are grouped by the layer they protect: scheduling and
theory semantics, context authorization, state contracts, replay integrity,
and cross-realization reporting.
"""

from __future__ import annotations

import hashlib

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


# ---------------------------------------------------------------------------
# State-history scaling, Phase 1 (docs/plans/2026-09-09-...-specification.md)
# ---------------------------------------------------------------------------


def test_state_history_iteration_does_not_scale_with_commit_count(tmp_path) -> None:
    """STH-001: iterating holds one snapshot; materialising holds them all.

    Measured as the evidence alive part-way through each call, attributed to the
    JSON decoder and this package. The whole-process peak this used to compare
    was fragile -- it depended on when a collection happened to run, and an
    unrelated 1.9 MB lru-cache resize inside pathlib landed in the window.
    """
    import gc
    import tracemalloc
    from json import decoder as json_decoder
    from pathlib import Path

    from genesis import persistence as persistence_module
    from genesis.persistence import PersistenceCoordinator

    # Attribute memory to the decoded evidence: the JSON decoder that builds
    # each snapshot, plus this package. A whole-process total also counts
    # unrelated allocations that land in the window -- an lru-cache resize
    # inside pathlib produced a single 1.9 MB block that looked exactly like a
    # retained history until it was itemised.
    traced = (
        tracemalloc.Filter(True, str(Path(persistence_module.__file__).parent / "*")),
        tracemalloc.Filter(True, json_decoder.__file__),
    )

    def alive_during(operation) -> int:
        """Evidence alive at the moment ``operation`` reports back.

        Sampled *during* the work, not after it: a generator that secretly
        accumulated every snapshot would drop them all when it was exhausted,
        so a measurement taken afterwards cannot tell the two apart.
        """
        gc.collect()
        tracemalloc.start()
        try:
            snapshot = operation()
            snapshot = snapshot.filter_traces(traced)
        finally:
            tracemalloc.stop()
        return sum(stat.size for stat in snapshot.statistics("filename"))

    def measure(commits: int) -> tuple[int, int]:
        store = PersistenceCoordinator(
            tmp_path / f"{commits}.sqlite", tmp_path / f"objects-{commits}"
        )
        try:
            store.create_run({"id": "r", "study_id": "s"})
            for version in range(1, commits + 1):
                # Distinct payloads: identical ones share their string object,
                # which would mask what materialising really costs.
                payload = {"field": f"{version:05d}" + "x" * 4096}
                store.commit_process_result(
                    {
                        "event_id": f"e{version}",
                        "invocation_id": f"i{version}",
                        "run_id": "r",
                        "kind": "process_completed",
                        "phase": version,
                        "state_version": version,
                    },
                    {"run_id": "r", "state_version": version, "payload": _json_bytes(payload)},
                    [],
                )

            def stream():
                sampled = None
                seen = 0
                for _version, _state in store.iter_state_history("r"):
                    seen += 1
                    if seen == commits // 2:
                        sampled = tracemalloc.take_snapshot()
                assert seen == commits
                assert sampled is not None
                return sampled

            def whole():
                history = store.list_state_history("r")
                assert len(history) == commits
                sampled = tracemalloc.take_snapshot()
                del history
                return sampled

            return alive_during(stream), alive_during(whole)
        finally:
            store.close()

    small_stream, small_whole = measure(100)
    large_stream, large_whole = measure(400)

    # Four times the commits, four times the history held.
    assert large_whole > small_whole * 3, (small_whole, large_whole)
    # Iterating holds one snapshot, so the extra 300 commits cost it only their
    # version-index entries -- a small fraction of what holding them all costs.
    assert (large_stream - small_stream) * 4 < large_whole - small_whole, (
        large_stream - small_stream,
        large_whole - small_whole,
    )
    # And at any one size it stays well below the materialised history.
    assert large_stream * 5 < large_whole, (large_stream, large_whole)
    assert small_stream * 5 < small_whole, (small_stream, small_whole)


def _json_bytes(value):
    import json as _json

    return _json.dumps(value, sort_keys=True).encode()


def test_iterated_snapshots_are_independent_objects(tmp_path) -> None:
    """A consumer mutating one yielded snapshot must not affect another."""
    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        for version in (1, 2):
            store.commit_process_result(
                {
                    "event_id": f"e{version}",
                    "invocation_id": f"i{version}",
                    "run_id": "r",
                    "kind": "process_completed",
                    "phase": version,
                    "state_version": version,
                },
                {
                    "run_id": "r",
                    "state_version": version,
                    "payload": _json_bytes({"shared": [1, 2, 3]}),
                },
                [],
            )
        seen = []
        for _version, snapshot in store.iter_state_history("r"):
            snapshot["shared"].append(99)
            seen.append(snapshot)
        assert seen[0] is not seen[1]
        assert seen[1]["shared"] == [1, 2, 3, 99], "a mutation must not leak between snapshots"
    finally:
        store.close()


def test_streamed_parquet_matches_a_whole_table_write(tmp_path) -> None:
    """STH-003: batching changes the file layout, never the rows."""
    import pyarrow.parquet as pq

    from genesis.analysis import AnalysisExporter

    rows = [{"a": index, "b": None if index % 3 else f"v{index}"} for index in range(50)]
    whole = AnalysisExporter.rows_to_parquet(list(rows), tmp_path / "whole.parquet")
    streamed = AnalysisExporter.stream_rows_to_parquet(
        iter(rows), tmp_path / "streamed.parquet", batch_rows=7
    )
    assert pq.read_table(whole).to_pylist() == pq.read_table(streamed).to_pylist()

    # A column first seen in a later batch is kept -- neither refused, which
    # failed exports of valid runs, nor dropped, which the whole-table writer
    # does silently because it infers columns from the first row.
    late = pq.read_table(
        AnalysisExporter.stream_rows_to_parquet(
            iter([{"a": 1}, {"a": 2, "late": 3}]), tmp_path / "late.parquet", batch_rows=1
        )
    )
    assert late.to_pylist() == [{"a": 1, "late": None}, {"a": 2, "late": 3}]

    # An empty relation still yields a readable file.
    empty = AnalysisExporter.stream_rows_to_parquet(iter([]), tmp_path / "empty.parquet")
    assert empty.exists()


def test_analysis_reads_project_away_fields_no_consumer_reads(tmp_path) -> None:
    """Evidence-only fields are served by export, not carried into analysis.

    `context` on an event and the provider exchange on an artifact are retained
    as evidence and read by nobody analysing a run, but they grow with the run.
    Export and the trace endpoints must still see the complete record.
    """
    import yaml

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = workspace / "pkg"
    source.mkdir()
    base = {"schema_version": "1.0", "study_id": "proj"}
    executor = {
        "mode": "computational",
        "parameters": {"entry_point": "tests.resume_executors:bump"},
    }
    files = {
        "study": {**base, "title": "p"},
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
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {**base, "outcomes": []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))

    service = GenesisService(workspace)
    try:
        build = service.compile_study(source, "builds/p")
        service.create_run({"id": "p1", "study_id": "proj", "build": build["path"]})
        service.execute_run("p1")

        evidence = service.trace_run("p1")
        analysis = service.trace_run("p1", evidence=False)
        assert any("context" in event for event in evidence), "evidence keeps the context"
        assert all("context" not in event for event in analysis)
        # Everything else is untouched.
        assert [
            {k: v for k, v in event.items() if k != "context"} for event in evidence
        ] == analysis

        art_evidence = service.artifacts_for_run("p1")
        art_analysis = service.artifacts_for_run("p1", evidence=False)
        assert len(art_evidence) == len(art_analysis)
        for full, projected in zip(art_evidence, art_analysis, strict=True):
            if isinstance(full["payload"], dict):
                assert not (
                    {"provider_attempts", "raw_response", "parsed_response"}
                    & set(projected["payload"])
                )
    finally:
        service.close()


def test_plan_aware_projection_is_conservative() -> None:
    """A plan that might read a field keeps it; only a provable miss drops it."""
    from genesis.service import GenesisService

    reads_state_delta = {
        "datasets": [{"id": "d", "source": {"kind": "events", "path": "state_delta.x"}}],
        "outcomes": [],
    }
    ignores_state_delta = {
        "datasets": [{"id": "d", "source": {"kind": "state", "snapshot": "final"}}],
        "outcomes": [],
    }
    assert GenesisService._unused_event_fields(reads_state_delta) == ()
    assert GenesisService._unused_event_fields(ignores_state_delta) == ("state_delta",)
    # No declared datasets: the legacy synthesis may read it, so keep everything.
    assert GenesisService._unused_event_fields({"datasets": [], "outcomes": []}) == ()

    assert GenesisService._plan_reads_artifact_relation({"datasets": [], "outcomes": []}) is True
    assert (
        GenesisService._plan_reads_artifact_relation(
            {"datasets": [{"id": "d", "source": {"kind": "artifacts"}}], "outcomes": []}
        )
        is True
    )
    assert (
        GenesisService._plan_reads_artifact_relation(
            {
                "datasets": [{"id": "d", "source": {"kind": "state"}}],
                "outcomes": [{"id": "o", "source": "artifacts"}],
            }
        )
        is True
    )
    assert (
        GenesisService._plan_reads_artifact_relation(
            {
                "datasets": [{"id": "d", "source": {"kind": "state"}}],
                "outcomes": [{"id": "o", "source": "d"}],
            }
        )
        is False
    )


# ---------------------------------------------------------------------------
# State-history scaling, Phase 2: append-aware patch storage
# ---------------------------------------------------------------------------


def _state_store(tmp_path, name="db"):
    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / f"{name}.sqlite", tmp_path / f"objects-{name}")
    store.create_run({"id": "r", "study_id": "s"})
    return store


def _commit_state(store, version, snapshot):
    from genesis.state_encoding import canonical_bytes

    store.commit_process_result(
        {
            "event_id": f"e{version}",
            "invocation_id": f"i{version}",
            "run_id": "r",
            "kind": "process_completed",
            "phase": version,
            "state_version": version,
        },
        {"run_id": "r", "state_version": version, "payload": canonical_bytes(snapshot)},
        [],
    )


def test_patch_encoding_round_trips_every_shape() -> None:
    """STH-006: a patch describes the transition exactly, whatever changed."""
    from genesis.state_encoding import apply_patch, canonical_bytes, encode_patch

    transitions = [
        ({}, {"a": 1}),
        ({"b": [1, 2]}, {"b": [1, 2, 3, 4]}),  # append: only the tail is stored
        ({"b": [1, 2, 3]}, {"b": [9, 9]}),  # rewritten, not extended
        ({"b": [1, 2, 3]}, {"b": [1, 2]}),  # shrunk
        ({"a": 1, "b": 2}, {"a": 1}),  # removal
        ({"a": None}, {"a": 1}),
        ({"a": 1}, {"a": None}),
        ({"x": {"k": [1]}}, {"x": {"k": [1, 2]}}),  # nested: replaced wholesale
    ]
    for previous, current in transitions:
        patch = encode_patch(previous, current)
        rebuilt = apply_patch(previous, patch)
        assert rebuilt == current, (previous, current, patch)
        # Byte-exactness matters: a commit's identity digests these bytes.
        assert canonical_bytes(rebuilt) == canonical_bytes(current)

    # A growing list is stored as its tail, not wholesale.
    grown = encode_patch({"b": [1, 2]}, {"b": [1, 2, 3, 4]})
    assert grown["fields"]["b"] == {"op": "append", "items": [3, 4]}


def test_patch_storage_reconstructs_exactly_and_shrinks(tmp_path) -> None:
    """STH-005/006 on an accumulating run: every version exact, far less stored."""
    from genesis.state_encoding import FORM_PATCH, canonical_bytes

    # Entries carry real content, as accumulating study state does; a ledger of
    # bare integers understates what patching saves.
    snapshots = [
        {"ledger": [{"i": i, "text": f"entry-{i}-" + "x" * 200} for i in range(n)], "n": n}
        for n in range(1, 121)
    ]
    store = _state_store(tmp_path)
    try:
        for version, snapshot in enumerate(snapshots, 1):
            _commit_state(store, version, snapshot)

        rows = store._state_rows("r")
        assert any(form == FORM_PATCH for _v, _r, form in rows), "patches must be used"

        stored = sum(store._object_size(ref) for _v, ref, _f in rows)
        whole = sum(len(canonical_bytes(s)) for s in snapshots)
        # Patching stores what changed; snapshots store the whole accumulation.
        assert stored * 10 < whole, (stored, whole)

        # Exhaustive, not sampled: every version, by value and by bytes.
        for version, expected in enumerate(snapshots, 1):
            rebuilt = store._reconstruct_state("r", version)
            assert rebuilt == expected, version
            assert canonical_bytes(rebuilt) == canonical_bytes(expected), version

        assert [s for _v, s in store.iter_state_history("r")] == snapshots
        _version, payload, _media = store.latest_state("r")
        assert payload == canonical_bytes(snapshots[-1])
    finally:
        store.close()


def test_commit_identity_is_unchanged_by_the_storage_form(tmp_path) -> None:
    """STH-007: a commit's hash must not depend on how its state was stored."""
    snapshots = [{"ledger": [{"i": i} for i in range(n)], "n": n} for n in range(1, 40)]

    def commit_hashes(force_whole: bool, name: str):
        store = _state_store(tmp_path, name)
        try:
            if force_whole:
                # Emulate the pre-change format: every commit stored whole.
                store._encode_state_for_storage = lambda run, prev, payload: ("base", payload)
            for version, snapshot in enumerate(snapshots, 1):
                _commit_state(store, version, snapshot)
            hashes = [
                row[0]
                for row in store.connection.execute("SELECT commit_hash FROM events ORDER BY rowid")
            ]
            states = [s for _v, s in store.iter_state_history("r")]
            return hashes, states
        finally:
            store.close()

    whole_hashes, whole_states = commit_hashes(True, "whole")
    patch_hashes, patch_states = commit_hashes(False, "patched")

    assert whole_hashes == patch_hashes, "storage form must not change commit identity"
    assert whole_states == patch_states == snapshots


def test_a_run_spanning_the_format_change_reconstructs(tmp_path) -> None:
    """STH-008: rows predating the encoding are bases; a resumed run mixes forms."""
    snapshots = [{"ledger": [{"i": i} for i in range(n)], "n": n} for n in range(1, 31)]
    store = _state_store(tmp_path)
    try:
        original = store._encode_state_for_storage
        store._encode_state_for_storage = lambda run, prev, payload: ("base", payload)
        for version, snapshot in enumerate(snapshots[:15], 1):
            _commit_state(store, version, snapshot)
        # Rows written before the encoding existed carry no form at all.
        store.connection.execute("UPDATE states SET form = NULL WHERE run_id='r'")

        store._encode_state_for_storage = original
        for version, snapshot in enumerate(snapshots[15:], 16):
            _commit_state(store, version, snapshot)

        forms = [form for _v, _r, form in store._state_rows("r")]
        assert forms.count(None) == 15, "the legacy half keeps a NULL form"
        assert any(form == "patch" for form in forms), "the resumed half uses patches"

        for version, expected in enumerate(snapshots, 1):
            assert store._reconstruct_state("r", version) == expected, version
        assert [s for _v, s in store.iter_state_history("r")] == snapshots
    finally:
        store.close()


def test_reconstruction_failure_names_the_run_and_version(tmp_path) -> None:
    """A patch whose base is gone must fail by name, never silently half-rebuild."""
    snapshots = [{"ledger": [{"i": i} for i in range(n)]} for n in range(1, 12)]
    store = _state_store(tmp_path)
    try:
        for version, snapshot in enumerate(snapshots, 1):
            _commit_state(store, version, snapshot)
        # Mark every row a patch: nothing is left to reconstruct from.
        store.connection.execute("UPDATE states SET form = 'patch' WHERE run_id='r'")
        with pytest.raises(ValueError, match="STATE_BASE_MISSING"):
            store._reconstruct_state("r", len(snapshots))
        with pytest.raises(ValueError, match="STATE_VERSION_MISSING"):
            store._reconstruct_state("r", 9999)
    finally:
        store.close()


# ---- Phase 3: event patch storage (STH-009..STH-013) ------------------------


def _phase3_store(tmp_path):
    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    store.create_run({"id": "r", "study_id": "s"})
    return store


def _commit(store, version, *, event_extra=None, state=None):
    event = {
        "event_id": f"e{version}",
        "invocation_id": f"i{version}",
        "run_id": "r",
        "kind": "process_completed",
        "phase": version,
        "state_version": version,
    }
    event.update(event_extra or {})
    store.commit_process_result(
        event,
        {
            "run_id": "r",
            "state_version": version,
            "payload": _json_bytes(state if state is not None else {"v": version}),
        },
        [],
    )
    return event


def test_events_store_as_patches_and_reconstruct_exactly(tmp_path) -> None:
    """An accumulating event ledger is patched, and reads are byte-exact."""
    store = _phase3_store(tmp_path)
    try:
        committed = []
        ledger: list[dict] = []
        for version in range(1, 25):
            ledger = [*ledger, {"n": version, "text": "y" * 200}]
            committed.append(_commit(store, version, event_extra={"context": {"ledger": ledger}}))
        forms = [form for _rowid, _ref, form in store._event_rows("r")]
        assert "patch" in forms, "an accumulating ledger must patch"

        read = store.list_events("r")
        assert read == committed
        # Identity is unaffected by storage form (STH-011).
        stored_hashes = [
            row[0]
            for row in store.connection.execute(
                "SELECT event_hash FROM events WHERE run_id = 'r' ORDER BY rowid"
            )
        ]
        assert stored_hashes == [
            hashlib.sha256(store._event_bytes(event)).hexdigest() for event in committed
        ]
    finally:
        store.close()


def test_event_patch_is_declined_when_it_would_be_larger(tmp_path) -> None:
    """STH-009: unrelated neighbours must not cause a storage regression."""
    store = _phase3_store(tmp_path)
    try:
        for version in range(1, 12):
            # Every event carries a wholly different payload, so a diff cannot
            # be smaller than the value itself.
            _commit(
                store,
                version,
                event_extra={"context": {f"k{version}": "z" * 300 * version}},
            )
        forms = [form for _rowid, _ref, form in store._event_rows("r")]
        assert set(forms) == {"base"}, forms
    finally:
        store.close()


def test_events_read_correctly_across_mixed_formats(tmp_path) -> None:
    """STH-008/STH-013: rows predating the encoding are read as whole payloads."""
    store = _phase3_store(tmp_path)
    try:
        ledger: list[dict] = []
        for version in range(1, 16):
            ledger = [*ledger, {"n": version, "text": "y" * 200}]
            _commit(store, version, event_extra={"context": {"ledger": ledger}})
        expected = store.list_events("r")
        # Simulate pre-migration rows: a NULL form must be read as a base.
        store.connection.execute(
            "UPDATE events SET form = NULL WHERE form = 'base' AND run_id = 'r'"
        )
        store.connection.commit()
        assert store.list_events("r") == expected
    finally:
        store.close()


def test_excluded_event_fields_do_not_corrupt_the_patch_chain(tmp_path) -> None:
    """STH-013: projection happens after reconstruction, not before."""
    store = _phase3_store(tmp_path)
    try:
        ledger: list[dict] = []
        for version in range(1, 20):
            ledger = [*ledger, {"n": version, "text": "y" * 200}]
            _commit(
                store,
                version,
                event_extra={"context": {"ledger": ledger}, "keep": version},
            )
        full = store.list_events("r")
        projected = store.list_events("r", exclude_fields=("context",))
        assert [event["keep"] for event in projected] == [event["keep"] for event in full]
        assert all("context" not in event for event in projected)
    finally:
        store.close()


def test_patched_snapshots_do_not_share_nested_values(tmp_path) -> None:
    """A patch leaves unchanged fields as the same objects; yields must not."""
    store = _phase3_store(tmp_path)
    try:
        big = ["x" * 100 for _ in range(200)]
        for version in (1, 2):
            _commit(
                store,
                version,
                state={"shared": [1, 2, 3], "big": big, "tick": version},
            )
        forms = [form for _version, _ref, form in store._state_rows("r")]
        assert "patch" in forms, "test needs a patched row to be meaningful"
        seen = []
        for _version, snapshot in store.iter_state_history("r"):
            snapshot["shared"].append(99)
            seen.append(snapshot)
        assert seen[0]["shared"] == [1, 2, 3, 99]
        assert seen[1]["shared"] == [1, 2, 3, 99]
    finally:
        store.close()


def _commit_growing(store, version, ledger):
    store.commit_process_result(
        {
            "event_id": f"e{version}",
            "invocation_id": f"i{version}",
            "run_id": "r",
            "kind": "process_completed",
            "phase": version,
            "state_version": version,
            "context": {"ledger": ledger},
        },
        {
            "run_id": "r",
            "state_version": version,
            "payload": _json_bytes({"ledger": ledger}),
        },
        [],
    )


def test_patch_chain_survives_a_cold_predecessor_cache(tmp_path) -> None:
    """Reopening mid-chain must reconstruct, not guess, the predecessor."""
    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    store.create_run({"id": "r", "study_id": "s"})
    ledger: list[dict] = []
    for version in range(1, 21):
        ledger = [*ledger, {"n": version, "text": "y" * 200}]
        _commit_growing(store, version, ledger)
    store.close()

    reopened = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        for version in range(21, 41):
            ledger = [*ledger, {"n": version, "text": "y" * 200}]
            _commit_growing(reopened, version, ledger)
        events = reopened.list_events("r")
        assert [event["phase"] for event in events] == list(range(1, 41))
        assert all(len(e["context"]["ledger"]) == e["phase"] for e in events)
        assert all(
            len(state["ledger"]) == version for version, state in reopened.iter_state_history("r")
        )
    finally:
        reopened.close()


def test_patch_chain_is_correct_when_another_writer_advances_it(tmp_path) -> None:
    """A cached predecessor must be ignored once it is no longer the tail."""
    from genesis.persistence import PersistenceCoordinator

    first = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    first.create_run({"id": "r", "study_id": "s"})
    ledger: list[dict] = []
    for version in range(1, 16):
        ledger = [*ledger, {"n": version, "text": "y" * 200}]
        _commit_growing(first, version, ledger)

    second = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        # The second writer advances the chain, so the first writer's cached
        # predecessor is stale and must not be used as a patch base.
        ledger = [*ledger, {"n": 16, "text": "y" * 200}]
        _commit_growing(second, 16, ledger)
        ledger = [*ledger, {"n": 17, "text": "y" * 200}]
        _commit_growing(first, 17, ledger)

        events = first.list_events("r")
        assert [event["phase"] for event in events] == list(range(1, 18))
        assert all(len(e["context"]["ledger"]) == e["phase"] for e in events)
    finally:
        second.close()
        first.close()


# ---- encoder: values that collide under == but differ in JSON --------------


def test_patch_encoding_distinguishes_booleans_from_the_integers_they_equal() -> None:
    """STH-007/STH-011: identity digests bytes, so `1` -> `true` is a change.

    Python holds ``True == 1`` and ``1 == 1.0``; JSON writes three different
    byte strings. Treating such a transition as unchanged drops it from the
    patch and the reconstructed record differs from the committed one in
    exactly the bytes its identity is taken from.
    """
    from genesis.state_encoding import apply_patch, canonical_bytes, encode_patch

    transitions = [
        ({"a": 1}, {"a": True}),
        ({"a": 0}, {"a": False}),
        ({"a": True}, {"a": 1}),
        ({"a": 1}, {"a": 1.0}),
        ({"a": 1.0}, {"a": 1}),
        ({"a": [1, 2]}, {"a": [True, 2]}),
        ({"a": {"b": 1}}, {"a": {"b": True}}),
        # An append whose retained prefix changed type is not an append.
        ({"a": [1, 2]}, {"a": [1.0, 2, 3]}),
    ]
    for previous, current in transitions:
        rebuilt = apply_patch(previous, encode_patch(previous, current))
        assert rebuilt == current, (previous, current, rebuilt)
        assert canonical_bytes(rebuilt) == canonical_bytes(current), (previous, current)


def test_patch_encoding_round_trips_arbitrary_nested_values() -> None:
    """Every transition must rebuild by value and by bytes, not most of them."""
    import random

    from genesis.state_encoding import apply_patch, canonical_bytes, encode_patch

    # Deliberately weighted towards values that compare equal across JSON types.
    ambiguous = [0, 1, 0.0, 1.0, True, False, 2, 2.0]
    random.seed(11)

    def value(depth: int = 0):
        roll = random.random()
        if depth > 3 or roll < 0.45:
            return random.choice([*ambiguous, "a", None, "x" * 10])
        if roll < 0.75:
            return [value(depth + 1) for _ in range(random.randint(0, 4))]
        return {f"k{i}": value(depth + 1) for i in range(random.randint(0, 4))}

    def mutate(item, depth: int = 0):
        if isinstance(item, dict):
            out = {
                key: (mutate(sub, depth + 1) if random.random() < 0.6 else sub)
                for key, sub in item.items()
            }
            if random.random() < 0.2:
                out[f"n{random.randint(0, 9)}"] = value(depth + 1)
            if out and random.random() < 0.2:
                out.pop(random.choice(list(out)))
            return out
        if isinstance(item, list):
            out = [mutate(sub, depth + 1) if random.random() < 0.5 else sub for sub in item]
            if random.random() < 0.5:
                out.extend(value(depth + 1) for _ in range(random.randint(1, 3)))
            return out
        return value(depth)

    for _ in range(2000):
        previous = {f"f{i}": value() for i in range(random.randint(0, 4))}
        current = mutate(previous)
        rebuilt = apply_patch(previous, encode_patch(previous, current))
        assert rebuilt == current
        assert canonical_bytes(rebuilt) == canonical_bytes(current)


def test_a_failing_commit_surfaces_its_own_error_not_the_rollback_s(tmp_path) -> None:
    """A failing COMMIT ends the transaction; rolling back then must not mask it."""
    import sqlite3

    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})

        class FailingCommit:
            """Ends the transaction and then fails, as a real I/O error does."""

            def __init__(self, connection):
                self._connection = connection
                self._armed = True

            def execute(self, sql, *args, **kwargs):
                if sql.strip() == "COMMIT" and self._armed:
                    self._armed = False
                    self._connection.execute("ROLLBACK")
                    raise sqlite3.OperationalError("disk I/O error")
                return self._connection.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._connection, name)

        real = store.connection
        store.connection = FailingCommit(real)
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            store.append_run_collection("r", "events", {"x": 1})
        store.connection = real
    finally:
        store.close()


def test_duckdb_and_python_emit_groups_in_the_same_order() -> None:
    """Enabling DuckDB must not reorder outcome rows.

    Outcome rows are exported to outcomes.json/csv/parquet and each exported
    file is hashed into the bundle's integrity manifest, so a different row
    order means the same run exports different digests depending on an
    environment variable.
    """
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")

    from genesis.analysis import AnalysisEngine, OutcomePlan, duckdb_aggregate

    cases = [
        [{"phase": 3, "x": 1}, {"phase": 1, "x": 2}, {"phase": 2, "x": 3}, {"phase": 1, "x": 4}],
        [{"g": "b", "x": 1}, {"g": "a", "x": 2}, {"g": "b", "x": None}],
        [{"g": None, "x": 1}, {"g": "a", "x": 2}],
        [{"phase": 2, "x": 1.5}, {"phase": 1, "x": None}, {"phase": 1, "x": 2}],
    ]
    for rows in cases:
        key = next(name for name in rows[0] if name != "x")
        for op in ("sum", "count", "mean"):
            plan = OutcomePlan(id="o", source="rows", select="x", aggregation=op, group_by=key)
            in_process = AnalysisEngine().evaluate(plan, {"rows": rows})
            through_duckdb = duckdb_aggregate(rows, select="x", op=op, group_by=key)
            assert through_duckdb == in_process, (op, rows, through_duckdb, in_process)


def test_empirically_seeded_follow_graph_survives_the_first_round(tmp_path) -> None:
    """An empirical initialization wraps its asset; consumers must unwrap it.

    ``follows`` is seeded as ``{origin, data_source, rows}``. Read as if it were
    the graph itself, that envelope iterates to its own keys -- ``"imported"``
    becomes ``['i','m','p',...]`` -- and because the writing process sets the
    field outright, one unwrapped read destroyed the graph in round 0 and it
    never recovered, leaving the recommender with nothing usable.
    """
    import json as _json

    # demos/ is local experiment scaffolding and is absent from a clean
    # checkout, so this regression runs only where the demo executors exist.
    demo_executors = pytest.importorskip("demos.demo_executors")
    from genesis.persistence import PersistenceCoordinator
    from genesis.runtime import (
        CallableExecutor,
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )
    from genesis.service import _load_empirical_data

    graph = {"u1": ["w3"], "u2": ["w3"], "u3": ["w4"]}
    asset = tmp_path / "initial-follows.json"
    asset.write_text(_json.dumps(graph))
    envelope = _load_empirical_data(asset, "data/initial-follows.json")
    assert set(envelope) == {"origin", "data_source", "rows"}

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        RunController(
            Scheduler(
                [
                    {
                        "id": "update-follow-relation",
                        "context_policy": "p",
                        "actors": [],
                        "state_effects": [{"field": "follows", "op": "set"}],
                        "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    }
                ]
            ),
            ExecutorRegistry(
                {
                    "update-follow-relation": CallableExecutor(
                        demo_executors.update_follow_relation, "computational"
                    )
                }
            ),
            ContextEngine({"p": {"allow": ["follows", "actions"]}}),
            state_store=StateStore(
                {"follows": dict, "actions": list},
                {"follows": envelope, "actions": []},
            ),
            persistence=store,
        ).run("r", phase_start=0, phase_end=2)

        latest = store.latest_json_state("r")
        follows = (latest[1] if latest else {}).get("follows")
        assert follows == graph, follows
    finally:
        store.close()


# ---- CTX-001..009: element-level context scoping ---------------------------


def _scoped_context(policy, actors=("w1",)):
    from genesis.runtime import ContextEngine, ProcessInvocation

    state = {
        "reflection": [{"actor": a, "note": f"n-{a}"} for a in ("w1", "w2", "u1", "u2")],
        "follows": {"w1": ["w2"], "u1": ["w1", "w2"]},
        "exposure-detail": {"u1": [{"article": "a1"}], "u2": [{"article": "a2"}]},
        "titles": [{"article_id": "a1"}, {"article_id": "a2"}],
    }
    invocation = ProcessInvocation(
        invocation_id="i", run_id="r", process_id="p", actor_ids=tuple(actors), phase=1
    )
    return ContextEngine({"p": policy}).build("p", invocation, state).data


def _seen(context):
    return [row["actor"] for row in context.get("reflection", ())]


def test_context_scope_expresses_own_selected_and_union(tmp_path) -> None:
    """CTX-002: one selector shape covers every visibility pattern."""
    own = {"field": "actor", "in": ["actor.ids"]}
    others = {"field": "actor", "in": ["state.follows.${actor}"]}
    both = {"field": "actor", "in": ["actor.ids", "state.follows.${actor}"]}

    assert _seen(_scoped_context({"allow": ["reflection"], "scope": {"reflection": own}})) == ["w1"]
    # CTX-003: self is never implicit -- w1 follows w2, and only w2 comes back.
    assert _seen(_scoped_context({"allow": ["reflection"], "scope": {"reflection": others}})) == [
        "w2"
    ]
    assert _seen(_scoped_context({"allow": ["reflection"], "scope": {"reflection": both}})) == [
        "w1",
        "w2",
    ]
    # CTX-009: absent scope is today's behaviour.
    assert _seen(_scoped_context({"allow": ["reflection"]})) == ["w1", "w2", "u1", "u2"]


def test_context_scope_handles_multi_actor_keys_and_missing_relations() -> None:
    """CTX-004/006: union over acting actors; an absent relation is empty, not an error."""
    own = {"field": "actor", "in": ["actor.ids"]}
    assert _seen(
        _scoped_context(
            {"allow": ["reflection"], "scope": {"reflection": own}}, actors=("w1", "u2")
        )
    ) == ["w1", "u2"]

    by_key = {"field": "__key__", "in": ["actor.ids"]}
    context = _scoped_context(
        {"allow": ["exposure-detail"], "scope": {"exposure-detail": by_key}}, actors=("u1",)
    )
    assert list(context["exposure-detail"]) == ["u1"]

    missing = {"field": "actor", "in": ["state.nowhere.${actor}"]}
    assert _seen(_scoped_context({"allow": ["reflection"], "scope": {"reflection": missing}})) == []


def test_context_scope_is_applied_before_the_cardinality_cap() -> None:
    """CTX-005: capping first would return a subset of an arbitrary prefix."""
    context = _scoped_context(
        {
            "allow": ["reflection"],
            "scope": {
                "reflection": {"field": "actor", "in": ["actor.ids", "state.follows.${actor}"]}
            },
            "cardinality": {"reflection": 1},
        }
    )
    # Scoped to w1 and w2, then capped to one: the first *entitled* row.
    assert _seen(context) == ["w1"]
    # Unscoped fields are untouched.
    full = _scoped_context(
        {
            "allow": ["reflection", "titles"],
            "scope": {"reflection": {"field": "actor", "in": ["actor.ids"]}},
        }
    )
    assert len(full["titles"]) == 2


def test_malformed_context_scope_fails_compilation(tmp_path) -> None:
    """CTX-008: a scope a run would depend on must be rejected at compile time."""
    from genesis.compiler import _validate_context_scope
    from genesis.specification.models import DomainSpec

    def errors_for(scope):
        domain = DomainSpec(
            schema_version="1.0",
            study_id="s",
            states=[{"id": "reflection", "value_type": "array", "initial": []}],
            visibility=[{"id": "p", "allow": ["reflection"], "scope": scope}],
        )
        return [item["message"] for item in _validate_context_scope(domain)]

    assert any(
        "does not allow" in m for m in errors_for({"absent": {"field": "a", "in": ["actor.ids"]}})
    )
    assert any("'field'" in m for m in errors_for({"reflection": {"in": ["actor.ids"]}}))
    assert any("'in'" in m for m in errors_for({"reflection": {"field": "actor"}}))
    assert any(
        "non-empty strings" in m for m in errors_for({"reflection": {"field": "actor", "in": [""]}})
    )
    assert errors_for({"reflection": {"field": "actor", "in": ["actor.ids"]}}) == []


def test_unbounded_context_is_reported_at_compilation() -> None:
    """CTX-008: handing a whole collection to every actor is a declared choice."""
    from genesis.compiler import _advise_unbounded_context
    from genesis.specification.models import DomainSpec

    def advisories(policy):
        domain = DomainSpec(
            schema_version="1.0",
            study_id="s",
            states=[
                {"id": "reflection", "value_type": "array", "initial": []},
                {"id": "headline", "value_type": "string", "initial": ""},
            ],
            visibility=[policy],
        )
        return [item["path"] for item in _advise_unbounded_context(domain)]

    assert advisories({"id": "p", "allow": ["reflection"]}) == [
        "domain.visibility.p.allow/reflection"
    ]
    # A scalar is not a collection, and any of scope/cardinality/aggregate settles it.
    assert advisories({"id": "p", "allow": ["headline"]}) == []
    assert (
        advisories(
            {
                "id": "p",
                "allow": ["reflection"],
                "scope": {"reflection": {"field": "actor", "in": ["actor.ids"]}},
            }
        )
        == []
    )
    assert advisories({"id": "p", "allow": ["reflection"], "cardinality": {"reflection": 5}}) == []


# ---- OUT-001..OUT-005: streaming outcome evaluation -------------------------


def test_outcome_sources_are_walked_once_per_relation_not_once_per_outcome() -> None:
    """OUT-002: eleven outcomes over one relation cost one pass, not eleven."""
    from genesis.analysis import AnalysisEngine, OutcomePlan

    walks = 0

    def events():
        nonlocal walks
        walks += 1
        for index in range(10):
            yield {"value": index, "group": index % 2, "time": index}

    plans = [
        OutcomePlan(id=f"o{n}", source="events", select="value", aggregation="sum")
        for n in range(11)
    ]
    results = AnalysisEngine().evaluate_many(plans, {"events": events})
    assert walks == 1, walks
    assert all(rows == [{"value_sum": 45, "value_missing": 0}] for rows in results)


def test_streaming_never_holds_the_source() -> None:
    """OUT-003: accumulator state is bounded by the output, not the input."""
    import tracemalloc

    from genesis.analysis import AnalysisEngine, OutcomePlan

    def events():
        for index in range(20000):
            yield {"value": index, "group": index % 4, "payload": "x" * 200}

    plan = OutcomePlan(id="o", source="events", select="value", aggregation="sum", group_by="group")
    tracemalloc.start()
    try:
        rows = AnalysisEngine().evaluate_many([plan], {"events": events})[0]
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(rows) == 4
    # 20,000 rows of ~200 bytes is ~4 MB of input; four groups of counters is not.
    assert peak < 1_000_000, peak


def test_group_rows_keep_first_appearance_order() -> None:
    """OUT-006: outcome files and their integrity digests depend on this order."""
    from genesis.analysis import AnalysisEngine, OutcomePlan

    rows = [
        {"group": "c", "value": 1},
        {"group": "a", "value": 2},
        {"group": "b", "value": 3},
        {"group": "a", "value": 4},
    ]
    plan = OutcomePlan(id="o", source="events", select="value", aggregation="sum", group_by="group")
    evaluated = AnalysisEngine().evaluate(plan, {"events": rows})
    assert [row["group"] for row in evaluated] == ["c", "a", "b"]


def test_hash_join_matches_the_nested_loop_and_scales_linearly() -> None:
    """OUT-005: same rows in the same order, without the quadratic."""
    import random
    import time

    from genesis.service import _hash_join

    left = [{"k": index % 3, "l": index} for index in range(6)]
    right = [{"k": index % 3, "r": index} for index in range(6)]

    def nested(left_rows, right_rows, on):
        merged = []
        for left_row in left_rows:
            for right_row in right_rows:
                if left_row.get(on) == right_row.get(on):
                    merged.append({**left_row, **right_row})
        return merged

    assert _hash_join(left, right, "k") == nested(left, right, "k")

    # An unhashable key cannot index, but it can still compare equal: {} == {}
    # is true and the nested loop matched such rows, so they must not vanish.
    for keys in ([{}], [[1]], [{"x": 1}], [(1,)], [None]):
        both = [{"k": keys[0], "l": 1}]
        other = [{"k": keys[0], "r": 2}]
        assert _hash_join(both, other, "k") == nested(both, other, "k"), keys

    mixed_keys = [1, "a", None, {}, {"x": 1}, [1], (1,), True, 0]
    random.seed(5)
    for _ in range(300):
        left_rows = [
            {"k": random.choice(mixed_keys), "l": index} for index in range(random.randint(0, 5))
        ]
        right_rows = [
            {"k": random.choice(mixed_keys), "r": index} for index in range(random.randint(0, 5))
        ]
        assert _hash_join(left_rows, right_rows, "k") == nested(left_rows, right_rows, "k")

    # Complexity, asserted with a margin noise cannot cross rather than a
    # ratio between two timings: 120,000 rows a side is 1.4e10 comparisons for
    # a nested loop -- hours -- and one pass each for a hash join.
    rows = [{"k": index, "v": index} for index in range(120_000)]
    start = time.perf_counter()
    joined = _hash_join(rows, rows, "k")
    duration = time.perf_counter() - start
    assert len(joined) == 120_000
    assert duration < 30, duration


def test_artifacts_are_read_one_at_a_time(tmp_path) -> None:
    """The payloads were read into a list, then parsed into a second list."""
    import json as _json

    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        for version in range(1, 6):
            store.commit_process_result(
                {
                    "event_id": f"e{version}",
                    "run_id": "r",
                    "kind": "process_completed",
                    "state_version": version,
                },
                {"run_id": "r", "state_version": version, "payload": _json_bytes({"v": version})},
                [
                    {
                        "artifact_id": f"a{version}",
                        "run_id": "r",
                        "payload": _json.dumps({"n": version, "bulk": "x" * 500}).encode(),
                    }
                ],
            )
        # The iterator yields the same rows the list form returned.
        assert list(store.iter_artifacts("r")) == store.list_artifacts("r")
        # And it is lazy: taking one row must not have read the rest.
        reads = 0
        original = PersistenceCoordinator._read_object

        def counting(self, digest):
            nonlocal reads
            reads += 1
            return original(self, digest)

        PersistenceCoordinator._read_object = counting
        try:
            first = next(iter(store.iter_artifacts("r")))
        finally:
            PersistenceCoordinator._read_object = original
        assert first["artifact_id"] == "a1"
        assert reads == 1, reads
    finally:
        store.close()


def test_round_snapshots_are_produced_one_at_a_time(tmp_path) -> None:
    """Only each round's winning snapshot is read, and none is held."""
    from genesis.persistence import PersistenceCoordinator
    from genesis.service import GenesisService

    workspace = tmp_path / "workspace"
    service = GenesisService(workspace)
    try:
        store: PersistenceCoordinator = service.persistence
        store.create_run({"id": "r", "study_id": "s"})
        ledger: list[int] = []
        for version in range(1, 13):
            ledger = [*ledger, version]
            store.commit_process_result(
                {
                    "event_id": f"e{version}",
                    "run_id": "r",
                    "kind": "process_completed",
                    "phase": (version - 1) // 4,
                    "state_version": version,
                },
                {
                    "run_id": "r",
                    "state_version": version,
                    "payload": _json_bytes({"ledger": ledger}),
                },
                [],
            )
        run = {**service.get_run("r"), "status": "completed"}
        annotations = [
            {"state_version": version, "phase": (version - 1) // 4, "kind": "process_completed"}
            for version in range(1, 13)
        ]
        rows = list(service._iter_round_annotated_state("r", run, annotations))
        # Three rounds of four commits: the final snapshot of each.
        assert [row["state_version"] for row in rows] == [4, 8, 12]
        assert [len(row["ledger"]) for row in rows] == [4, 8, 12]
        # The list wrapper must agree with the iterator exactly.
        assert service._round_annotated_state("r", run, annotations) == rows
    finally:
        service.close()


def test_cardinality_can_keep_the_most_recent_entries() -> None:
    """A bare cap keeps the oldest N, which is wrong for an append-ordered field."""
    from genesis.runtime import ContextEngine, ProcessInvocation

    state = {
        "analytics": [{"n": index, "round": index} for index in range(1, 9)],
        "detail": {f"u{index}": [index] for index in range(1, 6)},
    }

    def seen(policy, field="analytics"):
        invocation = ProcessInvocation(
            invocation_id="i", run_id="r", process_id="p", actor_ids=("a1",), phase=1
        )
        data = ContextEngine({"p": policy}).build("p", invocation, state).data
        value = data.get(field, ())
        return [row["n"] for row in value] if field == "analytics" else list(value)

    allow = {"allow": ["analytics"]}
    # The integer form is unchanged.
    assert seen({**allow, "cardinality": {"analytics": 3}}) == [1, 2, 3]
    assert seen({**allow, "cardinality": {"analytics": {"limit": 3, "keep": "first"}}}) == [1, 2, 3]
    # The most recent three, still in source order.
    assert seen({**allow, "cardinality": {"analytics": {"limit": 3, "keep": "last"}}}) == [6, 7, 8]
    assert seen(
        {**allow, "cardinality": {"analytics": {"limit": 3, "keep": "last", "by": "round"}}}
    ) == [6, 7, 8]
    assert seen({**allow, "cardinality": {"analytics": {"limit": 0, "keep": "last"}}}) == []
    by_key = {"limit": 2, "keep": "last", "by": "__key__"}
    assert seen({"allow": ["detail"], "cardinality": {"detail": by_key}}, field="detail") == [
        "u4",
        "u5",
    ]

    for malformed in ({"limit": 3, "keep": "middle"}, {"limit": -1}, {"limit": 3, "by": ""}):
        with pytest.raises(ValueError):
            seen({**allow, "cardinality": {"analytics": malformed}})


def test_empirically_seeded_fields_are_flagged_at_compilation() -> None:
    """The shape change that destroyed a follow graph should not be silent."""
    from genesis.compiler import _advise_empirical_envelope
    from genesis.specification.models import DomainSpec, OpennessSpec

    openness = OpennessSpec(
        schema_version="1.0",
        study_id="s",
        processes=[
            {
                "id": "update-follows",
                "executor": {"mode": "computational"},
                "context_policy": "p",
                "state_effects": [{"field": "follows", "op": "set"}],
            }
        ],
    )
    seeded = DomainSpec(
        schema_version="1.0",
        study_id="s",
        states=[{"id": "follows", "value_type": "object", "initial": {}}],
        initialization={
            "mode": "empirical",
            "state_field": "follows",
            "data_source": "data/initial-follows.json",
        },
    )
    advisories = _advise_empirical_envelope(seeded, openness)
    assert [item["code"] for item in advisories] == ["EMPIRICAL_ENVELOPE"]
    assert "update-follows" in advisories[0]["message"]
    assert "unwrap" in advisories[0]["message"]

    bare = DomainSpec(
        schema_version="1.0",
        study_id="s",
        states=[{"id": "follows", "value_type": "object", "initial": {}}],
    )
    assert _advise_empirical_envelope(bare, openness) == []


def test_a_purge_during_artifact_iteration_says_so(tmp_path) -> None:
    """Streaming releases the lock between rows; retention deletes artifacts.

    The list form read every payload under one lock hold, so a purge could not
    land mid-read. It can now, and reporting it as a missing object sends the
    reader looking for a corrupt store instead of a concurrent purge.
    """
    import json as _json

    from genesis.persistence import PersistenceCoordinator

    def populated():
        store = PersistenceCoordinator(tmp_path / f"{id(object())}.sqlite", tmp_path / "objects")
        store.create_run({"id": "r", "study_id": "s"})
        for version in range(1, 6):
            store.commit_process_result(
                {
                    "event_id": f"e{version}",
                    "run_id": "r",
                    "kind": "process_completed",
                    "state_version": version,
                },
                {"run_id": "r", "state_version": version, "payload": _json_bytes({"v": version})},
                [
                    {
                        "artifact_id": f"a{version}",
                        "run_id": "r",
                        "payload": _json.dumps({"n": version, "response": "raw"}).encode(),
                    }
                ],
            )
        return store

    store = populated()
    try:
        walk = store.iter_artifacts("r")
        next(walk)
        assert store.retention_purge("r") == 5
        with pytest.raises(ValueError, match="ARTIFACT_PURGED_DURING_READ"):
            list(walk)
    finally:
        store.close()

    # A genuinely unreadable object still reports itself as one.
    store = populated()
    try:
        walk = store.iter_artifacts("r")
        next(walk)
        (digest,) = store.connection.execute(
            "SELECT payload_ref FROM artifacts WHERE artifact_id = 'a3'"
        ).fetchone()
        (tmp_path / "objects" / digest[:2] / digest[2:]).unlink()
        with pytest.raises(ValueError, match="integrity check failed"):
            list(walk)
    finally:
        store.close()


def test_mistyped_context_declarations_are_refused_not_ignored() -> None:
    """A silently-ignored key here produces a plausible but wrong view.

    ``{"limit": 50, "kep": "last"}`` reads as "the fifty most recent" and, with
    the typo ignored, delivers the fifty oldest. Both return fifty rows, so
    nothing about the result says the declaration did not take effect.
    """
    from genesis.compiler import _validate_context_scope
    from genesis.runtime import ContextEngine, ProcessInvocation
    from genesis.specification.models import DomainSpec

    state = {"a": [{"actor": "w1", "n": index} for index in range(1, 6)]}

    def build(policy):
        invocation = ProcessInvocation(
            invocation_id="i", run_id="r", process_id="p", actor_ids=("w1",), phase=1
        )
        return ContextEngine({"p": policy}).build("p", invocation, state).data

    with pytest.raises(ValueError, match="unknown keys"):
        build({"allow": ["a"], "cardinality": {"a": {"limit": 2, "kep": "last"}}})
    with pytest.raises(ValueError, match="unknown keys"):
        build({"allow": ["a"], "scope": {"a": {"field": "actor", "in": ["actor.ids"], "wen": 1}}})
    # The correct spellings still work.
    assert [
        row["n"]
        for row in build({"allow": ["a"], "cardinality": {"a": {"limit": 2, "keep": "last"}}})["a"]
    ] == [4, 5]

    # And both are caught at compilation, not only during a run.
    def errors_for(policy):
        domain = DomainSpec(
            schema_version="1.0",
            study_id="s",
            states=[{"id": "a", "value_type": "array", "initial": []}],
            visibility=[{"id": "p", "allow": ["a"], **policy}],
        )
        return [item["message"] for item in _validate_context_scope(domain)]

    mistyped_cap = {"cardinality": {"a": {"limit": 2, "kep": 1}}}
    assert any("unknown keys" in m for m in errors_for(mistyped_cap))
    assert any(
        "unknown keys" in m
        for m in errors_for({"scope": {"a": {"field": "actor", "in": ["actor.ids"], "wen": 1}}})
    )
    assert errors_for({"cardinality": {"a": {"limit": 2, "keep": "last"}}}) == []


# ---- answer pool: local generative stand-in --------------------------------


def _pool_request(prompt, *, context="ctx", process="create-article"):
    from genesis.providers import ProviderRequest

    return ProviderRequest(model="m", prompt=prompt, context_hash=context, process_id=process)


def test_answer_pool_is_deterministic_and_follows_the_context() -> None:
    """The pool must vary with the context, or it cannot test that scoping lands.

    A canned answer does not read the context, but keying the *choice* on the
    context hash means a policy change that alters what an actor sees produces a
    different answer -- which is what makes "the context reached the model"
    observable rather than assumed.
    """
    from genesis.providers import AnswerPoolProvider

    pool = {"create-article": [{"creator": "w1", "phase": 0, "n": index} for index in range(8)]}
    provider = AnswerPoolProvider(pool, seed=3)
    prompt = "creator id: w1\nround (phase + 1): 2"

    first = provider.generate(_pool_request(prompt, context="A"))
    assert first.text == provider.generate(_pool_request(prompt, context="A")).text

    # Compared as mappings over many contexts, not as one pair: with eight
    # answers two draws coincide often enough that a single inequality would be
    # a coin flip.
    def drawn(engine):
        return {
            key: engine.generate(_pool_request(prompt, context=key)).text
            for key in (f"ctx-{index}" for index in range(20))
        }

    same_seed = drawn(AnswerPoolProvider(pool, seed=3))
    assert drawn(provider) == same_seed
    assert len(set(same_seed.values())) > 1, "the context must change the answer"
    assert drawn(AnswerPoolProvider(pool, seed=4)) != same_seed
    # It never claims to be a model.
    assert first.provider == "answer-pool"
    assert first.metadata["answer_pool"] is True


def test_answer_pool_echoes_the_acting_identity() -> None:
    """A pooled answer must not attribute itself to the actor it was recorded for."""
    from genesis.providers import AnswerPoolProvider

    pool = {"create-article": [{"creator": "w1", "phase": 0, "title": "t"}]}
    provider = AnswerPoolProvider(
        pool,
        seed=1,
        echo={"creator": r"creator id: (\S+)", "phase": r"round \(phase \+ 1\): (\d+)"},
    )
    answer = provider.generate(_pool_request("creator id: w4\nround (phase + 1): 3")).parsed
    assert answer["creator"] == "w4"
    # A recorded integer stays an integer.
    assert answer["phase"] == 3 and isinstance(answer["phase"], int)


def test_answer_pool_answers_a_process_it_has_never_seen() -> None:
    """Replay refuses an unknown invocation; the pool must not stop a run dead."""
    from genesis.providers import AnswerPoolProvider

    provider = AnswerPoolProvider({"create-article": [{"a": 1}]}, seed=0)
    answer = provider.generate(_pool_request("p", process="a-process-added-later"))
    assert answer.parsed == {"a": 1}

    with pytest.raises(ValueError, match="ANSWER_POOL"):
        AnswerPoolProvider({}, seed=0)


def test_answer_pool_profile_requires_a_pool_and_rejects_a_credential_shape(tmp_path) -> None:
    """A pooled profile is declared, so a pooled run is a property of the record."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        with pytest.raises(ValueError, match="ANSWER_POOL"):
            service.create_model_profile({"id": "no-pool", "provider": "answer-pool", "model": "m"})
        with pytest.raises(ValueError, match="ANSWER_POOL"):
            service.create_model_profile(
                {
                    "id": "bad-seed",
                    "provider": "answer-pool",
                    "model": "m",
                    "pool": "p.json",
                    "seed": "later",
                }
            )
        stored = service.create_model_profile(
            {"id": "pooled", "provider": "answer-pool", "model": "m", "pool": "p.json", "seed": 5}
        )
        assert stored["provider"] == "answer-pool"
        assert "api_key_env" not in stored and "base_url" not in stored
    finally:
        service.close()


# ---- Codex review of 7cf400b: verified findings, fail-first ----------------


def test_streamed_parquet_widens_types_instead_of_truncating(tmp_path) -> None:
    """H1: a column integral for its first batch must not truncate later fractions."""
    import pyarrow.parquet as pq

    from genesis.analysis import AnalysisExporter

    rows = [{"v": index} for index in range(512)] + [{"v": 7.5} for _ in range(88)]
    path = AnalysisExporter.stream_rows_to_parquet(rows, tmp_path / "s.parquet")
    values = pq.read_table(path).column("v").to_pylist()
    assert values[-88:] == [7.5] * 88
    assert values[:3] == [0.0, 1.0, 2.0]


def test_streamed_parquet_keeps_late_and_late_filled_columns(tmp_path) -> None:
    """M2: a valid run exports every column; a refused write leaves no file."""
    import pyarrow.parquet as pq

    from genesis.analysis import AnalysisExporter

    late = [{"a": index} for index in range(512)] + [{"a": 1, "b": 2}]
    table = pq.read_table(AnalysisExporter.stream_rows_to_parquet(late, tmp_path / "late.parquet"))
    assert table.column("b").to_pylist()[-1] == 2
    assert table.column("b").to_pylist()[0] is None

    null_first = [{"a": index, "b": None} for index in range(512)] + [{"a": 1, "b": 5}]
    table = pq.read_table(
        AnalysisExporter.stream_rows_to_parquet(null_first, tmp_path / "null.parquet")
    )
    assert table.column("b").to_pylist()[-1] == 5

    incompatible = [{"a": index} for index in range(512)] + [{"a": "text"}]
    target = tmp_path / "bad.parquet"
    with pytest.raises(ValueError, match="PARQUET_SCHEMA"):
        AnalysisExporter.stream_rows_to_parquet(incompatible, target)
    assert not list(tmp_path.glob("bad.parquet*"))


def test_streamed_parquet_accepts_a_row_factory(tmp_path) -> None:
    """Two passes need a re-iterable source; a factory keeps the history off-heap."""
    import pyarrow.parquet as pq

    from genesis.analysis import AnalysisExporter

    rows = [{"v": index} for index in range(600)] + [{"v": 0.5}]
    path = AnalysisExporter.stream_rows_to_parquet(lambda: iter(rows), tmp_path / "f.parquet")
    assert pq.read_table(path).column("v").to_pylist()[-1] == 0.5


def test_state_delta_is_kept_whenever_the_plan_can_reach_it() -> None:
    """H2: declaring any dataset must not blind outcomes that read state_delta."""
    from genesis.service import GenesisService

    unused = GenesisService._unused_event_fields
    state_only = {"id": "s", "source": {"kind": "state"}}
    assert (
        unused(
            {
                "datasets": [state_only],
                "outcomes": [
                    {
                        "id": "o",
                        "source": "events",
                        "aggregation": {"type": "count", "field": "state_delta"},
                    }
                ],
            }
        )
        == ()
    )
    reaches = [
        {"fields": [{"name": "tier", "op": "copy", "field": "state_delta.tier"}]},
        {"where": [{"field": "state_delta.tier", "op": "eq", "value": "premium"}]},
        {"deduplicate_on": ["state_delta.id"]},
    ]
    for extra in reaches:
        dataset = {"id": "d", "source": {"kind": "events", "path": "records"}, **extra}
        assert unused({"datasets": [dataset], "outcomes": []}) == (), extra
    # Nothing can reach it: it is still dropped.
    assert unused(
        {
            "datasets": [{"id": "d", "source": {"kind": "events", "path": "records"}}],
            "outcomes": [
                {"id": "o", "source": "d", "aggregation": {"type": "sum", "field": "score"}}
            ],
        }
    ) == ("state_delta",)


def test_keep_last_never_returns_fewer_rows_than_exist() -> None:
    """H5: a cap larger than the collection must keep all of it, from either end."""
    from genesis.runtime import _cap_cardinality

    for size in range(0, 12):
        for limit in range(0, 15):
            items = list(range(size))
            expected = min(size, limit)
            last = _cap_cardinality(items, {"limit": limit, "keep": "last"})
            first = _cap_cardinality(items, {"limit": limit, "keep": "first"})
            by = _cap_cardinality(
                [{"r": index} for index in items], {"limit": limit, "keep": "last", "by": "r"}
            )
            assert len(last) == len(first) == len(by) == expected, (size, limit)
            assert last == items[size - expected :]


def test_final_state_snapshot_is_the_last_commit_whatever_the_collapse_order(tmp_path) -> None:
    """H6: collapsed round rows must yield the same datasets as the full history.

    Compared against main's pipeline -- every committed version, in version
    order, annotated with its round -- for both snapshot kinds, over random
    phase sequences including interleaved rounds and unfinished runs.
    """
    import json as _json
    import random

    from genesis.outcome_plan import materialize_datasets
    from genesis.service import _INCOMPLETE_ROUND, GenesisService

    random.seed(21)
    for trial in range(120):
        service = GenesisService(tmp_path / f"ws-{trial}")
        try:
            store = service.persistence
            store.create_run({"id": "r", "study_id": "s"})
            count = random.randint(1, 8)
            phases = [random.choice([0, 1, 2, None]) for _ in range(count)]
            if all(phase is None for phase in phases):
                phases[0] = 0
            for version, phase in enumerate(phases, start=1):
                store.commit_process_result(
                    {
                        "event_id": f"e{version}",
                        "run_id": "r",
                        "kind": "process_completed",
                        "state_version": version,
                        **({"phase": phase} if phase is not None else {}),
                    },
                    {
                        "run_id": "r",
                        "state_version": version,
                        "payload": _json.dumps({"x": version * 100}, sort_keys=True).encode(),
                    },
                    [],
                )
            status = random.choice(["completed", "created"])
            run = {**service.get_run("r"), "status": status}
            annotations = [
                {
                    "event_id": f"e{version}",
                    "state_version": version,
                    "phase": phase,
                    "kind": "process_completed",
                }
                for version, phase in enumerate(phases, start=1)
            ]
            version_phase = {
                version: phase
                for version, phase in enumerate(phases, start=1)
                if isinstance(phase, int)
            }
            incomplete = (
                {max(version_phase.values())} if status != "completed" and version_phase else set()
            )
            full_history = []
            for version, snapshot in store.iter_state_history("r"):
                row = dict(snapshot)
                row["state_version"] = version
                phase = version_phase.get(version)
                row["_round"] = _INCOMPLETE_ROUND if phase in incomplete else phase
                full_history.append(row)
            collapsed = list(service._iter_round_annotated_state("r", run, annotations))
            for snapshot in ("final", "each_completed_round"):
                plan = {
                    "datasets": [{"id": "d", "source": {"kind": "state", "snapshot": snapshot}}]
                }
                want = materialize_datasets(plan, {"state": full_history})["d"]
                got = materialize_datasets(plan, {"state": collapsed})["d"]
                assert got == want, (trial, phases, status, snapshot, got, want)
        finally:
            service.close()


def test_mean_matches_statistics_mean_by_type_and_bytes() -> None:
    """M3: the streaming engine changed mean's type (3 -> 3.0) and its last bit."""
    import json as _json
    import random
    import statistics

    from genesis.analysis import AnalysisEngine, OutcomePlan

    random.seed(9)
    makers = [
        lambda: random.randint(-20, 20),
        lambda: random.uniform(-1e3, 1e3),
        lambda: random.choice([True, False, 1, 2.5, -3]),
    ]
    for _ in range(1500):
        make = random.choice(makers)
        values = [make() for _ in range(random.randint(1, 9))]
        missing = random.randint(0, 3)
        rows = [{"v": value} for value in values] + [{"v": None} for _ in range(missing)]
        for missingness in ("exclude", "zero"):
            plan = OutcomePlan(
                id="o", source="e", select="v", aggregation="mean", missingness=missingness
            )
            got = AnalysisEngine().evaluate(plan, {"e": rows})[0]["v_mean"]
            pool = values + ([0] * missing if missingness == "zero" else [])
            want = statistics.mean(pool)
            assert type(got) is type(want), (values, missingness, got, want)
            assert _json.dumps(got) == _json.dumps(want), (values, missingness, got, want)
        distribution = OutcomePlan(id="d", source="e", select="v", operation="distribution")
        got = AnalysisEngine().evaluate(distribution, {"e": rows})[0]["v_mean"]
        numeric = [value for value in values if isinstance(value, int | float)]
        want = statistics.mean(numeric) if numeric else None
        assert type(got) is type(want) and _json.dumps(got) == _json.dumps(want), (
            values,
            got,
            want,
        )


def test_state_history_refuses_to_rebuild_on_a_missing_object(tmp_path) -> None:
    """M4: a lost base must fail, not silently become {} under every later patch."""
    import json as _json

    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(tmp_path / "db.sqlite", tmp_path / "objects")
    try:
        store.create_run({"id": "r", "study_id": "s"})
        bulk = ["x" * 100 for _ in range(50)]
        for version in (1, 2, 3):
            store.commit_process_result(
                {"event_id": f"e{version}", "run_id": "r", "kind": "k", "state_version": version},
                {
                    "run_id": "r",
                    "state_version": version,
                    "payload": _json.dumps(
                        {"bulk": bulk, "tick": version}, sort_keys=True
                    ).encode(),
                },
                [],
            )
        rows = store._state_rows("r")
        assert [form for _v, _r, form in rows][1:] == ["patch", "patch"]
        digest = rows[0][1]
        (tmp_path / "objects" / digest[:2] / digest[2:]).unlink()
        with pytest.raises(ValueError):
            list(store.iter_state_history("r"))
    finally:
        store.close()


def test_selector_paths_must_name_a_real_root_and_state_field() -> None:
    """M5: 'stait.follows' compiled cleanly and selected nothing, silently."""
    from genesis.compiler import _advise_unbounded_context, _validate_context_scope
    from genesis.specification.models import DomainSpec

    def errors_for(selector):
        domain = DomainSpec(
            schema_version="1.0",
            study_id="s",
            states=[
                {"id": "a", "value_type": "array", "initial": []},
                {"id": "follows", "value_type": "object", "initial": {}},
            ],
            visibility=[
                {"id": "c", "allow": ["a"], "scope": {"a": {"field": "actor", "in": [selector]}}}
            ],
        )
        return _validate_context_scope(domain)

    for wrong in ("stait.follows", "actor.idz", "state.nowhere.${actor}", "follows"):
        assert errors_for(wrong), wrong
    for right in ("actor.ids", "state.follows", "state.follows.${actor}"):
        assert errors_for(right) == [], right

    namespaced = DomainSpec(
        schema_version="1.0",
        study_id="s",
        states=[{"id": "follows", "value_type": "object", "initial": {}}],
        visibility=[{"id": "c", "allow": ["state.follows"]}],
    )
    assert [item["path"] for item in _advise_unbounded_context(namespaced)] == [
        "domain.visibility.c.allow/state.follows"
    ]


def test_duckdb_path_matches_the_in_process_engine_by_bytes() -> None:
    """M6 and beyond: key order, key type, float sums and means all diverged."""
    import json as _json
    import random

    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")

    from genesis.analysis import AnalysisEngine, OutcomePlan, duckdb_aggregate

    random.seed(4)
    keys = [1, 2, 1.5, 2.5, "a", "b", True]
    values = [
        lambda: random.randint(-9, 9),
        lambda: random.uniform(-100, 100),
        lambda: random.choice([random.randint(-9, 9), random.uniform(-9, 9)]),
    ]
    for _ in range(400):
        key_pool = random.choice([keys[:2], keys[2:4], keys[:4], keys[4:6]])
        make = random.choice(values)
        rows = [{"g": random.choice(key_pool), "x": make()} for _ in range(random.randint(1, 10))]
        for op in ("sum", "count", "mean"):
            for group_by in (None, "g"):
                plan = OutcomePlan(
                    id="o", source="r", select="x", aggregation=op, group_by=group_by
                )
                in_process = AnalysisEngine().evaluate(plan, {"r": rows})
                through_duckdb = duckdb_aggregate(rows, select="x", op=op, group_by=group_by)
                assert _json.dumps(through_duckdb) == _json.dumps(in_process), (op, group_by, rows)


def _review_package(workspace, *, scope=None, cardinality=None, datasets=None, outcomes=None):
    import yaml

    workspace.mkdir(parents=True, exist_ok=True)
    source = workspace / "pkg"
    source.mkdir(exist_ok=True)
    base = {"schema_version": "1.0", "study_id": "review"}
    policy = {"id": "c", "allow": ["topic"]}
    if scope is not None:
        policy["scope"] = scope
    if cardinality is not None:
        policy["cardinality"] = cardinality
    files = {
        "study": {**base, "title": "review"},
        "openness": {
            **base,
            "processes": [
                {
                    "id": "stamp",
                    "executor": {
                        "mode": "computational",
                        "parameters": {"entry_point": "tests.review_executors:stamp"},
                    },
                    "context_policy": "c",
                    "actors": {"ids": ["a1", "a2"]},
                    "trigger": {"type": "phase", "phase": 0, "repeat": True},
                    "state_effects": [
                        {"field": "topic", "op": "set"},
                        {"field": "profile", "op": "set"},
                    ],
                }
            ],
        },
        "theory": {**base, "theory_family": "exploratory"},
        "domain": {
            **base,
            "visibility": [policy],
            "states": [
                {"id": "topic", "value_type": "string", "initial": ""},
                {"id": "profile", "value_type": "object", "initial": {}},
            ],
        },
        "protocol": {**base, "time_model": {"type": "rounds", "start": 0, "end": 2}},
        "outcomes": {**base, "datasets": datasets or [], "outcomes": outcomes or []},
        "models": {**base, "models": []},
    }
    for name, value in files.items():
        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return source


def test_analysis_keeps_the_context_a_dataset_declares(tmp_path) -> None:
    """H3: 'no analysis path reads context' was assumed, never enforced."""
    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    datasets = [
        {
            "id": "ev",
            "source": {"kind": "events"},
            "fields": [{"name": "ctx_topic", "op": "copy", "field": "context.topic"}],
        }
    ]
    outcomes = [
        {
            "id": "ctx",
            "source": "ev",
            "grouping": [],
            "aggregation": {"op": "count", "field": "ctx_topic"},
        }
    ]
    package = _review_package(workspace, datasets=datasets, outcomes=outcomes)
    service = GenesisService(workspace)
    try:
        build = service.compile_study(package, "builds/b")
        service.create_run({"id": "r", "study_id": "review", "build": build["path"]})
        service.execute_run("r")
        stamped = [event for event in service.trace_run("r") if event.get("process_id") == "stamp"]
        assert stamped and all("topic" in (event.get("context") or {}) for event in stamped)
        (row,) = service.evaluate_outcomes("r")
        assert row["ctx_topic_count"] == len(stamped), row
        assert row["ctx_topic_missing"] == 0, row
        assert all(item.get("ctx_topic") is not None for item in service._dataset_rows("r", "ev"))
    finally:
        service.close()


def test_malformed_context_declarations_are_reported_by_code(tmp_path) -> None:
    """H4: a malformed scope crashed the compiler with KeyError('code')."""
    from genesis.compiler import ValidationIssue
    from genesis.service import GenesisService

    cases = (
        ({"scope": {"topic.idds": {"field": "x", "in": ["actor.ids"]}}}, "CONTEXT_SCOPE_INVALID"),
        ({"scope": {"topic": {"field": "x", "in": ["stait.follows"]}}}, "CONTEXT_SCOPE_INVALID"),
        ({"cardinality": {"topic": {"limit": 2, "kep": "last"}}}, "CONTEXT_CARDINALITY_INVALID"),
    )
    for index, (declaration, code) in enumerate(cases):
        workspace = tmp_path / f"ws-{index}"
        package = _review_package(workspace, **declaration)
        service = GenesisService(workspace)
        try:
            with pytest.raises(ValidationIssue) as caught:
                service.compile_study(package, "builds/b")
            issue = caught.value
            records = next(
                (getattr(issue, name) for name in ("records", "issues") if hasattr(issue, name)),
                [],
            )
            codes = {getattr(record, "code", None) for record in records}
            assert code in codes or code in str(issue), (declaration, codes, str(issue))
        finally:
            service.close()


def test_answer_pool_echo_patterns_must_capture(tmp_path) -> None:
    """Low: a capture-less echo raised IndexError on every pooled answer."""
    from genesis.providers import AnswerPoolProvider
    from genesis.service import GenesisService

    for pattern in (r"creator id: \S+", r"creator id: ("):
        with pytest.raises(ValueError, match="ANSWER_POOL"):
            AnswerPoolProvider({"x": [{"creator": "w1"}]}, echo={"creator": pattern})
    service = GenesisService(tmp_path / "ws")
    try:
        with pytest.raises(ValueError, match="ANSWER_POOL"):
            service.create_model_profile(
                {
                    "id": "pooled",
                    "provider": "answer-pool",
                    "model": "m",
                    "pool": "p.json",
                    "echo": {"creator": r"creator id: \S+"},
                }
            )
    finally:
        service.close()


def test_pooled_profile_identity_follows_the_pool(tmp_path) -> None:
    """Low: two different pools resolved to the same recorded identity."""
    import json as _json

    from genesis.service import GenesisService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.json").write_text('{"x": [{"n": 1}]}')
    (workspace / "b.json").write_text('{"x": [{"n": 2}]}')
    service = GenesisService(workspace)
    try:
        base = {"provider": "answer-pool", "model": "m", "seed": 1, "echo": {}}
        identities = {
            service._profile_identity({**base, "pool": "a.json"}),
            service._profile_identity({**base, "pool": "b.json"}),
            service._profile_identity({**base, "pool": "a.json", "seed": 2}),
        }
        assert len(identities) == 3
        live = {
            "provider": "openai-compatible",
            "base_url": "https://example.test",
            "model": "m",
            "timeout": 60,
        }
        recorded_before = hashlib.sha256(
            _json.dumps(
                {
                    "base_url": "https://example.test",
                    "model": "m",
                    "timeout": 60,
                    "provider": "openai-compatible",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        assert service._profile_identity(live) == recorded_before
    finally:
        service.close()


def test_run_level_artifacts_are_projected_like_stored_ones(tmp_path) -> None:
    """Low: projection depended on where an artifact happened to be stored."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "ws")
    try:
        service.persistence.create_run({"id": "r", "study_id": "s"})
        service.persistence.append_run_collection(
            "r",
            "artifacts",
            {"artifact_id": "legacy", "payload": {"outputs": {"a": 1}, "raw_response": "raw"}},
        )
        (projected,) = [
            item
            for item in service.artifacts_for_run("r", evidence=False)
            if item.get("artifact_id") == "legacy"
        ]
        assert "raw_response" not in projected["payload"]
        (whole,) = [
            item for item in service.artifacts_for_run("r") if item.get("artifact_id") == "legacy"
        ]
        assert whole["payload"]["raw_response"] == "raw"
    finally:
        service.close()


def test_outcome_evaluation_hands_relations_over_without_listing_them(
    tmp_path, monkeypatch
) -> None:
    """M1: both evaluation branches listed whole relations before using them.

    The dataset branch listed the ledger, every artifact and every round
    snapshot before materialising any dataset. The legacy branch listed every
    artifact row -- each carrying its outputs -- before synthesising flat rows;
    on an accumulating study that was 342 MB at 40 rounds.
    """
    import genesis.service as service_module
    from genesis.service import GenesisService

    seen: dict[str, object] = {}
    real_materialize = service_module.materialize_datasets

    def spy_materialize(plan, sources):
        seen["dataset_sources"] = {
            name: callable(value)
            for name, value in sources.items()
            if name in ("events", "artifacts", "state")
        }
        return real_materialize(plan, sources)

    real_legacy = GenesisService._legacy_derived_rows

    def spy_legacy(event_rows, artifact_rows):
        seen["legacy_artifacts_listed"] = isinstance(artifact_rows, list)
        return real_legacy(event_rows, artifact_rows)

    monkeypatch.setattr(service_module, "materialize_datasets", spy_materialize)
    monkeypatch.setattr(GenesisService, "_legacy_derived_rows", staticmethod(spy_legacy))

    shapes = {
        "datasets": dict(
            datasets=[{"id": "fin", "source": {"kind": "state", "snapshot": "final"}}],
            outcomes=[
                {
                    "id": "n",
                    "source": "fin",
                    "grouping": [],
                    "aggregation": {"op": "count", "field": "topic"},
                }
            ],
        ),
        "legacy": dict(
            datasets=[],
            outcomes=[
                {
                    "id": "n",
                    "source": "events",
                    "grouping": [],
                    "aggregation": {"op": "count", "field": "phase"},
                }
            ],
        ),
    }
    for label, declaration in shapes.items():
        workspace = tmp_path / label
        package = _review_package(workspace, **declaration)
        service = GenesisService(workspace)
        try:
            build = service.compile_study(package, "builds/b")
            service.create_run({"id": "r", "study_id": "review", "build": build["path"]})
            service.execute_run("r")
            assert service.evaluate_outcomes("r")
        finally:
            service.close()

    assert seen["dataset_sources"] == {"events": True, "artifacts": True, "state": True}
    assert seen["legacy_artifacts_listed"] is False

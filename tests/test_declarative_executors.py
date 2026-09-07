from __future__ import annotations

from genesis.providers import ProviderExecutor, ProviderResponse
from genesis.runtime import (
    CallableExecutor,
    ContextEngine,
    ExecutorRegistry,
    ProcessInvocation,
    ProcessResult,
    RuleExecutor,
    RunController,
    Scheduler,
    StateStore,
    StateTransitionExecutor,
)


def _call(*, inputs=None, context=None) -> ProcessInvocation:
    return ProcessInvocation(
        "invocation-1",
        "run-1",
        "process-1",
        actor_ids=("creator-1",),
        inputs=inputs or {},
        context=context or {},
    )


def test_rule_executor_emits_declared_result_for_first_matching_rule() -> None:
    executor = RuleExecutor(
        {
            "rules": [
                {
                    "when": {"path": "inputs.detection.value.detected", "op": "eq", "value": True},
                    "outputs": {"governance-decision": {"penalty": 0.5}},
                    "events": [{"type": "penalty-applied"}],
                }
            ],
            "default": {"outputs": {"governance-decision": {"penalty": 0.0}}},
        }
    )

    result = executor.execute(_call(inputs={"detection": {"value": {"detected": True}}}))

    assert result.outputs["governance-decision"] == {"penalty": 0.5}
    assert result.events == ({"type": "penalty-applied"},)
    assert result.metadata["mode"] == "rule"
    assert result.metadata["matched_rule"] == 0


def test_rule_executor_combines_condition_and_typed_artifact_predicates() -> None:
    executor = RuleExecutor(
        {
            "rules": [
                {
                    "when": {
                        "all": [
                            {
                                "path": "condition.factors.governance",
                                "op": "eq",
                                "value": "disclosed",
                            },
                            {
                                "path": "artifacts.clickbait-detection.0.detected",
                                "op": "eq",
                                "value": True,
                            },
                        ]
                    },
                    "outputs": {"governance-decision": {"penalty": 0.5, "disclosed": True}},
                    "state_effects": {"revenue": {"creator-1": -1}},
                }
            ]
        }
    )
    invocation = ProcessInvocation(
        "governance-1",
        "run-1",
        "apply-governance",
        inputs={
            "detection-1": {
                "artifact_type": "clickbait-detection",
                "value": {"detected": True},
            }
        },
        condition={"factors": {"governance": "disclosed"}},
    )

    result = executor.execute(invocation)

    assert result.outputs["governance-decision"] == {
        "penalty": 0.5,
        "disclosed": True,
    }
    assert result.state_effects["revenue"] == {"creator-1": -1}


def test_state_transition_executor_computes_declared_state_operations() -> None:
    executor = StateTransitionExecutor(
        {
            "operations": [
                {"op": "increment", "state": "revenue", "value": -2},
                {"op": "append", "state": "memory", "value_from": "inputs.action.value"},
                {
                    "op": "add-relation",
                    "state": "follows",
                    "value": {"source": "user-1", "target": "creator-1"},
                },
            ]
        }
    )

    result = executor.execute(
        _call(
            inputs={"action": {"value": {"kind": "share"}}},
            context={"revenue": 10, "memory": [], "follows": []},
        )
    )

    assert result.state_effects["revenue"] == 8
    assert result.state_effects["memory"] == ({"kind": "share"},)
    assert result.state_effects["follows"] == ({"source": "user-1", "target": "creator-1"},)
    assert result.metadata["mode"] == "state-transition"


def test_state_transition_executor_removes_a_declared_relation() -> None:
    executor = StateTransitionExecutor(
        {
            "operations": [
                {
                    "op": "remove-relation",
                    "state": "follows",
                    "value": {"source": "user-1", "target": "creator-1"},
                }
            ]
        }
    )

    result = executor.execute(
        _call(
            context={
                "follows": [
                    {"source": "user-1", "target": "creator-1"},
                    {"source": "user-2", "target": "creator-2"},
                ]
            }
        )
    )

    assert result.state_effects["follows"] == ({"source": "user-2", "target": "creator-2"},)


def test_callable_executor_preserves_a_complete_process_result() -> None:
    expected = ProcessResult(
        outputs={"x": 1},
        state_effects={"counter": 2},
        events=({"type": "changed"},),
        scheduling_effects=({"type": "signal_event", "event": "changed"},),
    )

    actual = CallableExecutor(lambda _invocation: expected, "extension").execute(_call())

    assert actual is expected


def test_state_transition_applies_nested_collection_effects_at_runtime_boundary() -> None:
    controller = RunController(
        Scheduler(
            [
                {
                    "id": "update-relations",
                    "context_policy": "state-context",
                    "state_effects": [{"field": "follows", "op": "add-relation"}],
                }
            ]
        ),
        ExecutorRegistry(
            {
                "update-relations": StateTransitionExecutor(
                    {
                        "operations": [
                            {
                                "op": "add-relation",
                                "state": "follows",
                                "value": {"source": "user-1", "target": "creator-1"},
                            }
                        ]
                    }
                )
            }
        ),
        ContextEngine({"state-context": {"allow": ["follows"]}}),
        state_store=StateStore({"follows": list}, {"follows": []}),
    )

    controller.run("run-1", phase_limit=1)

    assert controller.state_store.snapshot()["follows"] == [
        {"source": "user-1", "target": "creator-1"}
    ]


class ParsedProvider:
    def generate(self, _request):
        return ProviderResponse(
            text='{"detected": true}',
            parsed={"detected": True},
            provider="fixture",
            model="detector-v1",
            request_id="request-1",
        )


def test_semantic_evaluator_preserves_raw_and_parsed_outputs() -> None:
    executor = ProviderExecutor(
        ParsedProvider(),
        model="detector-v1",
        mode="semantic-evaluator",
        output_key="clickbait-detection",
        output_schema={
            "type": "object",
            "properties": {"detected": {"type": "boolean"}},
            "required": ["detected"],
        },
    )

    result = executor.execute(_call(context={"article": {"title": "Example"}}))

    assert result.outputs == {"clickbait-detection": {"detected": True}}
    assert result.metadata["mode"] == "semantic-evaluator"
    assert result.metadata["raw_response"] == '{"detected": true}'
    assert result.metadata["parsed_response"] == {"detected": True}

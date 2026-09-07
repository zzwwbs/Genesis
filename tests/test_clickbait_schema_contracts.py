from __future__ import annotations

import pytest
from pydantic import ValidationError

from genesis.specification.models import ExecutorBinding, ProcessSpec, ProtocolSpec


@pytest.mark.parametrize(
    ("mode", "parameters"),
    [
        ("rule", {"rules": [{"when": {"path": "inputs.score", "op": "gt", "value": 0}}]}),
        (
            "state-transition",
            {"operations": [{"op": "increment", "state": "revenue", "value": 1}]},
        ),
    ],
)
def test_declarative_executor_modes_accept_required_programs(
    mode: str, parameters: dict[str, object]
) -> None:
    binding = ExecutorBinding.model_validate({"mode": mode, "parameters": parameters})

    assert binding.mode == mode


def test_semantic_evaluator_requires_a_frozen_model_profile() -> None:
    binding = ExecutorBinding.model_validate(
        {"mode": "semantic-evaluator", "model_profile": "clickbait-detector"}
    )

    assert binding.model_profile == "clickbait-detector"
    with pytest.raises(ValidationError, match="semantic-evaluator executor requires model_profile"):
        ExecutorBinding.model_validate({"mode": "semantic-evaluator"})


def test_rule_and_transition_modes_require_declarative_programs() -> None:
    with pytest.raises(ValidationError, match="rule executor requires parameters.rules"):
        ExecutorBinding.model_validate({"mode": "rule"})
    with pytest.raises(
        ValidationError, match="state-transition executor requires parameters.operations"
    ):
        ExecutorBinding.model_validate({"mode": "state-transition"})


def test_process_accepts_state_backed_actor_selector() -> None:
    process = ProcessSpec.model_validate(
        {
            "id": "creator-strategy",
            "actors": {"source": "population.creators", "id_field": "creator_id"},
            "executor": {
                "mode": "semantic-evaluator",
                "model_profile": "creator-model",
            },
            "context_policy": "creator-context",
        }
    )

    assert process.actors is not None
    assert process.actors.source == "population.creators"
    assert process.actors.id_field == "creator_id"
    assert process.actors.fan_out is True


def test_actor_selector_requires_exactly_one_source_form() -> None:
    base = {
        "id": "creator-strategy",
        "executor": {"mode": "deterministic"},
        "context_policy": "creator-context",
    }
    with pytest.raises(ValidationError, match="exactly one of ids or source"):
        ProcessSpec.model_validate({**base, "actors": {}})
    with pytest.raises(ValidationError, match="exactly one of ids or source"):
        ProcessSpec.model_validate(
            {**base, "actors": {"ids": ["creator-1"], "source": "population.creators"}}
        )


def test_protocol_accepts_factorial_design_and_treatment_phases() -> None:
    protocol = ProtocolSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "clickbait-mini",
            "time_model": {"type": "rounds", "start": 1, "end": 6, "step": 1},
            "factors": [
                {"id": "governance", "levels": ["none", "opaque", "disclosed"]},
                {"id": "peer-visibility", "levels": ["low", "high"]},
            ],
            "phases": [
                {"id": "baseline", "start": 1, "end": 3},
                {"id": "treatment", "start": 4, "end": 6},
            ],
            "matching": {
                "enabled": True,
                "shared_streams": ["initial-world", "conventional"],
            },
        }
    )

    assert [factor.id for factor in protocol.factors] == ["governance", "peer-visibility"]
    assert protocol.phases[1].start == 4


def test_protocol_rejects_empty_factor_levels_and_overlapping_phases() -> None:
    base = {
        "schema_version": "1.0",
        "study_id": "clickbait-mini",
        "time_model": {"type": "rounds", "start": 1, "end": 6},
    }
    with pytest.raises(ValidationError):
        ProtocolSpec.model_validate({**base, "factors": [{"id": "governance", "levels": []}]})
    with pytest.raises(ValidationError, match="protocol phases must not overlap"):
        ProtocolSpec.model_validate(
            {
                **base,
                "phases": [
                    {"id": "baseline", "start": 1, "end": 4},
                    {"id": "treatment", "start": 4, "end": 6},
                ],
            }
        )

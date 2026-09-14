"""A declared run length the engine would ignore is refused at compile.

The runtime reads `time_model.end` only as a whole number and never reads
`termination`, so `end: 12.0` compiled clean and the run went on to the default
100 rounds, calling models in every one.
"""

from __future__ import annotations

from typing import Any

from genesis.compiler import _validate_run_length
from genesis.specification.models import ProtocolSpec


def _codes(time_model: dict[str, Any], termination: list[Any] | None = None) -> list[str]:
    protocol = ProtocolSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "s",
            "time_model": time_model,
            "termination": termination or [],
        }
    )
    warnings: list[dict[str, Any]] = []
    errors = _validate_run_length(protocol, warnings)
    return [item["code"] for item in errors + warnings]


def test_a_non_integer_end_is_refused() -> None:
    assert _codes({"type": "rounds", "end": 12.0}) == ["TIME_MODEL_INVALID"]
    assert _codes({"type": "rounds", "end": "12"}) == ["TIME_MODEL_INVALID"]
    assert _codes({"type": "rounds", "start": 1, "end": 12}) == []


def test_a_missing_end_warns_that_the_default_applies() -> None:
    assert _codes({"type": "rounds"}) == ["TIME_MODEL_END_DEFAULT"]


def test_only_an_end_time_matching_the_time_model_is_accepted() -> None:
    rounds = {"type": "rounds", "end": 6}
    assert _codes(rounds, [{"type": "end_time", "at": 6, "early_stopping": False}]) == []
    assert _codes(rounds, [{"type": "end_time", "at": 4}]) == ["TERMINATION_UNSUPPORTED"]
    assert _codes(rounds, [{"type": "phase-end", "value": 6}]) == ["TERMINATION_UNSUPPORTED"]
    assert _codes(rounds, [{"type": "end_time", "at": 6, "early_stopping": True}]) == [
        "TERMINATION_UNSUPPORTED"
    ]

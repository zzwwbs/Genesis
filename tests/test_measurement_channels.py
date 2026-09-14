"""The compile-time isolation check must know every artifact a measurement produces.

produced_refs read only declared ``outputs``, but the runtime stores an artifact
for any catalog-named output key -- including one supplied by a retry policy's
fallback, or declared by a recorded_artifact executor. A consumer gated off a
measurement was not flagged when it read the measurement through one of those
channels. (The run-time gate is now by producer and does not depend on this.)
"""

from __future__ import annotations

from genesis.measurement import produced_refs

GATE_OFF = {"id": "c", "factors": {"governance": "none"}}
USE = {"source": "detect", "when": {"path": "condition.governance", "op": "eq", "value": "strict"}}


def _consumer() -> dict:
    return {"id": "sanction", "inputs": ["detection"], "measurement_use": [dict(USE)]}


def test_a_fallback_only_artifact_is_withheld_when_the_gate_is_off() -> None:
    detect = {
        "id": "detect",
        "outputs": [{"artifact_type": "score"}],
        "retry_policy": {"max_attempts": 2, "fallback_outputs": {"detection": {"missing": True}}},
    }
    assert "detection" in produced_refs(detect)


def test_a_recorded_artifact_output_is_withheld_too() -> None:
    detect = {
        "id": "detect",
        "executor": {"mode": "recorded_artifact", "parameters": {"outputs": {"detection": {}}}},
    }
    assert "detection" in produced_refs(detect)


def test_the_ordinary_channel_is_unchanged() -> None:
    detect = {"id": "detect", "outputs": [{"artifact_type": "detection"}]}
    assert produced_refs(detect) == {"detect", "detection"}

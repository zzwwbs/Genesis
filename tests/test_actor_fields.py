"""An output field that names the actor is written by the engine, not the model.

The clickbait detector is drawn once per article and was asked to write back
which article it scored. It was never shown the id, and wrote "unknown", "" or
the title in most calls, so settlement joined its scores to nothing and the
sanction missed most of what it should have judged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from genesis.compiler import _advise_rounds_without_repeat, _validate_engine_fields
from genesis.runtime import (
    ArtifactStore,
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    _stamp_engine_fields,
)
from genesis.specification.models import OpennessSpec, OutputSpec

DETECT = {
    "id": "detect",
    "actors": ["a-1-w1", "a-1-w2"],
    "context_policy": "none",
    "outputs": [
        {"artifact_type": "detection", "schema_ref": "detection", "actor_fields": ["article_id"]}
    ],
}


class _Careless:
    def execute(self, invocation: Any) -> ProcessResult:
        return ProcessResult(outputs={"detection": {"article_id": "unknown", "score": 9}})


class _Reader:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    def execute(self, invocation: Any) -> ProcessResult:
        self.seen.extend(record["value"] for record in invocation.inputs.values())
        return ProcessResult()


def test_the_engine_writes_the_actor_over_what_the_executor_returned() -> None:
    reader = _Reader()
    validated: list[Any] = []

    def validator(schema_ref: str, value: Any) -> list[str]:
        validated.append(value)
        return []

    processes = [
        DETECT,
        {"id": "settle", "context_policy": "none", "after": ["detect"], "inputs": ["detection"]},
    ]
    RunController(
        Scheduler(processes),
        ExecutorRegistry({"detect": _Careless(), "settle": reader}),
        ContextEngine({"none": {"allow": []}}),
        artifact_store=ArtifactStore({"detection": {"id": "detection", "schema_ref": "detection"}}),
        output_schema_validator=validator,
    ).run("r", phase_limit=1)

    assert sorted(item["article_id"] for item in reader.seen) == ["a-1-w1", "a-1-w2"]
    assert all(item["score"] == 9 for item in reader.seen)
    # The schema is checked against what will be committed, not the model's text.
    assert sorted(item["article_id"] for item in validated) == ["a-1-w1", "a-1-w2"]


def test_what_the_executor_wrote_is_kept_when_it_differed() -> None:
    stamped = _stamp_engine_fields(_Careless().execute(None), DETECT, ("a-1-w1",), 1)
    assert stamped.outputs["detection"]["article_id"] == "a-1-w1"
    assert stamped.metadata["engine_fields_replaced"] == {"detection": {"article_id": "unknown"}}
    agreeing = ProcessResult(outputs={"detection": {"article_id": "a-1-w1"}})
    unchanged = _stamp_engine_fields(agreeing, DETECT, ("a-1-w1",), 1)
    assert "engine_fields_replaced" not in unchanged.metadata


def test_an_invocation_with_several_actors_fails_rather_than_guessing() -> None:
    result = _stamp_engine_fields(_Careless().execute(None), DETECT, ("a-1-w1", "a-1-w2"), 1)
    assert result.status == "failed"
    assert result.metadata["code"] == "OUTPUT_ACTOR_FIELD_AMBIGUOUS"


def test_an_undeclared_actor_field_leaves_the_canonical_form_unchanged() -> None:
    assert OutputSpec(artifact_type="a", schema_ref="s").model_dump() == {
        "artifact_type": "a",
        "schema_ref": "s",
    }


def _openness(process: dict[str, Any]) -> OpennessSpec:
    base = {"id": "p", "executor": {"mode": "deterministic"}, "context_policy": "none"}
    return OpennessSpec.model_validate(
        {"schema_version": "1.0", "study_id": "s", "processes": [{**base, **process}]}
    )


def test_the_compiler_refuses_an_actor_field_it_could_not_write(tmp_path: Path) -> None:
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "detection.json").write_text(
        json.dumps(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"article_id": {"type": "string"}, "score": {"type": "integer"}},
            }
        )
    )

    def codes(actors: Any, fields: list[str]) -> list[str]:
        output = {"artifact_type": "detection", "schema_ref": "detection", "actor_fields": fields}
        spec = _openness({"actors": actors, "outputs": [output]})
        return [error["message"] for error in _validate_engine_fields(spec, tmp_path)]

    assert codes(["a"], ["article_id"]) == []
    assert "no actors" in codes(None, ["article_id"])[0]
    assert "no field 'nope'" in codes(["a"], ["nope"])[0]
    assert "typed 'integer'" in codes(["a"], ["score"])[0]


def test_the_compiler_refuses_a_phase_field_that_is_not_a_number(tmp_path: Path) -> None:
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "reflection.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"phase": {"type": "integer"}, "text": {"type": "string"}},
            }
        )
    )

    def messages(fields: list[str]) -> list[str]:
        output = {"artifact_type": "reflection", "schema_ref": "reflection", "phase_fields": fields}
        spec = _openness({"outputs": [output]})
        return [error["message"] for error in _validate_engine_fields(spec, tmp_path)]

    assert messages(["phase"]) == []
    assert "a round is integer or number" in messages(["text"])[0]


class _Misdated:
    def execute(self, invocation: Any) -> ProcessResult:
        return ProcessResult(outputs={"reflection": {"phase": invocation.phase + 1, "text": "t"}})


def test_the_engine_writes_the_round_over_what_the_executor_returned() -> None:
    reader = _Reader()
    processes = [
        {
            "id": "reflect",
            "context_policy": "none",
            "trigger": {"phase": 0, "repeat": True},
            "outputs": [
                {"artifact_type": "reflection", "schema_ref": "r", "phase_fields": ["phase"]}
            ],
        },
        {
            "id": "read",
            "context_policy": "none",
            "trigger": {"phase": 0, "repeat": True},
            "after": ["reflect"],
            "inputs": ["reflection"],
        },
    ]
    RunController(
        Scheduler(processes),
        ExecutorRegistry({"reflect": _Misdated(), "read": reader}),
        ContextEngine({"none": {"allow": []}}),
        artifact_store=ArtifactStore({"reflection": {"id": "reflection", "schema_ref": "r"}}),
    ).run("r", phase_limit=3)
    assert sorted({item["phase"] for item in reader.seen}) == [0, 1, 2]


def test_a_list_of_rounds_without_repeat_is_flagged() -> None:
    rounds = {"path": "protocol.phase", "op": "in", "value": [3, 6, 9]}
    once = _openness({"trigger": {"type": "condition", "predicate": {"all": [rounds]}}})
    [warning] = _advise_rounds_without_repeat(once)
    assert warning["code"] == "TRIGGER_ROUNDS_WITHOUT_REPEAT" and "[3, 6, 9]" in warning["message"]
    repeating = _openness({"trigger": {"type": "condition", "predicate": rounds, "repeat": True}})
    single = _openness({"trigger": {"type": "condition", "predicate": {**rounds, "value": [3]}}})
    assert _advise_rounds_without_repeat(repeating) == []
    assert _advise_rounds_without_repeat(single) == []


def test_a_state_effect_derived_from_the_output_carries_the_stamped_id() -> None:
    from genesis.runtime import StateStore

    store = StateStore({"detections": list}, {"detections": []})
    process = {
        **DETECT,
        "executor": {"mode": "generative"},
        "state_effects": [{"field": "detections", "op": "append", "from": "detection"}],
    }
    RunController(
        Scheduler([process]),
        ExecutorRegistry({"detect": _Careless()}),
        ContextEngine({"none": {"allow": []}}),
        state_store=store,
    ).run("r", phase_limit=1)
    assert sorted(item["article_id"] for item in store.snapshot()["detections"]) == [
        "a-1-w1",
        "a-1-w2",
    ]


def test_a_closed_schema_without_properties_refuses_an_engine_field(tmp_path: Path) -> None:
    """Skipped when there was no properties block (2026-09-14 M11)."""
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "closed.json").write_text(
        json.dumps({"type": "object", "additionalProperties": False})
    )
    output = {"artifact_type": "c", "schema_ref": "closed", "phase_fields": ["phase"]}
    [error] = _validate_engine_fields(_openness({"outputs": [output]}), tmp_path)
    assert "no field 'phase'" in error["message"]


def test_a_union_type_admitting_the_engine_value_is_accepted(tmp_path: Path) -> None:
    """["string", "null"] was refused as not a string (2026-09-14 M12)."""
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "u.json").write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "who": {"type": ["string", "null"]},
                    "when": {"type": ["integer", "null"]},
                    "bad": {"type": ["boolean", "null"]},
                },
            }
        )
    )

    def messages(kind: str, name: str) -> list[str]:
        output = {"artifact_type": "u", "schema_ref": "u", kind: [name]}
        spec = _openness({"actors": ["a"], "outputs": [output]})
        return [error["message"] for error in _validate_engine_fields(spec, tmp_path)]

    assert messages("actor_fields", "who") == []
    assert messages("phase_fields", "when") == []
    assert "typed" in messages("actor_fields", "bad")[0]

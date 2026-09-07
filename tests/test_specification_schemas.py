from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from genesis.specification import (
    DomainSpec,
    ModelsSpec,
    OpennessSpec,
    OutcomesSpec,
    ProtocolSpec,
    StudySpec,
    TheorySpec,
)

FIXTURES = Path(__file__).parent / "fixtures" / "specification"


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        ("study.yaml", StudySpec),
        ("openness.yaml", OpennessSpec),
        ("theory.yaml", TheorySpec),
        ("domain.yaml", DomainSpec),
        ("protocol.yaml", ProtocolSpec),
        ("outcomes.yaml", OutcomesSpec),
        ("models.yaml", ModelsSpec),
    ],
)
def test_valid_canonical_artifact_fixtures_parse(filename, model):
    value = model.model_validate(yaml.safe_load((FIXTURES / filename).read_text()))
    assert value.schema_version == "1.0"
    assert value.study_id == "platform-governance"


def test_unknown_top_level_field_is_rejected():
    with pytest.raises(ValidationError, match="extra_field"):
        StudySpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "platform-governance",
                "title": "x",
                "extra_field": True,
            }
        )


def test_namespaced_extensions_are_allowed():
    value = StudySpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "title": "x",
            "extensions": {"acme.demo": {"enabled": True}},
        }
    )
    assert value.extensions["acme.demo"]["enabled"] is True


@pytest.mark.parametrize("filename", ["study.yaml", "openness.yaml", "protocol.yaml"])
def test_invalid_fixture_is_rejected(filename):
    model = {
        "study.yaml": StudySpec,
        "openness.yaml": OpennessSpec,
        "protocol.yaml": ProtocolSpec,
    }[filename]
    with pytest.raises(ValidationError):
        model.model_validate(yaml.safe_load((FIXTURES / "invalid" / filename).read_text()))


def test_json_schema_is_strict_and_has_stable_identifiers():
    schema = StudySpec.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "study_id" in schema["required"]


def test_all_canonical_json_schemas_are_available():
    from genesis.specification.schemas import all_schemas

    schemas = all_schemas()
    assert set(schemas) == {
        "study",
        "openness",
        "theory",
        "domain",
        "protocol",
        "outcomes",
        "models",
    }
    assert all(schema["additionalProperties"] is False for schema in schemas.values())


def test_stable_ids_round_trip_across_nested_objects():
    value = OpennessSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "processes": [{"id": "make-decision", "executor": {}, "context_policy": "private"}],
        }
    )
    assert value.processes[0].id == "make-decision"


def test_generative_process_requires_declared_openness_and_trace_contract():
    with pytest.raises(ValidationError, match="openness_rationale"):
        OpennessSpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "platform-governance",
                "processes": [
                    {
                        "id": "choose",
                        "executor": {"mode": "generative", "model_profile": "gpt"},
                        "context_policy": "private",
                        "prompt_ref": "choose-prompt",
                        "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                        "trace_policy": {"record_context": True},
                    }
                ],
            }
        )


def test_generative_process_has_typed_executor_prompt_output_and_trace_policy():
    value = OpennessSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "processes": [
                {
                    "id": "choose",
                    "openness_rationale": "strategy is generated",
                    "closure_rationale": "a bounded menu would remove the observed variation",
                    "executor": {"mode": "generative", "model_profile": "gpt"},
                    "context_policy": "private",
                    "prompt_ref": "choose-prompt",
                    "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                    "trace_policy": {"record_context": True},
                }
            ],
        }
    )
    process = value.processes[0]
    assert process.executor.mode == "generative"
    assert process.outputs[0].schema_ref == "strategy-schema"


@pytest.mark.parametrize("field", ["id", "context_policy", "prompt_ref"])
def test_nested_identifier_fields_reject_non_stable_ids(field):
    payload = {
        "id": "choose",
        "openness_rationale": "r",
        "executor": {"mode": "generative", "model_profile": "gpt"},
        "context_policy": "private",
        "prompt_ref": "choose-prompt",
        "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
        "trace_policy": {},
    }
    payload[field] = "Not Stable"
    with pytest.raises(ValidationError):
        OpennessSpec.model_validate(
            {"schema_version": "1.0", "study_id": "platform-governance", "processes": [payload]}
        )


def test_extension_namespace_and_semantic_version_are_validated():
    with pytest.raises(ValidationError):
        StudySpec.model_validate(
            {"schema_version": "v1", "study_id": "platform-governance", "title": "x"}
        )
    with pytest.raises(ValidationError):
        StudySpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "platform-governance",
                "title": "x",
                "extensions": {"bad namespace!": {}},
            }
        )


@pytest.mark.parametrize("mode", ["extension", "recorded_artifact"])
def test_specialized_executor_modes_are_supported(mode):
    value = OpennessSpec.model_validate(
        {
            "schema_version": "1.0",
            "study_id": "platform-governance",
            "processes": [
                {
                    "id": "replay",
                    "executor": (
                        {"mode": mode, "extension_ref": "fixture-extension"}
                        if mode == "extension"
                        else {"mode": mode}
                    ),
                    "context_policy": "private",
                }
            ],
        }
    )
    assert value.processes[0].executor.mode == mode


def test_generative_executor_requires_model_profile():
    with pytest.raises(ValidationError, match="model_profile"):
        OpennessSpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "platform-governance",
                "processes": [
                    {
                        "id": "choose",
                        "openness_rationale": "r",
                        "executor": {"mode": "generative"},
                        "context_policy": "private",
                        "prompt_ref": "choose-prompt",
                        "outputs": [{"artifact_type": "strategy", "schema_ref": "strategy-schema"}],
                    }
                ],
            }
        )


def test_approval_confirmation_is_cross_field_consistent():
    base = {"schema_version": "1.0", "study_id": "platform-governance", "title": "x"}
    with pytest.raises(ValidationError):
        StudySpec.model_validate({**base, "approval": {"status": "confirmed"}})
    with pytest.raises(ValidationError):
        StudySpec.model_validate({**base, "approval": {"status": "proposed", "confirmed_by": "r"}})


def test_exported_json_schema_patterns_are_anchored():
    schema = StudySpec.model_json_schema()
    assert schema["properties"]["study_id"]["pattern"].startswith("^")
    assert schema["properties"]["study_id"]["pattern"].endswith("$")
    assert schema["properties"]["schema_version"]["pattern"].startswith("^")
    assert schema["properties"]["schema_version"]["pattern"].endswith("$")

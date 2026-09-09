"""SCH — authoritative JSON Schema Draft 2020-12 enforcement (G5).

Covers the fixed-dialect validator, compile-time schema checks, local-only
reference resolution, runtime no-state-write enforcement, and the typed
diagnostic/limit contract from the research-control gap-fixing specification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.schema_validation import (
    SCHEMA_DEFINITION_INVALID,
    SCHEMA_DIALECT_UNSUPPORTED,
    SCHEMA_REFERENCE_INVALID,
    SCHEMA_VALIDATION_LIMIT,
    PackageSchemaCatalog,
    SchemaDiagnostic,
    SchemaValidationError,
    validate_schema,
)

SCORE_SCHEMA = {
    "type": "integer",
    "minimum": 0,
    "maximum": 10,
}


def _catalog(schemas: dict[str, object] | None = None) -> PackageSchemaCatalog:
    return PackageSchemaCatalog(schemas if schemas is not None else {"score": SCORE_SCHEMA})


# ---------------------------------------------------------------------------
# Fixed dialect and schema definition validation (SCH-001, SCH-004)
# ---------------------------------------------------------------------------


def test_catalog_validates_schemas_against_2020_12_metaschema() -> None:
    with pytest.raises(SchemaValidationError) as exc:
        PackageSchemaCatalog({"bad": {"type": "integer", "minimum": "not-a-number"}})
    assert exc.value.code == SCHEMA_DEFINITION_INVALID


def test_catalog_rejects_unsupported_declared_dialect() -> None:
    with pytest.raises(SchemaValidationError) as exc:
        PackageSchemaCatalog(
            {"schema7": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "string"}}
        )
    assert exc.value.code == SCHEMA_DIALECT_UNSUPPORTED


def test_catalog_accepts_absent_dialect_as_package_dialect() -> None:
    catalog = _catalog()
    assert catalog.validate("score", 5) == []
    # An absent $schema is treated as the package dialect, not rejected.


def test_catalog_accepts_boolean_and_empty_schemas() -> None:
    catalog = _catalog({"anything": {}, "never": False, "all": True})
    assert catalog.validate("anything", {"x": 1}) == []
    assert catalog.validate("all", "whatever") == []
    assert catalog.validate("never", "whatever") != []
    with pytest.raises(SchemaValidationError):
        catalog.validate("missing-schema", 1)


# ---------------------------------------------------------------------------
# Bounds, enums, composition and conditionals (SCH-003)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("score", [-1, 11])
def test_rejects_score_outside_declared_bounds(score: int) -> None:
    diagnostics = _catalog().validate("score", score)
    assert diagnostics
    assert diagnostics[0].keyword == "minimum" or diagnostics[0].keyword == "maximum"
    assert isinstance(diagnostics[0], SchemaDiagnostic)


def test_accepts_boundary_scores() -> None:
    catalog = _catalog()
    assert catalog.validate("score", 0) == []
    assert catalog.validate("score", 10) == []


def test_boolean_value_does_not_satisfy_integer_declaration() -> None:
    diagnostics = _catalog().validate("score", True)
    assert diagnostics
    assert diagnostics[0].keyword == "type"


def test_wrong_enum_is_rejected() -> None:
    catalog = _catalog({"mode": {"enum": ["full", "partial"]}})
    assert catalog.validate("mode", "branch") != []
    assert catalog.validate("mode", "full") == []


def test_unexpected_field_is_rejected_when_additional_properties_false() -> None:
    catalog = _catalog(
        {
            "detection": {
                "type": "object",
                "properties": {"detected": {"type": "boolean"}},
                "required": ["detected"],
                "additionalProperties": False,
            }
        }
    )
    assert catalog.validate("detection", {"detected": True, "stray": 1}) != []
    assert catalog.validate("detection", {"detected": False}) == []


def test_null_is_rejected_for_declared_integer() -> None:
    assert _catalog().validate("score", None) != []


def test_nested_array_items_are_validated() -> None:
    catalog = _catalog(
        {
            "list": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            }
        }
    )
    assert catalog.validate("list", [1, 2, -3]) != []
    assert catalog.validate("list", [1, 2, 3]) == []


def test_local_defs_and_references_resolve_within_package() -> None:
    catalog = _catalog(
        {
            "bounded": {
                "$defs": {"score": {"type": "integer", "minimum": 0, "maximum": 5}},
                "$ref": "#/$defs/score",
            },
            "alias": {"$ref": "bounded"},
        }
    )
    assert catalog.validate("bounded", 3) == []
    assert catalog.validate("bounded", 6) != []
    # A package-internal reference to another schema id resolves locally.
    assert catalog.validate("alias", 3) == []
    assert catalog.validate("alias", 6) != []


def test_local_fragment_under_nested_id_scopes_to_the_subschema() -> None:
    """F5: a subschema declaring its own ``$id`` is an embedded resource; a
    local fragment (``#/$defs/x``) inside it must resolve against that
    ``$id`` as the new document base, exactly like the Draft 2020-12
    validator — not against the enclosing schema's ``$defs``."""
    catalog = _catalog(
        {
            "a": {
                "type": "object",
                "properties": {
                    "leaf": {
                        "$id": "https://schema.example.org/nested",
                        "$defs": {"x": {"type": "integer"}},
                        "type": "object",
                        "properties": {"value": {"$ref": "#/$defs/x"}},
                    }
                },
            }
        }
    )
    assert catalog.validate("a", {"leaf": {"value": 9}}) == []
    assert catalog.validate("a", {"leaf": {"value": "not-an-int"}}) != []


def test_external_reference_is_rejected_at_construction() -> None:
    # F13: an external/unresolvable reference must fail at catalog
    # construction, before any model invocation, not only during validation.
    with pytest.raises(SchemaValidationError) as exc:
        _catalog({"escaped": {"$ref": "https://example.com/outside-schema"}})
    assert exc.value.code == SCHEMA_REFERENCE_INVALID


def test_one_of_ambiguity_and_conditionals() -> None:
    catalog = _catalog(
        {
            "value": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
            },
            "conditional": {
                "if": {"properties": {"a": {"const": 1}}, "required": ["a"]},
                "then": {"required": ["b"]},
                "else": {"required": ["c"]},
            },
        }
    )
    assert catalog.validate("value", True) != []
    assert catalog.validate("value", "ok") == []
    assert catalog.validate("conditional", {"a": 1}) != []
    assert catalog.validate("conditional", {"a": 1, "b": 2}) == []
    assert catalog.validate("conditional", {"a": 2, "c": 3}) == []
    assert catalog.validate("conditional", {"a": 2}) != []


def test_empty_schema_is_permissive_and_warns() -> None:
    catalog = _catalog({"anything": {}})
    assert catalog.permissive_schemas() == ["anything"]
    assert "anything" not in catalog.permissive_schemas()[:0]


# ---------------------------------------------------------------------------
# Non-JSON values (SCH-003): NaN and infinity are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_numbers(value: float) -> None:
    catalog = _catalog({"num": {"type": "number"}})
    diagnostics = catalog.validate("num", value)
    assert diagnostics
    assert diagnostics[0].keyword == "non-finite"


# ---------------------------------------------------------------------------
# Typed diagnostics and structured pointers
# ---------------------------------------------------------------------------


def test_diagnostics_carry_structured_metadata() -> None:
    catalog = _catalog(
        {
            "wrapper": {
                "type": "object",
                "properties": {"score": SCORE_SCHEMA},
                "required": ["score"],
            }
        }
    )
    diagnostics = catalog.validate("wrapper", {"score": 99})
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "OUTPUT_SCHEMA_INVALID"
    assert diagnostic.schema_id == "wrapper"
    assert diagnostic.instance_pointer == "/score"
    assert diagnostic.keyword == "maximum"
    assert diagnostic.schema_pointer  # non-empty JSON pointer into the schema


def test_missing_required_field_reports_path_and_keyword() -> None:
    catalog = _catalog(
        {
            "wrapper": {
                "type": "object",
                "properties": {"score": SCORE_SCHEMA},
                "required": ["score"],
            }
        }
    )
    diagnostics = catalog.validate("wrapper", {})
    assert diagnostics
    assert diagnostics[0].keyword == "required"
    assert diagnostics[0].instance_pointer == ""


# ---------------------------------------------------------------------------
# Validation limits (SCH acceptance: bounded pathological input)
# ---------------------------------------------------------------------------


def test_excessively_deep_instance_hits_typed_limit() -> None:
    depth = PackageSchemaCatalog.INSTANCE_MAX_DEPTH + 5
    nested = 1
    for _ in range(depth):
        nested = {"next": nested}
    with pytest.raises(SchemaValidationError) as exc:
        _catalog().validate("score", nested)
    assert exc.value.code == SCHEMA_VALIDATION_LIMIT


def test_excessively_deep_schema_hits_typed_limit() -> None:
    schema: dict[str, object] = {"type": "integer"}
    for _ in range(PackageSchemaCatalog.SCHEMA_MAX_DEPTH + 5):
        schema = {"$defs": {"n": schema}, "$ref": "#/$defs/n"}
    with pytest.raises(SchemaValidationError) as exc:
        PackageSchemaCatalog({"deep": schema})
    assert exc.value.code == SCHEMA_VALIDATION_LIMIT


# ---------------------------------------------------------------------------
# Compatibility: the legacy human-readable validator still enforces bounds
# ---------------------------------------------------------------------------


def test_legacy_validate_schema_enforces_numeric_bounds() -> None:
    errors = validate_schema(SCORE_SCHEMA, 99)
    assert errors
    assert "maximum" in errors[0]


def test_legacy_validate_schema_reports_paths() -> None:
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "integer"}},
    }
    errors = validate_schema(schema, {"answer": "not-an-int"})
    assert errors and "answer" in errors[0]
    assert validate_schema(schema, {"answer": 4}) == []
    assert validate_schema({"type": "array", "items": {"type": "integer"}}, [1, "x"]) == [
        "root[1]: expected integer, got str"
    ]


# ---------------------------------------------------------------------------
# Runtime no-state-write enforcement (SCH-002 acceptance)
# ---------------------------------------------------------------------------


class _BoundedProvider:
    """Returns an out-of-bounds score on the first call, valid afterwards."""

    provider = "openai-compatible"

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, request: object) -> object:
        import json as _json

        from genesis.providers import ProviderResponse

        value = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return ProviderResponse(
            _json.dumps(value),
            self.provider,
            getattr(request, "model", "m"),
            f"req-{self.calls}",
            parsed=value,
        )


def test_invalid_output_through_execution_does_not_mutate_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCH-002: a schema-invalid output must not commit state or a successful artifact."""
    from genesis.providers import ProviderExecutor
    from genesis.service import GenesisService

    object_score = {
        "type": "object",
        "required": ["score"],
        "properties": {"score": dict(SCORE_SCHEMA)},
        "additionalProperties": False,
    }
    catalog = _catalog({"score": object_score})
    provider = _BoundedProvider([{"score": 99}])
    executor = ProviderExecutor(
        provider,
        model="m",
        output_schema=object_score,
        output_schema_validator=lambda value: catalog.validate("score", value),
        mode="generative",
        max_repairs=1,
    )
    invocation = object.__new__(type("I", (), {"context": {"x": 1}, "actor_ids": (), "phase": 0}))
    result = executor.execute(invocation)
    assert result.status == "failed"
    assert result.metadata["code"] == "OUTPUT_VALIDATION_FAILED"
    assert result.metadata["schema_valid"] is False
    assert "maximum" in " ".join(str(e) for e in result.metadata.get("validation_errors", []))

    # And a full run: an out-of-bounds documented output must not be committed.
    # The study is trivially approvable; the real bound executor is injected as
    # the run's override so enforcement flows through the runtime commit path.
    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "schema-bound-study",
                "title": "schema bound",
                "processes": [
                    {
                        "id": "measure",
                        "executor": {"mode": "deterministic"},
                        "context_policy": "public",
                        "state_effects": [{"field": "counter", "op": "set"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                "domain": {
                    "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "models": [],
            }
        )
        service.approve_specification("schema-bound-study", draft["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/schema-bound-study", specification_id="schema-bound-study"
        )
        service.create_run(
            {"id": "bound-run", "study_id": "schema-bound-study", "build": compiled["path"]}
        )
        with pytest.raises(Exception, match="process measure failed"):
            service.execute_run("bound-run", executor_overrides={"measure": executor})
        assert service.get_run("bound-run")["status"] == "failed"
        assert any(
            event.get("kind") == "process_failed"
            and event.get("process_id") == "measure"
            and event.get("metadata", {}).get("code") == "OUTPUT_VALIDATION_FAILED"
            for event in service.trace_run("bound-run")
        )
        # Persisted state remains untouched: initial snapshot only.
        history = service.persistence.list_state_history("bound-run")
        states = [snapshot for _version, snapshot in history]
        assert len(states) == 1
        assert states[0].get("counter") == 0
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F11 (effect): depth limits measure true nesting, not sibling count
# ---------------------------------------------------------------------------


def test_flat_schema_with_many_properties_is_not_deep() -> None:
    """F11: sibling properties must not count as nesting depth."""
    flat = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(60)}}
    catalog = PackageSchemaCatalog({"flat": flat})
    assert catalog.validate("flat", {f"p{i}": "x" for i in range(60)}) == []


def test_flat_array_of_many_empty_objects_is_not_deep() -> None:
    """F11: a list of many shallow siblings must not trip the depth limit."""
    catalog = _catalog({"anything": {}})
    assert catalog.validate("anything", [{} for _ in range(90)]) == []


# ---------------------------------------------------------------------------
# F13 (effect): boolean schemas compile and unresolved refs fail early
# ---------------------------------------------------------------------------


def test_boolean_schema_compiles_into_build(tmp_path: Path) -> None:
    """F13: a boolean schema asset must pass the compiler loader."""
    from genesis.compiler import StudyCompiler

    source = tmp_path / "package"
    source.mkdir(parents=True)
    for name, value in {
        "study": {"schema_version": "1.0", "study_id": "bool-schema", "title": "x"},
        "openness": {"schema_version": "1.0", "study_id": "bool-schema", "processes": []},
        "theory": {
            "schema_version": "1.0",
            "study_id": "bool-schema",
            "theory_family": "institutional",
        },
        "domain": {"schema_version": "1.0", "study_id": "bool-schema"},
        "protocol": {
            "schema_version": "1.0",
            "study_id": "bool-schema",
            "time_model": {"type": "rounds"},
        },
        "outcomes": {"schema_version": "1.0", "study_id": "bool-schema", "outcomes": []},
        "models": {"schema_version": "1.0", "study_id": "bool-schema", "models": []},
    }.items():
        import yaml

        (source / f"{name}.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    schema_dir = source / "schemas"
    schema_dir.mkdir()
    (schema_dir / "never.yaml").write_text("false\n")
    build = StudyCompiler(source).compile(tmp_path / "bool-build")
    assert build.study_id == "bool-schema"
    schemas = __import__("json").loads((build.path / "schemas.json").read_text())
    assert schemas["never"] is False


# ---------------------------------------------------------------------------
# F1 (effect): process.outputs[].schema_ref is enforced at the commit boundary
# ---------------------------------------------------------------------------


def test_process_declared_output_schema_is_enforced(tmp_path: Path) -> None:
    """F1: a schema declared on process.outputs[] is enforced even when the
    domain artifact catalog entry lacks a schema_ref."""
    from genesis.service import GenesisService

    service = GenesisService(tmp_path / "workspace")
    try:
        draft = service.create_specification(
            {
                "id": "process-schema-bound",
                "title": "psb",
                "processes": [
                    {
                        "id": "measure",
                        "executor": {
                            "mode": "rule",
                            "parameters": {
                                "rules": [
                                    {
                                        "when": [{"path": "inputs.x", "op": "eq", "value": 1}],
                                        "outputs": {"score": 99},
                                    }
                                ]
                            },
                        },
                        "context_policy": "public",
                        "outputs": [{"artifact_type": "score", "schema_ref": "score"}],
                        "state_effects": [{"field": "counter", "op": "set"}],
                    }
                ],
                "theory": {"theory_family": "exploratory"},
                # domain artifact entry deliberately has NO schema_ref
                "domain": {
                    "artifacts": [{"id": "score", "artifact_type": "object"}],
                    "states": [{"id": "counter", "value_type": "integer", "initial": 0}],
                },
                "protocol": {"time_model": {"type": "rounds", "end": 1}},
                "outcomes": [],
                "models": [],
            }
        )
        schema_dir = (
            tmp_path
            / "workspace"
            / ".genesis"
            / "specifications"
            / "process-schema-bound"
            / "schemas"
        )
        schema_dir.mkdir(parents=True)
        (schema_dir / "score.yaml").write_text(
            "type: object\nrequired: [score]\nproperties:\n"
            "  score: {type: integer, minimum: 0, maximum: 10}\n"
        )
        rev = service.update_specification(
            "process-schema-bound", {"description": "x"}, draft["version"]
        )
        service.approve_specification("process-schema-bound", rev["version"], "researcher")
        compiled = service.compile_study(
            None, "builds/process-schema-bound", specification_id="process-schema-bound"
        )
        service.create_run(
            {"id": "psb-run", "study_id": "process-schema-bound", "build": compiled["path"]}
        )
        with pytest.raises(Exception, match="process measure failed"):
            service.execute_run(
                "psb-run", executor_overrides={"measure": lambda inv: {"score": 99}}
            )
        assert service.get_run("psb-run")["status"] == "failed"
        # Nothing committed: no score artifact, state stays at initial 0.
        artifacts = [a for a in service.artifacts_for_run("psb-run")]
        assert not any(
            isinstance(a.get("payload"), dict)
            and a["payload"].get("declared_artifact_id") == "score"
            for a in artifacts
        )
        history = service.persistence.list_state_history("psb-run")
        states = [snapshot for _version, snapshot in history]
        assert len(states) == 1 and states[0].get("counter") == 0
    finally:
        service.close()


# ---------------------------------------------------------------------------
# F11 (effect): reference preflight resolves fragments via the registry
# ---------------------------------------------------------------------------


def test_package_scoped_fragment_reference_is_accepted() -> None:
    """F11: b#/$defs/value resolves through the registry."""
    catalog = PackageSchemaCatalog(
        {
            "b": {"$defs": {"value": {"type": "string"}}},
            "a": {"$ref": "b#/$defs/value"},
        }
    )
    assert catalog.validate("a", "ok") == []


def test_missing_local_fragment_is_rejected_at_construction() -> None:
    """F11: #/$defs/missing fails at catalog construction, not at validate."""
    with pytest.raises(SchemaValidationError) as exc:
        _catalog({"a": {"$defs": {"ok": {"type": "integer"}}, "$ref": "#/$defs/missing"}})
    assert exc.value.code == SCHEMA_REFERENCE_INVALID


def test_valid_local_fragment_is_accepted() -> None:
    """F11: #/$defs/ok resolves within the document."""
    catalog = _catalog({"a": {"$defs": {"ok": {"type": "integer"}}, "$ref": "#/$defs/ok"}})
    assert catalog.validate("a", 7) == []

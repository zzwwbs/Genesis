"""Authoritative JSON Schema Draft 2020-12 validation for GENESIS (G5).

One validator and one resolver for the whole package dialect: schemas are
checked against the 2020-12 metaschema at catalog construction (``SCH-001``),
``$ref`` resolution is confined to the pinned package registry (``SCH-004``),
``format`` stays annotation-only (``SCH-005``), boolean and empty schemas are
supported with a permissive-schema warning (``SCH-006``), and NaN/Infinity are
rejected as non-JSON values (``SCH-003``). Validation is bounded by typed
limits (``SCHEMA_VALIDATION_LIMIT``) and returns structured diagnostics with
instance/schema JSON Pointers and the failed keyword.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from referencing import Registry, Resource
from referencing.exceptions import (
    InvalidAnchor,
    NoSuchAnchor,
    NoSuchResource,
    PointerToNowhere,
    Unresolvable,
)
from referencing.jsonschema import DRAFT202012

PACKAGE_DIALECT = "https://json-schema.org/draft/2020-12/schema"

SCHEMA_DEFINITION_INVALID = "SCHEMA_DEFINITION_INVALID"
SCHEMA_REFERENCE_INVALID = "SCHEMA_REFERENCE_INVALID"
SCHEMA_DIALECT_UNSUPPORTED = "SCHEMA_DIALECT_UNSUPPORTED"
OUTPUT_SCHEMA_INVALID = "OUTPUT_SCHEMA_INVALID"
SCHEMA_VALIDATION_LIMIT = "SCHEMA_VALIDATION_LIMIT"

# Keywords that carry no validation semantics and never constrain an output.
_ANNOTATION_KEYWORDS = {
    "$id",
    "$schema",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$vocabulary",
}

# Bounded validation limits (SCH acceptance: pathological input terminates).
MAX_SCHEMA_DEPTH = 40
MAX_INSTANCE_DEPTH = 80
MAX_DIAGNOSTICS = 64


@dataclass(frozen=True)
class SchemaDiagnostic:
    """One structured validation failure (spec §3.2)."""

    code: str
    schema_id: str
    instance_pointer: str
    schema_pointer: str
    keyword: str
    message: str


class SchemaValidationError(Exception):
    """Typed failure during schema definition, reference resolution or limits."""

    def __init__(self, code: str, schema_id: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.schema_id = schema_id


def _json_pointer(segments: Iterable[int | str]) -> str:
    parts = [str(segment).replace("~", "~0").replace("/", "~1") for segment in segments]
    return "" if not parts else "/" + "/".join(parts)


def _schema_depth(schema: Any, depth: int = 0) -> int:
    """Measure true nesting depth, not sibling count.

    Every sibling at the same level recurses at ``depth + 1`` relative to the
    frame's original depth; sibling count must not inflate the reported depth.
    """
    if depth > MAX_SCHEMA_DEPTH:
        return depth
    deepest = depth
    children: Any
    if isinstance(schema, dict):
        children = schema.values()
    elif isinstance(schema, list):
        children = schema
    else:
        return depth
    for value in children:
        if isinstance(value, (dict, list)):
            child_depth = _schema_depth(value, depth + 1)
            if child_depth > deepest:
                deepest = child_depth
    return deepest


def _instance_depth(value: Any, depth: int = 0) -> int:
    """Measure true nesting depth, not sibling count (F11 fix)."""
    if depth > MAX_INSTANCE_DEPTH:
        return depth
    deepest = depth
    children: Any
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return depth
    for item in children:
        if isinstance(item, (dict, list)):
            child_depth = _instance_depth(item, depth + 1)
            if child_depth > deepest:
                deepest = child_depth
    return deepest


def _non_finite_numbers(value: Any, segments: tuple[int | str, ...] = ()) -> list[SchemaDiagnostic]:
    """Locate NaN/Infinity literals, which are not valid JSON values (SCH-003)."""
    found: list[SchemaDiagnostic] = []
    if isinstance(value, float) and not _finite(value):
        found.append(
            SchemaDiagnostic(
                code=OUTPUT_SCHEMA_INVALID,
                schema_id="",
                instance_pointer=_json_pointer(segments),
                schema_pointer="",
                keyword="non-finite",
                message=(
                    f"non-finite number at {_json_pointer(segments) or 'root'} is not valid JSON"
                ),
            )
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_non_finite_numbers(item, (*segments, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_non_finite_numbers(item, (*segments, index)))
    return found


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _is_permissive(schema: Any) -> bool:
    """An unconstrained schema: the empty schema or boolean ``true`` (SCH-006)."""
    if schema is True:
        return True
    if not isinstance(schema, dict):
        return False
    return all(key in _ANNOTATION_KEYWORDS for key in schema)


class PackageSchemaCatalog:
    """A validated, self-contained registry of package schemas under Draft 2020-12.

    Construction validates every schema against the 2020-12 metaschema and the
    package dialect, so a malformed schema, an unsupported ``$schema`` value or
    an over-deep declaration fails before any model invocation. The registry
    exposes each schema under both a canonical ``genesis://schemas/<id>`` URI
    and its bare id, so package-internal ``$ref`` values resolve locally while
    external references stay unresolvable and fail visibly (``SCH-004``).
    """

    INSTANCE_MAX_DEPTH = MAX_INSTANCE_DEPTH
    SCHEMA_MAX_DEPTH = MAX_SCHEMA_DEPTH

    def __init__(self, schemas: Mapping[str, Any]) -> None:
        self._schemas: dict[str, Any] = dict(schemas)
        registry = Registry()
        for schema_id, schema in self._schemas.items():
            if _schema_depth(schema) > MAX_SCHEMA_DEPTH:
                raise SchemaValidationError(
                    SCHEMA_VALIDATION_LIMIT,
                    schema_id,
                    f"schema '{schema_id}' exceeds the supported nesting depth",
                )
            if isinstance(schema, dict) and "$schema" in schema:
                dialect = schema["$schema"]
                if dialect != PACKAGE_DIALECT:
                    raise SchemaValidationError(
                        SCHEMA_DIALECT_UNSUPPORTED,
                        schema_id,
                        f"schema '{schema_id}' declares unsupported dialect '{dialect}'; "
                        f"the package dialect is {PACKAGE_DIALECT}",
                    )
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:
                raise SchemaValidationError(
                    SCHEMA_DEFINITION_INVALID,
                    schema_id,
                    f"schema '{schema_id}' is not a valid {PACKAGE_DIALECT} schema: {exc}",
                ) from exc
            resource = Resource.from_contents(schema, default_specification=DRAFT202012)
            registry = registry.with_resource(f"genesis://schemas/{schema_id}", resource)
            registry = registry.with_resource(schema_id, resource)
        self._registry = registry
        # F13/F11: reject unresolved references at catalog construction,
        # before any model invocation. Each schema's document base is its
        # registered package key, so local fragments and package-scoped
        # references resolve through the same registry resolver the real
        # validator uses.
        for schema_id, schema in self._schemas.items():
            self._assert_resolvable_refs(schema_id, schema, document_uri=schema_id)
        self._validators: dict[str, Draft202012Validator] = {}

    def _assert_resolvable_refs(
        self, schema_id: str, schema: Any, document_uri: str, pointer: str = "#"
    ) -> None:
        if not isinstance(schema, (dict, list)):
            return
        if isinstance(schema, dict):
            ref = schema.get("$ref")
            if isinstance(ref, str):
                self._resolve_reference(schema_id, ref, document_uri, pointer)
            for key, value in schema.items():
                self._assert_resolvable_refs(schema_id, value, document_uri, f"{pointer}/{key}")
        else:
            for index, item in enumerate(schema):
                self._assert_resolvable_refs(schema_id, item, document_uri, f"{pointer}/{index}")

    def _resolve_reference(self, schema_id: str, ref: str, document_uri: str, pointer: str) -> None:
        # F11: resolve the reference through the registry resolver with the
        # document as the base URI, so local fragments (#/$defs/x) and
        # package-scoped references (b#/$defs/value) resolve the same way
        # the real validator would. Raising on an unresolvable reference at
        # construction catches missing fragments before any model invocation.
        try:
            resolver = self._registry.resolver(document_uri)
            resolver.lookup(ref)
        except (
            Unresolvable,
            PointerToNowhere,
            NoSuchResource,
            NoSuchAnchor,
            InvalidAnchor,
        ) as exc:
            raise SchemaValidationError(
                SCHEMA_REFERENCE_INVALID,
                schema_id,
                f"schema '{schema_id}' references unresolved schema '{ref}' at {pointer}: {exc}",
            ) from exc

    def __contains__(self, schema_id: str) -> bool:
        return schema_id in self._schemas

    def schema(self, schema_id: str) -> Any:
        try:
            return self._schemas[schema_id]
        except KeyError as exc:
            raise SchemaValidationError(
                SCHEMA_REFERENCE_INVALID,
                schema_id,
                f"schema '{schema_id}' is not in the package registry",
            ) from exc

    def permissive_schemas(self) -> list[str]:
        """Schema ids that impose no constraints and need an approval warning (SCH-006)."""
        return [schema_id for schema_id, schema in self._schemas.items() if _is_permissive(schema)]

    def validate(self, schema_id: str, value: Any) -> list[SchemaDiagnostic]:
        """Validate one instance against a registered schema with bounded limits."""
        schema = self.schema(schema_id)
        if _instance_depth(value) > MAX_INSTANCE_DEPTH:
            raise SchemaValidationError(
                SCHEMA_VALIDATION_LIMIT,
                schema_id,
                f"instance exceeds the supported nesting depth for '{schema_id}'",
            )
        diagnostics = _non_finite_numbers(value)
        if diagnostics:
            return [
                SchemaDiagnostic(
                    code=diag.code,
                    schema_id=schema_id,
                    instance_pointer=diag.instance_pointer,
                    schema_pointer=diag.schema_pointer,
                    keyword=diag.keyword,
                    message=diag.message,
                )
                for diag in diagnostics
            ]
        validator = self._validators.get(schema_id)
        if validator is None:
            validator = Draft202012Validator(schema, registry=self._registry)
            self._validators[schema_id] = validator
        diagnostics = []
        try:
            for error in validator.iter_errors(value):
                diagnostics.append(
                    SchemaDiagnostic(
                        code=OUTPUT_SCHEMA_INVALID,
                        schema_id=schema_id,
                        instance_pointer=_json_pointer(error.absolute_path),
                        schema_pointer=_json_pointer(error.absolute_schema_path),
                        keyword=str(error.validator) or "schema",
                        message=error.message,
                    )
                )
                if len(diagnostics) >= MAX_DIAGNOSTICS:
                    raise SchemaValidationError(
                        SCHEMA_VALIDATION_LIMIT,
                        schema_id,
                        f"validation of '{schema_id}' exceeded the diagnostic limit",
                    )
        except Unresolvable as exc:
            raise SchemaValidationError(
                SCHEMA_REFERENCE_INVALID,
                schema_id,
                f"schema '{schema_id}' references an external or unresolved schema: {exc}",
            ) from exc
        return diagnostics


def validate_schema(schema: dict[str, Any], value: Any, path: str = "root") -> list[str]:
    """Legacy human-readable validation for standalone schemas (repair prompts).

    Backed by the same Draft 2020-12 engine as :class:`PackageSchemaCatalog`,
    with messages formatted for bounded-repair prompts and existing tests:
    ``root[1]: expected integer, got str``. Local ``$defs`` resolve within the
    passed schema; external/unresolvable references fail loudly.
    """
    validator = Draft202012Validator(schema)
    return [_legacy_message(error) for error in validator.iter_errors(value)]


def _legacy_message(error: Any) -> str:
    path = _legacy_path(error.absolute_path)
    expected = error.validator_value
    if error.validator == "type":
        return f"{path}: expected {expected}, got {type(error.instance).__name__}"
    if error.validator == "required":
        missing = expected[0] if isinstance(expected, list) and expected else expected
        return f"{path}.{missing}: required property is missing"
    if error.validator == "enum":
        return f"{path}: value is not one of the declared options"
    return f"{path}: {error.message}"


def _legacy_path(segments: Iterable[int | str]) -> str:
    out = "root"
    for segment in segments:
        out += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return out

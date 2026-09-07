"""JSON Schema accessors for canonical artifact types."""

from typing import Any

from .models import (
    DomainSpec,
    ModelsSpec,
    OpennessSpec,
    OutcomesSpec,
    ProtocolSpec,
    StrictModel,
    StudySpec,
    TheorySpec,
)

_MODELS: dict[str, type[StrictModel]] = {
    "study": StudySpec,
    "openness": OpennessSpec,
    "theory": TheorySpec,
    "domain": DomainSpec,
    "protocol": ProtocolSpec,
    "outcomes": OutcomesSpec,
    "models": ModelsSpec,
}


def schema_for(name: str) -> dict[str, Any]:
    """Return a fresh JSON Schema for one canonical artifact (without shared mutable state)."""
    try:
        return _MODELS[name].model_json_schema()
    except KeyError as exc:
        raise ValueError(f"unknown canonical artifact: {name}") from exc


def all_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON Schemas keyed by canonical YAML basename."""
    return {name: schema_for(name) for name in _MODELS}

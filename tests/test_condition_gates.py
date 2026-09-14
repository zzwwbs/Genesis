"""A condition gate must be validated against the shape the scheduler resolves.

The first version of this check was written from a belief about that shape
rather than from the shape itself, and got it exactly backwards: it refused
``condition.factors.<id>``, which fires, and accepted ``condition.<id>``, which
never does. Every clickbait package in the workspace used the accepted spelling,
so their governance treatment never arrived in any run.

So these tests establish the runtime's behaviour by evaluating predicates
against real expanded conditions, and hold the compiler to what they find. The
agreement test is the important one: it derives both sides from the same
expansion, so the check cannot be inverted again without failing.
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.compiler import _validate_condition_factors
from genesis.runtime import _evaluate_condition, expand_protocol_conditions
from genesis.specification.models import DomainSpec, OpennessSpec, ProtocolSpec

FACTORED = {"factors": [{"id": "governance", "levels": ["none", "strict"]}]}
EXPLICIT_NESTED = {"conditions": [{"id": "c1", "factors": {"governance": "strict"}}]}
EXPLICIT_FLAT = {"conditions": [{"id": "c2", "governance": "strict"}]}


def _fires(protocol: dict[str, Any], path: str) -> bool:
    """Does a gate on ``path`` ever open, in any cell the runtime will run?"""
    predicate = {"path": path, "op": "eq", "value": "strict"}
    return any(
        _evaluate_condition(predicate, {"condition": condition})
        for condition in expand_protocol_conditions(protocol)
    )


# --- what the runtime actually does -----------------------------------------------


@pytest.mark.parametrize("protocol", [FACTORED, EXPLICIT_NESTED])
def test_a_declared_factor_is_reached_through_factors(protocol: dict[str, Any]) -> None:
    assert _fires(protocol, "condition.factors.governance")
    assert not _fires(protocol, "condition.governance")


def test_a_flat_condition_key_is_reached_directly() -> None:
    """A condition may also carry its levels at the top level."""
    assert _fires(EXPLICIT_FLAT, "condition.governance")
    assert not _fires(EXPLICIT_FLAT, "condition.factors.governance")


# --- and what the compiler must therefore accept ----------------------------------


def _refusals(protocol: dict[str, Any], path: str) -> list[str]:
    loaded = {
        "protocol": ProtocolSpec.model_validate(
            {"schema_version": "1.0", "study_id": "s", "time_model": {"type": "rounds"}, **protocol}
        ),
        "openness": OpennessSpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "s",
                "processes": [
                    {
                        "id": "gated",
                        "executor": {},
                        "context_policy": "public",
                        "trigger": {
                            "type": "condition",
                            "predicate": {"path": path, "op": "eq", "value": "strict"},
                        },
                    }
                ],
            }
        ),
        "domain": DomainSpec.model_validate({"schema_version": "1.0", "study_id": "s"}),
    }
    return [issue["code"] for issue in _validate_condition_factors(loaded)]


@pytest.mark.parametrize(
    ("protocol", "path"),
    [
        (FACTORED, "condition.factors.governance"),
        (EXPLICIT_NESTED, "condition.factors.governance"),
        (EXPLICIT_FLAT, "condition.governance"),
    ],
)
def test_the_spelling_that_fires_compiles(protocol: dict[str, Any], path: str) -> None:
    assert _fires(protocol, path), "the probe itself must reach the gate"
    assert _refusals(protocol, path) == []


@pytest.mark.parametrize(
    ("protocol", "path"),
    [
        (FACTORED, "condition.governance"),
        (EXPLICIT_NESTED, "condition.governance"),
        (EXPLICIT_FLAT, "condition.factors.governance"),
        (FACTORED, "condition.nonsense"),
        (FACTORED, "condition.factors.nonsense"),
    ],
)
def test_a_spelling_that_never_fires_is_refused(protocol: dict[str, Any], path: str) -> None:
    assert not _fires(protocol, path)
    assert _refusals(protocol, path) == ["CONDITION_FACTOR_UNKNOWN"]


def test_the_condition_id_is_always_addressable() -> None:
    assert _refusals(FACTORED, "condition.id") == []


def test_the_check_agrees_with_the_runtime_on_every_spelling() -> None:
    """The property that matters: refused exactly when the gate cannot open.

    Both sides derive from expand_protocol_conditions, so an inversion fails
    here rather than in a study six weeks later.
    """
    for protocol in (FACTORED, EXPLICIT_NESTED, EXPLICIT_FLAT):
        for path in (
            "condition.governance",
            "condition.factors.governance",
            "condition.id",
            "condition.absent",
            "condition.factors.absent",
        ):
            fires = _fires(protocol, path) or path == "condition.id"
            refused = _refusals(protocol, path) != []
            assert fires != refused, f"{path} on {protocol}: fires={fires} refused={refused}"


# --- the grammar must describe what the compiler accepts ---------------------------


def test_the_workflow_grammar_does_not_advertise_a_refused_selector() -> None:
    """instructions.md and the domain template offered '*' and a literal list;
    the compiler refuses both, so a researcher following them wrote a package
    that would not compile."""
    from pathlib import Path as _Path

    from genesis.service import _workflows_root

    root = _Path(_workflows_root()) / "three-layer-study"
    for relative in ("instructions.md", "templates/domain.yaml"):
        text = (root / relative).read_text()
        if "in:" not in text and "`in`" not in text:
            continue
        assert "refused" in text, f"{relative} still offers selectors without saying which fail"


def test_the_refused_selectors_really_are_refused() -> None:
    from genesis.compiler import _selector_problem

    assert _selector_problem("*", {"feeds"})
    assert _selector_problem(["u1", "u2"], {"feeds"})
    assert _selector_problem("actor.ids", {"feeds"}) is None
    assert _selector_problem("state.feeds", {"feeds"}) is None


def test_a_dead_gate_inside_a_rule_executor_is_refused_too() -> None:
    """A rule's own `when` reads the condition exactly as a trigger does."""
    loaded = {
        "protocol": ProtocolSpec.model_validate(
            {"schema_version": "1.0", "study_id": "s", "time_model": {"type": "rounds"}, **FACTORED}
        ),
        "openness": OpennessSpec.model_validate(
            {
                "schema_version": "1.0",
                "study_id": "s",
                "processes": [
                    {
                        "id": "settle",
                        "executor": {
                            "mode": "rule",
                            "parameters": {
                                "rules": [
                                    {
                                        "when": {
                                            "path": "condition.governance",
                                            "op": "eq",
                                            "value": "strict",
                                        }
                                    }
                                ]
                            },
                        },
                        "context_policy": "public",
                    }
                ],
            }
        ),
        "domain": DomainSpec.model_validate({"schema_version": "1.0", "study_id": "s"}),
    }
    assert [issue["code"] for issue in _validate_condition_factors(loaded)] == [
        "CONDITION_FACTOR_UNKNOWN"
    ]


# --- an actor source names a collection that has to be there ----------------------


def _actor_source_warnings(tmp_path: Any, source: str) -> list[str]:
    import json as _json

    from genesis.compiler import StudyCompiler
    from tests.test_engine_gaps import _write_package

    root = tmp_path / source.replace(".", "-")
    package = _write_package(
        root,
        {
            "openness": {
                "processes": [
                    {
                        "id": "act",
                        "actors": {"source": source},
                        "information_timing": {"mode": "sequential"},
                        "executor": {"mode": "deterministic"},
                        "context_policy": "p",
                        "trigger": {"type": "phase", "phase": 0},
                    }
                ]
            },
            "domain": {
                "visibility": [{"id": "p", "allow": []}],
                "states": [
                    {
                        "id": "population",
                        "value_type": "object",
                        "initial": {"creators": [{"id": "w1"}]},
                    }
                ],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 0}},
        },
        "src-study",
    )
    build = StudyCompiler(package).compile(root / "build")
    report = _json.loads((build.path / "validation_report.json").read_text())
    return sorted({warning["code"] for warning in report.get("warnings", [])})


def test_an_actor_source_naming_an_undeclared_key_is_flagged(tmp_path: Any) -> None:
    """Only the root segment was checked, so naming a collection that is not
    there compiled and aborted before the run's first round."""
    assert "ACTOR_SOURCE_UNDECLARED_KEY" in _actor_source_warnings(tmp_path, "population.authors")


def test_a_key_the_state_declares_is_not_flagged(tmp_path: Any) -> None:
    assert "ACTOR_SOURCE_UNDECLARED_KEY" not in _actor_source_warnings(
        tmp_path, "population.creators"
    )

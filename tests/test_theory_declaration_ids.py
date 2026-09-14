"""A theory declaration id names one declaration.

The coverage report is keyed by declaration id, and the schema accepts two
declarations sharing one. Both executed -- their edges are appended to lists --
but the report kept only the last, so the build recorded one resolved
declaration where two ran. A generated id (``relation-0``) can collide with an
explicit one in the same way, since all three kinds share the id space.
"""

from __future__ import annotations

from genesis.theory_execution import compile_theory_execution


def _precedence(declared_id: str | None, producer: str, consumer: str) -> dict:
    relation = {
        "execution": {
            "kind": "precedence",
            "producer_process": producer,
            "consumer_process": consumer,
        }
    }
    if declared_id is not None:
        relation["id"] = declared_id
    return relation


def _codes(theory: dict) -> list[str]:
    plan = compile_theory_execution(theory, known_processes={"a", "b", "c"}, known_mechanisms=set())
    return [issue.code for issue in plan.issues]


def test_two_declarations_sharing_an_explicit_id_are_refused() -> None:
    theory = {"relations": [_precedence("r", "a", "b"), _precedence("r", "b", "c")]}
    assert "THEORY_DECLARATION_DUPLICATE" in _codes(theory)


def test_an_explicit_id_colliding_with_a_generated_one_is_refused() -> None:
    """The first unnamed relation is 'relation-0'; naming another that too
    collapses the two in the report just the same."""
    theory = {"relations": [_precedence(None, "a", "b"), _precedence("relation-0", "b", "c")]}
    assert "THEORY_DECLARATION_DUPLICATE" in _codes(theory)


def test_the_same_id_across_kinds_is_refused() -> None:
    theory = {
        "relations": [_precedence("shared", "a", "b")],
        "feedback": [{"id": "shared"}],
    }
    assert "THEORY_DECLARATION_DUPLICATE" in _codes(theory)


def test_distinct_ids_still_resolve_every_declaration() -> None:
    theory = {"relations": [_precedence("r1", "a", "b"), _precedence("r2", "b", "c")]}
    plan = compile_theory_execution(theory, known_processes={"a", "b", "c"}, known_mechanisms=set())
    assert set(plan.resolved) == {"r1", "r2"}
    assert "THEORY_DECLARATION_DUPLICATE" not in [issue.code for issue in plan.issues]

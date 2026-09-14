"""An availability rule written as the grammar documents must actually gate.

instructions.md and the domain template teach ``available_when`` as a predicate
-- ``{path, op, value}``, or ``all``/``any``/``not`` -- and the compiler and the
timing and measurement analyses all read it that way. The runtime recognised a
predicate only under a ``predicate:`` key, so a rule written as documented
matched none of its keys, skipped every check, and left the item available in
every round of every condition. The positive control found it: the control actor
received a context item gated to the treated cell. The clickbait study gated
its reflection context this way, so reflection was offered every round.
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.runtime import ContextEngine, ProcessInvocation

TREATED = {"path": "condition.factors.treatment", "op": "eq", "value": "treated"}


def _visible(rule: Any, cell: str, phase: int = 1, *, per_path: bool = True) -> bool:
    available_when = {"hint": rule} if per_path else rule
    engine = ContextEngine({"v": {"allow": ["hint"], "available_when": available_when}})
    invocation = ProcessInvocation(
        "i",
        "r",
        "p",
        actor_ids=("a",),
        phase=phase,
        condition={"id": cell, "factors": {"treatment": cell}},
    )
    return "hint" in engine.build("v", invocation, {"hint": 1}).data


@pytest.mark.parametrize(
    "rule",
    [
        TREATED,
        {"predicate": TREATED},
        {"all": [TREATED, {"path": "protocol.phase", "op": "gte", "value": 1}]},
        {"not": {"path": "condition.factors.treatment", "op": "eq", "value": "control"}},
    ],
    ids=["bare", "wrapped", "all", "not"],
)
def test_every_documented_form_gates_by_condition(rule: Any) -> None:
    assert _visible(rule, "treated") is True
    assert _visible(rule, "control") is False


def test_a_whole_policy_predicate_gates_too() -> None:
    assert _visible(TREATED, "treated", per_path=False) is True
    assert _visible(TREATED, "control", per_path=False) is False


def test_a_predicate_combines_with_a_round_window() -> None:
    rule = {**TREATED, "after_round": 3}
    assert _visible(rule, "treated", phase=2) is False
    assert _visible(rule, "treated", phase=3) is True
    assert _visible(rule, "control", phase=3) is False


def test_the_round_window_forms_are_unchanged() -> None:
    assert _visible({"after_round": 2}, "control", phase=1) is False
    assert _visible({"after_round": 2}, "control", phase=2) is True


def _inert_codes(tmp_path: Any, available_when: dict[str, Any]) -> list[str]:
    from genesis.compiler import StudyCompiler, ValidationIssue
    from tests.test_engine_gaps import _write_package

    source = _write_package(
        tmp_path,
        {
            "openness": {"processes": []},
            "domain": {
                "states": [{"id": "hint", "value_type": "integer", "initial": 0}],
                "visibility": [{"id": "v", "allow": ["hint"], "available_when": available_when}],
            },
            "protocol": {"time_model": {"type": "rounds", "start": 0, "end": 1}},
        },
        "avail-study",
    )
    try:
        StudyCompiler(source).compile(tmp_path / "build")
    except ValidationIssue as issue:
        return [item.code for item in issue.issues]
    return []


def test_a_mistyped_rule_key_is_refused_rather_than_left_open(tmp_path: Any) -> None:
    assert "AVAILABILITY_RULE_INERT" in _inert_codes(tmp_path, {"hint": {"after_rounds": 3}})


def test_a_gate_on_a_path_the_policy_does_not_allow_is_refused(tmp_path: Any) -> None:
    assert "AVAILABILITY_RULE_INERT" in _inert_codes(tmp_path, {"hnit": {"after_round": 3}})


@pytest.mark.parametrize(
    "rule",
    [
        {"hint": {"after_round": 1}},
        {"hint": {"path": "protocol.phase", "op": "gte", "value": 1}},
        {"after_round": 1},
    ],
)
def test_documented_rules_compile(tmp_path: Any, rule: dict[str, Any]) -> None:
    assert "AVAILABILITY_RULE_INERT" not in _inert_codes(tmp_path, rule)

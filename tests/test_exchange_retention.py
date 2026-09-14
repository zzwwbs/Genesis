"""An exchange log must not outgrow what any policy can read from it.

The declared cardinality cap was applied when a prompt was built, never to the
log itself, so a policy reading "the last three rounds" still held every round
for the life of the run -- O(rounds x actor groups x context size) in memory.

Only a cap that keeps the tail in recorded order can be applied at record time:
one that keeps the head, or orders by a field, selects entries a tail-trim would
already have discarded, so those stay unbounded rather than quietly wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.runtime import _exchange_retention


def _policy(**cardinality: Any) -> dict[str, Any]:
    return {
        "p": {
            "allow": ["exchanges.diary"],
            **({"cardinality": cardinality} if cardinality else {}),
        }
    }


def test_a_tail_cap_bounds_the_log() -> None:
    assert _exchange_retention(_policy(**{"exchanges.diary": {"limit": 3, "keep": "last"}})) == {
        "diary": 3
    }


@pytest.mark.parametrize(
    "rule",
    [
        {"limit": 3, "keep": "first"},
        {"limit": 3, "keep": "last", "by": "phase"},
        3,
    ],
)
def test_a_cap_a_tail_trim_would_break_leaves_the_log_unbounded(rule: Any) -> None:
    """Keeping the head, or ordering by a field, selects entries the trim drops."""
    assert _exchange_retention(_policy(**{"exchanges.diary": rule})) == {}


def test_no_declared_cap_leaves_the_log_unbounded() -> None:
    assert _exchange_retention(_policy()) == {}


def test_one_unbounded_reader_defeats_another_policy_s_cap() -> None:
    """A second policy reading the same exchanges without a cap still needs
    everything, so the bound cannot be applied."""
    policies = {
        "capped": {
            "allow": ["exchanges.diary"],
            "cardinality": {"exchanges.diary": {"limit": 2, "keep": "last"}},
        },
        "uncapped": {"allow": ["exchanges.diary"]},
    }
    assert _exchange_retention(policies) == {}


def test_the_log_is_trimmed_as_it_is_written() -> None:
    from genesis.runtime import ProcessInvocation, ProcessResult, RunController

    controller = RunController.__new__(RunController)
    controller._exchange_log = {}
    controller._exchange_processes = frozenset({"diary"})
    controller._exchange_retention = {"diary": 2}
    turn = type("T", (), {"process_id": "diary", "phase": 0, "process": {}})()
    for phase in range(5):
        turn.phase = phase
        controller._record_exchange(
            turn,
            ProcessInvocation("i", "run", "diary", actor_ids=("u1",)),
            ProcessResult(outputs={"note": f"n{phase}"}),
            1,
        )
    kept = controller._exchange_log[("diary", ("u1",))]
    assert [entry["phase"] for entry in kept] == [3, 4], kept

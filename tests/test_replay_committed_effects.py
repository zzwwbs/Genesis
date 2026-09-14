"""A replayed prefix applies every assignment its source committed (2026-09-14 M17).

Deltas with a None value were dropped as "removals", which the state store cannot
make; a field the run set to None replayed as untouched.
"""

from __future__ import annotations

from genesis.service import GenesisService


def test_a_none_assignment_is_replayed() -> None:
    service = GenesisService.__new__(GenesisService)
    service.trace_run = lambda run_id, evidence=False: [  # type: ignore[method-assign]
        {
            "kind": "process_completed",
            "invocation_id": "i1",
            "attempt": 1,
            "state_delta": {"leader": None, "count": 3},
        }
    ]
    effects = service._committed_effects("r")
    assert effects[("i1", 1)]["state_effects"] == {"leader": None, "count": 3}

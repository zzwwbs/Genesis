"""Legacy derived rows sit beside their evidence, and carry no second copy of it.

Every synthesized row was put before the first raw event, whatever its round,
and each carried its source event's whole state_delta (2026-09-14 M1).
"""

from __future__ import annotations

from genesis.service import GenesisService

EVENTS = [
    {"event_id": "e1", "phase": 1, "state_delta": {"analytics": [{"user": "u1", "phase": 1}]}},
    {"event_id": "e2", "phase": 1, "state_delta": {}},
    {"event_id": "e3", "phase": 2, "state_delta": {"analytics": [{"user": "u1", "phase": 2}]}},
]
ARTIFACTS = [
    {
        "artifact_id": "a1",
        "process_id": "evaluate-clickbait",
        "phase": 1,
        "value": {"detected": True},
    }
]


def test_derived_rows_follow_their_source_in_order() -> None:
    derived = GenesisService._legacy_derived_rows(EVENTS, ARTIFACTS)
    stream = list(GenesisService._with_derived_rows(EVENTS, derived))
    labels = [row.get("event_id") if "kind" not in row else f"{row['kind']}" for row in stream]
    assert labels == ["e1", "analytics", "e2", "measurement", "e3", "analytics"]
    assert all("state_delta" not in row for row in stream if "kind" in row)
    assert all("__derived_from_event__" not in row for row in stream)

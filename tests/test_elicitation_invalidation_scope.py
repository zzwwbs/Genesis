"""An upstream change is reported to a downstream stage still being answered (2026-09-14 M16)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from genesis.service import GenesisService


def _invalidations(status: str) -> list[dict[str, Any]]:
    service = GenesisService.__new__(GenesisService)
    marked: list[str] = []
    service._elicitation_store = SimpleNamespace(  # type: ignore[attr-defined]
        mark_stage_needs_review=lambda _sid, stage, _rule: marked.append(stage)
    )
    service._stage_for_section = lambda _workflow, target: target  # type: ignore[method-assign]
    workflow = SimpleNamespace(
        stages=[SimpleNamespace(id="openness"), SimpleNamespace(id="theory")],
        invalidation={"openness": ["theory"]},
    )
    session = SimpleNamespace(
        session_id="s",
        current_stage="openness",
        stages={"theory": SimpleNamespace(status=status)},
    )
    return service._invalidate_downstream_stages(session, workflow, ["openness"])


def test_a_clarifying_downstream_stage_is_listed() -> None:
    assert [item["stage"] for item in _invalidations("clarifying")] == ["theory"]


def test_a_stage_not_yet_started_is_not() -> None:
    assert _invalidations("not_started") == []

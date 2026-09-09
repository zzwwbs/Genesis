"""Executors for the resume/round-snapshot regression.

Kept in a module (not a closure) because a compiled package references
executors by ``module:attribute`` entry point.
"""

from __future__ import annotations

from typing import Any

SEEN: list[tuple[int, int | None]] = []


def bump(invocation: Any) -> dict[str, Any]:
    current = invocation.context.data.get("counter", 0) if invocation.context else 0
    return {"counter": current + 1}


def read(invocation: Any) -> dict[str, Any]:
    feedback = (invocation.context.data.get("feedback") or {}).get("last") or {}
    SEEN.append((int(invocation.phase), dict(feedback).get("counter")))
    return {}

"""Declared executor functions used by AW-07 wiring tests."""

from __future__ import annotations

import random


def stochastic_tick(invocation, rng: random.Random) -> dict[str, object]:
    """A declared stochastic function: seeded per invocation."""
    return {"counter": rng.randint(0, 5)}


def computational_double(invocation) -> dict[str, object]:
    """A declared computational function over the authorised context."""
    value = 0
    if invocation.context is not None:
        value = invocation.context.data.get("counter", 0)
    return {"counter": int(value) * 2}


def extension_probe(invocation) -> dict[str, object]:
    """A declared workspace extension factory."""
    return {"extension": True, "value": 7}

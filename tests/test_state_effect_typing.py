"""A state write must be judged by what JSON records, not by Python's ``==``.

The list-form effect branch filtered unchanged fields with ``!=``. Python holds
``1 == 1.0`` and ``True == 1``, so a write that changes the recorded type was
dropped as a no-op: the JSON bytes a record's identity is digested from differed
from the committed value, and an invalid type was discarded silently instead of
being refused. Model-call effects take this branch, so it was the normal path.

state_encoding.identical exists precisely for this comparison; the mapping-form
branch already refused what the list form let through.
"""

from __future__ import annotations

from typing import Any

import pytest

from genesis.runtime import StateStore


def _store() -> StateStore:
    return StateStore({"f": int}, {"f": 1})


@pytest.mark.parametrize("form", ["list", "mapping"])
def test_a_type_changing_write_is_refused_in_either_form(form: str) -> None:
    store = _store()
    effect: Any = [{"field": "f", "op": "set", "value": 1.0}] if form == "list" else {"f": 1.0}
    with pytest.raises(TypeError):
        store.apply(effect, {"f"})


@pytest.mark.parametrize("form", ["list", "mapping"])
def test_a_bool_over_a_number_is_refused_in_either_form(form: str) -> None:
    """bool subclasses int, so '== 1' hid this one entirely."""
    store = _store()
    effect: Any = [{"field": "f", "op": "set", "value": True}] if form == "list" else {"f": True}
    with pytest.raises(TypeError):
        store.apply(effect, {"f"})


def test_writing_the_same_value_is_still_a_no_op() -> None:
    store = _store()
    before = store.version
    store.apply([{"field": "f", "op": "set", "value": 1}], {"f"})
    assert store.snapshot() == {"f": 1}
    assert store.version == before + 1  # the call is still recorded


def test_a_real_change_still_applies() -> None:
    store = _store()
    store.apply([{"field": "f", "op": "set", "value": 7}], {"f"})
    assert store.snapshot() == {"f": 7}

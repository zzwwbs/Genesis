"""Append-aware encoding of committed simulation state (STH-005, STH-006).

A committed state row is stored either as a **base** — the whole snapshot — or
as a **patch** against the previous version. Simulation state accumulates across
rounds, so storing a full snapshot per commit is quadratic in population ×
horizon. Storing whole changed *fields* only shrinks the constant: a list
growing to N entries still costs 1+2+…+N. Storing the appended tail removes the
quadratic term outright — measured on real evidence, 303.83 MB of snapshots is
12.02 MB of whole-field deltas and 0.41 MB of append-aware patches.

The canonical serialization here must reproduce byte-for-byte what the runtime
commits, because a commit's identity digests those bytes (STH-007).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# Storage forms recorded on a state row. A row written before this encoding
# existed carries no form and is read as a base (STH-008).
FORM_BASE = "base"
FORM_PATCH = "patch"

# A base is rewritten once the patches since the last base have cost more than
# this fraction of it. Cost-based rather than a fixed interval: base size itself
# grows with the run, so a fixed interval would retain a quadratic component.
BASE_COST_RATIO = 0.5

# The event ledger tolerates a longer chain before rewriting a base. Measured
# across three recorded runs, 0.5 gave 4.7x/5.9x/3.4x while an unbounded chain
# gave 7.0x/13.4x/5.0x; 4.0 recovers ~96% of that gap with the longest chain
# still bounded near 90, where unbounded reached 179. State needs no such
# relaxation — it already reaches 197x at 0.5.
EVENT_BASE_COST_RATIO = 4.0


def canonical_bytes(state: Mapping[str, Any]) -> bytes:
    """The serialization the runtime commits, reproduced exactly.

    ``json.dumps(..., sort_keys=True)`` with default separators is what the
    commit path writes. A commit's identity digests these bytes, so
    reconstruction must produce them unchanged rather than an equivalent
    encoding (STH-007).
    """
    return json.dumps(dict(state), sort_keys=True).encode()


def identical(left: Any, right: Any) -> bool:
    """Whether two decoded JSON values serialize identically.

    ``==`` is the wrong test here. Python holds ``True == 1`` and ``1 == 1.0``,
    but JSON writes ``true``, ``1`` and ``1.0`` — three different byte strings.
    Treating such a transition as "unchanged" drops it from the patch, so the
    reconstructed record differs from the committed one in exactly the bytes its
    identity is digested from (STH-007, STH-011). Types must match as well as
    values.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            identical(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            identical(value, right[index]) for index, value in enumerate(left)
        )
    return bool(left == right)


def encode_patch(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Describe ``current`` as a change to ``previous``.

    Each changed field is one of three ops: ``append`` — the new value extends
    the old elementwise, so only the tail is stored; ``patch`` — both values are
    mappings, so the change is described recursively; or ``replace``, carrying
    the whole value. ``append`` and ``patch`` are used only when provably
    correct; anything else falls back to ``replace``. Removed keys are recorded
    explicitly so a patch is a complete description of the transition.

    Recursion matters for event payloads, where the growing value sits inside a
    single top-level key (``context``): describing that key flatly replaces it
    whole, measured at 1.5x against 6.9x recursive (STH-005, §7.1b).
    """
    changed: dict[str, Any] = {}
    for key, value in current.items():
        old = previous.get(key, _MISSING)
        if old is not _MISSING and identical(old, value):
            continue
        if (
            isinstance(value, list)
            and isinstance(old, list)
            and len(value) > len(old)
            and all(identical(item, old[index]) for index, item in enumerate(value[: len(old)]))
        ):
            changed[key] = {"op": "append", "items": value[len(old) :]}
        elif isinstance(value, Mapping) and isinstance(old, Mapping):
            changed[key] = {"op": "patch", "patch": encode_patch(old, value)}
        else:
            changed[key] = {"op": "replace", "value": value}
    removed = [key for key in previous if key not in current]
    patch: dict[str, Any] = {"fields": changed}
    if removed:
        patch["removed"] = sorted(removed)
    return patch


def apply_patch(previous: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the state a patch describes. Inverse of ``encode_patch``."""
    result = dict(previous)
    for key in patch.get("removed") or ():
        result.pop(str(key), None)
    fields = patch.get("fields") or {}
    if not isinstance(fields, Mapping):
        raise ValueError("STATE_PATCH_MALFORMED: patch fields must be a mapping")
    for key, change in fields.items():
        if not isinstance(change, Mapping):
            raise ValueError(f"STATE_PATCH_MALFORMED: change for '{key}' must be a mapping")
        op = str(change.get("op", ""))
        if op == "append":
            base = result.get(key)
            if not isinstance(base, list):
                raise ValueError(
                    f"STATE_PATCH_MALFORMED: append to '{key}' requires a list, "
                    f"found {type(base).__name__}"
                )
            result[key] = [*base, *(change.get("items") or ())]
        elif op == "patch":
            base = result.get(key)
            if not isinstance(base, Mapping):
                raise ValueError(
                    f"STATE_PATCH_MALFORMED: patch of '{key}' requires a mapping, "
                    f"found {type(base).__name__}"
                )
            nested = change.get("patch")
            if not isinstance(nested, Mapping):
                raise ValueError(f"STATE_PATCH_MALFORMED: patch of '{key}' missing body")
            result[key] = apply_patch(base, nested)
        elif op == "replace":
            result[key] = change.get("value")
        else:
            raise ValueError(f"STATE_PATCH_MALFORMED: unknown op '{op}' for '{key}'")
    return result


def should_write_base(
    patch_bytes_since_base: int, base_size: int, ratio: float = BASE_COST_RATIO
) -> bool:
    """Whether this commit should be stored whole rather than as a patch."""
    if base_size <= 0:
        return True
    return patch_bytes_since_base >= base_size * ratio


def copy_value(value: Any) -> Any:
    """A deep copy of decoded JSON, sharing no mutable container.

    ``copy.deepcopy`` is correct here but general: its memo table and identity
    bookkeeping dominated the read path, measured at 95% of the cost of reading
    a run's events. Committed values are decoded JSON — dicts, lists and
    immutable scalars — so a specialized walk is equivalent and far cheaper.
    """
    if isinstance(value, dict):
        return {key: copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_value(item) for item in value]
    return value


class _Missing:
    """Sentinel distinguishing an absent key from one holding ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()

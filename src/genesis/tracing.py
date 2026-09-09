"""Study-agnostic trajectory tracing over recorded provenance.

A trace answers "how did this come about, and what followed from it" for one
recorded invocation. Every ingredient is already in the run's own records: each
event carries its causal ``parent_events`` and the ``input_refs`` it consumed,
and each artifact carries its ``lineage``, ``producer_event`` and
``producer_process``. Walking those needs no knowledge of a particular study's
processes or state fields.

What IS study-specific is only which invocation to start from and what to call
each step. Both are declared in the package (``outcomes.traces``) rather than
recognised by name in this module, so a trace is part of the inspectable Study
Package and works for any study.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# A trace is a local explanation, not the whole run; without a bound a densely
# connected study would return most of its trajectory as one "chain".
DEFAULT_DEPTH = 12
MAX_DEPTH = 200
# A densely connected study reaches hundreds of events within a few
# generations. A chain is meant to be read, so the nearest relations are kept
# and the result says plainly that it was cut.
DEFAULT_MAX_STEPS = 40
# A caller may widen the chain, but not to the point of returning the run.
MAX_STEPS_LIMIT = 1000


def _order_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (event.get("phase", 0), event.get("commit_order", 0), str(event.get("event_id", "")))


def _step(
    event: Mapping[str, Any],
    *,
    relation: str,
    distance: int,
    labels: Mapping[str, str],
    artifacts_by_event: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    """One rendered step: what ran, for whom, on what, and what it produced."""
    process_id = str(event.get("process_id", ""))
    produced = artifacts_by_event.get(str(event.get("event_id", "")), [])
    return {
        "step": labels.get(process_id, process_id),
        "relation": relation,
        "distance": distance,
        "process": process_id,
        "event": event.get("event_id"),
        "phase": event.get("phase"),
        "actors": list(event.get("actors") or ()),
        "consumed_artifacts": list(event.get("input_refs") or ()),
        "produced_artifacts": [
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_type": (artifact.get("payload") or {}).get("declared_artifact_id"),
                "value": (artifact.get("payload") or {}).get("value"),
            }
            for artifact in produced
        ],
        "state_delta": dict(event.get("state_delta") or {}),
    }


def build_chain(
    events: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    seed_event_id: str,
    *,
    labels: Mapping[str, str] | None = None,
    depth: int = DEFAULT_DEPTH,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The causal chain around ``seed_event_id``, with a coverage report.

    Ancestors are followed through ``parent_events`` and descendants through the
    events that name this chain's events as parents, each bounded by ``depth``
    generations. When more events are reachable than ``max_steps``, the nearest
    by causal distance are kept and the report says so. Steps are ordered by
    their committed position so the chain reads in the order the run produced it.
    """
    bounded = max(1, min(int(depth), MAX_DEPTH))
    by_id = {str(event.get("event_id", "")): event for event in events if event.get("event_id")}
    if seed_event_id not in by_id:
        raise ValueError(f"TRACE_EVENT_NOT_FOUND: no recorded event '{seed_event_id}'")

    children: dict[str, list[str]] = {}
    for event in events:
        child = str(event.get("event_id", ""))
        for parent in event.get("parent_events") or ():
            children.setdefault(str(parent), []).append(child)

    artifacts_by_event: dict[str, list[Mapping[str, Any]]] = {}
    for artifact in artifacts:
        payload = artifact.get("payload")
        if not isinstance(payload, Mapping) or not payload.get("declared_artifact_id"):
            continue
        producer = str(payload.get("producer_event", ""))
        if producer:
            artifacts_by_event.setdefault(producer, []).append(artifact)

    def traverse(start: str, edges: Mapping[str, list[str]] | None) -> tuple[dict[str, int], bool]:
        """Events within ``depth`` of ``start``, and whether depth cut the walk.

        The flag distinguishes "the graph ended" from "the bound stopped us",
        which the caller reports: a depth-limited chain is partial even when
        every step it found was returned.
        """
        distance: dict[str, int] = {}
        frontier = [start]
        cut_by_depth = False
        for generation in range(1, bounded + 1):
            following: list[str] = []
            for node in frontier:
                neighbours = (
                    (edges or {}).get(node, [])
                    if edges is not None
                    else [str(p) for p in (by_id.get(node, {}).get("parent_events") or ())]
                )
                for neighbour in neighbours:
                    if neighbour in by_id and neighbour not in distance and neighbour != start:
                        distance[neighbour] = generation
                        following.append(neighbour)
            if not following:
                break
            frontier = following
        else:
            # The loop ran to the bound with a live frontier: more lies beyond.
            cut_by_depth = any(
                neighbour in by_id and neighbour not in distance and neighbour != start
                for node in frontier
                for neighbour in (
                    (edges or {}).get(node, [])
                    if edges is not None
                    else [str(p) for p in (by_id.get(node, {}).get("parent_events") or ())]
                )
            )
        return distance, cut_by_depth

    ancestors, ancestors_cut = traverse(seed_event_id, None)
    descendants, descendants_cut = traverse(seed_event_id, children)
    relations: dict[str, tuple[str, int]] = {seed_event_id: ("seed", 0)}
    for event_id, generation in ancestors.items():
        relations[event_id] = ("ancestor", generation)
    for event_id, generation in descendants.items():
        # An event reachable both ways sits on a cycle through rounds; the
        # forward reading is the informative one.
        relations[event_id] = ("descendant", generation)
    relations[seed_event_id] = ("seed", 0)

    cap = max(1, min(int(max_steps), MAX_STEPS_LIMIT))

    def nearest(event_id: str) -> tuple[Any, ...]:
        return (relations[event_id][1], *_order_key(by_id[event_id]))

    kept = sorted(relations, key=nearest)
    truncated = len(kept) > cap
    selected = set(kept[:cap])
    chain = sorted((by_id[event_id] for event_id in selected), key=_order_key)
    steps = [
        _step(
            event,
            relation=relations[str(event["event_id"])][0],
            distance=relations[str(event["event_id"])][1],
            labels=labels or {},
            artifacts_by_event=artifacts_by_event,
        )
        for event in chain
    ]
    depth_limited = ancestors_cut or descendants_cut
    return steps, {
        # "within_depth", not "in the run": the walk stops at the depth bound.
        "reachable": len(relations),
        "returned": len(steps),
        "truncated": truncated or depth_limited,
        "truncated_by": sorted(
            {name for name, hit in (("steps", truncated), ("depth", depth_limited)) if hit}
        ),
        "depth": bounded,
    }


def resolve_seed(
    rows: Sequence[Mapping[str, Any]],
    *,
    order_by: Sequence[str] = ("phase", "commit_order"),
    select: str = "first",
) -> str:
    """The seed event id from a declared dataset's rows.

    Rows come from the package's own ``outcomes.datasets``, so choosing a seed
    never requires this module to recognise a study's vocabulary.
    """
    candidates = [row for row in rows if row.get("event_id")]
    if not candidates:
        raise ValueError(
            "TRACE_SEED_EMPTY: the declared seed dataset produced no row carrying an event_id"
        )

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(_sortable(row.get(field)) for field in order_by)

    ordered = sorted(candidates, key=key)
    chosen = ordered[-1] if select == "last" else ordered[0]
    return str(chosen["event_id"])


def _sortable(value: Any) -> tuple[int, Any]:
    """Order mixed row values without raising on None or mixed types."""
    if value is None:
        return (0, 0)
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, (int, float)):
        return (1, value)
    return (2, str(value))

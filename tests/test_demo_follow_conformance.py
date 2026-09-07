"""Conformance test: user-follow transition machinery (separate from scientific runs).

The follow transition is exercised with a deterministic fixture - forced
u3 -> w1 in round 2 - and must be observed as (1) the follow state delta at
round 3, and (2) a later recommendation delivering the new relation's
article with delivery_source follower. This machinery is kept OUT of the
scientific package (demos/large-chain-package) where follow decisions arise
only from the generative user interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from genesis.runtime import (  # noqa: E402
    CallableExecutor,
    ContextEngine,
    ExecutorRegistry,
    ProcessResult,
    RunController,
    Scheduler,
    StateStore,
)

CREATORS = ["w1", "w2", "w3", "w4", "w5", "w6"]


def _follow_request(invocation):
    user = invocation.actor_ids[0]
    phase = invocation.phase or 0
    if (user, phase) == ("u3", 1):  # forced diagnostic decision (fixture only)
        action = {"user": "u3", "follow": True, "creator": "w1", "phase": 1}
        return ProcessResult(
            outputs={"action": action},
            state_effects=[{"field": "actions", "op": "append", "value": action}],
        )
    return ProcessResult(outputs={})


def _apply_follow(invocation):
    from collections.abc import Mapping as _Mapping

    context = getattr(invocation.context, "data", {}) or {}
    actions = [a for a in context.get("actions") or () if isinstance(a, _Mapping)]
    follows_value = context.get("follows")
    follows = (
        {str(k): list(v) for k, v in follows_value.items()}
        if isinstance(follows_value, _Mapping)
        else {}
    )
    for action in actions:
        user = str(action.get("user", ""))
        creator = str(action.get("creator", ""))
        current = [str(i) for i in (follows.get(user) or [])]
        if creator and creator not in current:
            current.append(creator)
        follows[user] = current
    return ProcessResult(
        outputs={"follows": follows},
        state_effects=[{"field": "follows", "op": "set", "value": follows}],
    )


def _recommend(invocation):
    from collections.abc import Mapping as _Mapping

    context = getattr(invocation.context, "data", {}) or {}
    follows_value = context.get("follows")
    follows = follows_value if isinstance(follows_value, _Mapping) else {}
    detail = {}
    for user in ("u3", "u4"):
        fans = [c for c in (follows.get(user) or ()) if c in CREATORS]
        others = [c for c in CREATORS if c not in fans]
        chosen = (fans + others[: max(0, 5 - len(fans))])[:5]
        round_no = (invocation.phase or 0) + 1
        detail[user] = [
            {
                "article_id": f"article-{round_no}-{creator}",
                "delivery_source": "follower" if creator in fans else "discovery",
            }
            for creator in chosen
        ]
    return ProcessResult(
        outputs={"exposure-detail": detail},
        state_effects=[{"field": "exposure-detail", "op": "set", "value": detail}],
    )


PROCESSES = [
    {
        "id": "recommend",
        "executor": {"mode": "computational", "parameters": {"entry_point": "conformance"}},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0, "repeat": True},
        "state_effects": [{"field": "exposure-detail", "op": "set"}],
    },
    {
        "id": "user-decision",
        "executor": {"mode": "computational", "parameters": {"entry_point": "conformance"}},
        "context_policy": "p",
        "actors": ["u3", "u4"],
        "trigger": {"type": "phase", "phase": 0, "repeat": True},
        "state_effects": [{"field": "actions", "op": "append"}],
    },
    {
        "id": "update-follow-relation",
        "executor": {"mode": "computational", "parameters": {"entry_point": "conformance"}},
        "context_policy": "p",
        "trigger": {"type": "phase", "phase": 0, "repeat": True},
        "dependencies": {"after": ["user-decision"]},
        "state_effects": [{"field": "follows", "op": "set"}],
    },
]


def test_follow_transition_machinery() -> None:
    registry = ExecutorRegistry(
        {
            "recommend": CallableExecutor(_recommend, "computational"),
            "user-decision": CallableExecutor(_follow_request, "computational"),
            "update-follow-relation": CallableExecutor(_apply_follow, "computational"),
        }
    )
    state = StateStore(
        {"follows": dict, "exposure-detail": dict, "actions": list},
        {"follows": {}, "exposure-detail": {}, "actions": []},
    )
    controller = RunController(
        Scheduler(PROCESSES),
        registry,
        ContextEngine({"p": {"allow": ["follows", "actions", "exposure-detail"]}}),
        state_store=state,
    )
    controller.run("conformance-follow", phase_limit=4)
    final = state.snapshot()
    assert "w1" in (final["follows"].get("u3") or []), "forced follow not applied"
    detail = final["exposure-detail"].get("u3") or []
    delivered = [d for d in detail if d.get("article_id", "").endswith("-w1")]
    assert delivered, "new relation not delivered in later recommendation"
    sources = {d.get("delivery_source") for d in delivered}
    assert "follower" in sources, "w1 article not delivered as follower source"


def test_terminal_skip_gates_content_processes() -> None:
    """Content processes must not fire at the terminal settlement phase."""
    from genesis.runtime import (
        CallableExecutor,
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )

    seen: list[tuple[str, int]] = []

    def probe(invocation):
        seen.append((invocation.process_id, invocation.phase or 0))
        return ProcessResult(outputs={})

    processes = [
        {
            "id": "content",
            "executor": {"mode": "computational", "parameters": {"entry_point": "x"}},
            "context_policy": "p",
            "terminal_skip": True,
            "trigger": {"type": "phase", "phase": 0, "repeat": True},
        },
        {
            "id": "settle",
            "executor": {"mode": "computational", "parameters": {"entry_point": "x"}},
            "context_policy": "p",
            "trigger": {"type": "phase", "phase": 0, "repeat": True},
            "state_effects": [{"field": "performance", "op": "append"}],
        },
    ]
    state = StateStore({"performance": list}, {"performance": []})
    registry = ExecutorRegistry(
        {p["id"]: CallableExecutor(probe, "computational") for p in processes}
    )
    controller = RunController(
        Scheduler(processes),
        registry,
        ContextEngine({"p": {"allow": ["performance"]}}),
        state_store=state,
    )
    controller.run("terminal-skip-probe", phase_limit=3, terminal_phase=2)
    # phase 2 = terminal: 'content' must be skipped; 'settle' must still run
    content_phases = [ph for pid, ph in seen if pid == "content"]
    settle_phases = [ph for pid, ph in seen if pid == "settle"]
    assert content_phases == [0, 1], content_phases
    assert settle_phases == [0, 1, 2], settle_phases

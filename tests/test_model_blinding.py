"""A model is shown what records say, never which run -- or condition -- it is in.

Engine ids are built from the run id, and a protocol's run id is
`<experiment>-<condition>-<replication>`. Records reach a model's context keyed
by instance id and carrying producer events and lineage, all run-derived, so
every model that read its inputs could read its experimental condition. In the
clickbait rehearsal a creator's prompt contained
`...-governance-hidden-sanction-peer-visibility-high-1-...`, which defeats a
manipulation that only works hidden.

What a model is sent is now a view: inputs grouped by record type, and engine
provenance removed from inputs, event history and past exchanges. Study code
still receives the full records.
"""

from __future__ import annotations

from typing import Any

from genesis.providers import ProviderExecutor, ProviderResponse, model_view
from genesis.runtime import ContextEnvelope, ProcessInvocation

RUN = "exp-governance-hidden-sanction-peer-visibility-high-1"
CONDITION = "governance-hidden-sanction-peer-visibility-high"


def _record(kind: str, phase: int, value: Any) -> dict[str, Any]:
    instance = f"{kind}-{RUN}-producer-{phase}-attempt-1"
    return {
        "artifact_type": kind,
        "instance_id": instance,
        "producer_event": f"{RUN}-producer-{phase}-attempt-1",
        "producer_process": "producer",
        "lineage": [f"note-{RUN}-earlier-attempt-1"],
        "actors": ["w1"],
        "phase": phase,
        "value": value,
    }


def _context() -> dict[str, Any]:
    first = _record("leaderboard", 1, {"top": ["a-1-w2"]})
    second = _record("leaderboard", 2, {"top": ["a-2-w3"]})
    return {
        "inputs": {first["instance_id"]: first, second["instance_id"]: second},
        "events": [{"type": "published", "producer_event": f"{RUN}-publish-1-attempt-1"}],
        "exchanges": {
            "publish": [
                {"phase": 1, "outputs": {"title": "t"}, "context": {"inputs": dict(first=first)}}
            ]
        },
        "revenue": {"w1": 3},
    }


def test_the_view_keeps_values_and_drops_run_derived_identity() -> None:
    view = model_view(_context())
    text = str(view)
    assert RUN not in text and CONDITION not in text
    assert view["inputs"]["leaderboard"] == [
        {
            "artifact_type": "leaderboard",
            "actors": ["w1"],
            "phase": 1,
            "value": {"top": ["a-1-w2"]},
        },
        {
            "artifact_type": "leaderboard",
            "actors": ["w1"],
            "phase": 2,
            "value": {"top": ["a-2-w3"]},
        },
    ]
    assert view["events"] == [{"type": "published"}]
    assert view["revenue"] == {"w1": 3}  # study-written state is left alone


class _Recording:
    provider = "openai-compatible"

    def __init__(self) -> None:
        self.sent: list[str] = []

    def generate(self, request: Any) -> ProviderResponse:
        self.sent.append(f"{request.system or ''}\n{request.prompt}")
        return ProviderResponse(
            '{"ok": true}', self.provider, request.model, "r", parsed={"ok": True}
        )


def test_no_rendered_prompt_contains_the_run_or_its_condition() -> None:
    provider = _Recording()
    executor = ProviderExecutor(
        provider, model="m", prompt_template="Round {phase} as {actor_ids}:\n{context}"
    )
    context = _context()
    invocation = ProcessInvocation(
        f"{RUN}-publish-w1-3",
        RUN,
        "publish",
        actor_ids=("w1",),
        phase=3,
        condition={"id": CONDITION, "factors": {"governance": "hidden-sanction"}},
        context=ContextEnvelope(policy_id="p", invocation_id="i", data=context, content_hash="h"),
    )
    executor.execute(invocation)
    assert provider.sent and all(RUN not in s and CONDITION not in s for s in provider.sent)
    assert "a-2-w3" in provider.sent[0]


def test_study_code_still_receives_the_full_records() -> None:
    context = _context()
    model_view(context)
    first = next(iter(context["inputs"].values()))
    assert first["instance_id"].startswith("leaderboard-" + RUN)


def test_a_short_run_id_is_not_redacted_out_of_ordinary_text() -> None:
    """A run id of "r" replaced as a substring rewrote every letter r."""
    provider = _Recording()
    executor = ProviderExecutor(provider, model="m", prompt_template="{context}")
    invocation = ProcessInvocation(
        "r-p-1",
        "r",
        "p",
        actor_ids=("u1",),
        phase=1,
        condition={"id": "base"},
        context=ContextEnvelope(
            policy_id="p", invocation_id="i", data={"note": "read carefully"}, content_hash="h"
        ),
    )
    executor.execute(invocation)
    assert "read carefully" in provider.sent[0]


def test_the_view_alone_hides_record_identity() -> None:
    """With no condition on the invocation the backstop cannot fire, so only the
    view can keep instance ids out of the prompt."""
    provider = _Recording()
    executor = ProviderExecutor(provider, model="m", prompt_template="{context}")
    invocation = ProcessInvocation(
        "x",
        RUN,
        "publish",
        actor_ids=("w1",),
        phase=3,
        context=ContextEnvelope(
            policy_id="p", invocation_id="i", data=_context(), content_hash="h"
        ),
    )
    executor.execute(invocation)
    assert "producer-2-attempt-1" not in provider.sent[0]
    assert "a-2-w3" in provider.sent[0]


def test_the_backstop_catches_a_run_id_quoted_in_study_state() -> None:
    """The view leaves study-written state as written; a value quoting the run
    id would still name the condition."""
    provider = _Recording()
    executor = ProviderExecutor(provider, model="m", prompt_template="{context}")
    invocation = ProcessInvocation(
        "x",
        RUN,
        "publish",
        actor_ids=("w1",),
        phase=3,
        condition={"id": CONDITION},
        context=ContextEnvelope(
            policy_id="p",
            invocation_id="i",
            data={"notice": f"logged under {RUN}"},
            content_hash="h",
        ),
    )
    executor.execute(invocation)
    assert RUN not in provider.sent[0] and CONDITION not in provider.sent[0]

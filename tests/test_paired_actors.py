"""A process may take one turn per record its actor produced.

``respond-to-read`` fanned out over readers, so a reader who opened two articles
reacted to one of them and the other open was never answered -- half the reading
in a 40-round run went unobserved, and the half that survived was the half the
model chose to write about. A ``per`` selector pairs each reader with each
article they opened, so every open gets its own turn and the engine, not the
model, says which article the reaction belongs to.
"""

from __future__ import annotations

import pytest

from genesis.runtime import (
    ProcessResult,
    _stamp_engine_fields,
    actor_roles,
    expand_actor_instances,
)
from genesis.specification.models import ActorSelector

STATE = {
    "population": {"rows": [{"users": [{"id": "u1"}, {"id": "u2"}, {"id": "u3"}]}]},
    "clicks": {
        "u1": {"opened": [{"article_id": "a-1-w2"}, {"article_id": "a-1-w1"}]},
        "u2": {"opened": [{"article_id": "a-1-w1"}]},
        # u3 never browsed, so it has no record at all.
    },
}

PAIRED = {
    "id": "respond-to-read",
    "actors": {
        "fan_out": True,
        "source": "population.rows.0.users",
        "id_field": "id",
        "role": "user",
        "per": {
            "source": "clicks.${actor}.opened",
            "id_field": "article_id",
            "role": "article",
        },
    },
}


def test_per_selector_yields_one_group_per_opened_article() -> None:
    assert expand_actor_instances(PAIRED, STATE) == [
        ("u1", "a-1-w1"),
        ("u1", "a-1-w2"),
        ("u2", "a-1-w1"),
    ]


def test_actor_with_no_inner_records_takes_no_turn() -> None:
    # The defect this replaces also fired for readers who opened nothing, which
    # asked a reader what they thought of an article they never saw.
    assert not any(group[0] == "u3" for group in expand_actor_instances(PAIRED, STATE))


def test_two_actors_may_share_an_inner_id() -> None:
    # Ids repeat across groups -- both readers opened a-1-w1 -- which the flat
    # duplicate check would have rejected. Only whole groups must be unique.
    groups = expand_actor_instances(PAIRED, STATE)
    assert [group for group in groups if group[1] == "a-1-w1"] == [
        ("u1", "a-1-w1"),
        ("u2", "a-1-w1"),
    ]


def test_expansion_is_deterministic_and_sorted() -> None:
    assert expand_actor_instances(PAIRED, STATE) == expand_actor_instances(PAIRED, STATE)
    # u1 opened w2 before w1; the group order follows the sorted ids, not the
    # order the reader happened to open them in.
    assert expand_actor_instances(PAIRED, STATE)[:2] == [("u1", "a-1-w1"), ("u1", "a-1-w2")]


def test_unpaired_selectors_are_unchanged() -> None:
    plain = {"id": "browse", "actors": {"source": "population.rows.0.users", "id_field": "id"}}
    assert expand_actor_instances(plain, STATE) == [("u1",), ("u2",), ("u3",)]
    assert actor_roles(plain) == ("actor",)
    assert actor_roles(PAIRED) == ("user", "article")


def test_per_requires_fan_out() -> None:
    declaration = {**PAIRED["actors"], "fan_out": False}
    with pytest.raises(ValueError, match="per requires fan_out"):
        expand_actor_instances({"id": "p", "actors": declaration}, STATE)


def test_engine_writes_each_role_into_its_own_field() -> None:
    process = {
        **PAIRED,
        "outputs": [
            {
                "artifact_type": "reader-response",
                "actor_fields": {"user_id": "user", "article_id": "article"},
                "phase_fields": ["phase"],
            }
        ],
    }
    result = _stamp_engine_fields(
        ProcessResult(outputs={"reader-response": {"user_id": "u9", "article_id": "guess"}}),
        process,
        ("u1", "a-1-w2"),
        7,
    )
    assert result.status == "succeeded"
    assert result.outputs["reader-response"] == {
        "user_id": "u1",
        "article_id": "a-1-w2",
        "phase": 7,
    }
    # What the model wrote is kept, so a reaction attributed to the wrong article
    # is visible in the trace rather than silently corrected.
    assert result.metadata["engine_fields_replaced"]["reader-response"]["article_id"] == "guess"


def test_actor_field_naming_an_undeclared_role_fails_the_invocation() -> None:
    process = {
        **PAIRED,
        "outputs": [{"artifact_type": "reader-response", "actor_fields": {"user_id": "reader"}}],
    }
    result = _stamp_engine_fields(
        ProcessResult(outputs={"reader-response": {}}), process, ("u1", "a-1-w2"), 1
    )
    assert result.status == "failed"
    assert result.metadata["code"] == "OUTPUT_ACTOR_FIELD_AMBIGUOUS"


def test_bare_actor_field_list_still_serves_a_single_actor() -> None:
    process = {
        "id": "detect",
        "actors": {"source": "articles"},
        "outputs": [{"artifact_type": "detection-result", "actor_fields": ["article_id"]}],
    }
    result = _stamp_engine_fields(
        ProcessResult(outputs={"detection-result": {"article_id": "unknown"}}),
        process,
        ("a-3-w1",),
        3,
    )
    assert result.outputs["detection-result"]["article_id"] == "a-3-w1"


def test_per_source_must_bind_the_outer_actor() -> None:
    with pytest.raises(ValueError, match=r"must bind \$\{actor\}"):
        ActorSelector(source="population", per={"source": "clicks.opened"})


def test_per_role_must_differ_from_the_outer_role() -> None:
    with pytest.raises(ValueError, match="must differ"):
        ActorSelector(
            source="population",
            role="user",
            per={"source": "clicks.${actor}.opened", "role": "user"},
        )


def test_unpaired_selector_serialises_without_the_new_fields() -> None:
    # Roles and per are omitted at their defaults, so every package compiled
    # before paired actors existed still hashes to the same build.
    assert ActorSelector(source="population.rows.0.users", id_field="id").model_dump() == {
        "ids": None,
        "source": "population.rows.0.users",
        "id_field": "id",
        "fan_out": True,
    }


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.ids: list[str] = []

    def execute(self, invocation):
        self.calls.append(invocation.actor_ids)
        self.ids.append(invocation.invocation_id)
        return ProcessResult(outputs={})


def test_a_paired_process_runs_end_to_end() -> None:
    from genesis.runtime import (
        ContextEngine,
        ExecutorRegistry,
        RunController,
        Scheduler,
        StateStore,
    )

    recorder = _Recorder()
    process = {**PAIRED, "context_policy": "private"}
    controller = RunController(
        Scheduler([process]),
        ExecutorRegistry({"respond-to-read": recorder}),
        ContextEngine({"private": {"allow": []}}),
        state_store=StateStore({"population": dict, "clicks": dict}, dict(STATE)),
    )

    executed = controller.run("paired-run", phase_limit=1, seed=7)

    # Three turns, not three readers: u1 answers twice and u3 not at all.
    assert executed == ["respond-to-read"] * 3
    assert recorder.calls == [("u1", "a-1-w1"), ("u1", "a-1-w2"), ("u2", "a-1-w1")]
    # Both of u1's turns must be separately addressable, or the second would
    # collide with the first on idempotency and be dropped.
    assert len(set(recorder.ids)) == 3
    assert recorder.ids[0] == "paired-run-respond-to-read-u1-a-1-w1-0"


def test_a_role_may_not_be_named_twice() -> None:
    """Both fields would take the same actor and the other id would be lost."""
    process = {
        **PAIRED,
        "outputs": [
            {"artifact_type": "reader-response", "actor_fields": {"x": "user", "y": "user"}}
        ],
    }
    result = _stamp_engine_fields(
        ProcessResult(outputs={"reader-response": {}}), process, ("u1", "a-1-w2"), 1
    )
    assert result.status == "failed"
    assert result.metadata["code"] == "OUTPUT_ACTOR_FIELD_AMBIGUOUS"
    assert "only once" in result.metadata["error"]


def test_a_paired_ids_selector_declares_no_reproducible_order() -> None:
    """Listed outer ids do not reproduce the order: the inner half comes from state.

    Returning an empty tuple rather than None read as "the declaration
    reproduces it", so batch_actors went unrecorded and an interrupted batch
    could not be reopened from the declared order (CON-008/CON-010).
    """
    from genesis.runtime import _declared_order

    paired_ids = {
        "id": "respond-to-read",
        "actors": {
            "ids": ["u1", "u2"],
            "role": "user",
            "per": {"source": "clicks.${actor}.opened", "role": "article"},
        },
    }
    assert _declared_order(paired_ids) is None
    # An unpaired ids-selector still reproduces its order from the declaration.
    assert _declared_order({"id": "p", "actors": {"ids": ["u1", "u2"]}}) == (("u1",), ("u2",))

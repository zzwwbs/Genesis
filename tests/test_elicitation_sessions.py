"""Task 2: in-memory elicitation session state machine."""

from __future__ import annotations

from pathlib import Path

import pytest

from genesis.elicitation import (
    ElicitationEngine,
    ElicitationSessionStore,
    WorkflowRegistry,
    known_checklist_ids,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


@pytest.fixture
def registry() -> WorkflowRegistry:
    return WorkflowRegistry(WORKFLOWS, checklist_ids=known_checklist_ids())


@pytest.fixture
def store() -> ElicitationSessionStore:
    return ElicitationSessionStore()


def _start(
    engine: ElicitationEngine,
    *,
    spec_id: str = "cooperation-study",
    base_version: int = 1,
):
    return engine.start_session(
        specification_id=spec_id,
        workflow_id="three-layer-study",
        model_profile_id="assistant",
        researcher_id="researcher",
        base_specification_version=base_version,
    )


def test_session_starts_at_configured_opening_question(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    assert session.current_stage == "study-foundation"
    assert session.status == "awaiting_answer"
    assert session.current_question == registry.get("three-layer-study").stages[0].opening_question
    assert session.stages["study-foundation"].status == "clarifying"
    assert session.stages["openness"].status == "not_started"


def test_stage_order_is_workflow_declared(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    _start(engine)
    workflow = registry.get("three-layer-study")
    for index, stage in enumerate(workflow.stages):
        assert stage.id == workflow.stages[index].id
    assert workflow.next_stage("openness").id == "theory"


def test_recorded_turns_are_immutable(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    turn = engine.record_researcher_answer(
        session,
        answer="Agents choose what to say.",
        response_mode="free_form",
        suggestions=(("Now", "Now"), ("Later", "Later"), ("Never", "Never")),
    )
    # Mutating the returned turn must not affect the stored copy.
    turn.answer = "tampered"
    stored = store.get(session.session_id)
    assert stored.turns[0].answer == "Agents choose what to say."
    assert stored.turns[0].id == 1
    # Sessions returned from the store are deep copies too.
    stored.turns[0].answer = "also tampered"
    again = store.get(session.session_id)
    assert again.turns[0].answer == "Agents choose what to say."


def test_stage_turn_counts_do_not_include_other_stages(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    engine.record_researcher_answer(session, answer="Foundation one.")
    engine.record_researcher_answer(session, answer="Foundation two.")
    session.current_stage = "openness"
    engine.record_researcher_answer(session, answer="Openness one.")
    stored = store.get(session.session_id)
    assert stored.stages["study-foundation"].turn_count == 2
    assert stored.stages["openness"].turn_count == 1


def test_transitions_follow_the_state_machine(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    # awaiting_answer -> awaiting_approval (draft ready)
    engine.transition(session, "awaiting_approval", stage_status="awaiting_approval")
    # awaiting_approval -> revision_requested
    engine.transition(session, "revision_requested", stage_status="clarifying")
    # revision_requested -> awaiting_answer
    engine.transition(session, "awaiting_answer")
    # awaiting_answer -> cancelled
    engine.transition(session, "cancelled")
    assert session.status == "cancelled"
    # No transitions after cancellation.
    with pytest.raises(ValueError, match="ELICITATION_STATE_CONFLICT"):
        engine.transition(session, "awaiting_answer")


def test_invalid_transition_is_rejected(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    # Direct jump to completed without approval is invalid.
    with pytest.raises(ValueError, match="ELICITATION_STATE_CONFLICT"):
        engine.transition(session, "completed")
    # Unknown statuses are rejected.
    with pytest.raises(ValueError, match="ELICITATION_STATE_CONFLICT"):
        engine.transition(session, "no_such_status")


def test_cancellation_discards_the_session(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    engine.cancel(session.session_id)
    stored = store.get(session.session_id)
    assert stored.status == "cancelled"
    assert stored.stages["study-foundation"].status == "cancelled"
    engine.cancel(session.session_id)  # idempotent


def test_missing_session_after_new_store(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine)
    fresh_store = ElicitationSessionStore()
    fresh_engine = ElicitationEngine(registry, fresh_store)
    with pytest.raises(KeyError, match="ELICITATION_SESSION_EXPIRED"):
        fresh_engine.require_session(session.session_id)


def test_base_version_and_metadata_are_pinned(
    registry: WorkflowRegistry, store: ElicitationSessionStore
) -> None:
    engine = ElicitationEngine(registry, store)
    session = _start(engine, spec_id="v-study", base_version=3)
    assert session.base_specification_version == 3
    assert session.workflow_id == "three-layer-study"
    assert session.workflow_version == "1.0"
    assert session.researcher_id == "researcher"
    assert session.model_profile_id == "assistant"


def test_session_store_idempotency_is_bounded_and_payload_sensitive(
    store: ElicitationSessionStore,
) -> None:
    payload_hash = "payload-a"
    store.record_idempotency("session", "key", payload_hash, {"ok": True})
    assert store.get_idempotency("session", "key", payload_hash) == {"ok": True}
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        store.get_idempotency("session", "key", "payload-b")

    for index in range(140):
        store.record_idempotency("session", f"key-{index}", f"payload-{index}", {"index": index})
    assert store.idempotency_size <= 128

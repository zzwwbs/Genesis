from __future__ import annotations

from genesis.runtime import ContextEngine, ProcessInvocation


def _article_inputs() -> dict[str, object]:
    return {
        "article-body-1": {
            "artifact_type": "article-body",
            "value": {"title": "A title", "body": "Full text"},
            "producer_event": "write-article-1",
        },
        "article-body-2": {
            "artifact_type": "article-body",
            "value": {"title": "Another title", "body": "Other text"},
            "producer_event": "write-article-2",
        },
    }


def test_context_can_select_state_inputs_and_condition_namespaces() -> None:
    engine = ContextEngine(
        {"detector": {"allow": ["state.public-rules", "inputs", "condition.factors.governance"]}}
    )
    invocation = ProcessInvocation(
        "detector-1",
        "run-1",
        "detect-clickbait",
        actor_ids=("platform-1",),
        inputs=_article_inputs(),
        condition={"id": "opaque", "factors": {"governance": "opaque"}},
    )

    envelope = engine.build(
        "detector",
        invocation,
        {"public-rules": {"threshold": 0.5}, "latent-strategy": {"intent": "deceive"}},
    )

    assert envelope.data["state"]["public-rules"] == {"threshold": 0.5}
    assert envelope.data["condition"]["factors"]["governance"] == "opaque"
    assert "latent-strategy" not in envelope.data["state"]
    assert {exposure["source_artifact_id"] for exposure in envelope.exposures} == {
        "article-body-1",
        "article-body-2",
    }
    assert all(exposure["recipient_ids"] == ("platform-1",) for exposure in envelope.exposures)


def test_body_entitlement_requires_a_selection_event_for_the_same_recipient() -> None:
    engine = ContextEngine(
        {
            "full-article": {
                "allow": ["inputs"],
                "available_when": {
                    "inputs": {
                        "event": "article-selected",
                        "recipient_match": True,
                        "source_match_event": True,
                    }
                },
            }
        }
    )
    history = (
        {
            "type": "article-selected",
            "recipient_id": "user-1",
            "source_artifact_id": "article-body-1",
        },
    )
    selected = ProcessInvocation(
        "read-1",
        "run-1",
        "read-article",
        actor_ids=("user-1",),
        inputs=_article_inputs(),
        event_history=history,
    )
    not_selected = ProcessInvocation(
        "read-2",
        "run-1",
        "read-article",
        actor_ids=("user-2",),
        inputs=_article_inputs(),
        event_history=history,
    )

    assert list(engine.build("full-article", selected, {}).data["inputs"]) == ["article-body-1"]
    assert engine.build("full-article", not_selected, {}).data == {}


def test_context_cardinality_caps_bound_candidate_artifacts() -> None:
    engine = ContextEngine({"titles": {"allow": ["inputs"], "cardinality": {"inputs": 2}}})
    invocation = ProcessInvocation(
        "recommend-1",
        "run-1",
        "recommend",
        actor_ids=("user-1",),
        inputs={
            f"title-{index}": {"artifact_type": "article-title", "value": str(index)}
            for index in range(4)
        },
    )

    envelope = engine.build("titles", invocation, {})

    assert list(envelope.data["inputs"]) == ["title-0", "title-1"]
    assert len(envelope.exposures) == 2

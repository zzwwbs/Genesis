"""The clickbait study's platform mechanics.

These four functions are the closed half of the design, so what they do is a
scientific claim and not an implementation detail: who receives an article, what
a click is worth, when the sanction applies and to whom, and what the leaderboard
shows. Each test below states one of those claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from genesis.runtime import _plain
from studies.clickbait_2x2 import (
    DETECTION_THRESHOLD,
    DISCOVERY_SHARE,
    SANCTION_ONSET_PHASE,
    SANCTION_RETENTION,
    apply_responses,
    distribute,
    settle,
    top_3_leaderboard,
)

SCHEMAS = Path("genesis-workspace/.genesis/specifications/clickbait-2x2-v4/schemas")


class _Context:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data


class _Invocation:
    def __init__(
        self,
        data: dict[str, Any],
        *,
        phase: int = 1,
        actor: str | None = None,
        factors: dict[str, str] | None = None,
    ) -> None:
        self.context = _Context(data)
        self.phase = phase
        self.actor_ids = (actor,) if actor else ()
        self.condition = {"factors": dict(factors or {})}


def _article(creator: str, phase: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "id": f"a-{phase}-{creator}",
        "creator_id": creator,
        "phase": phase,
        "category": "news",
        "track": "science",
        "topic": "topic",
        "title": "A title",
        "body": "A body",
        **extra,
    }


def _outputs(result: Any) -> dict[str, Any]:
    """ProcessResult freezes its payloads; the runtime normalises the same way."""
    return dict(_plain(result.outputs))


def _effects(result: Any) -> dict[str, Any]:
    return {effect["field"]: _plain(effect) for effect in _plain(result.state_effects)}


def _validate(schema_name: str, value: Any) -> None:
    """Hold the executors to the schemas the package actually declares."""
    import jsonschema

    schema = json.loads((SCHEMAS / f"{schema_name}.json").read_text())
    jsonschema.validate(value, schema)


# --- distribution -----------------------------------------------------------------


def test_every_follower_receives_the_article_and_the_channel_says_so() -> None:
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": ["u1"]},
        "population": {"users": [{"id": "u1"}, {"id": "u2"}]},
    }
    result = distribute(_Invocation(data, actor="u1"))
    feed = _outputs(result)["user-feed"]
    _validate("feed-schema", feed)
    assert [item["channel"] for item in feed["items"]] == ["follow"]
    assert feed["items"][0]["article_id"] == "a-1-w1"


def test_a_feed_item_never_carries_the_body() -> None:
    """A user choosing what to click must see the title, never what it delivers."""
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": ["u1"]},
        "population": {"users": [{"id": "u1"}]},
    }
    feed = _outputs(distribute(_Invocation(data, actor="u1")))["user-feed"]
    assert all("body" not in item for item in feed["items"])


def test_non_followers_are_sampled_at_roughly_the_declared_share() -> None:
    users = [f"u{index}" for index in range(400)]
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": []},
        "population": {"users": [{"id": user} for user in users]},
    }
    reached = sum(
        1
        for user in users
        if _outputs(distribute(_Invocation(data, actor=user)))["user-feed"]["items"]
    )
    # Independent per (article, user), so the share is approximate, not exact.
    assert abs(reached / len(users) - DISCOVERY_SHARE) < 0.06


def test_the_same_user_and_article_always_draw_the_same_way() -> None:
    """Matched conditions must differ in the treatment, not in the draw."""
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": []},
        "population": {"users": [{"id": "u7"}]},
    }
    first = _outputs(distribute(_Invocation(data, actor="u7")))["user-feed"]
    second = _outputs(distribute(_Invocation(data, actor="u7")))["user-feed"]
    assert first == second


def test_each_user_writes_only_their_own_feed_entry() -> None:
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": ["u1"]},
        "population": {"users": [{"id": "u1"}]},
    }
    effects = _effects(distribute(_Invocation(data, actor="u1")))
    assert [(e["field"], e["op"], e["key"]) for e in effects.values()] == [("feeds", "put", "u1")]


# --- settlement -------------------------------------------------------------------


def _settle_data(score: int, clicks: int = 4) -> dict[str, Any]:
    return {
        "articles": [_article("w1")],
        "detection-result": [
            {"article_id": "a-1-w1", "score": score, "checks": [], "rationale": "r"}
        ],
        "reader-response": [
            {
                "user_id": f"u{index}",
                "phase": 1,
                "article_id": "a-1-w1",
                "liked": index == 0,
                "favourited": False,
                "commented": False,
                "reported": False,
                "follow_action": "keep",
            }
            for index in range(clicks)
        ],
        "feeds": {"u0": [{"article_id": "a-1-w1"}], "u1": [{"article_id": "a-1-w1"}]},
        "revenue": {"w1": 10.0},
    }


def _performance(result: Any) -> dict[str, Any]:
    rows = _effects(result)["article-performance"]["value"]
    return list(rows.values())[-1] if isinstance(rows, dict) else rows[-1]


@pytest.mark.parametrize(
    ("governance", "phase", "penalised"),
    [
        ("hidden-sanction", SANCTION_ONSET_PHASE, True),
        ("hidden-sanction", SANCTION_ONSET_PHASE - 1, False),  # before onset
        ("none", SANCTION_ONSET_PHASE, False),  # no governance
    ],
)
def test_the_sanction_applies_only_under_governance_and_from_the_onset(
    governance: str, phase: int, penalised: bool
) -> None:
    data = _settle_data(score=DETECTION_THRESHOLD)
    data["articles"] = [_article("w1", phase=phase)]
    for response in data["reader-response"]:
        response["article_id"] = f"a-{phase}-w1"
    data["detection-result"][0]["article_id"] = f"a-{phase}-w1"
    row = _performance(settle(_Invocation(data, phase=phase, factors={"governance": governance})))
    assert row["penalised"] is penalised
    expected = row["gross_revenue"] * (SANCTION_RETENTION if penalised else 1.0)
    assert row["actual_revenue"] == expected


def test_an_article_below_the_threshold_is_never_penalised() -> None:
    row = _performance(
        settle(
            _Invocation(
                _settle_data(score=DETECTION_THRESHOLD - 1),
                factors={"governance": "hidden-sanction"},
                phase=1,
            )
        )
    )
    assert row["penalised"] is False
    assert row["actual_revenue"] == row["gross_revenue"]


def test_a_missing_detection_leaves_the_article_unsanctioned() -> None:
    """A failed detection is recorded as missing, never as a positive."""
    data = _settle_data(score=DETECTION_THRESHOLD)
    data["detection-result"] = []
    row = _performance(
        settle(_Invocation(data, phase=1, factors={"governance": "hidden-sanction"}))
    )
    assert row["penalised"] is False


def test_revenue_is_one_per_click_and_accumulates_onto_the_prior_total() -> None:
    result = settle(_Invocation(_settle_data(score=0, clicks=4), factors={"governance": "none"}))
    row = _performance(result)
    assert row["clicks"] == 4
    assert row["gross_revenue"] == 4.0
    revenue = _effects(result)["revenue"]["value"]
    assert revenue["w1"] == 14.0  # 10.0 carried in, plus this round's 4


def test_settlement_emits_no_artifact_and_the_creator_reads_state_instead() -> None:
    """The ledger schemas are per-creator; this step runs once per round."""
    result = settle(
        _Invocation(
            _settle_data(score=9),
            factors={"governance": "hidden-sanction"},
            phase=1,
        )
    )
    assert _outputs(result) == {}
    assert sorted(_effects(result)) == ["article-performance", "revenue"]


def test_the_creator_facing_revenue_carries_no_score_and_no_gross_figure() -> None:
    """The sanction is hidden: the creator sees a lower number and no reason."""
    result = settle(
        _Invocation(
            _settle_data(score=9),
            factors={"governance": "hidden-sanction"},
            phase=SANCTION_ONSET_PHASE,
        )
    )
    revenue = json.dumps(_effects(result)["revenue"]["value"])
    for leaked in ("score", "penalised", "gross"):
        assert leaked not in revenue


def test_exposures_count_every_delivered_title() -> None:
    row = _performance(settle(_Invocation(_settle_data(score=0), factors={"governance": "none"})))
    assert row["exposures"] == 2


# --- responses --------------------------------------------------------------------


def test_follow_and_unfollow_are_applied_and_impressions_accumulate() -> None:
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": ["u9"]},
        "impressions": {"w1": [{"user_id": "u9", "phase": 0, "impression": "earlier"}]},
        "reader-response": [
            {
                "user_id": "u1",
                "phase": 1,
                "article_id": "a-1-w1",
                "liked": True,
                "favourited": False,
                "commented": False,
                "reported": False,
                "follow_action": "follow",
                "author_impression": "Delivered what it promised.",
            },
            {
                "user_id": "u9",
                "phase": 1,
                "article_id": "a-1-w1",
                "liked": False,
                "favourited": False,
                "commented": False,
                "reported": True,
                "follow_action": "unfollow",
                "author_impression": "Title oversold it.",
            },
        ],
    }
    effects = {k: v["value"] for k, v in _effects(apply_responses(_Invocation(data))).items()}
    assert effects["follower-network"]["w1"] == ["u1"]
    # Permanent: the earlier impression survives alongside the new ones.
    notes = [entry["impression"] for entry in effects["impressions"]["w1"]]
    assert notes == ["earlier", "Delivered what it promised.", "Title oversold it."]


def test_a_response_to_an_unknown_article_changes_nothing() -> None:
    data = {
        "articles": [_article("w1")],
        "follower-network": {"w1": []},
        "reader-response": [
            {
                "user_id": "u1",
                "phase": 1,
                "article_id": "missing",
                "liked": False,
                "favourited": False,
                "commented": False,
                "reported": False,
                "follow_action": "follow",
            }
        ],
    }
    effects = {k: v["value"] for k, v in _effects(apply_responses(_Invocation(data))).items()}
    assert effects["follower-network"] == {"w1": []}


# --- leaderboard ------------------------------------------------------------------


def _perf_row(article_id: str, author: str, clicks: int, **extra: Any) -> dict[str, Any]:
    return {
        "phase": 1,
        "article_id": article_id,
        "author": author,
        "track": "science",
        "exposures": 10,
        "clicks": clicks,
        "likes": 0,
        "favourites": 0,
        "follows": 0,
        "unfollows": 0,
        "actual_revenue": float(clicks),
        "penalised": False,
        **extra,
    }


def test_the_leaderboard_is_the_three_most_clicked_of_this_round() -> None:
    data = {
        "article-performance": [
            _perf_row("a1", "w1", 1),
            _perf_row("a2", "w2", 9),
            _perf_row("a3", "w3", 5),
            _perf_row("a4", "w4", 7),
            _perf_row("old", "w5", 99, phase=0),
        ]
    }
    board = _outputs(top_3_leaderboard(_Invocation(data)))["leaderboard-top3"]
    _validate("leaderboard-schema", board)
    assert [entry["article_id"] for entry in board["entries"]] == ["a2", "a4", "a3"]
    assert [entry["rank"] for entry in board["entries"]] == [1, 2, 3]


def test_a_penalised_article_still_reaches_the_leaderboard() -> None:
    """The conflict between the two signals is the mechanism under study."""
    data = {
        "article-performance": [
            _perf_row("clickbait", "w1", 20, penalised=True),
            _perf_row("honest", "w2", 3),
        ]
    }
    board = _outputs(top_3_leaderboard(_Invocation(data)))["leaderboard-top3"]
    assert board["entries"][0]["article_id"] == "clickbait"


def test_the_leaderboard_never_carries_private_or_detector_fields() -> None:
    data = {
        "article-performance": [
            _perf_row("a1", "w1", 5, strategy_note="my angle", reason="because", score=9)
        ]
    }
    board = _outputs(top_3_leaderboard(_Invocation(data)))["leaderboard-top3"]
    shown = json.dumps(board)
    for private in ("strategy_note", "reason", "score", "penalised"):
        assert private not in shown


def test_an_artifact_input_is_read_as_readily_as_a_state_field() -> None:
    """A state arrives under its own key; an artifact arrives under 'inputs'.

    Reading only the top-level key found the states and silently found nothing
    for the artifacts, so settlement priced a round having seen none of its
    responses: every article ended with exposures, no clicks and no revenue.
    """
    phase = SANCTION_ONSET_PHASE
    article_id = f"a-{phase}-w1"
    data = {
        "articles": [_article("w1", phase=phase)],
        "inputs": {
            "resp-1": {
                "artifact_type": "reader-response",
                "value": {
                    "user_id": "u1",
                    "phase": phase,
                    "article_id": article_id,
                    "liked": True,
                    "favourited": False,
                    "commented": False,
                    "reported": False,
                    "follow_action": "keep",
                },
            },
            "det-1": {
                "artifact_type": "detection-result",
                "value": {
                    "article_id": article_id,
                    "score": 9,
                    "checks": {},
                    "rationale": "r",
                },
            },
        },
        "feeds": {"u1": [{"article_id": article_id}]},
    }
    row = _performance(
        settle(_Invocation(data, phase=phase, factors={"governance": "hidden-sanction"}))
    )
    assert row["clicks"] == 1
    assert row["exposures"] == 1
    # The detection came through the same way, so the sanction could apply.
    assert row["penalised"] is True


def test_a_detection_joins_on_the_article_it_was_invoked_for_not_the_id_it_wrote() -> None:
    """In paid calls the detector wrote "unknown", "" or the title as article_id."""
    phase = SANCTION_ONSET_PHASE
    article_id = f"a-{phase}-w1"
    data = {
        "articles": [_article("w1", phase=phase)],
        "inputs": {
            "resp": {
                "artifact_type": "reader-response",
                "value": {"user_id": "u1", "phase": phase, "article_id": article_id},
            },
            "det": {
                "artifact_type": "detection-result",
                "actors": [article_id],
                "value": {"article_id": "unknown", "score": 9, "checks": {}, "rationale": "r"},
            },
        },
    }
    row = _performance(
        settle(_Invocation(data, phase=phase, factors={"governance": "hidden-sanction"}))
    )
    assert row["penalised"] is True

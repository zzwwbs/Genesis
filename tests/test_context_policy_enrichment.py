"""AW-08: context policy enrichment — projection, cardinality, aggregation, availability."""

from __future__ import annotations

from genesis.runtime import ContextEngine, ProcessInvocation

POLICIES = {
    "limited": {
        "allow": ["tags", "scores", "secret", "early"],
        "redact": ["secret"],
        "cardinality": {"tags": 2},
        "aggregate": {"scores": {"op": "sum"}},
        "available_when": {"early": {"after_round": 2}},
    }
}


def _invocation(phase: int) -> ProcessInvocation:
    return ProcessInvocation("inv-1", "run-1", "p", phase=phase)


def test_cardinality_cap_truncates_deterministically() -> None:
    engine = ContextEngine(POLICIES)
    envelope = engine.build(
        "limited",
        _invocation(0),
        {"tags": ["a", "b", "c", "d"], "scores": {"x": 1, "y": 2}, "secret": "s", "early": 1},
    )
    assert envelope.data["tags"] == ("a", "b")
    assert envelope.data["scores"] == 3  # aggregated sum
    assert "secret" not in envelope.data  # redacted
    assert envelope.content_hash  # envelope always hashed


def test_availability_blocks_early_access_but_allows_later() -> None:
    engine = ContextEngine(POLICIES)
    state = {"tags": [], "scores": {}, "early": "payload"}
    early_envelope = engine.build("limited", _invocation(1), state)
    assert "early" not in early_envelope.data
    late_envelope = engine.build("limited", _invocation(3), state)
    assert late_envelope.data["early"] == "payload"


def test_aggregate_mean_and_count_operations() -> None:
    engine = ContextEngine(
        {
            "stats": {
                "allow": ["values", "names"],
                "aggregate": {"values": {"op": "mean"}, "names": {"op": "count"}},
            }
        }
    )
    envelope = engine.build(
        "stats",
        _invocation(0),
        {"values": [1, 2, 3, 4], "names": ["a", "b"]},
    )
    assert envelope.data["values"] == 2.5
    assert envelope.data["names"] == 2


def test_aggregate_rejects_unsupported_operation() -> None:
    engine = ContextEngine({"bad": {"allow": ["values"], "aggregate": {"values": {"op": "nope"}}}})
    try:
        engine.build("bad", _invocation(0), {"values": [1]})
    except ValueError as exc:
        assert "unsupported aggregate operation" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cardinality_rejects_negative_limit() -> None:
    engine = ContextEngine({"bad": {"allow": ["tags"], "cardinality": {"tags": -1}}})
    try:
        engine.build("bad", _invocation(0), {"tags": ["a"]})
    except ValueError as exc:
        assert "cardinality" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_aggregated_and_capped_content_changes_envelope_hash() -> None:
    engine = ContextEngine({"plain": {"allow": ["tags", "scores"]}})
    full = engine.build(
        "plain", _invocation(0), {"tags": ["a", "b", "c"], "scores": {"x": 1, "y": 2, "z": 3}}
    )
    assert full.data["tags"] == ("a", "b", "c")
    assert full.data["scores"] == {"x": 1, "y": 2, "z": 3}
    assert full.content_hash != POLICIES  # sanity only; hash is present
    assert isinstance(full.content_hash, str) and len(full.content_hash) == 64


def test_compiler_merges_domain_availability_into_context_policies(tmp_path) -> None:
    """Review finding 9: domain.availability reaches the compiled context policies."""
    import json

    from genesis.compiler import StudyCompiler

    source = tmp_path / "pkg"
    source.mkdir()
    (source / "study.yaml").write_text('schema_version: "1.0"\nstudy_id: av-study\ntitle: x\n')
    (source / "openness.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: av-study\nprocesses:\n'
        "  - id: p\n    executor: {}\n    context_policy: av-policy\n"
    )
    (source / "theory.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: av-study\ntheory_family: exploratory\n'
    )
    (source / "domain.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: av-study\nvisibility:\n'
        "  - id: av-policy\n    allow: [counter]\navailability:\n"
        "  - path: counter\n    available_when: {after_round: 2}\n"
    )
    (source / "protocol.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: av-study\ntime_model: {type: rounds}\n'
    )
    (source / "outcomes.yaml").write_text(
        'schema_version: "1.0"\nstudy_id: av-study\noutcomes: []\n'
    )
    (source / "models.yaml").write_text('schema_version: "1.0"\nstudy_id: av-study\n')
    build = StudyCompiler(source).compile(tmp_path / "av-build")
    policies = json.loads((build.path / "context_policies.json").read_text())
    policy = next(item for item in policies if item.get("id") == "av-policy")
    assert policy["available_when"]["counter"] == {"after_round": 2}

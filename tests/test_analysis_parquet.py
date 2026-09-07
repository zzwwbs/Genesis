import json
from pathlib import Path

import pyarrow.parquet as pq

from genesis.analysis import AnalysisEngine, AnalysisExporter, OutcomePlan


def test_outcome_evaluation_reports_missing_values_explicitly() -> None:
    result = AnalysisEngine().evaluate(
        OutcomePlan("score", "events", "score", "mean"),
        {"events": [{"score": 2}, {"score": None}, {}]},
    )
    assert result == [{"score_mean": 2, "score_missing": 2}]
    assert AnalysisEngine().evaluate(
        OutcomePlan("score", "events", "score", "mean"), {"events": []}
    ) == [{"score_mean": None, "score_missing": 0}]


def test_export_bundle_contains_real_parquet_and_verifiable_lineage(tmp_path: Path) -> None:
    rows = [{"outcome_id": "score", "score_mean": 2.0, "score_missing": 1}]
    paths = AnalysisExporter().export_bundle(
        rows,
        tmp_path,
        methods={"aggregation": "mean"},
        replay_lineage={"source_run_id": "run-1"},
    )
    assert "outcomes.parquet" in {path.name for path in paths}
    assert pq.read_table(tmp_path / "outcomes.parquet").to_pylist() == rows
    assert json.loads((tmp_path / "replay_lineage.json").read_text())["source_run_id"] == "run-1"
    assert AnalysisExporter.verify_bundle(tmp_path)
    (tmp_path / "methods.json").write_text("tampered")
    assert not AnalysisExporter.verify_bundle(tmp_path)


def test_duckdb_aggregates_over_100k_rows_quickly() -> None:
    import time

    from genesis.analysis import AnalysisEngine, OutcomePlan, duckdb_aggregate

    rows = [{"group": f"g{i % 10}", "value": i, "time": i} for i in range(100_000)]
    started = time.perf_counter()
    result = duckdb_aggregate(rows, select="value", op="sum", group_by="group")
    elapsed = time.perf_counter() - started
    assert len(result) == 10
    assert result[0]["value_sum"] > 0
    assert elapsed < 5.0, f"duckdb aggregation took {elapsed:.2f}s"
    # Cross-check against the in-process engine.
    reference = AnalysisEngine().evaluate(
        OutcomePlan("o", "rows", "value", "sum", "group"),
        {"rows": rows},
    )
    assert {row["group"]: row["value_sum"] for row in result} == {
        row["group"]: row["value_sum"] for row in reference
    }

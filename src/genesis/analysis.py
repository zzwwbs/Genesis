"""Small declarative outcome evaluator and portable export helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutcomePlan:
    id: str
    source: str
    select: str
    aggregation: str = "count"
    group_by: tuple[str, ...] | str | None = None
    filters: tuple[dict[str, Any], ...] = ()
    operation: str = "aggregate"  # aggregate | distribution | trajectory
    missingness: str = "exclude"  # exclude | zero
    window: dict[str, Any] | None = None


class AnalysisEngine:
    def evaluate(
        self, plan: OutcomePlan, sources: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        rows = list(sources.get(plan.source, []))
        if plan.window:
            rows = self._window_filter(rows, plan.window)
        for predicate in plan.filters:
            rows = [
                row
                for row in rows
                if all(row.get(key) == value for key, value in predicate.items())
            ]
        group_keys = (plan.group_by,) if isinstance(plan.group_by, str) else (plan.group_by or ())
        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row.get(g) for g in group_keys) if group_keys else None
            groups.setdefault(key, []).append(row)
        if not groups and not group_keys:
            groups[None] = []
        if plan.operation == "trajectory":
            return self._trajectory(plan, groups)
        if plan.operation == "distribution":
            return self._distribution(plan, groups)
        return self._aggregate(plan, groups)

    @staticmethod
    def _window_filter(rows: list[dict[str, Any]], window: dict[str, Any]) -> list[dict[str, Any]]:
        time_field = str(window.get("time_field", "time"))
        start = window.get("start")
        end = window.get("end")
        out = []
        for row in rows:
            if time_field not in row:
                continue
            time = row[time_field]
            if start is not None and time < start:
                continue
            if end is not None and time > end:
                continue
            out.append(row)
        return out

    @staticmethod
    def _group_result(plan: OutcomePlan, key: Any) -> dict[str, Any]:
        keys = (plan.group_by,) if isinstance(plan.group_by, str) else (plan.group_by or ())
        if not keys:
            return {}
        if len(keys) == 1:
            return {keys[0]: key if not isinstance(key, tuple) else key[0]}
        return {f"group_{i}": item for i, item in enumerate(key)}

    def _aggregate(
        self, plan: OutcomePlan, groups: dict[Any, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        out = []
        for key, group in groups.items():
            values = [row[plan.select] for row in group if row.get(plan.select) is not None]
            missing = len(group) - len(values)
            if plan.missingness == "zero" and missing and plan.aggregation in {"sum", "mean"}:
                values = [*values, *([0] * missing)]
            if plan.aggregation == "mean":
                if any(not isinstance(item, int | float) for item in values):
                    raise ValueError(f"mean requires numeric values: {plan.select}")
                value = statistics.mean(values) if values else None
                name = f"{plan.select}_mean"
            elif plan.aggregation == "sum":
                if any(not isinstance(item, int | float) for item in values):
                    raise ValueError(f"sum requires numeric values: {plan.select}")
                value, name = sum(values), f"{plan.select}_sum"
            elif plan.aggregation == "count":
                value, name = len(values), f"{plan.select}_count"
            else:
                raise ValueError(f"unsupported aggregation: {plan.aggregation}")
            result = self._group_result(plan, key)
            result[name] = value
            result[f"{plan.select}_missing"] = missing
            out.append(result)
        return out

    def _distribution(
        self, plan: OutcomePlan, groups: dict[Any, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        out = []
        for key, group in groups.items():
            values = [row[plan.select] for row in group if row.get(plan.select) is not None]
            result = self._group_result(plan, key)
            if not values:
                result.update(
                    {
                        f"{plan.select}_count": 0,
                        f"{plan.select}_min": None,
                        f"{plan.select}_max": None,
                        f"{plan.select}_mean": None,
                    }
                )
            else:
                numeric = [item for item in values if isinstance(item, int | float)]
                result.update(
                    {
                        f"{plan.select}_count": len(values),
                        f"{plan.select}_min": min(values),
                        f"{plan.select}_max": max(values),
                        f"{plan.select}_mean": statistics.mean(numeric) if numeric else None,
                    }
                )
            result[f"{plan.select}_missing"] = len(group) - len(values)
            out.append(result)
        return out

    def _trajectory(
        self, plan: OutcomePlan, groups: dict[Any, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        out = []
        for key, group in groups.items():
            ordered = sorted(
                group,
                key=lambda item: item.get(
                    plan.window.get("time_field", "time") if plan.window else "time", 0
                ),
            )
            trajectory = [row[plan.select] for row in ordered if row.get(plan.select) is not None]
            result = self._group_result(plan, key)
            result[f"{plan.select}_trajectory"] = trajectory
            result[f"{plan.select}_missing"] = len(group) - len(trajectory)
            out.append(result)
        return out

    def window(
        self,
        rows: list[dict[str, Any]],
        field: str,
        *,
        start: int | float,
        end: int | float,
        time_field: str = "time",
    ) -> list[dict[str, Any]]:
        """Select rows whose declared temporal coordinate is within [start, end]."""
        if end < start:
            raise ValueError("window end must not precede start")
        return [
            dict(row) for row in rows if start <= row.get(time_field, 0) <= end and field in row
        ]

    def distribution(self, rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
        values = [row[field] for row in rows if row.get(field) is not None]
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }

    def trajectory(
        self, rows: list[dict[str, Any]], group_by: str, field: str, *, time_field: str = "time"
    ) -> dict[Any, list[Any]]:
        grouped: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get(field) is not None:
                grouped.setdefault(row.get(group_by), []).append(row)
        return {
            key: [row[field] for row in sorted(group, key=lambda item: item.get(time_field, 0))]
            for key, group in grouped.items()
        }


def duckdb_aggregate(
    rows: list[dict[str, Any]], *, select: str, op: str, group_by: str | None = None
) -> list[dict[str, Any]]:
    """Aggregate rows through DuckDB when available (AW-11), else exact Python rows."""

    def fallback() -> list[dict[str, Any]]:
        plan = OutcomePlan(
            id="duckdb",
            source="rows",
            select=select,
            aggregation=op,
            group_by=group_by,
        )
        return AnalysisEngine().evaluate(plan, {"rows": rows})

    try:
        import duckdb
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError:
        return fallback()
    connection = duckdb.connect()
    try:
        connection.register("_genesis_rows", pa.Table.from_pylist(rows))
        expression = {"count": "count", "sum": "sum", "mean": "avg"}[op]
        quoted_select = f'"{select}"'
        if group_by:
            result = connection.sql(
                f'SELECT "{group_by}" AS g, {expression}({quoted_select}) AS v, '
                f"count(*) AS total, count({quoted_select}) AS non_null "
                "FROM _genesis_rows GROUP BY g ORDER BY g"
            ).fetchall()
            rows = []
            for key, value, total, non_null in result:
                row: dict[str, Any] = {group_by: str(key), f"{select}_{op}": value}
                row[f"{select}_missing"] = int(total) - int(non_null)
                rows.append(row)
            return rows
        value = connection.sql(
            f"SELECT {expression}({quoted_select}) AS v, count(*) AS total, "
            f"count({quoted_select}) AS non_null FROM _genesis_rows"
        ).fetchone()
        if value is None:
            return [{f"{select}_{op}": None, f"{select}_missing": 0}]
        return [{f"{select}_{op}": value[0], f"{select}_missing": int(value[1]) - int(value[2])}]
    finally:
        connection.close()


class AnalysisExporter:
    def export_json(self, rows: list[dict[str, Any]], path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n")
        return target

    def export_csv(self, rows: list[dict[str, Any]], path: str | Path) -> Path:
        target = Path(path)
        fields = sorted({key for row in rows for key in row})
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return target

    @staticmethod
    def rows_to_parquet(rows: list[dict[str, Any]], path: str | Path) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        target = Path(path)
        pq.write_table(pa.Table.from_pylist(rows), target)
        return target

    def export_parquet(self, rows: list[dict[str, Any]], path: str | Path) -> Path:
        return self.rows_to_parquet(rows, path)

    def export_bundle(
        self,
        rows: list[dict[str, Any]],
        directory: str | Path,
        *,
        methods: dict[str, Any] | None = None,
        replay_lineage: dict[str, Any] | None = None,
    ) -> list[Path]:
        """Write portable outcome evidence with a small integrity manifest."""
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        json_path = self.export_json(rows, root / "outcomes.json")
        csv_path = self.export_csv(rows, root / "outcomes.csv")
        parquet_path = self.export_parquet(rows, root / "outcomes.parquet")
        fields = sorted({key for row in rows for key in row})
        dictionary_path = root / "data_dictionary.json"
        dictionary_path.write_text(
            json.dumps(
                {
                    field: {
                        "observed_type": type(
                            next((row[field] for row in rows if field in row), None)
                        ).__name__
                    }
                    for field in fields
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        methods_path = root / "methods.json"
        methods_path.write_text(json.dumps(methods or {}, indent=2, sort_keys=True) + "\n")
        lineage_path = root / "replay_lineage.json"
        lineage_path.write_text(json.dumps(replay_lineage or {}, indent=2, sort_keys=True) + "\n")
        outputs = [
            json_path,
            csv_path,
            parquet_path,
            dictionary_path,
            methods_path,
            lineage_path,
        ]
        integrity_path = root / "integrity.json"
        integrity_path.write_text(
            json.dumps(
                {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return [*outputs, integrity_path]

    @staticmethod
    def verify_bundle(directory: str | Path) -> bool:
        root = Path(directory)
        try:
            manifest = json.loads((root / "integrity.json").read_text())
            if not manifest:
                return False
            return all(
                (root / name).is_file()
                and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                for name, digest in manifest.items()
            )
        except (OSError, json.JSONDecodeError):
            return False

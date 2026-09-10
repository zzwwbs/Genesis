"""Small declarative outcome evaluator and portable export helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
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


class _GroupState:
    """Per-group accumulator state. Holds summaries, never the rows."""

    __slots__ = (
        "count",
        "values",
        "total",
        "numeric",
        "low",
        "high",
        "series",
        "non_numeric",
        "exact",
        "float_seen",
        "nonfinite",
        "nonfinite_seen",
    )

    def __init__(self) -> None:
        self.count = 0  # rows in the group, including missing
        self.values = 0  # rows whose selected field is present
        self.total: Any = 0  # running left-to-right sum, as builtin sum() gives
        self.numeric = 0  # count of numeric values
        self.low: Any = None
        self.high: Any = None
        self.series: list[tuple[Any, Any]] = []  # trajectory only: (order key, value)
        self.non_numeric = False
        self.exact = Fraction(0)  # exact sum of finite numeric values, for means
        self.float_seen = False
        self.nonfinite = 0.0
        self.nonfinite_seen = False

    def mean(self, count: int) -> Any:
        """The mean of ``count`` values, as ``statistics.mean`` reports it.

        statistics.mean sums exactly, rounds once, and returns an int when
        every value is an integer and the mean is whole. Running float division
        reproduces neither: integer inputs came back as 3.0 instead of 3, and
        about a fifth of float means differed in the last bit -- enough to
        change exported outcome files and their digests.
        """
        if count <= 0:
            return None
        if self.nonfinite_seen:
            # statistics.mean's total is the non-finite part alone in this case.
            return self.nonfinite / count
        quotient = self.exact / count
        if self.float_seen or quotient.denominator != 1:
            return float(quotient)
        return int(quotient)


class _Accumulator:
    """One outcome's state, fed a row at a time (OUT-001..OUT-003).

    The engine used to materialise the whole source, group it into lists, then
    reduce. Nothing it computes needs the rows: an aggregate needs three
    counters per group, and a distribution needs five. Only a trajectory retains
    values, and those values are its own output.
    """

    def __init__(self, plan: OutcomePlan) -> None:
        self.plan = plan
        self.keys = (
            (plan.group_by,) if isinstance(plan.group_by, str) else tuple(plan.group_by or ())
        )
        # Insertion-ordered, so groups are emitted in first-appearance order —
        # the order the materialising engine produced, which reaches exported
        # files and their integrity digests.
        self.groups: dict[Any, _GroupState] = {}
        # Exact summation costs more than a running total; only a mean needs it.
        self.tracks_mean = plan.operation == "distribution" or (
            plan.operation == "aggregate" and plan.aggregation == "mean"
        )

    def _admits(self, row: Mapping[str, Any]) -> bool:
        window = self.plan.window
        if window:
            time_field = str(window.get("time_field", "time"))
            if time_field not in row:
                return False
            moment = row[time_field]
            start, end = window.get("start"), window.get("end")
            if start is not None and moment < start:
                return False
            if end is not None and moment > end:
                return False
        return all(
            all(row.get(key) == value for key, value in predicate.items())
            for predicate in self.plan.filters
        )

    def add(self, row: Mapping[str, Any]) -> None:
        if not self._admits(row):
            return
        key = tuple(row.get(name) for name in self.keys) if self.keys else None
        state = self.groups.get(key)
        if state is None:
            state = self.groups[key] = _GroupState()
        state.count += 1
        value = row.get(self.plan.select)
        if value is None:
            return
        state.values += 1
        if isinstance(value, int | float):
            state.total += value
            state.numeric += 1
            if self.tracks_mean:
                if isinstance(value, float) and not math.isfinite(value):
                    state.nonfinite += value
                    state.nonfinite_seen = True
                    state.float_seen = True
                else:
                    state.exact += Fraction(value)
                    if isinstance(value, float):
                        state.float_seen = True
        else:
            state.non_numeric = True
        # Only a distribution compares values. An aggregate never did, and its
        # selected field may hold values that do not order at all (a dict).
        if self.plan.operation == "distribution":
            if state.low is None or value < state.low:
                state.low = value
            if state.high is None or value > state.high:
                state.high = value
        elif self.plan.operation == "trajectory":
            window = self.plan.window
            field = str(window.get("time_field", "time")) if window else "time"
            state.series.append((row.get(field, 0), value))

    def finish(self) -> list[dict[str, Any]]:
        if not self.groups and not self.keys:
            self.groups[None] = _GroupState()
        if self.plan.operation == "trajectory":
            return [self._trajectory_row(key, st) for key, st in self.groups.items()]
        if self.plan.operation == "distribution":
            return [self._distribution_row(key, st) for key, st in self.groups.items()]
        return [self._aggregate_row(key, st) for key, st in self.groups.items()]

    def _labelled(self, key: Any) -> dict[str, Any]:
        return AnalysisEngine._group_result(self.plan, key)

    def _aggregate_row(self, key: Any, state: _GroupState) -> dict[str, Any]:
        plan = self.plan
        missing = state.count - state.values
        values, total = state.values, state.total
        # "zero" imputes a zero per missing row, so those rows join the sum and
        # the mean's denominator, but never the count.
        if plan.missingness == "zero" and missing and plan.aggregation in {"sum", "mean"}:
            values += missing
        if plan.aggregation in {"sum", "mean"} and state.non_numeric:
            raise ValueError(f"{plan.aggregation} requires numeric values: {plan.select}")
        if plan.aggregation == "mean":
            value: Any = state.mean(values)
            name = f"{plan.select}_mean"
        elif plan.aggregation == "sum":
            value, name = total, f"{plan.select}_sum"
        elif plan.aggregation == "count":
            value, name = state.values, f"{plan.select}_count"
        else:
            raise ValueError(f"unsupported aggregation: {plan.aggregation}")
        result = self._labelled(key)
        result[name] = value
        result[f"{plan.select}_missing"] = missing
        return result

    def _distribution_row(self, key: Any, state: _GroupState) -> dict[str, Any]:
        plan = self.plan
        result = self._labelled(key)
        if not state.values:
            result.update(
                {
                    f"{plan.select}_count": 0,
                    f"{plan.select}_min": None,
                    f"{plan.select}_max": None,
                    f"{plan.select}_mean": None,
                }
            )
        else:
            result.update(
                {
                    f"{plan.select}_count": state.values,
                    f"{plan.select}_min": state.low,
                    f"{plan.select}_max": state.high,
                    f"{plan.select}_mean": state.mean(state.numeric),
                }
            )
        result[f"{plan.select}_missing"] = state.count - state.values
        return result

    def _trajectory_row(self, key: Any, state: _GroupState) -> dict[str, Any]:
        plan = self.plan
        # Stable, as the materialising engine's sort was: ties keep arrival order.
        ordered = sorted(state.series, key=lambda item: item[0])
        trajectory = [value for _moment, value in ordered]
        result = self._labelled(key)
        result[f"{plan.select}_trajectory"] = trajectory
        result[f"{plan.select}_missing"] = state.count - len(trajectory)
        return result


def _iter_source(sources: Mapping[str, Any], name: str) -> Iterable[Mapping[str, Any]]:
    """A fresh walk of one relation.

    A source may be a sequence or a callable returning an iterator, so a
    relation can be read more than once without being held (OUT-001).
    """
    source = sources.get(name)
    if source is None:
        return ()
    rows: Iterable[Mapping[str, Any]] = source() if callable(source) else source
    return rows


class AnalysisEngine:
    def evaluate(
        self, plan: OutcomePlan, sources: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        return self.evaluate_many([plan], sources)[0]

    def evaluate_many(
        self, plans: Sequence[OutcomePlan], sources: Mapping[str, Any]
    ) -> list[list[dict[str, Any]]]:
        """Evaluate several outcomes with one walk per distinct relation (OUT-002).

        Every outcome of the illustrative study reads ``events``; evaluating them
        one at a time walked that relation once per outcome and held it
        throughout.
        """
        accumulators = [_Accumulator(plan) for plan in plans]
        by_source: dict[str, list[_Accumulator]] = {}
        for accumulator in accumulators:
            by_source.setdefault(accumulator.plan.source, []).append(accumulator)
        for name, group in by_source.items():
            for row in _iter_source(sources, name):
                for accumulator in group:
                    accumulator.add(row)
        return [accumulator.finish() for accumulator in accumulators]

    @staticmethod
    def _group_result(plan: OutcomePlan, key: Any) -> dict[str, Any]:
        """Label one aggregated group with its declared grouping keys.

        Multi-key groupings also carry the declared field names, not only
        positional ``group_N`` labels: without them a downstream consumer (the
        protocol summary, an exported outcome row) cannot tell which condition
        or phase a value belongs to, and pooling across them is undetectable.
        The positional labels are retained for compatibility.
        """
        keys = (plan.group_by,) if isinstance(plan.group_by, str) else (plan.group_by or ())
        if not keys:
            return {}
        if len(keys) == 1:
            return {keys[0]: key if not isinstance(key, tuple) else key[0]}
        values = key if isinstance(key, tuple) else (key,)
        result: dict[str, Any] = {f"group_{i}": item for i, item in enumerate(values)}
        for name, item in zip(keys, values, strict=False):
            result.setdefault(str(name), item)
        return result

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


def _duckdb_is_exact(
    rows: Iterable[Mapping[str, Any]], *, select: str, op: str, group_by: str | None
) -> bool:
    """Whether DuckDB produces the in-process engine's bytes for this call.

    Measured, not assumed. AVG differs from statistics.mean in type and in
    rounding; a float SUM rounds differently from a left-to-right sum; and a
    group key mixing int and float is coerced to DOUBLE, changing both the key's
    type and its order. Counts, integer sums and uniformly typed keys agree.
    """
    rows = list(rows)
    if op == "count":
        exact = True
    elif op == "sum":
        exact = all(type(row.get(select)) is int for row in rows if row.get(select) is not None)
    else:
        exact = False
    if exact and group_by:
        key_types = {type(row.get(group_by)) for row in rows if row.get(group_by) is not None}
        exact = len(key_types) <= 1
    return exact


def _order_key(value: Any) -> Any:
    """A hashable stand-in for a grouping key, for ordering only.

    Keys come back from the database as their own types; ``None`` and
    unhashable values still need a stable slot.
    """
    try:
        hash(value)
    except TypeError:
        return ("unhashable", repr(value))
    return (type(value).__name__, value)


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

    if not _duckdb_is_exact(rows, select=select, op=op, group_by=group_by):
        return fallback()
    try:
        import duckdb
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError:
        return fallback()
    connection = duckdb.connect()

    def _normalize(value: Any) -> Any:
        """Match the in-process engine's empty-aggregate value.

        SQL ``sum`` over an all-NULL group is NULL; Python's ``sum([])`` is 0.
        Left as-is, enabling DuckDB changed a reported outcome from 0 to None.
        ``avg`` and Python's mean both yield None for an empty group.
        """
        return 0 if op == "sum" and value is None else value

    try:
        connection.register("_genesis_rows", pa.Table.from_pylist(rows))
        expression = {"count": "count", "sum": "sum", "mean": "avg"}[op]
        quoted_select = f'"{select}"'
        if group_by:
            result = connection.sql(
                f'SELECT "{group_by}" AS g, {expression}({quoted_select}) AS v, '
                f"count(*) AS total, count({quoted_select}) AS non_null "
                "FROM _genesis_rows GROUP BY g"
            ).fetchall()
            # Emit groups in first-appearance order, as the in-process engine
            # does. Ordering by the key instead made the same declared plan
            # export outcome files -- and therefore bundle integrity digests --
            # that depended on which engine happened to run.
            appearance: dict[Any, int] = {}
            for row_index, source_row in enumerate(rows):
                appearance.setdefault(_order_key(source_row.get(group_by)), row_index)
            result.sort(key=lambda item: appearance.get(_order_key(item[0]), len(rows)))
            rows = []
            for key, value, total, non_null in result:
                # Keep the grouping key's native type. Stringifying it made the
                # same declared plan produce `phase: "1"` under DuckDB and
                # `phase: 1` in-process, so the recorded evidence depended on
                # which engine happened to run.
                row: dict[str, Any] = {group_by: key, f"{select}_{op}": _normalize(value)}
                row[f"{select}_missing"] = int(total) - int(non_null)
                rows.append(row)
            return rows
        value = connection.sql(
            f"SELECT {expression}({quoted_select}) AS v, count(*) AS total, "
            f"count({quoted_select}) AS non_null FROM _genesis_rows"
        ).fetchone()
        if value is None:
            return [{f"{select}_{op}": None, f"{select}_missing": 0}]
        return [
            {
                f"{select}_{op}": _normalize(value[0]),
                f"{select}_missing": int(value[1]) - int(value[2]),
            }
        ]
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

    @staticmethod
    def stream_rows_to_parquet(
        rows: Iterable[dict[str, Any]] | Callable[[], Iterable[dict[str, Any]]],
        path: str | Path,
        *,
        batch_rows: int = 512,
    ) -> Path:
        """Write rows to parquet in batches, holding one batch at a time.

        Two passes. The first infers every batch's column types over all of its
        values and unifies them, widening where types are compatible; the
        second writes each batch against that one schema. Fixing the schema
        from the first batch instead silently truncated a column that was
        integral for 512 rows and fractional afterwards, and refused -- or
        crashed on -- a column that appeared, or first held a value, later.

        ``rows`` may be a factory returning a fresh iterator, which is what
        keeps a long history off the heap across both passes. A plain iterable
        is buffered, because it cannot be walked twice.

        Written to a sibling temporary file and moved into place only on
        success, so a refused write leaves nothing behind.
        """
        import os
        import tempfile

        import pyarrow as pa
        import pyarrow.parquet as pq

        target = Path(path)
        size = max(1, int(batch_rows))
        if callable(rows):
            factory = rows
        else:
            buffered = list(rows)

            def factory() -> Iterable[dict[str, Any]]:
                return iter(buffered)

        def batches() -> Iterator[list[dict[str, Any]]]:
            batch: list[dict[str, Any]] = []
            for row in factory():
                batch.append(row)
                if len(batch) >= size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def columns(batch: list[dict[str, Any]], names: Iterable[str]) -> dict[str, list[Any]]:
            return {name: [row.get(name) for row in batch] for name in names}

        def refused(exc: Exception) -> ValueError:
            return ValueError(
                f"PARQUET_SCHEMA: {target.name} holds values with incompatible types: {exc}"
            )

        schema: Any = None
        for batch in batches():
            names = list(dict.fromkeys(key for row in batch for key in row))
            try:
                inferred = pa.Table.from_pydict(columns(batch, names)).schema
                schema = (
                    inferred
                    if schema is None
                    else pa.unify_schemas([schema, inferred], promote_options="permissive")
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
                raise refused(exc) from exc
        if schema is None:
            schema = pa.schema([])

        handle, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(handle)
        writer: Any = None
        try:
            writer = pq.ParquetWriter(temporary, schema)
            for batch in batches():
                try:
                    table = pa.Table.from_pydict(columns(batch, schema.names), schema=schema)
                except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError) as exc:
                    raise refused(exc) from exc
                writer.write_table(table)
            writer.close()
            writer = None
            os.replace(temporary, target)
        except BaseException:
            if writer is not None:
                writer.close()
            Path(temporary).unlink(missing_ok=True)
            raise
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

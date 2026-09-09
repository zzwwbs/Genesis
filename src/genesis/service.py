"""Durable local application service shared by HTTP and command-line adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import statistics
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

import yaml  # type: ignore[import-untyped]
from pydantic import Field, ValidationError, model_validator

from genesis import __version__
from genesis.analysis import AnalysisEngine, AnalysisExporter, OutcomePlan
from genesis.assistant import StudyAssistant
from genesis.checklists import CHECKLIST_ITEMS, checklist_record, persist_checklist
from genesis.compiler import StudyCompiler
from genesis.elicitation import (
    ElicitationAssistant,
    ElicitationEngine,
    ElicitationSessionStore,
    PatchPreviewService,
    WorkflowRegistry,
    completion_checks,
    require_completion_rules,
)
from genesis.elicitation import (
    SpecificationPatch as ElicitationSpecificationPatch,
)
from genesis.evidence import (
    ExportMode,
    _is_contained_relative,
    evaluate_capabilities,
    publish_bundle,
    stage_bundle,
    verify_bundle_manifest_and_size,
    write_bundle_manifest,
)
from genesis.execution_manifest import (
    canonical_json,
    resolve_execution_manifest,
    scientific_config_digest,
)
from genesis.extensions import ExtensionManifest, ExtensionRegistry
from genesis.outcome_plan import (
    _INCOMPLETE_ROUND,
    compile_outcome_plan,
    materialize_datasets,
    outcome_plan_digest,
)
from genesis.persistence import ObjectRef, PersistenceCoordinator
from genesis.providers import (
    OpenAICompatibleProvider,
    ProviderExecutor,
    ProviderRequest,
)
from genesis.replay import ReplayMode
from genesis.runtime import (
    STATE_VALUE_TYPES,
    ArtifactStore,
    CallableExecutor,
    ContextEngine,
    DeterministicExecutor,
    ExecutorRegistry,
    ProcessInvocation,
    RecordedArtifactExecutor,
    RuleExecutor,
    RunController,
    Scheduler,
    StateStore,
    StateTransitionExecutor,
    StochasticExecutor,
    derive_seed,
    expand_protocol_conditions,
)
from genesis.schema_validation import PackageSchemaCatalog, SchemaDiagnostic, SchemaValidationError
from genesis.specification.models import StrictModel
from genesis.tracing import DEFAULT_DEPTH, DEFAULT_MAX_STEPS, build_chain, resolve_seed

# One declared vocabulary, shared with the compiler's validation.
_STATE_TYPES = STATE_VALUE_TYPES


class _ElicitationOutputFailure(ValueError):
    """Typed failure carrying safe diagnostics for bounded model-output repair."""

    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = [dict(item) for item in attempts]


class SpecificationPatchOperation(StrictModel):
    """One deterministic, typed patch operation (AW-02).

    Paths address specification or checklist sections; only ``set`` operations
    are supported so every patch step is a full-value assignment.
    """

    path: str
    op: Literal["set"]
    value: Any = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class SpecificationPatch(StrictModel):
    """Typed assistant-change envelope with evidence and review metadata."""

    operations: list[SpecificationPatchOperation]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    affected_checklist_items: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_shape(self) -> SpecificationPatch:
        if not self.operations:
            raise ValueError("PATCH_EMPTY: a patch must carry at least one operation")
        for operation in self.operations:
            if not operation.path.startswith("specification") and not operation.path.startswith(
                "checklist"
            ):
                raise ValueError(f"PATCH_PATH: unsupported patch path '{operation.path}'")
        return self

    def validate_deterministic(self) -> None:
        """Deterministic validation of the envelope shape (AW-02)."""
        if not self.operations:
            raise ValueError("PATCH_EMPTY: a patch must carry at least one operation")
        for operation in self.operations:
            if not operation.path.startswith("specification") and not operation.path.startswith(
                "checklist"
            ):
                raise ValueError(f"PATCH_PATH: unsupported patch path '{operation.path}'")


_FORM_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "title",
        "description",
        "owners",
        "source_citations",
        "artifact_refs",
        "package_compatibility",
        "origin",
        "approval",
        "extensions",
        "processes",
        "models",
        "artifacts",
        "prompts",
        "schemas",
        "research_question",
        "theory",
        "domain",
        "protocol",
        "theory_family",
        "constructs",
        "process_mappings",
        "relations",
        "feedback",
        "delays",
        "observables",
        "actors",
        "attributes",
        "states",
        "mechanisms",
        "institutions",
        "initialization",
        "visibility",
        "availability",
        "updates",
        "persistence",
        "time_model",
        "termination",
        "conditions",
        "factors",
        "phases",
        "replications",
        "matching",
        "random_streams",
        "model_freezing",
        "budgets",
        "checkpoints",
        "replay_retention",
        "outcomes",
        "datasets",
    }
)

_THEORY_BLOCK_FIELDS = frozenset(
    {
        "theory_family",
        "constructs",
        "process_mappings",
        "relations",
        "feedback",
        "delays",
        "observables",
    }
)
_DOMAIN_BLOCK_FIELDS = frozenset(
    {
        "actors",
        "attributes",
        "states",
        "artifacts",
        "mechanisms",
        "institutions",
        "initialization",
        "visibility",
        "availability",
        "updates",
        "persistence",
    }
)
_PROTOCOL_BLOCK_FIELDS = frozenset(
    {
        "time_model",
        "termination",
        "conditions",
        "factors",
        "phases",
        "replications",
        "matching",
        "random_streams",
        "model_freezing",
        "budgets",
        "checkpoints",
        "replay_retention",
    }
)


def _parquet_safe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project rows to parquet-writable scalars (nested values as JSON strings)."""
    safe = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                item[key] = json.dumps(value, default=str, sort_keys=True)
            else:
                item[key] = value
        safe.append(item)
    return safe


def _load_empirical_data(path: Path, data_ref: str) -> dict[str, Any]:
    """Load an empirical data asset into an origin-tagged initial-state envelope (AW-06)."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    elif suffix == ".json":
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = [rows]
    elif suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            table = pq.read_table(path)
            rows = table.to_pylist()
        except ImportError as exc:  # pragma: no cover - installation error path
            raise RuntimeError("PyArrow is required for Parquet data assets") from exc
    else:
        raise ValueError(f"DATA_SOURCE_INVALID: unsupported data format {suffix}")
    return {"origin": "imported", "data_source": data_ref, "rows": list(rows)}


def _apply_field_operations(
    payload: dict[str, Any], operations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply field-level ``specification`` operations to a form payload (AW-02).

    Paths are dot segments with ``[n]`` list indices, e.g.
    ``specification.processes[0].executor.mode``. Only existing targets can be
    set; unknown segments raise ``PATCH_PATH`` so application is deterministic.
    """
    result: dict[str, Any] = json.loads(json.dumps(payload))
    for operation in operations:
        path = str(operation.get("path", ""))
        if not path.startswith("specification"):
            raise ValueError(f"PATCH_PATH: unsupported patch path '{path}'")
        if str(operation.get("op")) != "set":
            raise ValueError(f"PATCH_OP: unsupported operation '{operation.get('op')}'")
        value = operation.get("value")
        segments = path.split(".")[1:]
        node: Any = result
        for index, segment in enumerate(segments):
            is_last = index == len(segments) - 1
            if "[" in segment and segment.endswith("]"):
                name, raw_index = segment[:-1].split("[", 1)
                if not raw_index.isdigit():
                    raise ValueError(f"PATCH_PATH: invalid list index in '{segment}'")
                container = node.get(name)
                if not isinstance(container, list) or int(raw_index) >= len(container):
                    raise ValueError(f"PATCH_PATH: unknown target '{path}'")
                if is_last:
                    container[int(raw_index)] = value
                else:
                    node = container[int(raw_index)]
            else:
                if is_last:
                    if not isinstance(node, dict):
                        raise ValueError(f"PATCH_PATH: unknown target '{path}'")
                    node[segment] = value
                else:
                    if not isinstance(node, dict) or segment not in node:
                        raise ValueError(f"PATCH_PATH: unknown target '{path}'")
                    node = node[segment]
    return result


def _diff_form_fields(current: dict[str, Any], proposal: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive field-level patch operations from a form proposal (AW-02)."""
    operations = []
    for top_level in ("processes", "theory", "domain", "protocol", "outcomes", "prompts"):
        if proposal.get(top_level) != current.get(top_level):
            operations.append(
                {
                    "path": f"specification.{top_level}",
                    "op": "set",
                    "value": proposal.get(top_level),
                }
            )
    return operations


def _dispatch_bounded(
    pool: Any, worker: Any, trials: list[str], max_in_flight: int
) -> list[tuple[bool, str]]:
    """Bounded result-queue dispatch (AW-13).

    At most ``max_in_flight`` futures are in flight at once; results arrive on
    the queue and are collected in submission order, giving both backpressure
    and a deterministic result order. Each result is (ok, diagnostic) so a
    failure is reported rather than reduced to a bare False.
    """
    from concurrent.futures import FIRST_COMPLETED, Future, wait

    pending: dict[Future[tuple[bool, str]], int] = {}
    results: list[tuple[int, tuple[bool, str]]] = []
    next_index = 0
    while next_index < len(trials) or pending:
        while next_index < len(trials) and len(pending) < max(1, int(max_in_flight)):
            pending[pool.submit(worker, trials[next_index])] = next_index
            next_index += 1
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            error = future.exception()
            if error is not None:
                results.append((index, (False, f"{type(error).__name__}: {error}")))
            else:
                outcome = future.result()
                results.append(
                    (index, outcome if isinstance(outcome, tuple) else (bool(outcome), ""))
                )
    results.sort(key=lambda item: item[0])
    return [outcome for _, outcome in results]


def _workflows_root() -> Path:
    """Resolve the workflow-package root (env override, repo, or workspace)."""
    configured = os.environ.get("GENESIS_WORKFLOWS")
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "workflows").is_dir():
        return repo_root / "workflows"
    return Path("workflows")


def _workflow_schema(name: str) -> dict[str, Any]:
    import json as _json

    path = _workflows_root() / "three-layer-study" / "schemas" / name
    if not path.is_file():
        return {}
    try:
        payload = _json.loads(path.read_text())
    except (OSError, _json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_trial_worker(workspace: str, trial_id: str) -> tuple[bool, str]:
    """Process-pool worker: opens its own coordinator and executes one trial.

    A fresh GenesisService per worker keeps SQLite connections process-local;
    the shared database is serialized by WAL plus busy timeout (AW-13).
    Returns (ok, diagnostic) so a failed realization reports WHY rather than
    disappearing into a bare "partial" protocol status.
    """
    service = GenesisService(workspace)
    try:
        service.execute_run(trial_id)
        status = str(service.get_run(trial_id)["status"])
        return (status == "completed", "" if status == "completed" else f"status={status}")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        service.close()


def _resolve_callable(reference: str | None) -> Any:
    """Resolve a declared ``module:attribute`` function reference (AW-07)."""
    if not reference or ":" not in reference:
        raise ValueError("EXECUTOR_UNAVAILABLE: function reference must use module:attribute form")
    module_name, _, attribute = reference.partition(":")
    try:
        import importlib

        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"EXECUTOR_UNAVAILABLE: cannot resolve function reference '{reference}'"
        ) from exc


def _parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object embedded in a model response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("PROPOSAL_INVALID: model response contains no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"PROPOSAL_INVALID: response JSON is malformed: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("PROPOSAL_INVALID: response JSON must be an object")
    return value


def _validated_block(payload: dict[str, Any], name: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Return the nested authoring block for one artifact, rejecting unknown keys."""
    block = payload.get(name)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"INVALID_FIELD: {name} must be an object")
    unsupported = sorted(set(block) - allowed)
    if unsupported:
        raise ValueError(f"INVALID_FIELD: unsupported {name} field(s): {', '.join(unsupported)}")
    return block


def _check_profile_drift(
    profile_id: str,
    configured: Mapping[str, Any],
    compiled: Mapping[str, Any],
    build_hash: str,
) -> None:
    """Fail loudly when the runtime profile drifts from the compiled package (AW-10).

    Compares provider, model, endpoint (base URL/timeout), and the scientific
    parameters that can affect execution (review finding 2).
    """
    mismatches = []
    if str(configured.get("provider", "")) != str(compiled.get("provider", "")):
        mismatches.append("provider")
    if str(configured.get("model", "")) != str(compiled.get("model", "")):
        mismatches.append("model")
    compiled_endpoint = str(compiled.get("endpoint_ref", "") or "")
    runtime_base = str(configured.get("base_url", "") or "")
    if compiled_endpoint and runtime_base and compiled_endpoint != runtime_base:
        mismatches.append("endpoint")
    if dict(configured.get("parameters", {})) != dict(compiled.get("parameters", {})):
        mismatches.append("parameters")
    if mismatches:
        raise ValueError(
            f"MODEL_PROFILE_DRIFT: runtime profile '{profile_id}' differs from the "
            f"compiled package on: {', '.join(mismatches)} (build {build_hash[:12]})"
        )


def _replayed_result(record: Mapping[str, Any], *, source_run_id: str, process_id: str) -> Any:
    """Rebuild a recorded invocation's full committed result, not only outputs."""
    from genesis.runtime import ProcessResult

    return ProcessResult(
        outputs=dict(record.get("outputs") or {}),
        state_effects=dict(record.get("state_effects") or {}),
        events=tuple(record.get("events") or ()),
        scheduling_effects=tuple(record.get("scheduling_effects") or ()),
        metadata={
            "recorded": True,
            "source_run_id": source_run_id,
            "process_id": process_id,
        },
    )


class _SelectiveExecutor:
    """Partial-replay dispatcher: frozen invocations replay recorded outputs,
    everything else executes through the live fallback executor (finding 5).

    The frozen prefix is a research guarantee, so a divergence inside it is an
    error rather than a silent live invocation: an invocation inside the frozen
    prefix with no matching recording means the replayed trajectory no longer
    follows the recorded one, and continuing would generate new content where
    the researcher expects fixed evidence.
    """

    def __init__(
        self,
        records: list[Any],
        *,
        frozen_keys: set[tuple[Any, ...]],
        fallback: Any,
        source_run_id: str,
        process_id: str,
        phase_boundary: int | float | None = None,
        inclusive: bool = False,
        known_keys: set[tuple[Any, ...]] | None = None,
    ):
        self._records = list(records)
        self._frozen_keys = frozen_keys
        self._fallback = fallback
        self.source_run_id = source_run_id
        self.process_id = process_id
        self._phase_boundary = phase_boundary
        # An event boundary sits INSIDE a phase, so the whole boundary phase
        # counts as prefix for an invocation with no source counterpart.
        self._inclusive = inclusive
        # Every invocation of this process the SOURCE performed, frozen or not.
        # An invocation the source made after the boundary is a legitimate
        # suffix invocation and re-executes live; one the source never made is
        # a divergence when it lands inside the prefix.
        self._known_keys = set(known_keys or frozen_keys)

    @staticmethod
    def _key(invocation: ProcessInvocation) -> tuple[Any, ...]:
        actors = tuple(invocation.actor_ids) if invocation.actor_ids else ()
        return (invocation.phase, invocation.attempt, actors)

    def _diverged(self, invocation: ProcessInvocation, key: tuple[Any, ...]) -> ValueError:
        return ValueError(
            "REPLAY_PREFIX_DIVERGED: the frozen prefix of run "
            f"'{self.source_run_id}' has no recorded invocation of "
            f"'{self.process_id}' at phase={invocation.phase} "
            f"attempt={invocation.attempt} actors={list(key[2])}; "
            "the replayed trajectory diverged from the recorded one before the "
            "boundary, so the prefix cannot be held fixed"
        )

    def execute(self, invocation: ProcessInvocation) -> Any:
        key = self._key(invocation)
        # A phase boundary freezes every invocation at phase < N (see
        # _frozen_invocation_keys); phase >= N is the re-executed suffix. An
        # event boundary cannot order a never-recorded invocation within its own
        # phase, so that phase is prefix too.
        within_prefix = self._phase_boundary is not None and (
            invocation.phase <= self._phase_boundary
            if self._inclusive
            else invocation.phase < self._phase_boundary
        )
        if key in self._frozen_keys:
            # Actor-parity matching, equivalent to _RecordedExecutor: prefer the
            # record with identical actors, then any record for the phase.
            for record in self._records:
                if (
                    record.get("phase") == invocation.phase
                    and record.get("attempt") == invocation.attempt
                    and (record.get("actors") == key[2] or not key[2])
                ):
                    return _replayed_result(
                        record,
                        source_run_id=self.source_run_id,
                        process_id=self.process_id,
                    )
            # Only actor-less (legacy) recordings may match on phase alone.
            for record in self._records:
                if (
                    record.get("phase") == invocation.phase
                    and record.get("attempt") == invocation.attempt
                    and not record.get("actors")
                ):
                    return _replayed_result(
                        record,
                        source_run_id=self.source_run_id,
                        process_id=self.process_id,
                    )
            # The key was declared frozen but carries no usable record.
            raise self._diverged(invocation, key)
        # An invocation the source also performed (after the boundary) is a
        # legitimate suffix invocation even inside the boundary phase.
        if within_prefix and key not in self._known_keys:
            raise self._diverged(invocation, key)
        if self._fallback is None:
            raise RuntimeError(
                f"REPLAY_EXECUTION: no live executor for unfrozen process '{self.process_id}'"
            )
        return self._fallback.execute(invocation)


class _FrozenPrefixGuard:
    """Reject generation inside the frozen prefix for an unrecorded process.

    A process the source run never performed (or performed only after the
    boundary) receives no recorded substitute, so nothing otherwise stopped its
    live executor from being invoked inside the prefix — a branch that changes
    a condition can make a previously idle generative process fire there. That
    would produce new content in the stretch of trajectory the researcher
    declared fixed, so it is an error; outside the prefix the live executor
    runs normally.
    """

    def __init__(
        self,
        fallback: Any,
        *,
        source_run_id: str,
        process_id: str,
        phase_boundary: int | float,
        inclusive: bool = False,
    ):
        self._fallback = fallback
        self.source_run_id = source_run_id
        self.process_id = process_id
        self._phase_boundary = phase_boundary
        # An event boundary sits INSIDE a phase. A never-recorded invocation
        # cannot be ordered against that event, so the whole boundary phase is
        # treated as prefix: refusing a suffix invocation is recoverable (pick a
        # phase boundary), generating inside a frozen prefix is not.
        self._inclusive = inclusive

    def execute(self, invocation: ProcessInvocation) -> Any:
        inside = (
            invocation.phase <= self._phase_boundary
            if self._inclusive
            else invocation.phase < self._phase_boundary
        )
        if inside:
            raise ValueError(
                "REPLAY_PREFIX_DIVERGED: generative process "
                f"'{self.process_id}' would run at phase={invocation.phase}, "
                f"inside the frozen prefix of run '{self.source_run_id}', but the "
                "source run recorded no invocation there; the prefix cannot be "
                "held fixed"
                + (
                    ". An event boundary cannot order an unrecorded invocation "
                    "within its own phase, so the whole phase is treated as "
                    "prefix; use a phase boundary to re-execute inside it."
                    if self._inclusive
                    else ""
                )
            )
        if self._fallback is None:
            raise RuntimeError(
                f"REPLAY_EXECUTION: no live executor for unfrozen process '{self.process_id}'"
            )
        return self._fallback.execute(invocation)


class _RecordedExecutor:
    """Replay substitution: returns recorded outputs without a provider call.

    Selects the recorded output whose invocation coordinates match the current
    invocation (phase, attempt, actors). An invocation the source run never
    performed has no faithful recording, so it raises rather than substituting
    an unrelated record: replay must not present another actor's or round's
    artifact as ``recorded``.
    """

    def __init__(
        self,
        records: list[Mapping[str, Any]],
        *,
        source_run_id: str,
        process_id: str,
    ):
        self._records = list(records)
        self.source_run_id = source_run_id
        self.process_id = process_id

    def _select(self, invocation: ProcessInvocation) -> Mapping[str, Any]:
        """The recorded invocation matching these coordinates (whole record)."""
        actors = tuple(invocation.actor_ids) if invocation.actor_ids else ()
        for record in self._records:
            if (
                record["phase"] == invocation.phase
                and record["attempt"] == invocation.attempt
                and (record["actors"] == actors or not actors)
            ):
                return record
        # Legacy recordings carry NO actor identity; only those may be matched
        # on phase and attempt alone. A recording that names a different actor
        # is another actor's evidence, not this invocation's.
        for record in self._records:
            if (
                record["phase"] == invocation.phase
                and record["attempt"] == invocation.attempt
                and not record["actors"]
            ):
                return record
        raise ValueError(
            "REPLAY_RECORD_MISSING: no recorded invocation of "
            f"'{self.process_id}' in run '{self.source_run_id}' matches "
            f"phase={invocation.phase} attempt={invocation.attempt} "
            f"actors={list(actors)}; the replayed trajectory diverged from the "
            "recorded one"
        )

    def execute(self, invocation: ProcessInvocation) -> Any:
        return _replayed_result(
            self._select(invocation),
            source_run_id=self.source_run_id,
            process_id=self.process_id,
        )


class GenesisService:
    """Own a contained local workspace and all durable GENESIS mutations."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.initialize()
        internal = self.workspace / ".genesis"
        self.persistence = PersistenceCoordinator(internal / "genesis.db", internal / "objects")
        self._extensions: ExtensionRegistry | None = None
        self._extension_store = internal / "extensions.json"
        self._load_extensions()
        self._workflow_registry = WorkflowRegistry(_workflows_root())
        self._elicitation_store = ElicitationSessionStore(internal / "elicitation-sessions.json")
        self._elicitation_engine = ElicitationEngine(
            self._workflow_registry, self._elicitation_store
        )
        self._elicitation_assistant = ElicitationAssistant(
            response_schema=_workflow_schema("assistant-evaluation.schema.json")
        )
        self._patch_preview = PatchPreviewService(self)
        self.last_outcome_engine = "python"

    def register_extension(
        self,
        extension_id: str,
        factory: Any,
        *,
        version: str = "1.0.0",
        genesis_range: str = ">=0.1.0",
        schema_range: str = ">=1.0",
        capabilities: tuple[str, ...] = ("extension",),
        entry_point: str = "inline:factory",
        enabled: bool = True,
    ) -> dict[str, Any]:
        """Register a trusted workspace extension (AW-07).

        The integrity hash covers the held module code, not just the declared
        entry-point name (finding 6), so implementation changes invalidate the
        hash and are rejected on reload.
        """
        if self._extensions is None:
            self._extensions = ExtensionRegistry(genesis_version=__version__, schema_version="1.0")
        integrity_material = entry_point.encode()
        hash_method = "entrypoint-sha256"
        if entry_point != "inline:factory":
            integrity_material = self._extension_integrity_material(entry_point)
            hash_method = "module-sha256"
        manifest = ExtensionManifest(
            id=extension_id,
            version=version,
            genesis_range=genesis_range,
            schema_range=schema_range,
            capabilities=capabilities,
            entry_point=entry_point,
            integrity_hash=hashlib.sha256(integrity_material).hexdigest(),
        )
        try:
            self._extensions.register(
                manifest, factory, enabled=enabled, integrity_material=integrity_material
            )
        except (PermissionError, ValueError) as exc:
            raise ValueError(f"EXTENSION_REGISTRATION: {exc}") from exc
        self._save_extensions()
        return {
            "id": extension_id,
            "version": version,
            "entry_point": entry_point,
            "integrity_hash": manifest.integrity_hash,
            "hash_method": hash_method,
        }

    @staticmethod
    def _extension_integrity_material(entry_point: str) -> bytes:
        """Digest of the module file implementing the entry point."""
        import inspect

        callable_ref = _resolve_callable(entry_point)
        module = inspect.getmodule(callable_ref)
        source_file = inspect.getsourcefile(callable_ref) or getattr(module, "__file__", None)
        if not source_file:
            raise ValueError(
                f"EXTENSION_REGISTRATION: cannot locate module code for '{entry_point}'"
            )
        return entry_point.encode() + b"\0" + Path(source_file).read_bytes()

    def _save_extensions(self) -> None:
        if self._extensions is None:
            return
        records = [
            {
                "id": manifest.id,
                "version": manifest.version,
                "entry_point": manifest.entry_point,
                "capabilities": list(manifest.capabilities),
                "integrity_hash": manifest.integrity_hash,
                "genesis_range": manifest.genesis_range,
                "schema_range": manifest.schema_range,
            }
            for manifest in self._extensions.manifests()
        ]
        self._extension_store.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")

    def _load_extensions(self) -> None:
        """Restore resolvable (module:attribute) extensions across restarts.

        Registration records are re-verified: the recorded integrity hash must
        still match the module code, and the recorded compatibility ranges are
        retained (finding 6).
        """
        if not self._extension_store.is_file():
            return
        try:
            records = json.loads(self._extension_store.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(records, list):
            return
        for record in records:
            if not isinstance(record, dict) or not record.get("entry_point"):
                continue
            entry_point = str(record["entry_point"])
            if entry_point == "inline:factory" or ":" not in entry_point:
                continue
            try:
                material = self._extension_integrity_material(entry_point)
            except ValueError:
                continue
            if str(record.get("integrity_hash", "")) and hashlib.sha256(
                material
            ).hexdigest() != str(record["integrity_hash"]):
                continue
            try:
                factory = _resolve_callable(entry_point)
            except ValueError:
                continue
            try:
                self.register_extension(
                    str(record["id"]),
                    factory,
                    version=str(record.get("version", "1.0.0")),
                    genesis_range=str(record.get("genesis_range", ">=0.1.0")),
                    schema_range=str(record.get("schema_range", ">=1.0")),
                    capabilities=tuple(record.get("capabilities", ("extension",))),
                    entry_point=entry_point,
                )
            except ValueError:
                continue

    def list_extensions(self) -> list[dict[str, Any]]:
        if self._extensions is None:
            return []
        return [
            {
                "id": manifest.id,
                "version": manifest.version,
                "capabilities": list(manifest.capabilities),
            }
            for manifest in self._extensions.manifests()
        ]

    def initialize(self) -> dict[str, str]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        for relative in (
            ".genesis",
            ".genesis/packages",
            "studies",
            "builds",
            "exports",
            "checkpoints",
        ):
            (self.workspace / relative).mkdir(exist_ok=True)
        return {"path": str(self.workspace), "status": "initialized"}

    def resolve_path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.workspace / candidate).resolve()
        if not resolved.is_relative_to(self.workspace):
            raise ValueError(f"WORKSPACE_CONTAINMENT: path escapes workspace: {value}")
        return resolved

    def create_study(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.persistence.create_study(payload)

    def get_study(self, study_id: str) -> dict[str, Any]:
        return self.persistence.get_study(study_id)

    def list_studies(self) -> list[dict[str, Any]]:
        return self.persistence.list_studies()

    def create_package(self, payload: dict[str, Any]) -> dict[str, Any]:
        package_id = payload["id"]
        path = self.resolve_path(Path(".genesis/packages") / f"{package_id}.json")
        if path.exists():
            raise ValueError(f"ALREADY_EXISTS: package '{package_id}' already exists")
        record = {**payload, "id": package_id, "version": 0}
        descriptor, temporary = tempfile.mkstemp(prefix=".package-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(record, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return record

    def list_packages(self) -> list[dict[str, Any]]:
        root = self.workspace / ".genesis" / "packages"
        return [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]

    def _model_profiles_path(self) -> Path:
        return self.resolve_path(".genesis/model-profiles.json")

    def _load_model_profiles(self) -> dict[str, dict[str, Any]]:
        path = self._model_profiles_path()
        if not path.exists():
            return {}
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("MODEL_PROFILES: profile store must be an object")
        return {
            str(profile_id): dict(profile)
            for profile_id, profile in value.items()
            if isinstance(profile, dict)
        }

    def _save_model_profiles(self, profiles: Mapping[str, Mapping[str, Any]]) -> None:
        path = self._model_profiles_path()
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(dict(profiles), indent=2, sort_keys=True) + "\n")
        # A profile may hold a pasted API key, so the file is owner-only rather
        # than whatever the umask allows.
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temporary, path)

    @staticmethod
    def _validate_model_profile(payload: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = payload.get("id")
        if not isinstance(profile_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", profile_id
        ):
            raise ValueError("INVALID_ID: model profile id must be a stable lowercase identifier")
        if payload.get("provider") != "openai-compatible":
            raise ValueError("MODEL_PROVIDER: only openai-compatible is supported")
        base_url = payload.get("base_url")
        parsed = urlparse(str(base_url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("MODEL_URL: base_url must be an HTTP(S) URL")
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("MODEL_NAME: model must be a non-empty string")
        api_key_env = payload.get("api_key_env")
        if not isinstance(api_key_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", api_key_env):
            raise ValueError("MODEL_CREDENTIAL: api_key_env must be an uppercase environment name")
        api_key = payload.get("api_key")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("MODEL_CREDENTIAL: api_key must be a non-empty string")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("MODEL_PARAMETERS: parameters must be an object")
        timeout = payload.get("timeout", 60)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("MODEL_TIMEOUT: timeout must be positive")
        stored: dict[str, Any] = {
            "id": profile_id,
            "provider": "openai-compatible",
            "base_url": str(base_url).rstrip("/"),
            "model": model.strip(),
            "api_key_env": api_key_env,
            "parameters": dict(parameters),
            "timeout": timeout,
            "version": int(payload.get("version", 1)),
        }
        if api_key is not None:
            stored["api_key"] = api_key
        return stored

    def _full_model_profile(self, profile_id: str) -> dict[str, Any]:
        """Stored record including any pasted api_key (never exposed to reads)."""
        profile = self._load_model_profiles().get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        return dict(profile)

    @staticmethod
    def _public_model_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in profile.items() if key != "api_key"}

    def list_model_profiles(self) -> list[dict[str, Any]]:
        profiles = self._load_model_profiles()
        return [self._public_model_profile(profiles[key]) for key in sorted(profiles)]

    def get_model_profile(self, profile_id: str) -> dict[str, Any]:
        self._validate_specification_id(profile_id)
        profile = self._load_model_profiles().get(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        return self._public_model_profile(profile)

    def create_model_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        profile = self._validate_model_profile(payload)
        profiles = self._load_model_profiles()
        if profile["id"] in profiles:
            raise ValueError(f"ALREADY_EXISTS: model profile '{profile['id']}' already exists")
        profiles[profile["id"]] = profile
        self._save_model_profiles(profiles)
        return self._public_model_profile(profile)

    def update_model_profile(
        self, profile_id: str, payload: Mapping[str, Any], expected_version: int
    ) -> dict[str, Any]:
        profiles = self._load_model_profiles()
        current = profiles.get(profile_id)
        if current is None:
            raise KeyError(profile_id)
        if current.get("version") != expected_version:
            raise ValueError(
                f"EXPECTED_VERSION: expected {expected_version}, found {current.get('version')}"
            )
        candidate = {**current, **dict(payload), "id": profile_id, "version": expected_version + 1}
        profile = self._validate_model_profile(candidate)
        profiles[profile_id] = profile
        self._save_model_profiles(profiles)
        return self._public_model_profile(profile)

    def model_profile_status(self, profile_id: str) -> dict[str, Any]:
        profile = self._full_model_profile(profile_id)
        return {
            "id": profile_id,
            "credential_present": bool(
                profile.get("api_key") or os.environ.get(str(profile["api_key_env"]))
            ),
        }

    def test_model_profile(self, profile_id: str, prompt: str = "Reply with OK.") -> dict[str, Any]:
        profile = self._full_model_profile(profile_id)
        provider = OpenAICompatibleProvider(
            base_url=str(profile["base_url"]),
            model=str(profile["model"]),
            api_key_env=str(profile["api_key_env"]),
            api_key=profile.get("api_key"),
            timeout=float(profile["timeout"]),
        )
        response = provider.generate(
            ProviderRequest(
                model=str(profile["model"]),
                prompt=prompt,
                parameters=dict(profile.get("parameters", {})),
            )
        )
        return {
            "id": profile_id,
            "provider": response.provider,
            "model": response.model,
            "request_id": response.request_id,
            "text": response.text,
            "usage": response.usage,
        }

    @staticmethod
    def _validate_specification_id(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value):
            raise ValueError("INVALID_ID: specification id must be a stable lowercase identifier")
        return value

    def _specification_exists(self, specification_id: str) -> bool:
        return (self._specification_dir(specification_id) / "metadata.json").is_file()

    def _current_form_payload(self, directory: Path, specification_id: str) -> dict[str, Any]:
        loaded = StudyCompiler(directory)._load()  # schema validation, raises on invalid
        return self._form_from_package(loaded, directory, specification_id)

    def _snapshot_package(self, directory: Path) -> str:
        """Persist the exact package files as one immutable, content-addressed bundle."""
        excluded = {"metadata.json", "checklist.json"}
        files: dict[str, str] = {}
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.name in excluded:
                    continue
                relative = path.relative_to(directory).as_posix()
                try:
                    files[relative] = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    files[relative] = "base64:" + base64.b64encode(path.read_bytes()).decode()
        bundle = json.dumps({"files": files}, sort_keys=True).encode()
        reference = self.persistence.object_store.put(
            bundle, "application/vnd.genesis.snapshot+json"
        )
        return reference.digest

    def get_package_snapshot(self, specification_id: str, version: int) -> dict[str, Any]:
        """Retrieve the immutable snapshot of an earlier accepted package version."""
        versions = self.persistence.list_package_versions(specification_id)
        for item in versions:
            if int(item["version"]) != int(version):
                continue
            digest = str(item.get("snapshot_digest", ""))
            if not digest:
                raise ValueError(
                    f"SNAPSHOT_MISSING: version {version} of '{specification_id}' "
                    "has no recorded snapshot"
                )
            payload = self.persistence.object_store.root / digest[:2] / digest[2:]
            if not payload.is_file():
                raise ValueError(f"SNAPSHOT_MISSING: snapshot object '{digest}' is absent")
            bundle = json.loads(payload.read_bytes())
            return {
                "specification_id": specification_id,
                "version": int(version),
                "content_hash": item.get("content_hash"),
                "snapshot_digest": digest,
                "files": bundle.get("files", {}),
            }
        raise KeyError(specification_id)

    def _specification_dir(self, specification_id: str) -> Path:
        self._validate_specification_id(specification_id)
        return self.resolve_path(Path(".genesis/specifications") / specification_id)

    @staticmethod
    def _canonical_specification(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        study_id = payload["id"]
        unsupported = sorted(set(payload) - _FORM_FIELDS)
        if unsupported:
            raise ValueError(f"INVALID_FIELD: unsupported form field(s): {', '.join(unsupported)}")
        theory_block = _validated_block(payload, "theory", _THEORY_BLOCK_FIELDS)
        domain_block = _validated_block(payload, "domain", _DOMAIN_BLOCK_FIELDS)
        protocol_block = _validated_block(payload, "protocol", _PROTOCOL_BLOCK_FIELDS)
        extensions = payload.get("extensions") or {}

        def pick(source: dict[str, Any], name: str, default: Any) -> Any:
            return source.get(name, payload.get(name, default))

        def artifact(**fields: Any) -> dict[str, Any]:
            value: dict[str, Any] = {"schema_version": "1.0", "study_id": study_id}
            if extensions:
                value["extensions"] = dict(extensions)
            value.update({name: item for name, item in fields.items() if item is not None})
            return value

        return {
            "study": artifact(
                title=payload.get("title", study_id),
                description=payload.get("description"),
                owners=payload.get("owners", []),
                source_citations=payload.get("source_citations", []),
                artifact_refs=payload.get("artifact_refs", []),
                package_compatibility=payload.get("package_compatibility"),
                origin=payload.get("origin"),
                approval=payload.get("approval"),
            ),
            "openness": artifact(processes=payload.get("processes", [])),
            "theory": artifact(
                theory_family=pick(theory_block, "theory_family", "exploratory"),
                constructs=pick(theory_block, "constructs", []),
                process_mappings=pick(theory_block, "process_mappings", []),
                relations=pick(theory_block, "relations", []),
                feedback=pick(theory_block, "feedback", []),
                delays=pick(theory_block, "delays", []),
                observables=pick(theory_block, "observables", []),
            ),
            "domain": artifact(
                actors=pick(domain_block, "actors", []),
                attributes=pick(domain_block, "attributes", []),
                states=pick(domain_block, "states", []),
                artifacts=pick(domain_block, "artifacts", []),
                mechanisms=pick(domain_block, "mechanisms", []),
                institutions=pick(domain_block, "institutions", []),
                initialization=pick(domain_block, "initialization", {}),
                visibility=pick(domain_block, "visibility", []),
                availability=pick(domain_block, "availability", []),
                updates=pick(domain_block, "updates", []),
                persistence=pick(domain_block, "persistence", []),
            ),
            "protocol": artifact(
                time_model=pick(protocol_block, "time_model", {"type": "rounds"}),
                termination=pick(protocol_block, "termination", []),
                conditions=pick(protocol_block, "conditions", []),
                factors=pick(protocol_block, "factors", []),
                phases=pick(protocol_block, "phases", []),
                replications=pick(protocol_block, "replications", 1),
                matching=pick(protocol_block, "matching", {}),
                random_streams=pick(protocol_block, "random_streams", []),
                model_freezing=pick(protocol_block, "model_freezing", True),
                budgets=pick(protocol_block, "budgets", {}),
                checkpoints=pick(protocol_block, "checkpoints", {}),
                replay_retention=pick(protocol_block, "replay_retention", {}),
            ),
            "outcomes": artifact(
                outcomes=payload.get("outcomes", []),
                datasets=payload.get("datasets", []),
            ),
            "models": artifact(models=payload.get("models", [])),
        }

    @staticmethod
    def _write_yaml(path: Path, value: dict[str, Any]) -> None:
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    def _write_specification_files(self, directory: Path, payload: dict[str, Any]) -> None:
        # Validate and serialize the entire candidate before touching live files.
        prompts = payload.get("prompts", {})
        if not isinstance(prompts, dict):
            raise ValueError("PROMPTS: prompts must be an object")
        for prompt_id, content in prompts.items():
            self._validate_specification_id(prompt_id)
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"PROMPT_CONTENT: prompt '{prompt_id}' must be non-empty text")
        schemas = payload.get("schemas", {})
        self._validate_schema_files(schemas)
        canonical = self._canonical_specification(payload)
        for value in canonical.values():
            yaml.safe_dump(value, sort_keys=False)
        for content in schemas.values():
            json.dumps(content)
        directory.mkdir(parents=True, exist_ok=True)
        for name, value in canonical.items():
            self._write_yaml(directory / f"{name}.yaml", value)
        if "prompts" in payload and (directory / "prompts").exists():
            for existing in (directory / "prompts").glob("*.txt"):
                if existing.stem not in prompts:
                    existing.unlink()
        if prompts:
            if not isinstance(prompts, dict):
                raise ValueError("PROMPTS: prompts must be an object")
            prompt_dir = directory / "prompts"
            prompt_dir.mkdir(exist_ok=True)
            for prompt_id, content in prompts.items():
                self._validate_specification_id(prompt_id)
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(f"PROMPT_CONTENT: prompt '{prompt_id}' must be non-empty text")
                (prompt_dir / f"{prompt_id}.txt").write_text(content)
        if "schemas" in payload:
            schema_dir = directory / "schemas"
            schema_dir.mkdir(exist_ok=True)
            for existing in schema_dir.iterdir():
                if existing.is_file() and existing.suffix in {".json", ".yaml", ".yml"}:
                    if existing.stem not in schemas or existing.suffix != ".json":
                        existing.unlink()
            for schema_id, content in schemas.items():
                (schema_dir / f"{schema_id}.json").write_text(json.dumps(content, indent=2))

    @staticmethod
    def _validate_schema_files(schemas: Any) -> None:
        """Metaschema-validate package schemas at write time (SCH-001/006).

        Boolean and empty schemas are valid under the package dialect; any
        other malformed declaration fails here with the typed schema code.
        """
        if not isinstance(schemas, dict):
            raise ValueError("SCHEMA_INVALID: schemas must map IDs to JSON Schema objects")
        for schema_id, content in schemas.items():
            GenesisService._validate_specification_id(schema_id)
            if not isinstance(content, (dict, bool)):
                raise ValueError(f"SCHEMA_INVALID: '{schema_id}' must be a schema object")
        try:
            PackageSchemaCatalog(schemas)
        except SchemaValidationError as exc:
            raise ValueError(f"SCHEMA_INVALID: {exc.code}: {exc}") from exc

    @contextmanager
    def _package_transaction(self, specification_id: str) -> Iterator[None]:
        """Restore files and registry rows if a package edit fails."""
        directory = self._specification_dir(specification_id)
        directory.parent.mkdir(parents=True, exist_ok=True)
        with (
            self.persistence._lock,
            tempfile.TemporaryDirectory(prefix=".package-edit-", dir=directory.parent) as temporary,
        ):
            backup = Path(temporary) / "before"
            if directory.exists():
                shutil.copytree(directory, backup, symlinks=True)
            connection = self.persistence.connection
            connection.execute("SAVEPOINT package_edit")
            try:
                yield
                connection.execute("RELEASE SAVEPOINT package_edit")
            except Exception:
                connection.execute("ROLLBACK TO SAVEPOINT package_edit")
                connection.execute("RELEASE SAVEPOINT package_edit")
                if directory.exists():
                    directory.rename(Path(temporary) / "failed")
                if backup.exists():
                    backup.rename(directory)
                raise

    def create_specification(self, payload: dict[str, Any]) -> dict[str, Any]:
        specification_id = self._validate_specification_id(payload.get("id"))
        with self._package_transaction(specification_id):
            return self._create_specification(payload)

    def _create_specification(self, payload: dict[str, Any]) -> dict[str, Any]:
        specification_id = self._validate_specification_id(payload.get("id"))
        directory = self._specification_dir(specification_id)
        if directory.exists():
            raise ValueError(f"ALREADY_EXISTS: specification '{specification_id}' already exists")
        self._write_specification_files(directory, payload)
        metadata = {
            "id": specification_id,
            "title": payload.get("title", specification_id),
            "description": payload.get("description"),
            "version": 1,
            "status": "draft",
            "form": dict(payload),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        persist_checklist(directory, checklist_record(directory))
        metadata["snapshot_digest"] = self._snapshot_package(directory)
        self.persistence.record_package_version(
            specification_id, 1, self._package_content_hash(directory), None, "draft", metadata
        )
        return metadata

    def get_specification(self, specification_id: str) -> dict[str, Any]:
        directory = self._specification_dir(specification_id)
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise KeyError(specification_id)
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("SPECIFICATION_METADATA: metadata must be an object")
        metadata["files"] = {
            path.name: path.read_text() for path in sorted(directory.glob("*.yaml"))
        }
        return metadata

    @staticmethod
    def _metadata_for_storage(metadata: Mapping[str, Any]) -> dict[str, Any]:
        """Remove the read-only YAML projection before persisting metadata."""

        return {key: value for key, value in metadata.items() if key != "files"}

    def list_specifications(self) -> list[dict[str, Any]]:
        root = self.workspace / ".genesis/specifications"
        return [json.loads(path.read_text()) for path in sorted(root.glob("*/metadata.json"))]

    def update_specification(
        self, specification_id: str, payload: dict[str, Any], expected_version: int
    ) -> dict[str, Any]:
        with self._package_transaction(specification_id):
            return self._update_specification(specification_id, payload, expected_version)

    def _update_specification(
        self, specification_id: str, payload: dict[str, Any], expected_version: int
    ) -> dict[str, Any]:
        current = self.get_specification(specification_id)
        if current["version"] != expected_version:
            raise ValueError(
                f"EXPECTED_VERSION: expected {expected_version}, found {current['version']}"
            )
        if payload.get("id", specification_id) != specification_id:
            raise ValueError("SPECIFICATION_ID: id cannot change during an edit")
        payload = {**current.get("form", {}), **payload, "id": specification_id}
        directory = self._specification_dir(specification_id)
        self._write_specification_files(directory, payload)
        updated = {
            **current,
            "title": payload.get("title", specification_id),
            "description": payload.get("description"),
            "version": current["version"] + 1,
            "status": "draft",
            "form": payload,
        }
        updated.pop("approved_version", None)
        updated.pop("approved_by", None)
        (directory / "metadata.json").write_text(
            json.dumps(self._metadata_for_storage(updated), indent=2, sort_keys=True)
        )
        # Later information may reopen earlier items (Section 6): re-evaluate rules.
        persist_checklist(directory, checklist_record(directory))
        # Editing an approved package creates a new draft with parent lineage (LIFE-001).
        parents = self.persistence.list_package_versions(specification_id)
        approved_parent = max(
            (int(item["version"]) for item in parents if item["status"] == "approved"),
            default=None,
        )
        updated["snapshot_digest"] = self._snapshot_package(directory)
        self.persistence.record_package_version(
            specification_id,
            int(updated["version"]),
            self._package_content_hash(directory),
            approved_parent,
            "draft",
            updated,
        )
        return updated

    def get_checklist(self, specification_id: str) -> dict[str, Any]:
        metadata = self.get_specification(specification_id)
        directory = self._specification_dir(specification_id)
        return {
            "specification_id": specification_id,
            "version": metadata["version"],
            "items": checklist_record(directory),
        }

    def update_checklist_item(
        self,
        specification_id: str,
        item_id: str,
        *,
        status: str | None = None,
        evidence: list[str] | None = None,
        confirmed_by: str | None = None,
    ) -> dict[str, Any]:
        directory = self._specification_dir(specification_id)
        if not (directory / "metadata.json").is_file():
            raise KeyError(specification_id)
        items = checklist_record(directory)
        for item in items:
            if str(item.get("id")) != item_id:
                continue
            if status is not None:
                if status not in {"unresolved", "partial", "complete", "not_applicable"}:
                    raise ValueError(f"CHECKLIST_STATUS: unknown status '{status}'")
                item["status"] = status
                item["manual"] = True
            if evidence is not None:
                item["evidence"] = list(evidence)
            if confirmed_by is not None:
                item["confirmed_by"] = confirmed_by
                item["researcher_confirmation"] = True
            persist_checklist(directory, items)
            return dict(item)
        raise KeyError(item_id)

    def refresh_checklist(self, specification_id: str) -> list[dict[str, Any]]:
        directory = self._specification_dir(specification_id)
        items = checklist_record(directory)
        for item in items:
            item.pop("manual", None)
            item.pop("confirmed_by", None)
            item["researcher_confirmation"] = False
        persist_checklist(directory, items)
        return checklist_record(directory)

    def _assistant_prompt(self, specification_id: str | None, instruction: str) -> str:
        if specification_id:
            current = json.dumps(
                self.get_specification(specification_id).get("form", {}), sort_keys=True
            )
            checklist = self.get_checklist(specification_id)["items"]
        else:
            current = "{}"
            checklist = [
                {"id": item["id"], "status": "unresolved", "question": item["question"]}
                for item in CHECKLIST_ITEMS
            ]
        lines = [
            "You are the GENESIS study-specification assistant.",
            "Checklist state (id: status | question):",
        ]
        lines.extend(
            f"- {item['id']}: {item.get('status', 'unresolved')} | {item['question']}"
            for item in checklist
        )
        lines.append("Current specification form (JSON):")
        lines.append(current)
        lines.append(f"Research instruction: {instruction}")
        lines.append(
            "Reply with only a JSON object. Allowed keys: id, title, description, owners, "
            "theory, domain, protocol, outcomes, models, processes, artifacts, prompts. "
            "id and title are required."
        )
        return "\n".join(lines)

    def draft_from_model(
        self,
        specification_id: str | None = None,
        instruction: str = "",
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        """LLM-mediated elicitation: propose a draft payload without writing it (AST-006)."""
        profiles = self._load_model_profiles()
        if not profiles:
            raise ValueError("MODEL_PROFILE: no model profile configured for the assistant")
        profile_id = profile_id or sorted(profiles)[0]
        configured = profiles.get(profile_id)
        if configured is None:
            raise KeyError(profile_id)
        provider = OpenAICompatibleProvider(
            base_url=str(configured["base_url"]),
            model=str(configured["model"]),
            api_key_env=str(configured["api_key_env"]),
            api_key=configured.get("api_key"),
            timeout=float(configured.get("timeout", 60)),
        )
        response = provider.generate(
            ProviderRequest(
                model=str(configured["model"]),
                prompt=self._assistant_prompt(specification_id, instruction),
                parameters=dict(configured.get("parameters", {})),
            )
        )
        proposal = _parse_json_object(response.text)
        if not isinstance(proposal, dict) or not proposal.get("id") or not proposal.get("title"):
            raise ValueError("PROPOSAL_INVALID: model response must contain id and title")
        unsupported = sorted(set(proposal) - _FORM_FIELDS)
        if unsupported:
            raise ValueError(f"INVALID_FIELD: unsupported form field(s): {', '.join(unsupported)}")
        proposal = dict(proposal)
        canonical = self._canonical_specification(proposal)
        preview = {
            name: yaml.safe_dump(value, sort_keys=False) for name, value in canonical.items()
        }
        operations: list[dict[str, Any]]
        target_id = (
            specification_id if specification_id is not None else str(proposal.get("id", ""))
        )
        if target_id and self._specification_exists(target_id):
            directory = self._specification_dir(target_id)
            current_payload = self._current_form_payload(directory, target_id)
            operations = _diff_form_fields(current_payload, proposal)
            if not operations:
                operations = [{"path": "specification", "op": "set", "value": proposal}]
        else:
            operations = [{"path": "specification", "op": "set", "value": proposal}]
        patch = SpecificationPatch(
            operations=[
                SpecificationPatchOperation(
                    path=str(operation["path"]),
                    op="set",
                    value=operation.get("value"),
                )
                for operation in operations
            ],
            evidence=[],
            assumptions=[],
            unresolved_questions=[],
            affected_checklist_items=[item["id"] for item in CHECKLIST_ITEMS],
        )
        patch.validate_deterministic()
        return {
            "proposal": proposal,
            "preview": preview,
            "profile_id": profile_id,
            "patch": patch.model_dump(mode="json"),
        }

    def accept_draft(
        self,
        specification_id: str,
        proposal: dict[str, Any],
        confirmed_by: str = "researcher",
        patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply an accepted assistant proposal as a draft (researcher-confirmed).

        When ``patch`` is supplied the operations are applied field-by-field
        over the current payload (AW-02); otherwise the proposal replaces the
        whole form, preserving the established flow.
        """
        if patch is not None:
            validated = SpecificationPatch.model_validate(patch)
            validated.validate_deterministic()
            patch = validated.model_dump(mode="json")
            specification_ops = [
                operation
                for operation in patch["operations"]
                if operation["path"].startswith("specification")
            ]
        try:
            current = self.get_specification(specification_id)
        except KeyError:
            current = None
        if patch is not None and current is not None:
            directory = self._specification_dir(specification_id)
            payload = self._current_form_payload(directory, specification_id)
            proposal = _apply_field_operations(payload, specification_ops)
            proposal["id"] = self._validate_specification_id(specification_id)
            self._canonical_specification(proposal)
            metadata = self.update_specification(specification_id, proposal, current["version"])
        else:
            if not isinstance(proposal, dict) or not proposal.get("title"):
                raise ValueError("PROPOSAL_INVALID: accepted proposal requires a title")
            unsupported = sorted(set(proposal) - _FORM_FIELDS)
            if unsupported:
                raise ValueError(
                    f"INVALID_FIELD: unsupported form field(s): {', '.join(unsupported)}"
                )
            proposal = dict(proposal)
            proposal["id"] = self._validate_specification_id(specification_id)
            # Validate through the canonical mapper; nested unknown fields are rejected.
            self._canonical_specification(proposal)
            if current is None:
                metadata = self.create_specification({**proposal, "id": specification_id})
            else:
                metadata = self.update_specification(specification_id, proposal, current["version"])
        metadata["assistant_confirmed_by"] = confirmed_by
        directory = self._specification_dir(specification_id)
        history = list(metadata.get("patch_history", []))
        if patch is not None:
            history.append(
                {
                    "at": datetime.now(UTC).isoformat(),
                    "confirmed_by": str(confirmed_by),
                    "operations": patch["operations"],
                    "evidence": patch.get("evidence", []),
                    "assumptions": patch.get("assumptions", []),
                    "unresolved_questions": patch.get("unresolved_questions", []),
                }
            )
        metadata["patch_history"] = history
        (directory / "metadata.json").write_text(
            json.dumps(self._metadata_for_storage(metadata), indent=2, sort_keys=True)
        )
        return metadata

    def inspect_specification(self, specification_id: str) -> dict[str, Any]:
        metadata = self.get_specification(specification_id)
        response = (
            StudyAssistant(theory_templates=self._workflow_registry.theory_templates())
            .inspect_package(self._specification_dir(specification_id))
            .as_dict()
        )
        return {**response, "status": metadata["status"], "version": metadata["version"]}

    def approve_specification(
        self,
        specification_id: str,
        expected_version: int,
        approved_by: str,
    ) -> dict[str, Any]:
        metadata = self.get_specification(specification_id)
        if metadata["version"] != expected_version:
            raise ValueError(
                f"EXPECTED_VERSION: expected {expected_version}, found {metadata['version']}"
            )
        unresolved = []
        for item in self.get_checklist(specification_id)["items"]:
            if not item.get("required"):
                continue
            if item.get("status") not in {"complete", "not_applicable"}:
                unresolved.append(item["id"])
                continue
            if (
                item.get("manual")
                and item.get("status") == "complete"
                and not item.get("researcher_confirmation")
            ):
                unresolved.append(f"{item['id']} (unconfirmed)")
        if unresolved:
            raise ValueError(
                "SPECIFICATION_INCOMPLETE: required checklist items unresolved: "
                + ", ".join(unresolved)
            )
        inspection = StudyAssistant(
            theory_templates=self._workflow_registry.theory_templates()
        ).inspect_package(self._specification_dir(specification_id))
        if not inspection.valid:
            raise ValueError("SPECIFICATION_INVALID: resolve assistant validation issues first")
        metadata = {
            **metadata,
            "status": "approved",
            "approved_version": expected_version,
            "approved_by": approved_by,
        }
        (self._specification_dir(specification_id) / "metadata.json").write_text(
            json.dumps(self._metadata_for_storage(metadata), indent=2, sort_keys=True)
        )
        directory = self._specification_dir(specification_id)
        metadata["snapshot_digest"] = self._snapshot_package(directory)
        self.persistence.record_package_version(
            specification_id,
            expected_version,
            self._package_content_hash(directory),
            None,
            "approved",
            metadata,
        )
        return metadata

    def compile_study(
        self,
        source: str | Path | None,
        output: str | Path,
        *,
        specification_id: str | None = None,
    ) -> dict[str, Any]:
        if specification_id:
            specification = self.get_specification(specification_id)
            if specification["status"] != "approved":
                raise ValueError("SPECIFICATION_NOT_APPROVED: approve the draft before compilation")
            source = Path(".genesis/specifications") / specification_id
        if source is None:
            raise ValueError("VALIDATION_ERROR: source or specification_id is required")
        source_path = self.resolve_path(source)
        output_path = self.resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build = StudyCompiler(
            source_path,
            theory_templates=self._workflow_registry.theory_templates(),
        ).compile(output_path)
        package_version = 0
        try:
            versions = self.persistence.list_package_versions(build.study_id)
            if versions:
                package_version = int(versions[-1]["version"])
        except ValueError:
            package_version = 0
        package_content_hash = self._package_content_hash(source_path)
        build_manifest_path = build.path / "build_manifest.json"
        if package_version:
            for recorded in self.persistence.list_package_versions(str(build.study_id)):
                if int(recorded["version"]) == int(package_version):
                    snapshot_digest = str(recorded.get("snapshot_digest", ""))
                    if snapshot_digest and build_manifest_path.is_file():
                        import stat as _stat

                        build_manifest_path.chmod(
                            _stat.S_IRUSR | _stat.S_IWUSR | _stat.S_IRGRP | _stat.S_IROTH
                        )
                        manifest_payload = json.loads(build_manifest_path.read_text())
                        manifest_payload["package_snapshot"] = snapshot_digest
                        build_manifest_path.write_text(
                            json.dumps(manifest_payload, indent=2, sort_keys=True)
                        )
                        # Recompute integrity for the touched manifest so
                        # verify_build remains consistent.
                        self._recompute_build_integrity(build.path)
                    break
        self.persistence.record_study_build(
            {
                "build_hash": build.build_hash,
                "study_id": build.study_id,
                "package_version": package_version,
                "package_content_hash": package_content_hash,
                "compiler_version": build.compiler_version
                if hasattr(build, "compiler_version")
                else "1.0",
                "created_at": datetime.now(UTC).isoformat(),
                "path": str(build.path),
            }
        )
        return {
            "study_id": build.study_id,
            "build_hash": build.build_hash,
            "path": str(build.path),
            "manifest": build.manifest,
        }

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.persistence.create_run(payload)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.persistence.get_run(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.persistence.list_runs()

    def transition_run(self, run_id: str, target: str, expected_version: int) -> dict[str, Any]:
        current = self.get_run(run_id)
        if target == "paused" and current["status"] == "created":
            current = self.persistence.transition_run(run_id, "running", expected_version)
            return self.persistence.transition_run(run_id, "paused", current["version"])
        return self.persistence.transition_run(run_id, target, expected_version)

    def _build_run_manifest(self, run: dict[str, Any]) -> dict[str, Any]:
        """Freeze the complete configuration governing this run (REP-001, 21.6).

        Covers package identity, build/protocol hashes, data provenance, named
        random streams, model/prompt versions, and the resolved runtime profile
        parameters (review finding 2).
        """
        manifest: dict[str, Any] = {
            "run_id": run["id"],
            "condition_id": run.get("condition_id", "base"),
            "replication": int(run.get("replication", 1)),
            "generated_at": datetime.now(UTC).isoformat(),
            "genesis_version": __version__,
            "condition": dict(run.get("condition", {})),
        }
        protocol: dict[str, Any] = {}
        build_manifest: dict[str, Any] = {}
        build_ref = run.get("build") or run.get("build_path")
        if build_ref:
            build_path = self.resolve_path(build_ref)
            build_manifest = json.loads((build_path / "build_manifest.json").read_text())
            manifest["study_id"] = build_manifest.get("study_id")
            manifest["build_hash"] = build_manifest.get("build_hash")
            # Exact package identity: the compile-time StudyBuild record, never
            # the latest package version (an older build may be executed later).
            for record in self.persistence.list_study_builds():
                if str(record.get("path", "")) == str(build_path) or str(
                    record.get("build_hash", "")
                ) == str(build_manifest.get("build_hash", "")):
                    manifest["package_version"] = int(record.get("package_version", 0))
                    manifest["package_content_hash"] = str(record.get("package_content_hash", ""))
                    break
            manifest["compiler_version"] = build_manifest.get("compiler_version")
            protocol = json.loads((build_path / "protocol.json").read_text())
            manifest["protocol_hash"] = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            manifest["random_streams"] = [
                dict(stream) for stream in protocol.get("random_streams", [])
            ]
            data_provenance: dict[str, Any] = {}
            data_manifest_path = build_path / "data_manifest.json"
            if data_manifest_path.is_file():
                data_provenance["assets"] = dict(json.loads(data_manifest_path.read_text()))
            init_path = build_path / "initialization.json"
            if init_path.is_file():
                data_provenance["initialization"] = json.loads(init_path.read_text())
            if data_provenance:
                manifest["data_provenance"] = data_provenance
            models_path = build_path / "model_profiles.json"
            if models_path.is_file():
                profiles = json.loads(models_path.read_text())
                manifest["model_versions"] = {
                    str(profile["id"]): str(profile.get("model", ""))
                    for profile in profiles
                    if isinstance(profile, dict) and profile.get("id")
                }
            prompts_path = build_path / "prompt_templates.json"
            if prompts_path.is_file():
                prompts = json.loads(prompts_path.read_text())
                manifest["prompt_versions"] = {
                    str(key): hashlib.sha256(str(value).encode()).hexdigest()[:16]
                    for key, value in prompts.items()
                }
            # Pin the predeclared observables alongside the rest of the
            # configuration, so a run records which measurement plan produced
            # its outcomes.
            if (build_path / "outcome_plan.json").is_file():
                manifest["outcome_plan_digest"] = outcome_plan_digest(
                    compile_outcome_plan(build_path)
                )
        # Resolved runtime profile parameters plus the endpoint identity.
        resolved: dict[str, Any] = {}
        for profile_id in manifest.get("model_versions", {}):
            try:
                configured = self.get_model_profile(str(profile_id))
            except KeyError:
                continue
            endpoint_identity = hashlib.sha256(
                json.dumps(
                    {
                        "base_url": configured.get("base_url"),
                        "model": configured.get("model"),
                        "timeout": configured.get("timeout"),
                        "provider": configured.get("provider"),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
            resolved[str(profile_id)] = {
                "parameters": dict(configured.get("parameters", {})),
                "endpoint_hash": endpoint_identity,
            }
        if resolved:
            manifest["resolved_profiles"] = resolved
        if self._extensions is not None:
            manifests = self._extensions.manifests()
            if manifests:
                manifest["extensions"] = {
                    extension.id: {
                        "version": extension.version,
                        "integrity_hash": extension.integrity_hash,
                    }
                    for extension in manifests
                }
        matching = protocol.get("matching", {})
        matching = matching if isinstance(matching, dict) else {}
        # RPL/§2.2: a replay child derives randomness from the ROOT source identity,
        # never from its own child run ID or from an intermediate replay in a
        # lineage, so recorded streams and runtime draws stay stable across
        # child IDs, replay-of-replay, imports and creation times.
        randomness = self._randomness_inputs(run)
        manifest["randomness_inputs"] = randomness
        # Retain the matching block so the seeds stay independently derivable
        # from the manifest alone (see verify_manifest_seeds).
        if matching:
            manifest["matching"] = dict(matching)
        streams = self.derive_manifest_seeds(randomness, manifest.get("random_streams"), matching)
        manifest["seeds"] = streams
        # Effective execution identity (spec §2.2): one immutable manifest
        # resolved before execution, with the package closure digest pinned at
        # compile time and a scientific configuration digest independent of the
        # local run ID.
        package_closure_digest = str(build_manifest.get("package_closure_digest", ""))
        manifest["package_closure_digest"] = package_closure_digest
        if package_closure_digest:
            model_configuration_digest = hashlib.sha256(
                canonical_json(
                    {
                        "model_versions": manifest.get("model_versions", {}),
                        "resolved_profiles": manifest.get("resolved_profiles", {}),
                    }
                ).encode()
            ).hexdigest()
            execution = resolve_execution_manifest(
                condition_id=str(manifest["condition_id"]),
                factors=manifest.get("condition", {}).get("factors", {}),
                replication=int(manifest["replication"]),
                build_manifest=build_manifest,
                protocol=protocol,
                package_closure_digest=package_closure_digest,
                protocol_digest=str(manifest.get("protocol_hash", "")),
                model_configuration_digest=model_configuration_digest,
                outcome_plan_digest=str(manifest.get("outcome_plan_digest", "")),
                origin_experiment_id=run.get("experiment_id"),
            )
            manifest["execution"] = execution
            manifest["scientific_config_digest"] = scientific_config_digest(execution)
        else:
            # Legacy/unverified build without a pinned closure.
            manifest["legacy_unverified"] = True
        return manifest

    @staticmethod
    def _recompute_build_integrity(build_path: Path) -> None:
        import stat as _stat

        integrity_path = build_path / "integrity_manifest.json"
        manifest_path = build_path / "build_manifest.json"
        integrity_path.chmod(_stat.S_IRUSR | _stat.S_IWUSR | _stat.S_IRGRP | _stat.S_IROTH)
        integrity = json.loads(integrity_path.read_text())
        integrity["build_manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        integrity["manifest_hash"] = integrity["build_manifest.json"]
        integrity_path.write_text(json.dumps(integrity, sort_keys=True, indent=2) + "\n")
        integrity_path.chmod(_stat.S_IRUSR | _stat.S_IRGRP | _stat.S_IROTH)

    @staticmethod
    def _package_content_hash(directory: Path) -> str:
        """Authoritative package digest (LIFE-002, review finding 1).

        Covers canonical YAML plus the execution-relevant assets that are not
        part of the top-level YAML: normalized prompts, schemas, and data
        digests. Extension namespaces live inside the canonical YAML files.
        """
        root = Path(directory)
        digest = hashlib.sha256()
        for name in sorted(root.glob("*.yaml")):
            digest.update(name.name.encode())
            digest.update(b"\0")
            digest.update(name.read_bytes())
        for relative in ("prompts", "schemas"):
            child = root / relative
            if not child.is_dir():
                continue
            for path in sorted(child.rglob("*")):
                if not path.is_file():
                    continue
                digest.update(relative.encode())
                digest.update(b"/")
                digest.update(path.relative_to(child).as_posix().encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
        data_dir = root / "data"
        if data_dir.is_dir():
            for path in sorted(data_dir.rglob("*")):
                if not path.is_file():
                    continue
                digest.update(b"data/")
                digest.update(path.relative_to(data_dir).as_posix().encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
        extensions_dir = root / "extensions"
        if extensions_dir.is_dir():
            for path in sorted(extensions_dir.rglob("*")):
                if not path.is_file():
                    continue
                digest.update(b"extensions/")
                digest.update(path.relative_to(extensions_dir).as_posix().encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _record_package_version(
        self,
        specification_id: str,
        *,
        parent_version: int | None,
        status: str,
    ) -> dict[str, Any]:
        directory = self._specification_dir(specification_id)
        metadata = json.loads((directory / "metadata.json").read_text())
        content_hash = self._package_content_hash(directory)
        payload = {"title": metadata.get("title"), "content_hash": content_hash}
        return self.persistence.record_package_version(
            specification_id,
            int(metadata["version"]),
            content_hash,
            parent_version,
            status,
            payload,
        )

    def list_package_versions(self, specification_id: str) -> list[dict[str, Any]]:
        self.get_specification(specification_id)
        return self.persistence.list_package_versions(specification_id)

    def _attach_run_manifest(self, run: dict[str, Any]) -> dict[str, Any]:
        if run.get("manifest") is not None:
            return run
        return self.persistence.attach_run_manifest(run["id"], self._build_run_manifest(run))

    @staticmethod
    def _build_schema_catalog(build_path: Path) -> dict[str, Any]:
        schemas_path = build_path / "schemas.json"
        return json.loads(schemas_path.read_text()) if schemas_path.is_file() else {}

    @staticmethod
    def _build_model_profiles(build_path: Path) -> dict[str, Any]:
        models_path = build_path / "model_profiles.json"
        return {
            profile["id"]: profile
            for profile in (json.loads(models_path.read_text()) if models_path.is_file() else [])
        }

    @staticmethod
    def _build_prompt_templates(build_path: Path) -> dict[str, Any]:
        prompts_path = build_path / "prompt_templates.json"
        return json.loads(prompts_path.read_text()) if prompts_path.is_file() else {}

    def _frozen_invocation_keys(
        self,
        run_id: str,
        recorded: dict[str, Any],
        *,
        phase_boundary: int | None,
        event_boundary: str | None,
    ) -> set[tuple[Any, ...]]:
        """Invocation keys (phase, attempt, actors) frozen by a phase/event boundary.

        A phase boundary freezes every invocation at phase < N. An event
        boundary freezes every invocation whose committed event precedes or is
        the given source event (review finding 5).
        """
        if phase_boundary is None and event_boundary is None:
            return set()
        target_order: int | None = None
        if event_boundary is not None:
            for order, event in enumerate(self.trace_run(run_id)):
                if str(event.get("event_id", "")) == event_boundary:
                    target_order = order
                    break
            if target_order is None:
                raise ValueError(f"REPLAY_BOUNDARY: unknown source event '{event_boundary}'")
        # Earliest committed order of each invocation in the source trace.
        # Only an event boundary needs it, so a phase boundary does not read
        # the trace at all.
        invocation_order: dict[str, int] = {}
        if event_boundary is not None:
            for order, event in enumerate(self.trace_run(run_id)):
                invocation = str(event.get("invocation_id", ""))
                if invocation and invocation not in invocation_order:
                    invocation_order[invocation] = order
        # Keys carry the process id: invocation coordinates alone are shared
        # across processes that run in the same phase for the same actors, so a
        # pooled key set froze recordings belonging to processes that were
        # never inside the prefix.
        frozen: set[tuple[Any, ...]] = set()
        for process_id, records in recorded.items():
            for record in records:
                if phase_boundary is not None and record["phase"] < phase_boundary:
                    frozen.add(
                        (str(process_id), record["phase"], record["attempt"], record["actors"])
                    )
        if event_boundary is not None:
            artifacts = self.artifacts_for_run(run_id)
            for artifact in artifacts:
                payload = artifact["payload"]
                if not isinstance(payload, dict):
                    continue
                invocation = str(payload.get("invocation_id", ""))
                claim_order = invocation_order.get(invocation)
                if (
                    invocation
                    and claim_order is not None
                    and target_order is not None
                    and claim_order <= target_order
                ):
                    frozen.add(
                        (
                            str(payload.get("process_id", "")),
                            int(payload.get("phase", 0)),
                            int(payload.get("attempt", 1)),
                            tuple(str(item) for item in payload.get("actors", [])),
                        )
                    )
        return frozen

    def _build_executors(
        self,
        build_path: Path | None,
        processes: list[Mapping[str, Any]],
        schema_catalog: dict[str, Any],
        model_profiles: dict[str, Any],
        prompt_templates: dict[str, Any],
        *,
        overrides: Mapping[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Construct the executor registry for one run (shared by run and replay)."""
        executors: dict[str, Any] = {}
        override_map = dict(overrides or {})
        try:
            schema_catalog_validator = PackageSchemaCatalog(schema_catalog)
        except SchemaValidationError:
            # Fall back to no per-process catalog binding; validate_schema still
            # enforces the declared dialect for a standalone schema.
            schema_catalog_validator = None
        for process in processes:
            binding = process.get("executor", {})
            if process["id"] in override_map:
                injected = override_map[process["id"]]
                executors[process["id"]] = (
                    injected
                    if hasattr(injected, "execute")
                    else CallableExecutor(injected, str(binding.get("mode", "deterministic")))
                )
                continue
            mode = str(binding.get("mode", "deterministic"))
            if mode == "stochastic":
                function = _resolve_callable(binding.get("parameters", {}).get("function"))
                executors[process["id"]] = StochasticExecutor(function)
                continue
            if mode == "computational":
                entry = _resolve_callable(binding.get("parameters", {}).get("entry_point"))
                executors[process["id"]] = CallableExecutor(entry, "computational")
                continue
            if mode == "rule":
                executors[process["id"]] = RuleExecutor(binding.get("parameters", {}))
                continue
            if mode == "state-transition":
                executors[process["id"]] = StateTransitionExecutor(binding.get("parameters", {}))
                continue
            if mode == "recorded_artifact":
                declared = binding.get("parameters", {}).get("outputs")
                executors[process["id"]] = (
                    RecordedArtifactExecutor(dict(declared))
                    if isinstance(declared, Mapping)
                    else DeterministicExecutor(
                        lambda _invocation: {"recorded": False, "code": "EXECUTOR_NO_RECORDING"}
                    )
                )
                continue
            if mode == "extension":
                extension_ref = str(binding.get("extension_ref", ""))
                if self._extensions is None:
                    raise ValueError(
                        f"EXECUTOR_UNREGISTERED: no extensions registered for '{extension_ref}'"
                    )
                try:
                    factory = self._extensions.get(extension_ref)
                except KeyError as exc:
                    raise ValueError(
                        f"EXECUTOR_UNREGISTERED: extension '{extension_ref}' is not registered"
                    ) from exc
                executors[process["id"]] = CallableExecutor(factory, "extension")
                continue
            if mode == "deterministic":
                executors[process["id"]] = DeterministicExecutor(lambda _invocation: {})
                continue
            if mode not in {"generative", "semantic-evaluator"}:
                raise ValueError(f"EXECUTOR_UNAVAILABLE: unknown executor mode '{mode}'")
            if build_path is None:
                raise ValueError(f"MODEL_PROFILE: {mode} processes require a compiled build")
            profile_id = binding.get("model_profile")
            if not isinstance(profile_id, str) or profile_id not in model_profiles:
                raise ValueError(
                    f"MODEL_PROFILE: compiled process references unknown profile {profile_id}"
                )
            configured = self._full_model_profile(profile_id)
            profile = model_profiles[profile_id]
            if profile.get("provider") != "openai-compatible":
                raise ValueError(f"MODEL_PROVIDER: unsupported provider {profile.get('provider')}")
            resolved_build = json.loads((build_path / "build_manifest.json").read_text())
            _check_profile_drift(
                profile_id, configured, profile, str(resolved_build.get("build_hash", ""))
            )
            provider = OpenAICompatibleProvider(
                base_url=str(configured["base_url"]),
                model=str(configured["model"]),
                api_key_env=str(configured["api_key_env"]),
                api_key=configured.get("api_key"),
                timeout=float(configured.get("timeout", 60)),
                cancel_event=cancel_event,
            )
            prompt_ref = process.get("prompt_ref")
            prompt_template = (
                prompt_templates.get(prompt_ref, "{context}")
                if isinstance(prompt_ref, str)
                else "{context}"
            )
            parameters = {
                **dict(configured.get("parameters", {})),
                **dict(profile.get("parameters", {})),
            }
            output_schema = None
            output_schema_validator = None
            output_key = "response"
            # Outputs are properties of the compiled process, not the executor binding.
            compiled_outputs = process.get("outputs")
            if isinstance(compiled_outputs, list) and compiled_outputs:
                schema_ref = compiled_outputs[0].get("schema_ref")
                artifact_type = compiled_outputs[0].get("artifact_type")
                if isinstance(artifact_type, str):
                    output_key = artifact_type
                if isinstance(schema_ref, str) and schema_ref in schema_catalog:
                    output_schema = schema_catalog[schema_ref]
                    if schema_catalog_validator is not None:
                        bound_schema_id = schema_ref
                        catalog_validator = schema_catalog_validator

                        def output_schema_validator(
                            value: Any,
                            _validator: PackageSchemaCatalog = catalog_validator,
                            _schema_id: str = bound_schema_id,
                        ) -> list[SchemaDiagnostic]:
                            return _validator.validate(_schema_id, value)

            if mode == "semantic-evaluator" and output_schema is None:
                raise ValueError(
                    "SEMANTIC_EVALUATOR_SCHEMA: semantic-evaluator requires "
                    "a declared output schema"
                )
            executors[process["id"]] = ProviderExecutor(
                provider,
                model=str(profile["model"]),
                prompt_template=str(prompt_template),
                parameters=parameters,
                output_schema=output_schema,
                output_schema_validator=output_schema_validator,
                output_key=output_key,
                mode=mode,
                max_repairs=int(profile.get("max_repairs", 1)),
                cancel_event=cancel_event,
            )
        return executors

    def execute_run(
        self, run_id: str, *, executor_overrides: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run["status"] == "completed":
            return run
        # Operational preflight happens before the state transition so a denied
        # dispatch never leaves the run persisted as "running" (finding 7).
        try:
            usage = shutil.disk_usage(self.workspace)
        except OSError:
            usage = None
        if usage is not None and usage.free < 100 * 1024 * 1024:
            raise ValueError(
                f"OPERATIONS_BLOCKED: disk space below dispatch threshold ({usage.free} free)"
            )
        if run["status"] == "paused":
            run = self.persistence.transition_run(run_id, "running", run["version"])
        elif run["status"] == "created":
            run = self.persistence.transition_run(run_id, "running", run["version"])
        elif run["status"] != "running":
            raise ValueError(f"RUN_TRANSITION: cannot execute run in {run['status']} status")
        if run.get("manifest") is None:
            run = self.persistence.attach_run_manifest(run_id, self._build_run_manifest(run))

        try:
            return self._dispatch_run(run_id, run, executor_overrides=executor_overrides)
        except Exception:
            latest = self.get_run(run_id)
            if latest["status"] == "running":
                self.persistence.transition_run(run_id, "failed", latest["version"])
            raise

    def _dispatch_run(
        self,
        run_id: str,
        run: Mapping[str, Any],
        *,
        executor_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cancel_event = threading.Event()

        def _status_provider() -> str:
            status = str(self.get_run(run_id).get("status", "running"))
            if status in {"cancelled", "paused"}:
                cancel_event.set()
            return status

        processes: list[Mapping[str, Any]] = []
        policies: dict[str, Any] = {
            "private": {"allow": []},
            "public": {"allow": []},
            "none": {"allow": []},
        }
        state_schema: dict[str, type | tuple[type, ...]] = {}
        initial_state: dict[str, Any] = {}
        artifact_catalog: dict[str, Any] = {}
        build_ref = run.get("build") or run.get("build_path")
        if build_ref:
            build_path = self.resolve_path(build_ref)
            StudyCompiler.verify_build(build_path)
            processes = json.loads((build_path / "processes.json").read_text())
            model_profiles_path = build_path / "model_profiles.json"
            model_profiles = {
                profile["id"]: profile
                for profile in (
                    json.loads(model_profiles_path.read_text())
                    if model_profiles_path.is_file()
                    else []
                )
            }
            prompts_path = build_path / "prompt_templates.json"
            prompt_templates = (
                json.loads(prompts_path.read_text()) if prompts_path.is_file() else {}
            )
            configured_policies = json.loads((build_path / "context_policies.json").read_text())
            if isinstance(configured_policies, list):
                policies.update(
                    {
                        item["id"]: item
                        for item in configured_policies
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    }
                )
            state_model = json.loads((build_path / "state_model.json").read_text())
            for state in state_model:
                if not isinstance(state, dict) or not state.get("id"):
                    continue
                field = str(state["id"])
                declared_type = str(state.get("value_type", "object"))
                if declared_type not in _STATE_TYPES:
                    raise ValueError(
                        f"STATE_VALUE_TYPE: state '{field}' declares unknown value_type "
                        f"'{declared_type}'; expected one of {sorted(_STATE_TYPES)}"
                    )
                state_schema[field] = _STATE_TYPES[declared_type]
                if state.get("initial") is not None:
                    initial_state[field] = state["initial"]
            artifact_entries = json.loads((build_path / "artifact_catalog.json").read_text())
            artifact_catalog = {
                str(entry["id"]): entry
                for entry in artifact_entries
                if isinstance(entry, dict) and entry.get("id")
            }
            schemas_path = build_path / "schemas.json"
            schema_catalog = json.loads(schemas_path.read_text()) if schemas_path.is_file() else {}
            init_path = build_path / "initialization.json"
            if init_path.is_file():
                initialization = json.loads(init_path.read_text())
                if (
                    isinstance(initialization, dict)
                    and str(initialization.get("mode", "")) == "empirical"
                ):
                    data_ref = str(initialization.get("data_source", ""))
                    field = str(initialization.get("state_field", "population"))
                    initial_state[field] = _load_empirical_data(build_path / data_ref, data_ref)

            protocol = json.loads((build_path / "protocol.json").read_text())

        executors = self._build_executors(
            build_path if build_ref else None,
            processes,
            schema_catalog if build_ref else {},
            model_profiles if build_ref else {},
            prompt_templates if build_ref else {},
            overrides=executor_overrides,
            cancel_event=cancel_event,
        )
        registry = ExecutorRegistry(executors)
        state_store = StateStore(state_schema, initial_state) if build_ref else None
        artifact_store = ArtifactStore(artifact_catalog) if build_ref else None
        # F3/SCH-002: schema enforcement at the common output-commit boundary.
        # Every declared artifact output is validated against its declared
        # schema before any state or artifact commit, regardless of executor
        # kind (generative, rule, computational, fallback, ...).
        output_schema_validator: Callable[[str, Any], list[str]] | None = None
        commit_catalog = None
        if build_ref and schema_catalog:
            try:
                commit_catalog = PackageSchemaCatalog(schema_catalog)
            except SchemaValidationError:
                # An unusable schema catalog fails validation per request
                # through the per-executor validator; leave the boundary
                # validator unbound for such malformed builds.
                commit_catalog = None
        if commit_catalog is not None:
            catalog = commit_catalog

            # F1: the validator resolves the schema REF directly (from the
            # executing process's output declaration); no domain-artifact
            # indirection is involved.
            def output_schema_validator(schema_ref: str, value: Any) -> list[str]:
                if schema_ref not in schema_catalog:
                    return []
                return [
                    f"{diagnostic.instance_pointer or 'root'}: {diagnostic.message}"
                    for diagnostic in catalog.validate(schema_ref, value)
                ]

        controller = RunController(
            Scheduler(processes),
            registry,
            ContextEngine(policies),
            persistence=self.persistence,
            state_store=state_store,
            artifact_store=artifact_store,
            status_provider=_status_provider,
            output_schema_validator=output_schema_validator,
        )
        try:
            # F5: randomness derives from the ROOT source's experiment,
            # condition and replication — never from a replay child's own
            # (possibly empty) experiment id or a derived branch condition id.
            # The intervention (effective) condition is still supplied as the
            # invocation input; only the seed inputs are root-inherited.
            randomness = self._randomness_inputs(run)
            controller.run(
                run_id,
                experiment_id=randomness["experiment_id"],
                condition_id=randomness["condition_id"],
                replication=randomness["replication"],
                seed_identity=randomness["run_id"],
                phase_start=int(protocol.get("time_model", {}).get("start") or 0)
                if build_ref
                else 0,
                phase_end=(
                    int(protocol.get("time_model", {}).get("end"))
                    if build_ref and isinstance(protocol.get("time_model", {}).get("end"), int)
                    else None
                ),
                terminal_phase=(
                    int(protocol.get("time_model", {}).get("end"))
                    if build_ref and isinstance(protocol.get("time_model", {}).get("end"), int)
                    else None
                ),
                max_events=(
                    int(protocol.get("budgets", {}).get("max_events"))
                    if build_ref and isinstance(protocol.get("budgets", {}).get("max_events"), int)
                    else None
                ),
                condition=dict(run.get("condition", {"id": run.get("condition_id", "base")})),
                matching=dict(protocol.get("matching", {})) if build_ref else {},
            )
        except Exception:
            latest = self.get_run(run_id)
            if latest["status"] == "running":
                self.persistence.transition_run(run_id, "failed", latest["version"])
            raise
        self._record_process_instances(run_id, controller)
        latest = self.get_run(run_id)
        if controller.status == "cancelled":
            return self.persistence.transition_run(run_id, "cancelled", latest["version"])
        if controller.status == "paused":
            if latest["status"] == "paused":
                return latest
            return self.persistence.transition_run(run_id, "paused", latest["version"])
        return self.persistence.transition_run(run_id, "completed", latest["version"])

    def _record_process_instances(self, run_id: str, controller: Any) -> None:
        """Persist ProcessInstance rows derivable from dispatch/commit logs (AW-15)."""
        statuses = {
            str(entry.get("invocation_id")): "failed"
            if entry.get("status") == "failed"
            else "committed"
            for entry in controller.commit_log
        }
        rows = []
        for entry in controller.dispatch_log:
            invocation_id = str(entry.get("invocation_id"))
            rows.append(
                {
                    "id": invocation_id,
                    "run_id": run_id,
                    "process_id": str(entry.get("process_id")),
                    "phase": int(entry.get("phase", 0)),
                    "attempt": int(entry.get("attempt", 1)),
                    "status": statuses.get(invocation_id, "dispatched"),
                }
            )
        if rows:
            self.persistence.record_process_instances(run_id, rows)

    def execute_protocol(
        self,
        run_id: str,
        *,
        executor_overrides: Mapping[str, Any] | None = None,
        parallel: bool = False,
        max_workers: int = 4,
        worker_kind: str | None = None,
    ) -> dict[str, Any]:
        """Expand protocol.conditions x replications into runs (REP-004, AW-13).

        The template run must reference a compiled build. Runs are dispatched
        sequentially by default; ``parallel`` dispatches through a bounded
        worker pool (backpressure) with the persistence coordinator as the
        sole writer.
        """
        template = self.get_run(run_id)
        build_ref = template.get("build") or template.get("build_path")
        if not build_ref:
            raise ValueError("RUN_PROTOCOL: the template run must reference a build")
        build_path = self.resolve_path(build_ref)
        protocol = json.loads((build_path / "protocol.json").read_text())
        protocol_hash = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        try:
            self.persistence.create_experiment(
                {
                    "id": run_id,
                    "study_id": str(template.get("study_id", "")),
                    "build_ref": build_ref,
                    "protocol_hash": protocol_hash,
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )
        except ValueError as exc:
            if not str(exc).startswith("ALREADY_EXISTS"):
                raise
        conditions = expand_protocol_conditions(protocol)
        replications = int(protocol.get("replications", 1))
        trial_ids: list[str] = []
        for condition in conditions:
            if not isinstance(condition, dict) or not condition.get("id"):
                raise ValueError("RUN_PROTOCOL: every condition requires a stable id")
            condition_id = str(condition["id"])
            for replication in range(1, replications + 1):
                trial_id = f"{run_id}-{condition_id}-{replication}"
                payload = {
                    "id": trial_id,
                    "study_id": template.get("study_id"),
                    "build": build_ref,
                    "experiment_id": run_id,
                    "condition_id": condition_id,
                    "condition": dict(condition),
                    "replication": replication,
                }
                try:
                    self.create_run(payload)
                except ValueError as exc:
                    if not str(exc).startswith("ALREADY_EXISTS"):
                        raise
                trial_ids.append(trial_id)

        def dispatch_one(trial_id: str) -> tuple[bool, str]:
            try:
                self.execute_run(trial_id, executor_overrides=executor_overrides)
                status = str(self.get_run(trial_id)["status"])
                return (status == "completed", "" if status == "completed" else f"status={status}")
            except Exception as exc:
                return False, f"{type(exc).__name__}: {exc}"

        worker_kind = str(worker_kind or ("thread" if executor_overrides else "process"))
        if parallel and max_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            if worker_kind == "process":
                # Executor overrides embed callables and are not picklable.
                if executor_overrides:
                    raise ValueError(
                        "RUN_PROTOCOL: process workers cannot carry executor overrides"
                    )
                from functools import partial

                worker = partial(_run_trial_worker, str(self.workspace))
                with ProcessPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
                    outcomes = _dispatch_bounded(pool, worker, trial_ids, max_workers)
            else:
                with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as thread_pool:
                    outcomes = _dispatch_bounded(thread_pool, dispatch_one, trial_ids, max_workers)
        else:
            outcomes = [dispatch_one(trial_id) for trial_id in trial_ids]
        run_ids = trial_ids
        failures = {
            trial_id: diagnostic
            for trial_id, (ok, diagnostic) in zip(trial_ids, outcomes, strict=True)
            if not ok
        }
        failed_ids = list(failures)
        aggregates = self._cross_run_aggregates(run_ids)
        summary = {
            "experiment_id": run_id,
            "runs": run_ids,
            "failed_runs": failed_ids,
            "failures": failures,
            "aggregates": aggregates,
        }
        self.append_run_collection(run_id, "outcomes", summary)
        return {**summary, "status": "completed" if not failed_ids else "partial"}

    def _outcome_groupings(self, run_id: str) -> dict[str, tuple[str, ...]]:
        """Declared grouping keys per outcome id, from the run's pinned plan."""
        run = self.get_run(run_id)
        build_ref = run.get("build") or run.get("build_path")
        if not build_ref:
            return {}
        try:
            plan = compile_outcome_plan(self.resolve_path(build_ref))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        groupings: dict[str, tuple[str, ...]] = {}
        for definition in plan.get("outcomes", []):
            if not isinstance(definition, Mapping) or not definition.get("id"):
                continue
            declared = definition.get("grouping", [])
            if isinstance(declared, str):
                declared = [declared]
            groupings[str(definition["id"])] = tuple(
                str(name) for name in declared if isinstance(name, str)
            )
        return groupings

    def _cross_run_aggregates(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Summarize one measure across realizations OF THE SAME GROUP.

        Outcome rows carry their declared grouping keys (condition, phase, ...).
        Pooling every row of an outcome into one mean mixed distinct
        experimental conditions and phases into a single number, which is
        exactly the conflation cross-realization stability must avoid. Values
        are therefore grouped by their declared keys, and each group reports
        dispersion alongside the mean so within-configuration variation is
        visible rather than averaged away.
        """
        # Reserved row keys are identity/annotation, not measures or groupings.
        reserved = {"outcome_id", "run_id"}
        collected: dict[tuple[Any, ...], dict[str, Any]] = {}
        for trial_id in run_ids:
            declared_groupings = self._outcome_groupings(trial_id)
            for row in self.evaluate_outcomes(trial_id):
                outcome_id = str(row.get("outcome_id", ""))
                group_names = declared_groupings.get(outcome_id, ())
                grouping = {name: row.get(name) for name in group_names if name in row}
                measures = {
                    key: value
                    for key, value in row.items()
                    if key not in reserved
                    and key not in grouping
                    and not key.startswith("group_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                group_key = tuple(
                    sorted((str(k), json.dumps(v, default=str)) for k, v in grouping.items())
                )
                for field, value in measures.items():
                    entry = collected.setdefault(
                        (outcome_id, field, group_key),
                        {
                            "outcome_id": outcome_id,
                            "field": field,
                            "group": dict(grouping),
                            "values": [],
                        },
                    )
                    entry["values"].append(float(value))
        aggregates: list[dict[str, Any]] = []
        for _key, entry in sorted(collected.items(), key=lambda item: str(item[0])):
            values = entry.pop("values")
            aggregates.append(
                {
                    **entry,
                    "runs": len(values),
                    "mean": float(sum(values) / len(values)) if values else None,
                    "stdev": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
            )
        return aggregates

    def trace_run(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        return [*self.persistence.list_events(run_id), *run.get("events", [])]

    def artifacts_for_run(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        artifacts = self.persistence.list_artifacts(run_id)
        decoded = []
        for artifact in artifacts:
            payload = artifact["payload"]
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                value = payload.hex()
            decoded.append({**artifact, "payload": value})
        return [*decoded, *run.get("artifacts", [])]

    def append_run_collection(
        self, run_id: str, field: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self.persistence.append_run_collection(run_id, field, payload)

    def checkpoint_run(self, run_id: str) -> str:
        run = self.get_run(run_id)
        latest = self.persistence.latest_json_state(run_id)
        state = latest[1] if latest else {}
        return self.persistence.create_checkpoint(
            run_id,
            {
                "run": run,
                "state": state,
                "event_count": len(self.trace_run(run_id)),
            },
        )

    def _branch_factors(
        self,
        protocol: Mapping[str, Any],
        source_condition: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve effective factors after applying only branchable overrides.

        For one clean line: branchable overrides are the only changes allowed.
        """
        source_factors = dict(source_condition.get("factors") or {})
        if not overrides:
            return source_factors
        declared: dict[str, Mapping[str, Any]] = {}
        for factor in protocol.get("factors", []):
            if isinstance(factor, Mapping):
                declared[str(factor.get("id", ""))] = factor
        for key, value in overrides.items():
            factor = declared.get(str(key))
            if factor is None:
                raise ValueError(
                    f"REPLAY_CONFIGURATION_INVALID: override '{key}' is not a "
                    "declared protocol factor"
                )
            if not factor.get("branchable"):
                raise ValueError(
                    f"REPLAY_CONFIGURATION_INVALID: factor '{key}' is not declared branchable"
                )
            levels = factor.get("levels")
            if isinstance(levels, list) and value not in levels:
                raise ValueError(
                    f"REPLAY_CONFIGURATION_INVALID: value '{value}' is not a declared level "
                    f"of branchable factor '{key}'"
                )
            source_factors[str(key)] = value
        return source_factors

    def replay_preview(
        self,
        run_id: str,
        *,
        mode: ReplayMode = ReplayMode.FULL,
        artifact_ids: tuple[str, ...] = (),
        boundary: str | None = None,
        overrides: dict[str, Any] | None = None,
        justification: str | None = None,
    ) -> dict[str, Any]:
        """Read-only replay preview (spec §5.4): no provider calls, no child run.

        Returns the normalized boundary, inherited manifest facts, the applied
        factor diff for branch, evidence requirements, warnings, and a
        digest-bound execution token the execution operation must confirm.
        """
        source = self.get_run(run_id)
        build_ref = source.get("build") or source.get("build_path")
        if not build_ref:
            raise ValueError("REPLAY_SOURCE_MISSING: source run has no compiled build")
        self._validate_replay_request(mode, artifact_ids, boundary, justification, overrides)
        build_path = self.resolve_path(build_ref)
        StudyCompiler.verify_build(build_path)
        protocol = json.loads((build_path / "protocol.json").read_text())
        processes = json.loads((build_path / "processes.json").read_text())
        process_ids = {str(process["id"]) for process in processes}
        dependency_map = {
            str(process["id"]): list(process.get("dependencies", {}).get("after", []))
            for process in processes
        }
        if (
            mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH}
            and boundary
            and boundary.startswith("event:")
        ):
            target_event = boundary.split(":", 1)[1]
            if not any(
                str(event.get("event_id", "")) == target_event for event in self.trace_run(run_id)
            ):
                raise ValueError(f"REPLAY_BOUNDARY: unknown source event '{target_event}'")
        normalized_boundary, phase_boundary, event_boundary, frozen_processes = (
            self._normalize_boundary(mode, boundary, process_ids, dependency_map)
        )
        source_condition = {
            "id": str(source.get("condition_id", "base")),
            "factors": dict((source.get("condition") or {}).get("factors", {})),
        }
        effective_factors = self._branch_factors(protocol, source_condition, overrides or {})
        effective_condition = dict(source_condition)
        derived = False
        if mode == ReplayMode.BRANCH:
            source_factor_map = dict(source_condition.get("factors") or {})
            diff = {
                key: value
                for key, value in effective_factors.items()
                if source_factor_map.get(key) != value
            }
            if not diff:
                raise ValueError(
                    "REPLAY_NO_EFFECTIVE_CHANGE: branch overrides produce no effective "
                    "factor change; use partial replay instead"
                )
            derived = True
            effective_condition = {
                "id": f"derived-{source_condition['id']}-branch",
                "factors": dict(effective_factors),
            }
        token_input = {
            "source_run_id": run_id,
            "mode": mode.value,
            "boundary": normalized_boundary,
            "overrides": dict(overrides or {}),
            "justification": justification or "",
            "effective_condition": effective_condition,
            "replication": int(source.get("replication", 1)),
        }
        import hashlib as _hashlib
        import json as _json

        preview_token = _hashlib.sha256(
            _json.dumps(token_input, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        warnings = []
        if derived:
            warnings.append("branch suffix regenerates with the source RNG stream identity")
        if event_boundary is not None:
            warnings.append("event boundary freezes every invocation committed before the event")
        # F9: a phase boundary must lie within the phases actually executed by
        # the source run, and checkpoint availability depends on retained
        # recorded evidence for the prefix — not merely on a non-negative
        # integer being supplied.
        executed_phases: set[int] = set()
        recorded_outputs = bool(self._recorded_process_outputs(run_id, artifact_ids))
        checkpoint_available = False
        if phase_boundary is not None:
            if phase_boundary < 0:
                raise ValueError("REPLAY_BOUNDARY_UNSUPPORTED: phase boundary must be non-negative")
            executed_phases = {int(event.get("phase", 0)) for event in self.trace_run(run_id)}
            # F9: a phase boundary requires actual source execution evidence.
            # An empty trace (a run that never executed) has no checkpoint at
            # any phase, and beyond the last executed phase there is nothing.
            if not executed_phases:
                raise ValueError(
                    "REPLAY_BOUNDARY_UNSUPPORTED: source run has no executed "
                    "phases; there is no checkpoint evidence for phase "
                    f"boundary {phase_boundary}"
                )
            if phase_boundary > (max(executed_phases) + 1):
                raise ValueError(
                    "REPLAY_BOUNDARY_UNSUPPORTED: phase boundary "
                    f"{phase_boundary} is beyond the source run's executed phases "
                    f"(max {max(executed_phases)}); there is no checkpoint evidence "
                    "for a boundary after the run completed"
                )
            frozen_keys = self._frozen_invocation_keys(
                run_id,
                self._recorded_process_outputs(run_id, artifact_ids),
                phase_boundary=phase_boundary,
                event_boundary=event_boundary,
            )
            checkpoint_available = bool(frozen_keys) and recorded_outputs
        return {
            "source_run_id": run_id,
            "mode": mode.value,
            "boundary": normalized_boundary,
            "inherited_condition": source_condition,
            "effective_condition": effective_condition,
            "effective_factors": dict(effective_factors),
            "inherited_replication": int(source.get("replication", 1)),
            "evidence_requirements": {
                "recorded_outputs": recorded_outputs,
                "checkpoint_available": checkpoint_available,
            },
            "warnings": warnings,
            "preview_token": preview_token,
        }

    @staticmethod
    def _validate_replay_request(
        mode: ReplayMode,
        artifact_ids: tuple[str, ...],
        boundary: str | None,
        justification: str | None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if overrides and mode != ReplayMode.BRANCH:
            raise ValueError(
                "REPLAY_CONFIGURATION_INVALID: overrides are only accepted for "
                f"branch replay, not {mode.value}"
            )
        if mode == ReplayMode.PARTIAL and not boundary:
            raise ValueError("partial replay requires a boundary")
        if mode == ReplayMode.BRANCH:
            if not boundary:
                raise ValueError("branch replay requires a boundary")
            if not justification:
                raise ValueError("branch replay requires a justification")
        if mode == ReplayMode.FULL and (artifact_ids or boundary):
            raise ValueError("full replay does not accept artifact_ids or a boundary")
        if artifact_ids and mode != ReplayMode.ARTIFACT:
            raise ValueError(
                f"artifact_ids are only accepted for artifact replay, not {mode.value}"
            )

    @staticmethod
    def _normalize_boundary(
        mode: ReplayMode,
        boundary: str | None,
        process_ids: set[str],
        dependency_map: Mapping[str, list[str]] | None = None,
    ) -> tuple[str | None, int | None, str | None, set[str]]:
        """Parse a boundary into a normalized description and its parts.

        First release accepts only ``phase:N`` and ``event:ID`` boundaries
        plus a dependency-closed comma-separated process selection; any other
        form fails as unsupported (spec §5.3). A negative phase boundary or a
        selection with an unsatisfied dependency is rejected rather than
        silently treated as a valid checkpoint.
        """
        phase_boundary: int | None = None
        event_boundary: str | None = None
        frozen_processes: set[str] = set()
        if mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH} and boundary:
            if boundary.startswith("phase:"):
                try:
                    phase_boundary = int(boundary.split(":", 1)[1])
                except ValueError as exc:
                    raise ValueError("REPLAY_BOUNDARY: phase boundary must be an integer") from exc
                if phase_boundary < 0:
                    raise ValueError(
                        "REPLAY_BOUNDARY_UNSUPPORTED: phase boundary must be non-negative"
                    )
            elif boundary.startswith("event:"):
                event_boundary = boundary.split(":", 1)[1]
            else:
                names = {name.strip() for name in boundary.split(",") if name.strip()}
                unknown = names - process_ids
                if unknown:
                    raise ValueError(
                        "REPLAY_BOUNDARY_UNSUPPORTED: unknown process selection: "
                        + ", ".join(sorted(unknown))
                    )
                # F9: a process selection is only a valid prefix boundary when
                # it is dependency-closed — every dependency of a frozen
                # process must also be frozen.
                for process_id, deps in (dependency_map or {}).items():
                    if process_id not in names:
                        continue
                    missing_deps = [dep for dep in deps if dep not in names]
                    if missing_deps:
                        raise ValueError(
                            "REPLAY_BOUNDARY_UNSUPPORTED: process selection '"
                            + process_id
                            + "' freezes a process whose dependencies are not frozen: "
                            + ", ".join(sorted(missing_deps))
                        )
                frozen_processes = names
        else:
            frozen_processes = set()
        return boundary, phase_boundary, event_boundary, frozen_processes

    @staticmethod
    def derive_manifest_seeds(
        randomness: Mapping[str, Any],
        random_streams: Any,
        matching: Mapping[str, Any] | None = None,
    ) -> dict[str, int]:
        """Derive every named stream seed from a manifest's own recorded inputs.

        One derivation shared by manifest construction and by import
        verification, so a manifest's ``seeds`` can always be checked against
        the identity it claims.
        """
        matching = matching if isinstance(matching, Mapping) else {}
        shared_streams = matching.get("shared_streams", [])
        matched = matching.get("enabled") is True and isinstance(shared_streams, list)
        identity = str(randomness["run_id"])
        experiment_id = str(randomness.get("experiment_id", ""))
        condition_id = str(randomness.get("condition_id", "base"))
        replication = int(randomness.get("replication", 1))
        seeds: dict[str, int] = {
            "conventional": derive_seed(
                0,
                identity,
                "run-manifest",
                experiment_id=experiment_id,
                condition_id=condition_id,
                replication=replication,
                matching_key=(
                    "conventional" if matched and "conventional" in shared_streams else None
                ),
            )
        }
        for stream in random_streams or []:
            if isinstance(stream, Mapping) and stream.get("id"):
                stream_id = str(stream["id"])
                seeds[stream_id] = derive_seed(
                    int(stream.get("seed", 0)),
                    identity,
                    stream_id,
                    experiment_id=experiment_id,
                    condition_id=condition_id,
                    replication=replication,
                    matching_key=(stream_id if matched and stream_id in shared_streams else None),
                )
        return seeds

    @classmethod
    def verify_manifest_seeds(cls, manifest: Mapping[str, Any]) -> None:
        """Reject a manifest whose seeds cannot be derived from its own identity.

        A manifest is the record of the configuration a realization arose from.
        If its ``seeds`` do not follow from the identity it states, it cannot
        reconstruct the run it names — the case a rekeyed or hand-edited copy
        produces. Manifests predating ``randomness_inputs`` carry no recorded
        identity and are left to the legacy path.
        """
        randomness = manifest.get("randomness_inputs")
        recorded = manifest.get("seeds")
        if not isinstance(randomness, Mapping) or not isinstance(recorded, Mapping):
            return
        streams = manifest.get("random_streams")
        if isinstance(manifest.get("matching"), Mapping):
            candidates = [cls.derive_manifest_seeds(randomness, streams, manifest["matching"])]
        else:
            # A manifest written before the matching block was retained cannot
            # say whether its streams were matched, and matched streams derive
            # differently. Accept either reading rather than rejecting evidence
            # this check simply cannot reconstruct.
            every_stream = [
                str(stream["id"])
                for stream in streams or []
                if isinstance(stream, Mapping) and stream.get("id")
            ]
            candidates = [
                cls.derive_manifest_seeds(randomness, streams, None),
                cls.derive_manifest_seeds(
                    randomness,
                    streams,
                    {"enabled": True, "shared_streams": [*every_stream, "conventional"]},
                ),
            ]
        for expected in candidates:
            mismatched = sorted(
                stream
                for stream, value in recorded.items()
                if stream in expected and int(value) != int(expected[stream])
            )
            if not mismatched:
                return
        raise ValueError(
            "MANIFEST_SEED_MISMATCH: seeds "
            f"{mismatched} do not derive from the recorded identity "
            f"'{randomness.get('run_id')}'; the manifest cannot reconstruct "
            "the run it names"
        )

    def _randomness_inputs(self, run: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve and freeze seed inputs independently of mutable local lineage.

        New bundles carry these inputs, so a foreign root need not exist in
        the receiving workspace. Legacy local children can still resolve them
        through their parent; imported legacy roots use their retained manifest.
        """
        current = run
        seen: set[str] = set()
        while True:
            manifest = current.get("manifest") or {}
            frozen = manifest.get("randomness_inputs")
            if isinstance(frozen, Mapping):
                return {
                    "run_id": str(frozen["run_id"]),
                    "experiment_id": str(frozen["experiment_id"]),
                    "condition_id": str(frozen["condition_id"]),
                    "replication": int(frozen["replication"]),
                }
            parent = current.get("replay_of")
            if not parent:
                return {
                    "run_id": self._root_source_run_id(current),
                    "experiment_id": str(
                        current.get("experiment_id")
                        or manifest.get("experiment_id")
                        or (manifest.get("execution") or {}).get("origin_experiment_id")
                        or ""
                    ),
                    "condition_id": str(
                        current.get("condition_id", manifest.get("condition_id", "base"))
                    ),
                    "replication": int(current.get("replication", manifest.get("replication", 1))),
                }
            if str(parent) in seen:
                raise ValueError("REPLAY_LINEAGE_INVALID: cyclic randomness lineage")
            seen.add(str(parent))
            current = self.get_run(str(parent))

    def _root_source_run_id(self, run: Mapping[str, Any]) -> str:
        """The original source run id for a replay lineage (F8/§2.2 fix).

        Randomness must derive from the root source identity, never from an
        intermediate replay child, a derived branch identity, or the LOCAL id
        used when a run was imported. Imported runs preserve the original
        source id in ``manifest.origin.source_run_id``; when that id does not
        exist in this workspace (a foreign import), the preserved id itself is
        the randomness identity.
        """
        source_id = str(run.get("replay_of") or "")
        if not source_id:
            source_id = str(
                ((run.get("manifest") or {}).get("origin") or {}).get("source_run_id") or ""
            )
        if not source_id:
            return str(run.get("id", ""))
        seen: set[str] = set()
        current = source_id
        while current and current not in seen:
            seen.add(current)
            try:
                source = self.get_run(current)
            except KeyError:
                # A preserved foreign source id: no local record to walk; it is
                # the root identity.
                return current
            next_hop = str(source.get("replay_of") or "")
            if not next_hop:
                # An imported intermediate: continue through its preserved
                # original id (which may itself be foreign or a local root).
                next_hop = str(
                    ((source.get("manifest") or {}).get("origin") or {}).get("source_run_id") or ""
                )
            if not next_hop or next_hop == current:
                return current
            current = next_hop
        return current

    def _generative_process_ids(self, build_ref: str | Path) -> set[str]:
        """Process ids that invoke an LLM: generative AND semantic-evaluator.

        F4: recorded-output retrieval and full-replay substitution must cover
        every LLM-invoking executor, not only ``mode == generative``.
        F3: build refs may be workspace-relative (e.g. restored imports), so
        resolve against this workspace before reading the build files.
        """
        build_path = self.resolve_path(
            Path(build_ref) if isinstance(build_ref, Path) else str(build_ref)
        )
        processes = json.loads((build_path / "processes.json").read_text())
        return {
            str(process["id"])
            for process in processes
            if process.get("executor", {}).get("mode") in {"generative", "semantic-evaluator"}
        }

    def _reconstruct_imported_build(
        self, source_path: Path, bundle_manifest: Mapping[str, Any] | None, target_id: str
    ) -> tuple[str, bool]:
        """Restore a verified executable build from a reproducibility bundle.

        F3: when the bundle carries the executable build files, reconstruct
        them under ``builds/``, write a fresh integrity manifest, register the
        build so replay/reexecution work, and return (build_ref, restored).
        When no executable files are present, import stays exploration-only.
        """
        executable_files = (
            "processes.json",
            "process_graph.json",
            "context_policies.json",
            "state_model.json",
            "artifact_catalog.json",
            "outcome_plan.json",
            "protocol.json",
            "build_manifest.json",
            "validation_report.json",
        )
        if not all((source_path / name).is_file() for name in executable_files):
            return "", False
        build_manifest = json.loads((source_path / "build_manifest.json").read_text())
        build_hash = str(build_manifest.get("build_hash", ""))
        if not build_hash:
            raise ValueError(
                "IMPORT_BUILD: reproducibility bundle has no build_hash in build_manifest.json"
            )
        build_dir = self.workspace / "builds" / f"{target_id}-imported-{build_hash[:8]}"
        build_dir.mkdir(parents=True, exist_ok=True)
        for name in executable_files:
            target = build_dir / name
            target.write_bytes((source_path / name).read_bytes())
        for name in (
            "model_profiles.json",
            "prompt_templates.json",
            "initialization.json",
            "data_manifest.json",
            "schemas.json",
            "package_closure.json",
            "theory_execution_plan.json",
        ):
            if (source_path / name).is_file():
                (build_dir / name).write_bytes((source_path / name).read_bytes())
        # F3: restore the run-pinned package closure bytes from the bundle's
        # ``package/`` tree into ``closure/`` so the reconstructed build can be
        # re-exported in reproducibility mode (package_closure.json lists the
        # package-relative asset paths used by _write_bundle).
        closure_manifest_path = build_dir / "package_closure.json"
        if closure_manifest_path.is_file():
            try:
                closure_manifest = json.loads(closure_manifest_path.read_text())
                assets = closure_manifest.get("assets") or []
            except json.JSONDecodeError:
                assets = []
            for asset in assets:
                if not isinstance(asset, Mapping):
                    continue
                relative = str(asset.get("path", ""))
                if not _is_contained_relative(relative):
                    raise ValueError(f"IMPORT_BUILD: unsafe closure asset path '{relative}'")
                source_asset = source_path / "package" / relative
                if not source_asset.is_file() or source_asset.is_symlink():
                    raise ValueError(f"IMPORT_BUILD: closure asset missing in bundle: {relative}")
                target = build_dir / "closure" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source_asset.read_bytes())
                # F2: empirical data assets must also land in the build's
                # ``data/`` directory — the runtime reads them there when
                # initialization.mode is "empirical" (data_source is
                # relative to the build root, e.g. data/population.csv), and
                # the closure tree alone is not consulted at dispatch time.
                if relative.startswith("data/"):
                    data_target = build_dir / relative
                    data_target.parent.mkdir(parents=True, exist_ok=True)
                    data_target.write_bytes(source_asset.read_bytes())
        # Regenerate the integrity manifest for the reconstructed build and
        # verify it (mirrors the compiler's per-file digest table).
        integrity: dict[str, str] = {}
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                integrity[path.relative_to(build_dir).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        if "build_manifest.json" in integrity:
            integrity["manifest_hash"] = integrity["build_manifest.json"]
        (build_dir / "integrity_manifest.json").write_text(
            json.dumps(integrity, sort_keys=True, indent=2) + "\n"
        )
        StudyCompiler.verify_build(build_dir)
        self.persistence.record_study_build(
            {
                "build_hash": build_hash,
                "study_id": str(build_manifest.get("study_id", "")),
                "package_version": 0,
                "package_content_hash": "",
                "compiler_version": str(build_manifest.get("compiler_version", "1.0")),
                "created_at": datetime.now(UTC).isoformat(),
                "path": str(build_dir),
            }
        )
        return str(build_dir.relative_to(self.workspace)), True

    def replay_run(
        self,
        run_id: str,
        *,
        mode: ReplayMode = ReplayMode.FULL,
        artifact_ids: tuple[str, ...] = (),
        boundary: str | None = None,
        overrides: dict[str, Any] | None = None,
        justification: str | None = None,
        preview_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute a replay (Section 12.3, REP-003) under spec §5.1 semantics.

        FULL reuses the recorded invocations for the complete trajectory and
        fails if any required record is unavailable; it rejects overrides.
        ARTIFACT retrieves the selected retained artifacts only: no child run
        is created and no provider is invoked, so it is never described as a
        new simulation. PARTIAL/BRANCH freeze the recorded prefix at
        ``boundary`` and re-execute the suffix; BRANCH additionally requires a
        justification and a digest-bound preview confirmation. PARTIAL and
        BRANCH inherit the source condition, factors and replication, and only
        declared branchable protocol factors may change in BRANCH (RPL-001..004).
        """
        source = self.get_run(run_id)
        if mode == ReplayMode.FULL and overrides:
            raise ValueError(
                "REPLAY_CONFIGURATION_INVALID: full replay reuses the recorded "
                "trajectory and does not accept overrides"
            )
        build_ref = source.get("build") or source.get("build_path")
        if not build_ref:
            raise ValueError("REPLAY_SOURCE_MISSING: source run has no compiled build")
        if mode == ReplayMode.ARTIFACT:
            # Spec §5.1: artifact replay is retrieval of retained artifacts,
            # not a new simulation. No child run, no provider, no preview, and
            # no intervention configuration (F10: overrides/justification are
            # rejected on the execution path just as the preview does).
            if overrides:
                raise ValueError(
                    "REPLAY_CONFIGURATION_INVALID: overrides are only accepted for "
                    "branch replay, not artifact"
                )
            if justification or boundary:
                raise ValueError(
                    "REPLAY_CONFIGURATION_INVALID: artifact replay is retrieval-only "
                    "and does not accept a boundary or justification"
                )
            retained = []
            if artifact_ids:
                retained = [
                    artifact
                    for artifact in self.artifacts_for_run(run_id)
                    if artifact.get("artifact_id") in artifact_ids
                ]
            else:
                recorded = self._recorded_process_outputs(run_id, ())
                generative_ids = self._generative_process_ids(build_ref)
                retained_processes = sorted(set(recorded) & generative_ids)
                if not retained_processes:
                    raise ValueError(
                        "REPLAY_EVIDENCE_INCOMPLETE: artifact replay has no retained "
                        "generative outputs to retrieve"
                    )
                retained = [
                    artifact
                    for artifact in self.artifacts_for_run(run_id)
                    if isinstance(artifact.get("payload"), dict)
                    and artifact["payload"].get("process_id") in retained_processes
                ]
            return {
                "run_id": run_id,
                "source_run_id": run_id,
                "mode": mode.value,
                "artifacts": retained,
                "overrides": {},
                "lineage": {
                    "source_run_id": run_id,
                    "mode": mode.value,
                    "retrieval": True,
                },
            }
        self._validate_replay_request(mode, artifact_ids, boundary, justification, overrides)
        build_path = self.resolve_path(build_ref)
        StudyCompiler.verify_build(build_path)
        processes = json.loads((build_path / "processes.json").read_text())
        process_ids = {str(process["id"]) for process in processes}
        dependency_map = {
            str(process["id"]): list(process.get("dependencies", {}).get("after", []))
            for process in processes
        }
        normalized_boundary, phase_boundary, event_boundary, frozen_processes = (
            self._normalize_boundary(mode, boundary, process_ids, dependency_map)
        )
        # F9: a phase boundary beyond the source's executed phases must fail
        # even on the execution path (preview and execution agree). The
        # terminal state (one past the last executed phase) is the final
        # valid boundary; anything further has no retained evidence.
        if phase_boundary is not None and mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH}:
            executed_phases = {int(e.get("phase", 0)) for e in self.trace_run(run_id)}
            if not executed_phases:
                raise ValueError(
                    "REPLAY_BOUNDARY_UNSUPPORTED: source run has no executed "
                    f"phases; there is no checkpoint evidence for phase boundary {phase_boundary}"
                )
            if phase_boundary > (max(executed_phases) + 1):
                raise ValueError(
                    "REPLAY_BOUNDARY_UNSUPPORTED: phase boundary "
                    f"{phase_boundary} is beyond the source run's executed phases "
                    f"(max {max(executed_phases)})"
                )
        source_condition = {
            "id": str(source.get("condition_id", "base")),
            "factors": dict((source.get("condition") or {}).get("factors", {})),
        }
        protocol = json.loads((build_path / "protocol.json").read_text())
        effective_factors = self._branch_factors(protocol, source_condition, overrides or {})
        effective_condition = dict(source_condition)
        if mode == ReplayMode.BRANCH:
            source_factor_map = dict(source_condition.get("factors") or {})
            diff = {
                key: value
                for key, value in effective_factors.items()
                if source_factor_map.get(key) != value
            }
            if not diff:
                raise ValueError(
                    "REPLAY_NO_EFFECTIVE_CHANGE: branch overrides produce no effective "
                    "factor change; use partial replay instead"
                )
            effective_condition = {
                "id": f"derived-{source_condition['id']}-branch",
                "factors": dict(effective_factors),
            }
            if not preview_token:
                raise ValueError(
                    "REPLAY_PREVIEW_STALE: branch replay requires a confirmed preview "
                    "token from replay_preview"
                )
        if mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH} and not preview_token:
            raise ValueError(
                "REPLAY_PREVIEW_STALE: partial/branch replay requires the confirmed "
                "preview token from replay_preview"
            )
        if preview_token is not None:
            expected = self.replay_preview(
                run_id,
                mode=mode,
                artifact_ids=artifact_ids,
                boundary=normalized_boundary,
                overrides=overrides,
                justification=justification,
            )["preview_token"]
            if preview_token != expected:
                raise ValueError(
                    "REPLAY_PREVIEW_STALE: preview token does not match the requested "
                    "configuration; re-run replay_preview and confirm"
                )
        inherited_replication = int(source.get("replication", 1))
        # F4: recorded substitution must cover every executor that invokes an
        # LLM — generative AND semantic-evaluator — so FULL replay never makes
        # a fresh provider call for either.
        llm_ids = {
            str(process["id"])
            for process in processes
            if process.get("executor", {}).get("mode") in {"generative", "semantic-evaluator"}
        }
        recorded = self._recorded_process_outputs(run_id, artifact_ids)
        frozen_keys = self._frozen_invocation_keys(
            run_id, recorded, phase_boundary=phase_boundary, event_boundary=event_boundary
        )
        # The phase up to which an unrecorded generative invocation counts as
        # inside the frozen prefix. A phase boundary states it directly; an
        # event boundary resolves to the phase that event was committed in.
        guard_phase: int | None = phase_boundary
        if guard_phase is None and event_boundary is not None:
            for event in self.trace_run(run_id):
                if str(event.get("event_id", "")) == event_boundary:
                    guard_phase = int(event.get("phase", 0))
                    break
        executor_overrides: dict[str, Any] = {}
        if mode == ReplayMode.FULL:
            # Spec §5.1: FULL reuses recorded invocations for the complete
            # trajectory and fails if required records are unavailable.
            missing = sorted(llm_ids - set(recorded))
            if missing:
                raise ValueError(
                    "REPLAY_EVIDENCE_INCOMPLETE: full replay requires recorded "
                    "invocations for: " + ", ".join(missing)
                )
            for process_id in sorted(llm_ids):
                executor_overrides[process_id] = _RecordedExecutor(
                    recorded[process_id]
                    if isinstance(recorded[process_id], list)
                    else [
                        {
                            "outputs": recorded[process_id],
                            "phase": 0,
                            "attempt": 1,
                            "actors": (),
                            "order": 0,
                        }
                    ],
                    source_run_id=run_id,
                    process_id=process_id,
                )
        # Built lazily: the live executors a replay child would otherwise use.
        live_executors: dict[str, Any] | None = None

        def _live(process_id: str) -> Any:
            nonlocal live_executors
            if live_executors is None:
                live_executors = self._build_executors(
                    build_path,
                    processes,
                    self._build_schema_catalog(build_path),
                    self._build_model_profiles(build_path),
                    self._build_prompt_templates(build_path),
                )
            return live_executors.get(process_id)

        for process in processes:
            process_id = str(process["id"])
            if mode == ReplayMode.FULL:
                continue
            guard_boundary = (
                guard_phase
                if mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH}
                and guard_phase is not None
                and process_id in llm_ids
                else None
            )
            if process_id not in recorded:
                # No recorded evidence at all. Its live executor would still be
                # installed, so guard the frozen prefix explicitly.
                if guard_boundary is not None:
                    executor_overrides[process_id] = _FrozenPrefixGuard(
                        _live(process_id),
                        source_run_id=run_id,
                        process_id=process_id,
                        phase_boundary=guard_boundary,
                        inclusive=phase_boundary is None,
                    )
                continue
            if mode in {ReplayMode.PARTIAL, ReplayMode.BRANCH}:
                if phase_boundary is not None or event_boundary is not None:
                    records = [dict(r) for r in recorded[process_id]]
                    matching = [
                        r
                        for r in records
                        if (process_id, r["phase"], r["attempt"], r["actors"]) in frozen_keys
                    ]
                    # Every invocation the SOURCE performed, frozen or not. An
                    # invocation the source made after the boundary re-executes
                    # live; one it never made is a divergence if it lands inside
                    # the prefix. Source position does not survive a branch, so
                    # a recorded suffix process is still bound by the boundary.
                    known = {(r["phase"], r["attempt"], r["actors"]) for r in records}
                    if not matching:
                        if guard_boundary is not None:
                            executor_overrides[process_id] = _SelectiveExecutor(
                                [],
                                frozen_keys=set(),
                                fallback=_live(process_id),
                                source_run_id=run_id,
                                process_id=process_id,
                                phase_boundary=guard_boundary,
                                inclusive=phase_boundary is None,
                                known_keys=known,
                            )
                        continue
                    executor_overrides[process_id] = _SelectiveExecutor(
                        matching,
                        known_keys=known,
                        phase_boundary=guard_boundary,
                        inclusive=phase_boundary is None,
                        frozen_keys={(r["phase"], r["attempt"], r["actors"]) for r in matching},
                        fallback=_live(process_id),
                        source_run_id=run_id,
                        process_id=process_id,
                    )
                    continue
                if process_id not in frozen_processes:
                    continue
            executor_overrides[process_id] = (
                _RecordedExecutor(recorded[process_id], source_run_id=run_id, process_id=process_id)
                if isinstance(recorded[process_id], list)
                else _RecordedExecutor(
                    [
                        {
                            "outputs": recorded[process_id],
                            "phase": 0,
                            "attempt": 1,
                            "actors": (),
                            "order": 0,
                        }
                    ],
                    source_run_id=run_id,
                    process_id=process_id,
                )
            )
        # RPL-T06: the child id is derived from the confirmed configuration, so a
        # duplicate confirmed submission returns the same child instead of a
        # fresh realization (idempotent retries).
        import hashlib as _replay_hash
        import json as _replay_json

        child_key = _replay_hash.sha256(
            _replay_json.dumps(
                {
                    "source_run_id": run_id,
                    "mode": mode.value,
                    "boundary": boundary,
                    "overrides": dict(overrides or {}),
                    "justification": justification or "",
                    "effective_condition": dict(effective_condition),
                    "replication": inherited_replication,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:12]
        replay_run_id = f"{run_id}-replay-{child_key}"
        try:
            self.create_run(
                {
                    "id": replay_run_id,
                    "study_id": source.get("study_id"),
                    "build": build_ref,
                    "replay_of": run_id,
                    "replay_mode": mode.value,
                    "replay_boundary": boundary,
                    "replay_justification": justification,
                    "replay_overrides": {
                        "requested": dict(overrides or {}),
                        "applied": dict(effective_factors),
                    },
                    # Effective-configuration inheritance (RPL-001): the branch keeps
                    # the source condition/factors/replication unless a branchable
                    # factor override produced a derived condition.
                    "condition_id": effective_condition["id"],
                    "condition": dict(effective_condition),
                    "replication": inherited_replication,
                }
            )
        except Exception as exc:
            if "ALREADY_EXISTS" in str(exc):
                # Duplicate confirmed request: return the existing child run.
                existing = self.get_run(replay_run_id)
                return {
                    "run_id": replay_run_id,
                    "source_run_id": run_id,
                    "mode": mode.value,
                    "artifacts": self.artifacts_for_run(replay_run_id),
                    "overrides": dict(overrides or {}),
                    "lineage": {
                        "source_run_id": run_id,
                        "mode": mode.value,
                        "boundary": boundary,
                        "justification": justification,
                        "effective_condition": dict(existing.get("condition", {})),
                        "replication": int(existing.get("replication", 1)),
                    },
                }
            raise
        self.execute_run(replay_run_id, executor_overrides=executor_overrides)
        return {
            "run_id": replay_run_id,
            "source_run_id": run_id,
            "mode": mode.value,
            "artifacts": self.artifacts_for_run(replay_run_id),
            "overrides": dict(overrides or {}),
            "lineage": {
                "source_run_id": run_id,
                "mode": mode.value,
                "boundary": boundary,
                "justification": justification,
                "effective_condition": dict(effective_condition),
                "replication": inherited_replication,
            },
        }

    def _recorded_process_outputs(
        self, run_id: str, artifact_ids: tuple[str, ...]
    ) -> dict[str, Any]:
        """Per-invocation recorded outputs keyed by process ID (review finding 5).

        Replay substitutions are keyed by the invocation coordinates
        (phase, attempt, actors) recorded with each committed artifact, so a
        process that ran for multiple rounds/actors replays each invocation
        with its own recorded output instead of the last one.
        """
        recorded: dict[str, Any] = {}
        effects = self._committed_effects(run_id)
        for artifact in self.artifacts_for_run(run_id):
            if artifact_ids and artifact["artifact_id"] not in artifact_ids:
                continue
            payload = artifact["payload"]
            if not isinstance(payload, dict) or "outputs" not in payload:
                continue
            process_id = str(payload.get("process_id", ""))
            if not process_id:
                continue
            entry = {
                "outputs": payload["outputs"],
                "phase": int(payload.get("phase", 0)),
                "attempt": int(payload.get("attempt", 1)),
                "actors": tuple(str(item) for item in payload.get("actors", [])),
                "order": len(recorded.get(process_id, [])),
            }
            committed = effects.get(
                (str(payload.get("invocation_id", "")), int(payload.get("attempt", 1)))
            )
            if committed:
                entry.update(committed)
            recorded.setdefault(process_id, []).append(entry)
        return recorded

    def _committed_effects(self, run_id: str) -> dict[tuple[str, int], dict[str, Any]]:
        """The consequences each recorded invocation actually committed.

        Outputs alone do not describe an invocation: its state changes, emitted
        events and scheduling effects are what the run carried forward. Replaying
        outputs only made a frozen prefix drop those consequences — a state
        transition that produced no artifact value replayed as a no-op, so a
        partial replay silently ended in a different state than its source while
        reporting success.
        """
        committed: dict[tuple[str, int], dict[str, Any]] = {}
        for event in self.trace_run(run_id):
            if event.get("kind") != "process_completed":
                continue
            invocation = str(event.get("invocation_id", ""))
            if not invocation:
                continue
            delta = event.get("state_delta")
            # A None value marks a removed key, which the state model cannot
            # represent; only real assignments are replayable.
            state_effects = (
                {key: value for key, value in delta.items() if value is not None}
                if isinstance(delta, Mapping)
                else {}
            )
            committed[(invocation, int(event.get("attempt", 1)))] = {
                "state_effects": state_effects,
                "events": tuple(
                    dict(item) for item in event.get("events", []) if isinstance(item, Mapping)
                ),
                "scheduling_effects": tuple(
                    dict(item)
                    for item in event.get("scheduling_effects", [])
                    if isinstance(item, Mapping)
                ),
            }
        return committed

    def _duckdb_outcome_rows(
        self, plan: OutcomePlan, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], str] | None:
        """DuckDB as the default aggregate engine when enabled (AW-11).

        Opt-in via ``GENESIS_USE_DUCKDB=1``: aggregate plans with a single
        grouping key evaluate through DuckDB; any conversion or schema drift
        falls back to the in-process engine for exact parity.
        """
        if os.environ.get("GENESIS_USE_DUCKDB", "0") != "1":
            return None
        if plan.operation != "aggregate" or plan.window is not None or not rows:
            return None
        # The SQL path has no missingness policy: it always excludes nulls.
        # Running it for a plan that declares "zero" would report a different
        # number for the same declared measurement depending on an environment
        # variable, so that plan stays on the in-process engine.
        if plan.missingness != "exclude":
            return None
        if plan.filters:
            return None
        self.last_outcome_engine = "python"
        declared_keys = (
            (plan.group_by,) if isinstance(plan.group_by, str) else tuple(plan.group_by or ())
        )
        # The SQL path groups by a single column. A multi-key grouping passed
        # through it silently lost its grouping entirely and pooled every
        # condition into one row, so those plans stay on the in-process engine.
        if len(declared_keys) > 1:
            return None
        group_key = declared_keys[0] if declared_keys else None
        try:
            from genesis.analysis import duckdb_aggregate

            return duckdb_aggregate(
                _parquet_safe(rows),
                select=plan.select,
                op=plan.aggregation,
                group_by=group_key,
            ), "duckdb"
        except Exception:
            self.last_outcome_engine = "python"
            return None

    def evaluate_outcomes(self, run_id: str) -> list[dict[str, Any]]:
        run = self.get_run(run_id)
        if "imported_outcomes" in run:
            return cast(list[dict[str, Any]], run["imported_outcomes"])
        build_ref = run.get("build") or run.get("build_path")
        if not build_ref:
            # Imported runs may carry only the build identity (build_hash), not
            # the path; resolve the path from the recorded build registry.
            build_hash = (run.get("manifest") or {}).get("build_hash")
            if build_hash:
                row = self.persistence.connection.execute(
                    "SELECT payload_json FROM study_builds WHERE build_hash=?", (build_hash,)
                ).fetchone()
                if row:
                    try:
                        build_path = json.loads(row[0]).get("path")
                    except (TypeError, json.JSONDecodeError):
                        build_path = None
                    if build_path:
                        build_ref = str(build_path)
        if not build_ref:
            return []
        build_path = self.resolve_path(build_ref)
        outcome_plan = compile_outcome_plan(build_path)
        definitions = outcome_plan["outcomes"]
        schema_catalog = self._build_schema_catalog(build_path)
        # A build compiled before the package dialect was tightened cannot bind
        # a catalog. Refusing to read it at all would make already-recorded,
        # integrity-verified evidence unevaluable and unexportable, so the
        # catalog is optional here exactly as it is on the execution path; an
        # outcome that actually asks for schema validation is refused below by
        # name, so nothing is silently left unvalidated.
        outcome_catalog: PackageSchemaCatalog | None = None
        catalog_error: str | None = None
        if schema_catalog:
            try:
                outcome_catalog = PackageSchemaCatalog(schema_catalog)
            except SchemaValidationError as exc:
                catalog_error = str(exc)
        event_rows = []
        for event in self.trace_run(run_id):
            row = dict(event)
            row["time"] = event.get("phase", 0)
            event_rows.append(row)
        artifacts = self.artifacts_for_run(run_id)
        declared_ids = {
            str(payload["declared_artifact_id"])
            for artifact in artifacts
            if isinstance((payload := artifact["payload"]), dict)
            and isinstance(payload.get("declared_artifact_id"), str)
        }
        artifact_rows = []
        artifact_sources: dict[str, list[dict[str, Any]]] = {}
        for artifact in artifacts:
            payload = artifact["payload"]
            if not isinstance(payload, dict):
                continue
            artifact_row: dict[str, Any] = {
                "artifact_id": artifact["artifact_id"],
                "run_id": run_id,
                "time": 0,
                "condition_id": run.get("condition_id", "base"),
                "replication": int(run.get("replication", 1)),
            }
            artifact_row.setdefault("invocation_id", payload.get("invocation_id"))
            artifact_row.setdefault("process_id", payload.get("process_id"))
            artifact_row.setdefault("phase", payload.get("phase"))
            # Synthetic trace rows defer to the declared row for declared keys.
            artifact_row.update(
                {
                    key: value
                    for key, value in payload.get("outputs", {}).items()
                    if key not in declared_ids
                }
            )
            if "value" in payload:
                artifact_row["value"] = payload["value"]
                if isinstance(payload["value"], dict):
                    artifact_row.update(payload["value"])
                if isinstance(payload.get("declared_artifact_id"), str):
                    artifact_row[payload["declared_artifact_id"]] = payload["value"]
            artifact_rows.append(artifact_row)
            declared_id = payload.get("declared_artifact_id")
            if isinstance(declared_id, str):
                artifact_sources.setdefault(declared_id, []).append(artifact_row)
        state_rows = self._round_annotated_state(run_id, run, event_rows)
        # OUT-005: generic outcome derivation. When the outcome plan declares
        # datasets, rows are materialized ONCE by the fixed operation registry
        # and exposed under the dataset ids; raw event rows are never enlarged
        # by derived rows, so an outcome counting events counts exactly the
        # raw evidence (F4 fix: no double-counting). The legacy flat-row
        # synthesis remains for packages that predate the datasets contract.
        raw_event_rows = event_rows
        dataset_rows: dict[str, list[dict[str, Any]]] = {}
        if outcome_plan.get("datasets"):
            dataset_rows = materialize_datasets(
                outcome_plan,
                {
                    "events": raw_event_rows,
                    "artifacts": artifacts,
                    "state": state_rows,
                },
            )
        lineage_rows = [
            {
                "event_id": event.get("event_id"),
                "invocation_id": event.get("invocation_id"),
                "process_id": event.get("process_id"),
                "phase": event.get("phase", 0),
                "parent_events": event.get("parent_events", []),
            }
            for event in raw_event_rows
        ]
        event_rows = raw_event_rows
        sources: dict[str, list[dict[str, Any]]] = {
            "events": event_rows,
            "artifacts": artifact_rows,
            "state": state_rows,
            "lineage": lineage_rows,
            **artifact_sources,
        }
        if outcome_plan.get("datasets"):
            sources.update(dataset_rows)
        else:
            # Legacy flat-row packages aggregate over a synthesized event
            # view; keep that behaviour only for pre-dataset packages.
            derived = self._legacy_derived_rows(event_rows, artifact_rows)
            if derived:
                event_rows = derived + event_rows
                sources["events"] = event_rows
        results: list[dict[str, Any]] = []
        for definition in definitions:
            join = definition.get("join")
            joined_name: str | None = None
            if isinstance(join, dict):
                left = str(join.get("left", "events"))
                right = str(join.get("right", "artifacts"))
                on = str(join.get("on", "invocation_id"))
                merged_rows: list[dict[str, Any]] = []
                for left_row in sources.get(left, []):
                    for right_row in sources.get(right, []):
                        if left_row.get(on) == right_row.get(on):
                            combined: dict[str, Any] = dict(left_row)
                            combined.update(right_row)
                            merged_rows.append(combined)
                joined_name = f"{left}+{right}"
                sources[joined_name] = merged_rows
            aggregation = definition.get("aggregation", {})
            op_type = str(
                aggregation.get(
                    "op", aggregation.get("type", aggregation.get("operation", "count"))
                )
            )
            field = aggregation.get("field") or aggregation.get("select")
            if not field and op_type == "count":
                field = "value"
            if not field:
                continue
            source = definition.get("source", "events")
            source_name = source[0] if isinstance(source, list) and source else source
            if joined_name is not None:
                # When a join is declared it is the evaluation source,
                # regardless of whether the named source also exists.
                source_name = joined_name
            grouping = definition.get("grouping", [])
            if op_type in {"trajectory", "distribution"}:
                operation, agg_op = op_type, "count"
            else:
                operation, agg_op = "aggregate", op_type
            missingness_raw = definition.get("missingness", {})
            missingness = (
                missingness_raw.get("policy", "exclude")
                if isinstance(missingness_raw, dict)
                else str(missingness_raw or "exclude")
            )
            window = definition.get("window")
            window = window if isinstance(window, dict) else None
            plan = OutcomePlan(
                id=definition["id"],
                source=str(source_name),
                select=str(field),
                aggregation=agg_op,
                group_by=tuple(grouping) if grouping else None,
                filters=tuple(definition.get("filters", [])),
                operation=operation,
                missingness=missingness,
                window=window,
            )
            evaluated = AnalysisEngine().evaluate(plan, sources)
            alternate = self._duckdb_outcome_rows(plan, sources.get(str(plan.source), []))
            if alternate is not None:
                evaluated, engine = alternate
                self.last_outcome_engine = engine
            else:
                self.last_outcome_engine = "python"
            output_schema_ref = definition.get("output_schema")
            if isinstance(output_schema_ref, str) and output_schema_ref in schema_catalog:
                if outcome_catalog is None:
                    raise ValueError(
                        "OUTCOME_SCHEMA_UNAVAILABLE: outcome "
                        f"{definition['id']} declares output_schema "
                        f"'{output_schema_ref}', but this run's build cannot bind a "
                        f"schema catalog: {catalog_error}"
                    )
                for row in evaluated:
                    diagnostics = outcome_catalog.validate(output_schema_ref, row)
                    errors = [
                        f"{diagnostic.instance_pointer or 'root'}: {diagnostic.message}"
                        for diagnostic in diagnostics
                    ]
                    if errors:
                        raise ValueError(
                            "OUTPUT_SCHEMA_VIOLATION: outcome "
                            f"{definition['id']} produced rows failing "
                            f"'{output_schema_ref}': {'; '.join(errors)}"
                        )
            for row in evaluated:
                results.append({"outcome_id": plan.id, **row})
        return results

    @staticmethod
    def _legacy_derived_rows(
        event_rows: list[dict[str, Any]], artifact_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Legacy flat-row synthesis for packages predating declared datasets.

        Retained for backward compatibility only: new packages use the generic
        dataset engine, and the core must not special-case study identifiers.
        """
        derived_rows: list[dict[str, Any]] = []
        seen_analytic: set[tuple[Any, Any]] = set()
        seen_articles: set[str] = set()
        for event in event_rows:
            delta = event.get("state_delta") or {}
            if not isinstance(delta, dict):
                continue
            for record in delta.get("analytics") or []:
                if not isinstance(record, dict):
                    continue
                key = (record.get("user"), record.get("phase"))
                if key in seen_analytic:
                    continue
                seen_analytic.add(key)
                flat = dict(event)
                flat.update(record)
                flat["kind"] = "analytics"
                flat["time"] = event.get("phase", 0)
                derived_rows.append(flat)
            for record in delta.get("titles") or []:
                if isinstance(record, dict) and record.get("article_id"):
                    if record["article_id"] in seen_articles:
                        continue
                    seen_articles.add(record["article_id"])
                    flat = dict(event)
                    flat["article_count"] = 1
                    flat["kind"] = "publication"
                    derived_rows.append(flat)
        seen_detections: set[str] = set()
        for row in artifact_rows:
            if row.get("process_id") != "evaluate-clickbait":
                continue
            artifact_id = str(row.get("artifact_id") or "")
            if artifact_id in seen_detections:
                continue
            seen_detections.add(artifact_id)
            value = row.get("value") or {}
            if isinstance(value, dict) and "detected" in value:
                flat = dict(row)
                flat["detected"] = 1 if value.get("detected") else 0
                flat["kind"] = "measurement"
                derived_rows.append(flat)
        return derived_rows

    @staticmethod
    def _retention_purges_raw(processes: list[Mapping[str, Any]]) -> bool:
        for process in processes:
            trace = process.get("trace_policy", {})
            retention = trace.get("retention") if isinstance(trace, Mapping) else None
            if isinstance(retention, str) and "purge" in retention:
                return True
        return False

    @staticmethod
    def _redact_raw_responses(value: Any) -> Any:
        """Replace raw provider-response fields with a purge marker (AW-20)."""
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                if key in {"response", "raw_response", "parsed_response"}:
                    redacted[key] = "<purged-by-retention>"
                else:
                    redacted[key] = GenesisService._redact_raw_responses(item)
            return redacted
        if isinstance(value, list):
            return [GenesisService._redact_raw_responses(item) for item in value]
        return value

    def _elicitation_provider(self, profile_id: str) -> Any:
        profile = self._full_model_profile(profile_id)
        return OpenAICompatibleProvider(
            base_url=str(profile["base_url"]),
            model=str(profile["model"]),
            api_key_env=str(profile.get("api_key_env") or "OPENAI_API_KEY"),
            api_key=profile.get("api_key"),
            timeout=float(profile.get("timeout", 60)),
        )

    def _approved_upstream_projections(self, session: Any) -> dict[str, str]:
        from genesis.compiler import StudyCompiler

        directory = self._specification_dir(session.specification_id)
        if not (directory / "metadata.json").is_file():
            return {}
        loaded = StudyCompiler(directory)._load()
        workflow = self._workflow_registry.get(session.workflow_id)
        projections: dict[str, str] = {}
        for stage in workflow.stages:
            if stage.id == session.current_stage:
                break
            progress = session.stages.get(stage.id)
            if progress is None or progress.status != "approved":
                continue
            for owned in stage.owned_paths:
                section = owned.strip("/") or ""
                if section not in loaded:
                    continue
                value = loaded[section]
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                if isinstance(value, dict):
                    projections[section] = yaml.safe_dump(value, sort_keys=False)
        return projections

    def _checklist_state(self, specification_id: str) -> dict[str, Any]:
        try:
            checklist = self.get_checklist(specification_id)["items"]
        except KeyError:
            return {}
        return {item["id"]: item for item in checklist}

    def start_elicitation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        specification_id = str(payload.get("specification_id", ""))
        workflow_id = str(payload.get("workflow_id", "three-layer-study"))
        model_profile_id = str(payload.get("model_profile_id", ""))
        researcher_id = str(payload.get("researcher_id", "researcher"))
        if not specification_id or not model_profile_id:
            raise ValueError("VALIDATION_ERROR: specification_id and model_profile_id are required")
        try:
            self.get_model_profile(model_profile_id)
        except KeyError as exc:
            raise ValueError(
                f"PROFILE_NOT_FOUND: unknown model profile '{model_profile_id}'"
            ) from exc
        try:
            base_version = int(
                payload.get(
                    "base_specification_version",
                    self.get_specification(specification_id)["version"],
                )
            )
        except KeyError:
            # A missing specification starts as a minimal canonical draft so the
            # browser flow can preview and approve the first stage immediately.
            self.create_specification(
                {
                    "id": specification_id,
                    "title": specification_id,
                    "description": "Initialized by the interactive elicitation session.",
                }
            )
            base_version = 1
        session = self._elicitation_engine.start_session(
            specification_id=specification_id,
            workflow_id=workflow_id,
            model_profile_id=model_profile_id,
            researcher_id=researcher_id,
            base_specification_version=base_version,
            session_id=(str(payload["session_id"]) if payload.get("session_id") else None),
        )
        return self.get_elicitation(session.session_id)

    def get_elicitation(self, session_id: str) -> dict[str, Any]:
        session = self._elicitation_engine.require_session(session_id)
        workflow, stage = self._elicitation_engine.stage_for(session)
        return {
            "session_id": session.session_id,
            "specification_id": session.specification_id,
            "workflow_id": session.workflow_id,
            "workflow_version": session.workflow_version,
            "model_profile_id": session.model_profile_id,
            "researcher_id": session.researcher_id,
            "current_stage": session.current_stage,
            "status": session.status,
            "base_specification_version": session.base_specification_version,
            "current_question": session.current_question,
            "current_suggestions": [
                {"label": label, "value": value} for label, value in session.current_suggestions
            ],
            "clarification": self._clarification_projection(session, stage),
            "stages": {
                stage_id: progress.model_dump() for stage_id, progress in session.stages.items()
            },
            "turns": [
                {
                    "id": turn.id,
                    "stage_id": turn.stage_id,
                    "question": turn.question,
                    "answer": turn.answer,
                    "response_mode": turn.response_mode,
                    "provider": turn.provider,
                    "model": turn.model,
                }
                for turn in session.turns
            ],
            "invalidations": session.invalidations,
            "pending_preview": session.pending_preview,
            "last_assistant_attempts": list(session.last_assistant_attempts),
            "allowed_actions": self._allowed_elicitation_actions(session),
        }

    @staticmethod
    def _clarification_projection(session: Any, stage: Any) -> dict[str, Any]:
        progress = session.stages[stage.id]
        required = [decision for decision in stage.critical_decisions if decision.required]
        remaining = [
            decision.id
            for decision in required
            if progress.decision_coverage.get(decision.id, "unresolved")
            not in {"covered", "defaulted"}
        ]
        covered = len(required) - len(remaining)
        return {
            "turns_used": progress.turn_count,
            "turns_remaining": max(0, stage.clarification.max_turns - progress.turn_count),
            "max_turns": stage.clarification.max_turns,
            "limit_reached": progress.limit_reached,
            "deferred_questions": list(progress.deferred_questions),
            "required_decisions": len(required),
            "covered_decisions": covered,
            "remaining_decisions": remaining,
        }

    @staticmethod
    def _allowed_elicitation_actions(session: Any) -> list[str]:
        if session.status == "cancelled":
            return []
        if session.status == "completed":
            return ["reopen_stage"]
        progress = session.stages[session.current_stage]
        actions: list[str] = []
        if session.status == "awaiting_answer":
            actions.append("submit_message")
        elif session.status == "awaiting_approval":
            if progress.status == "draft_ready":
                actions.append("draft")
            elif progress.status == "needs_review":
                actions.append("approve")
            elif session.pending_preview is None:
                actions.append("preview")
            else:
                actions.append("approve")
            actions.append("revise")
            if progress.review_mode:
                actions.extend(["submit_message", "draft", "edit_draft"])
        if any(item.status == "approved" for item in session.stages.values()):
            actions.append("reopen_stage")
        actions.append("cancel")
        return actions

    def execute_elicitation_mutation(
        self,
        session_id: str,
        *,
        operation: str,
        payload: Mapping[str, Any],
        expected_version: Any,
        idempotency_key: str | None,
        mutation: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        material = json.dumps(
            {"operation": operation, "payload": dict(payload)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        payload_hash = hashlib.sha256(material.encode()).hexdigest()

        def _execute() -> dict[str, Any]:
            self._require_expected_version(session_id, expected_version)
            return mutation()

        return self._elicitation_store.run_idempotent(
            session_id, idempotency_key, payload_hash, _execute
        )

    def submit_elicitation_message(
        self,
        session_id: str,
        answer: str,
        *,
        response_mode: str = "free_form",
        suggestion_index: int | None = None,
    ) -> dict[str, Any]:
        session = self._elicitation_engine.require_session(session_id)
        if session.status not in {"awaiting_answer", "awaiting_approval"}:
            raise ValueError(
                f"ELICITATION_STATE_CONFLICT: session is {session.status}, not awaiting_answer"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("VALIDATION_ERROR: answer must be non-empty text")
        workflow, stage = self._elicitation_engine.stage_for(session)
        clarification = self._clarification_projection(session, stage)
        progress = session.stages[stage.id]
        if (
            progress.review_mode
            or session.status == "awaiting_approval"
            or (progress.turn_count + 1 >= stage.clarification.max_turns)
        ):
            before = session.model_copy(deep=True)
            self._elicitation_engine.record_researcher_answer(
                session, answer=answer, response_mode="free_form"
            )
            progress.review_mode = True
            progress.limit_reached = progress.turn_count >= stage.clarification.max_turns
            self._elicitation_engine.transition(
                session,
                "awaiting_approval",
                stage_status="draft_ready",
                question="Review the draft, edit it, or send feedback. Approve when ready.",
                suggestions=(),
            )
            self._elicitation_store.put_pending_preview(session_id, None)
            try:
                self.draft_elicitation(session_id)
                return self.preview_elicitation_stage(session_id)
            except Exception:
                self._elicitation_store.update(
                    session_id, lambda stored: stored.__dict__.update(before.__dict__)
                )
                raise
        suggestions = session.current_suggestions
        effective_answer = answer
        effective_mode = response_mode
        if suggestion_index is not None and 0 <= int(suggestion_index) < len(suggestions):
            label, value = suggestions[int(suggestion_index)]
            effective_mode = "suggested" if answer == value else "edited"
        provider_name, provider = self._provider_fail_safe(
            self._elicitation_provider, session.model_profile_id
        )
        request = self._elicitation_assistant.assemble_evaluation_request(
            workflow,
            stage,
            session,
            projections=self._approved_upstream_projections(session),
            checklist_state=self._checklist_state(session.specification_id),
            pending_answer=effective_answer,
            clarification_state={
                **clarification,
                "decision_coverage": dict(session.stages[stage.id].decision_coverage),
            },
        )
        pending_turn_id = len(session.turns) + 1
        stage_turn_ids = {turn.id for turn in session.turns if turn.stage_id == stage.id} | {
            pending_turn_id
        }
        # The model may number this stage's rendered "Conversation turns" from 1;
        # map stage-local ids (1..k) onto the stage's global ids.
        stage_globals = sorted(turn.id for turn in session.turns if turn.stage_id == stage.id)
        # the pending (not-yet-recorded) turn is the stage's next local index
        stage_globals = [*stage_globals, pending_turn_id]
        local_map = {index: global_id for index, global_id in enumerate(stage_globals, 1)}
        profile = self.get_model_profile(session.model_profile_id)

        def _validate_evaluation(text: str) -> Any:
            evaluation = self._elicitation_assistant.parse_evaluation(
                text, stage_turn_ids, local_map
            )
            evaluation.validate_decisions(stage, stage_turn_ids, local_map)
            if evaluation.status == "needs_clarification":
                proposed_question = evaluation.next_question or (
                    evaluation.ambiguities[0].question if evaluation.ambiguities else ""
                )
                if proposed_question in set(session.stages[stage.id].asked_questions):
                    raise ValueError(
                        "ASSISTANT_DECISION_INVALID: this question was already asked; "
                        "ask something new"
                    )
            return evaluation

        try:
            evaluation, response, attempts = self._generate_validated_elicitation_output(
                provider=provider,
                profile=profile,
                prompt=request,
                schema=_workflow_schema("assistant-evaluation.schema.json"),
                validator=_validate_evaluation,
            )
        except _ElicitationOutputFailure as exc:
            self._elicitation_store.put_assistant_attempts(session.session_id, exc.attempts)
            raise ValueError(str(exc)) from exc
        session.last_assistant_attempts = attempts
        turn = self._elicitation_engine.record_researcher_answer(
            session,
            answer=effective_answer,
            response_mode=effective_mode,  # type: ignore[arg-type]
            question=session.current_question,
            suggestions=suggestions,
        )
        self._elicitation_store.set_answer_metadata(
            session.session_id, turn.id, provider_name, getattr(response, "model", None)
        )
        self._elicitation_store.put_turn_evaluation(session.session_id, turn.id, evaluation)
        session = self._merge_decision_coverage(session.session_id, stage, evaluation)
        remaining = self._remaining_required_decisions(session, stage)
        if not remaining:
            self._elicitation_engine.transition(
                session, "awaiting_approval", stage_status="draft_ready"
            )
            return self.get_elicitation(session_id)

        if evaluation.status == "needs_clarification":
            question = evaluation.next_question or evaluation.ambiguities[0].question
            next_suggestions = evaluation.next_suggestions or (
                evaluation.ambiguities[0].suggestions if evaluation.ambiguities else ()
            )
            self._elicitation_engine.transition(
                session,
                "awaiting_answer",
                stage_status="clarifying",
                question=question,
                suggestions=tuple((s.label, s.value) for s in next_suggestions),
            )
            self._elicitation_store.append_asked_question(session.session_id, stage.id, question)
            return self.get_elicitation(session_id)
        raise ValueError(
            "ASSISTANT_DECISION_INVALID: assistant reported readiness while required "
            "decisions remain unresolved: " + ", ".join(remaining)
        )

    def _merge_decision_coverage(self, session_id: str, stage: Any, evaluation: Any) -> Any:
        def _mutate(stored: Any) -> None:
            progress = stored.stages[stage.id]
            for item in evaluation.decision_coverage:
                previous = progress.decision_coverage.get(item.decision_id, "unresolved")
                if previous in {"covered", "defaulted"} and item.status == "unresolved":
                    continue
                progress.decision_coverage[item.decision_id] = item.status

        return self._elicitation_store.update(session_id, _mutate)

    @staticmethod
    def _remaining_required_decisions(session: Any, stage: Any) -> list[str]:
        progress = session.stages[stage.id]
        return [
            decision.id
            for decision in stage.critical_decisions
            if decision.required
            and progress.decision_coverage.get(decision.id, "unresolved")
            not in {"covered", "defaulted"}
        ]

    def draft_elicitation(self, session_id: str) -> dict[str, Any]:
        session = self._elicitation_engine.require_session(session_id)
        workflow, stage = self._elicitation_engine.stage_for(session)
        progress = session.stages[session.current_stage]
        if progress.status not in {"draft_ready", "awaiting_approval"}:
            raise ValueError(
                "STAGE_INCOMPLETE: the stage is not draft-ready; answer the opening "
                "question until the assistant reports readiness"
            )
        remaining = self._remaining_required_decisions(session, stage)
        if remaining and not progress.review_mode:
            raise ValueError(
                "STAGE_INCOMPLETE: simulation-critical decisions remain unresolved: "
                + ", ".join(remaining)
            )
        provider = self._elicitation_provider(session.model_profile_id)
        request = self._assemble_draft_request(session, workflow, stage)
        profile = self.get_model_profile(session.model_profile_id)

        def _validate_patch(text: str) -> ElicitationSpecificationPatch:
            from genesis.elicitation import _extract_json_object

            payload = _extract_json_object(text)
            if payload is None:
                raise ValueError("draft is not valid JSON")
            operations = list(payload.get("operations", []))
            kept_operations = []
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
                path = str(operation.get("path", ""))
                if path == "/study/study_id":
                    continue
                if path == "/study" and isinstance(operation.get("value"), dict):
                    operation = {
                        **operation,
                        "value": {
                            key: value
                            for key, value in operation["value"].items()
                            if key != "study_id"
                        },
                    }
                kept_operations.append(operation)
            payload["operations"] = kept_operations
            patch = ElicitationSpecificationPatch.model_validate(payload)
            if patch.stage_id != stage.id:
                raise ValueError("ASSISTANT_OUTPUT_INVALID: patch must target the current stage")
            stage_for_patch = self._elicitation_engine.stage_for(session)[1]
            patch_globals = sorted(
                turn.id for turn in session.turns if turn.stage_id == stage_for_patch.id
            )
            patch_local_map = {index: global_id for index, global_id in enumerate(patch_globals, 1)}
            patch.validate_turn_references(session.turn_ids, patch_local_map)
            patch.validate_owned_paths(stage.owned_paths)
            patch.validate_base_version(session.base_specification_version)
            patch.validate_evidence()
            patch.validate_checklist_items(stage.checklist_items)
            # Validate the candidate inside the repair loop, not only its patch envelope.
            from genesis.compiler import CANONICAL
            from genesis.elicitation import apply_operations

            current_form = self._current_form_payload(
                self._specification_dir(session.specification_id), session.specification_id
            )
            projection = self._canonical_specification(current_form)
            projection["schemas"] = current_form.get("schemas", {})
            candidate = apply_operations(projection, patch.operations)
            self._validate_schema_files(candidate.get("schemas", {}))
            for section, model in CANONICAL.items():
                model.model_validate(candidate[section])
            return patch

        try:
            patch, _response, attempts = self._generate_validated_elicitation_output(
                provider=provider,
                profile=profile,
                prompt=request,
                schema=_workflow_schema("specification-patch.schema.json"),
                validator=_validate_patch,
            )
        except _ElicitationOutputFailure as exc:
            self._elicitation_store.put_assistant_attempts(session.session_id, exc.attempts)
            raise ValueError(str(exc)) from exc
        session.last_assistant_attempts = attempts
        session.stages[stage.id].review_mode = True
        self._elicitation_engine.transition(
            session,
            "awaiting_approval",
            stage_status="awaiting_approval",
            question="Review the draft, edit it, or send feedback. Approve when ready.",
            suggestions=(),
        )
        self._elicitation_store.put_pending_patch(session.session_id, patch.model_dump(mode="json"))
        self._elicitation_store.put_pending_preview(session.session_id, None)
        return self.get_elicitation(session_id)

    @staticmethod
    def _structured_elicitation_parameters(
        provider: Any,
        configured: Mapping[str, Any],
    ) -> dict[str, Any]:
        parameters = dict(configured)
        try:
            capabilities = provider.capabilities()
        except (AttributeError, TypeError, ValueError):
            capabilities = None
        if capabilities is not None and bool(getattr(capabilities, "structured_output", False)):
            parameters.setdefault(
                "response_format",
                {"type": "json_object"},
            )
        return parameters

    def _generate_validated_elicitation_output(
        self,
        *,
        provider: Any,
        profile: Mapping[str, Any],
        prompt: str,
        schema: dict[str, Any],
        validator: Callable[[str], Any],
        max_repairs: int = 2,
    ) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Generate typed output with at most two deterministic repair calls."""
        parameters = self._structured_elicitation_parameters(
            provider, profile.get("parameters", {})
        )
        attempts: list[dict[str, Any]] = []
        current_prompt = prompt
        last_error = "unknown validation error"
        response: Any = None
        for attempt_number in range(1, max_repairs + 2):
            try:
                response = provider.generate(
                    ProviderRequest(
                        model=str(profile["model"]),
                        prompt=current_prompt,
                        parameters=parameters,
                    )
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "unavailable",
                        "error": str(exc)[:1000],
                    }
                )
                raise _ElicitationOutputFailure(
                    f"ASSISTANT_UNAVAILABLE: model call failed: {exc}", attempts
                ) from exc
            text = response.text if hasattr(response, "text") else str(response)
            try:
                validated = validator(text)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "invalid",
                        "request_id": getattr(response, "request_id", None),
                        "error": last_error[:1000],
                    }
                )
                if attempt_number > max_repairs:
                    break
                current_prompt = (
                    prompt
                    + "\n\n## Correct your previous response\n"
                    + "The response failed deterministic validation:\n"
                    + last_error[:4000]
                    + "\n\nPrevious response:\n"
                    + text[:12000]
                    + "\n\nRespond with ONLY one corrected JSON object matching this schema:\n"
                    + json.dumps(schema, indent=2)
                )
                continue
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "valid",
                    "request_id": getattr(response, "request_id", None),
                }
            )
            return validated, response, attempts
        raise _ElicitationOutputFailure(
            "ASSISTANT_OUTPUT_INVALID: model output remained invalid after "
            f"{max_repairs + 1} attempts. Last validation error: {last_error}",
            attempts,
        )

    def _assemble_draft_request(self, session: Any, workflow: Any, stage: Any) -> str:
        sections = [
            workflow.instructions_text().strip(),
            "",
            f"## Stage: {stage.title} ({stage.id})",
            stage.instructions.strip(),
            "",
            "## Approved upstream projections",
        ]
        projections = self._approved_upstream_projections(session)
        if projections:
            for section, text in projections.items():
                sections.append(f"### {section}\n{text}")
        else:
            sections.append("(none yet)")
        template_contents = []
        if stage.templates:
            package_root = _workflows_root() / workflow.id
            for key, relpath in stage.templates.items():
                template_path = package_root / relpath
                if template_path.is_file():
                    template_contents.append(
                        f"### template {key} ({relpath})\n{template_path.read_text()}"
                    )
        theory_templates = []
        if "/theory" in stage.owned_paths:
            for template in workflow.theory_templates.values():
                theory_templates.append(
                    f"### {template.id}\n"
                    f"Required functions: {', '.join(template.functions)}\n"
                    + "\n".join(f"- {question}" for question in template.questions)
                )
        live_sections = {}
        try:
            live_form = self._current_form_payload(
                self._specification_dir(session.specification_id),
                session.specification_id,
            )
            live_canonical = self._canonical_specification(live_form)
            live_canonical["schemas"] = live_form.get("schemas", {})
        except Exception:
            live_canonical = {}
        for owned in stage.owned_paths:
            section = owned.strip("/") or ""
            if section in live_canonical and isinstance(live_canonical[section], dict):
                live_sections[section] = yaml.safe_dump(live_canonical[section], sort_keys=False)
        sections += [
            "Schema files: /schemas maps stable IDs to nonempty JSON Schema objects. "
            "When this stage owns /schemas, create each schema_ref required by approved "
            "process outputs, with type, properties and required describing the intended payload.",
            "## Canonical schemas (processes must be objects, never prose strings)",
            self._draft_canonical_schemas(stage),
            "## Researcher-driven draft review",
            "Generate a complete candidate from the available answers. Do not ask another "
            "clarification question. Propose minimal executable choices for missing details "
            "and list them as explicit assumptions for researcher review. "
            "Never invent empirical evidence.",
            "## Previous draft (return the complete cumulative patch against the base version)",
            json.dumps(session.pending_patch, ensure_ascii=False),
            "",
            "## Live canonical projection (stage-owned sections)",
            "\n".join(f"### {name}\n{text}" for name, text in live_sections.items())
            if live_sections
            else "(no live package yet)",
            "",
            (
                "Refine ONLY fields shown in the live projection and templates. Never "
                "invent field names. study_id is derived from the specification id - "
                "never include study_id in operations."
            ),
            "",
            "## Draft target template contents",
            "\n".join(template_contents) if template_contents else "(none)",
            "",
            "## Registered theory templates",
            "\n\n".join(theory_templates) if theory_templates else "(not applicable)",
            "",
            "## Theory mapping mandate" if "/theory" in stage.owned_paths else "",
            (
                "If you select a REGISTERED theory family (listed above), the "
                "theory.yaml process_mappings MUST map EVERY required function of "
                "that family to an existing Layer-1 process id from the approved "
                "'### openness' projection. Use ONLY process ids present there. If "
                "any required function cannot be mapped to a real process, use an "
                "unregistered/exploratory theory_family instead."
            )
            if "/theory" in stage.owned_paths
            else "",
            "",
            "## Conversation turns",
            "\n".join(
                f"turn {turn.id} ({turn.response_mode}): {turn.answer or ''}"
                for turn in session.turns
                if turn.stage_id == session.current_stage
            ),
            "",
            (
                "Produce a field-level SpecificationPatch for the stage-owned paths "
                f"{list(stage.owned_paths)}. Base specification version: "
                f"{session.base_specification_version}. Every proposed consequential "
                "value must cite supporting turns or be covered by an assumption object "
                "with the operation path in `target` and the rationale in `statement`. "
                "Supported operations: add, replace, remove. "
                f"affected_checklist_items must ONLY reference these stage checklist "
                f"ids: {list(stage.checklist_items or ())}. Respond with ONLY the "
                "patch object itself; do not echo schema metadata such as $schema "
                "or $id, and add no prose."
            ),
            "",
            "## Response schema",
            json.dumps(_workflow_schema("specification-patch.schema.json"), indent=2),
        ]
        return "\n".join(sections)

    @staticmethod
    def _draft_canonical_schemas(stage: Any) -> str:
        from genesis.compiler import CANONICAL

        return json.dumps(
            {
                name: model.model_json_schema()
                for name, model in CANONICAL.items()
                if "/" + name in stage.owned_paths
            }
        )

    def approve_elicitation_stage(self, session_id: str, *, approved_by: str) -> dict[str, Any]:
        """Explicit researcher approval writing a new immutable package version (IEL-010/025)."""
        session = self._elicitation_engine.require_session(session_id)
        if not approved_by or approved_by == "assistant" or approved_by == session.model_profile_id:
            raise ValueError("STAGE_APPROVAL_REQUIRED: only the researcher can approve a stage")
        workflow, stage = self._elicitation_engine.stage_for(session)
        progress = session.stages[session.current_stage]
        if workflow.next_stage(session.current_stage) is None:
            incomplete_stages = [
                stage_id
                for stage_id, stage_progress in session.stages.items()
                if stage_id != session.current_stage and stage_progress.status != "approved"
            ]
            if incomplete_stages:
                raise ValueError(
                    "STAGE_INCOMPLETE: final approval requires every prior stage to be "
                    "approved: " + ", ".join(incomplete_stages)
                )
        if not (session.status == "awaiting_approval" or progress.status == "needs_review"):
            raise ValueError("STAGE_INCOMPLETE: request a draft and preview before approving")
        # 1. Completion rules (generic engine over the CANDIDATE package).
        # Only the LATEST assistant evaluation decides readiness: an earlier
        # clarification is superseded when readiness was subsequently reported.
        latest_evaluation = None
        for turn in reversed(session.turns):
            if turn.evaluation:
                latest_evaluation = turn.evaluation
                break
        if progress.review_mode:
            any_consequential = False
        elif stage.critical_decisions:
            any_consequential = bool(self._remaining_required_decisions(session, stage))
        else:
            any_consequential = bool(
                latest_evaluation
                and latest_evaluation.get("status") == "needs_clarification"
                and any(
                    ambiguity.get("consequential")
                    for ambiguity in latest_evaluation.get("ambiguities", [])
                )
            )
        try:
            current = self.get_specification(session.specification_id)
        except KeyError:
            current = None
        if progress.status == "needs_review":
            # Re-approval of already-published YAML: no patch required.
            if current is None:
                raise ValueError("STAGE_INCOMPLETE: no accepted package to re-approve")
            live_validation = self._inspect_current_package(session)
            live_checks = completion_checks(
                checklist_state={
                    item["id"]: item["status"]
                    for item in self.get_checklist(session.specification_id)["items"]
                },
                checklist_items=stage.checklist_items,
                any_consequential_ambiguity=any_consequential,
                canonical_errors=live_validation["errors"],
            )
            failed_rules = require_completion_rules(stage, live_checks)
            if failed_rules:
                raise ValueError(
                    f"STAGE_INCOMPLETE: completion rules unresolved: {', '.join(failed_rules)}"
                )
            form = self._current_form_payload(
                self._specification_dir(session.specification_id), session.specification_id
            )
            form["id"] = session.specification_id
            validation = self._inspect_current_package(session)
            if validation["errors"]:
                raise ValueError(f"SPECIFICATION_INVALID: {validation['errors'][0]['message']}")
            updated = self.update_specification(session.specification_id, form, current["version"])
        else:
            if session.pending_patch is None or session.pending_preview is None:
                raise ValueError("STAGE_INCOMPLETE: request a draft and preview before approving")
            try:
                patch = ElicitationSpecificationPatch.model_validate(session.pending_patch)
            except ValidationError as exc:
                raise ValueError(f"ASSISTANT_OUTPUT_INVALID: {exc}") from exc
            patch.validate_base_version(session.base_specification_version)
            if patch.unresolved_questions:
                if any_consequential:
                    raise ValueError(
                        "STAGE_INCOMPLETE: unresolved questions must be clarified before approval: "
                        + "; ".join(patch.unresolved_questions)
                    )
                # Clarification budget exhausted (or the questions are advisory): the
                # required simulation decisions are all covered/defaulted, so approval
                # proceeds and the questions are recorded as explicit deferred notes
                # instead of deadlocking the session at the turn limit.
                self._elicitation_store.record_deferred_questions(
                    session.session_id, stage.id, list(patch.unresolved_questions)
                )
            preview = session.pending_preview
            if int(preview.get("base_package_version", -1)) != session.base_specification_version:
                raise ValueError(
                    f"PATCH_BASE_STALE: preview targets version "
                    f"{preview.get('base_package_version')}, session is at "
                    f"{session.base_specification_version}"
                )
            if current is None or int(current["version"]) != session.base_specification_version:
                raise ValueError(
                    "PATCH_BASE_STALE: the package version changed after the preview was "
                    "generated; regenerate the preview before approving"
                )
            live_hash = self._package_content_hash(
                self._specification_dir(session.specification_id)
            )
            if live_hash != preview.get("package_hash"):
                raise ValueError(
                    "PATCH_BASE_STALE: the package changed after the preview was "
                    "generated; regenerate the preview before approving"
                )
            # Persisted previews can outlive compiler/workflow updates. Recheck
            # the same patch against the unchanged base before trusting either
            # cached errors or cached success; do not bypass the stale guards.
            preview = self.preview_elicitation_stage(session_id)["pending_preview"]
            if preview.get("validation", {}).get("errors"):
                first = preview["validation"]["errors"][0]
                raise ValueError(f"SPECIFICATION_INVALID: {first.get('message')}")
            checks = completion_checks(
                checklist_state=preview.get("validation", {}).get("checklist", {}),
                checklist_items=stage.checklist_items,
                any_consequential_ambiguity=any_consequential,
                canonical_errors=preview.get("validation", {}).get("errors", []),
            )
            failed_rules = require_completion_rules(stage, checks)
            if failed_rules:
                raise ValueError(
                    f"STAGE_INCOMPLETE: completion rules unresolved: {', '.join(failed_rules)}"
                )
            candidate_form = dict(preview.get("candidate_form", {}))
            candidate_form["id"] = session.specification_id
            if current is None:
                updated = self.create_specification(candidate_form)
            else:
                updated = self.update_specification(
                    session.specification_id, candidate_form, current["version"]
                )
            touched = self._touched_sections(patch)
        new_version = int(updated["version"])
        # 2. Advance the session base to the accepted immutable version.
        self._elicitation_store.put_base_version(session_id, new_version)
        # 3. Record approval metadata (IEL-025: identity and timestamp).
        directory = self._specification_dir(session.specification_id)
        metadata = self.get_specification(session.specification_id)
        approvals = list(metadata.get("elicitation_approvals", []))
        approvals.append(
            {
                "stage": session.current_stage,
                "approved_by": str(approved_by),
                "at": datetime.now(UTC).isoformat(),
                "revision": new_version,
            }
        )
        metadata["elicitation_approvals"] = approvals
        (directory / "metadata.json").write_text(
            json.dumps(self._metadata_for_storage(metadata), indent=2, sort_keys=True)
        )
        # 3. Invalidate dependent downstream stages from data-defined rules.
        invalidations: list[dict[str, Any]] = []
        if progress.status != "needs_review":
            invalidations = self._invalidate_downstream_stages(session, workflow, touched)
        summary = None
        for turn in reversed(session.turns):
            if turn.evaluation and turn.evaluation.get("summary"):
                summary = turn.evaluation["summary"]
                break
        self._elicitation_engine.mark_approved(session, revision=new_version, summary=summary)
        if workflow.next_stage(session.current_stage) is None:
            # Final approval: complete the session and mark the package approved.
            self.approve_specification(session.specification_id, new_version, approved_by)
            self._elicitation_store.rename_to_completed(session_id)
            self._elicitation_store.put_pending_patch(session_id, None)
            return self.get_elicitation(session_id)
        existing = list(session.invalidations)
        self._elicitation_store.put_invalidations(session_id, existing + invalidations)
        self._advance_stage(session, workflow.next_stage(session.current_stage))
        self._elicitation_store.put_pending_patch(session_id, None)
        self._elicitation_store.put_pending_preview(session_id, None)
        return self.get_elicitation(session_id)

    def _advance_stage(self, session: Any, next_stage: Any) -> None:
        def _mutate(stored: Any) -> None:
            stored.current_stage = next_stage.id
            stored.status = "awaiting_answer"
            stored.current_question = next_stage.opening_question
            stored.current_suggestions = ()
            progress = stored.stages[next_stage.id]
            if progress.status != "needs_review":
                progress.status = "clarifying"

        self._elicitation_store.update(session.session_id, _mutate)

    def _touched_sections(self, patch: Any) -> list[str]:
        sections: list[str] = []
        for operation in patch.operations:
            parts = [part for part in str(operation.path).split("/") if part]
            if len(parts) >= 2:
                sections.append(f"{parts[0]}.{parts[1]}")
            elif parts:
                sections.append(parts[0])
        return sections

    def _stage_for_section(self, workflow: Any, section: str) -> str | None:
        """Map a canonical section (e.g. 'theory.process_mappings') to its owner."""
        head = section.split(".", 1)[0]
        for stage in workflow.stages:
            for owned in stage.owned_paths:
                if owned.strip("/") == head:
                    return str(stage.id)
        return None

    def _invalidate_downstream_stages(
        self, session: Any, workflow: Any, touched: list[str]
    ) -> list[dict[str, Any]]:
        invalidations: list[dict[str, Any]] = []
        stage_ids = [stage.id for stage in workflow.stages]
        current_index = stage_ids.index(session.current_stage)
        for rule_key, targets in workflow.invalidation.items():
            if not any(
                touched_section == rule_key or touched_section.startswith(rule_key + ".")
                for touched_section in touched
            ):
                continue
            for target in targets:
                downstream = self._stage_for_section(workflow, target)
                if downstream is None or downstream not in stage_ids:
                    continue
                if stage_ids.index(downstream) <= current_index:
                    continue
                progress = session.stages.get(downstream)
                if progress is None or progress.status not in {
                    "approved",
                    "awaiting_approval",
                    "draft_ready",
                    "needs_review",
                }:
                    continue
                self._elicitation_store.mark_stage_needs_review(
                    session.session_id, downstream, rule_key
                )
                invalidations.append(
                    {
                        "stage": downstream,
                        "reason": f"{rule_key} changed during {session.current_stage} revision",
                        "affected_paths": [rule_key],
                    }
                )
        return invalidations

    def _predicted_invalidations(self, workflow: Any, patch: Any) -> list[dict[str, Any]]:
        touched: list[str] = []
        for operation in patch.operations:
            parts = [part for part in str(operation.path).split("/") if part]
            if len(parts) >= 2:
                touched.append(f"{parts[0]}.{parts[1]}")
            elif parts:
                touched.append(parts[0])
        result: list[dict[str, Any]] = []
        for rule_key, targets in workflow.invalidation.items():
            if not any(
                section == rule_key or section.startswith(rule_key + ".") for section in touched
            ):
                continue
            for target in targets:
                mapped = self._stage_for_section(workflow, target)
                if mapped:
                    result.append(
                        {
                            "stage": mapped,
                            "reason": f"{rule_key} changes; {target} becomes stale",
                            "affected_paths": [target],
                        }
                    )
        return result

    def _inspect_current_package(self, session: Any) -> dict[str, Any]:

        directory = self._specification_dir(session.specification_id)
        report = (
            StudyAssistant(theory_templates=self._workflow_registry.theory_templates())
            .inspect_package(directory)
            .as_dict()
        )
        issues = report.get("issues", []) if isinstance(report, dict) else []
        return {
            "errors": [
                {
                    "code": item.get("code"),
                    "path": item.get("path", ""),
                    "message": item.get("message", ""),
                }
                for item in issues
                if item.get("severity") == "error"
            ],
            "warnings": [
                {
                    "code": item.get("code"),
                    "path": item.get("path", ""),
                    "message": item.get("message", ""),
                }
                for item in issues
                if item.get("severity") == "warning"
            ],
        }

    def _require_expected_version(self, session_id: str, expected_version: Any) -> None:
        if expected_version is None:
            return
        session = self._elicitation_engine.require_session(session_id)
        if int(expected_version) != session.base_specification_version:
            raise ValueError(
                f"PATCH_BASE_STALE: expected package version {expected_version}, "
                f"session is at {session.base_specification_version}"
            )

    def revise_elicitation_stage(self, session_id: str) -> dict[str, Any]:
        session = self._elicitation_engine.require_session(session_id)
        if session.status != "awaiting_approval":
            raise ValueError(
                f"ELICITATION_STATE_CONFLICT: session is {session.status}, not awaiting_approval"
            )
        session.stages[session.current_stage].review_mode = True
        self._elicitation_engine.transition(
            session,
            "awaiting_approval",
            question="Send feedback to regenerate the draft, or edit and approve it.",
            suggestions=(),
        )
        return self.get_elicitation(session_id)

    def edit_elicitation_draft(
        self, session_id: str, filename: str, content: str
    ) -> dict[str, Any]:
        """Preview a researcher's YAML edit without changing the accepted package."""
        session = self._elicitation_engine.require_session(session_id)
        if session.status != "awaiting_approval" or session.pending_patch is None:
            raise ValueError("ELICITATION_STATE_CONFLICT: generate a draft before editing")
        _, stage = self._elicitation_engine.stage_for(session)
        section = filename.removesuffix(".yaml")
        path = "/" + section
        if filename != section + ".yaml" or path not in stage.owned_paths:
            raise ValueError("PATCH_PATH_FORBIDDEN: edit a section owned by this stage")
        value = yaml.safe_load(content)
        if not isinstance(value, dict):
            raise ValueError("VALIDATION_ERROR: YAML section must be a mapping")
        patch = ElicitationSpecificationPatch.model_validate(session.pending_patch)
        payload = patch.model_dump(mode="json")
        payload["operations"] = [
            op
            for op in payload["operations"]
            if op["path"] != path and not op["path"].startswith(path + "/")
        ] + [{"op": "replace", "path": path, "value": value}]
        before = session.model_copy(deep=True)
        turn = self._elicitation_engine.record_researcher_answer(
            session, answer=f"Manual edit of {filename}:\n{content}", response_mode="edited"
        )
        targets = {op["path"] for op in payload["operations"]}
        payload["evidence"] = [e for e in payload["evidence"] if e["target"] in targets]
        payload["assumptions"] = [a for a in payload["assumptions"] if a["target"] in targets]
        payload["evidence"].append({"target": path, "source_turns": [turn.id]})
        edited = ElicitationSpecificationPatch.model_validate(payload)
        edited.validate_owned_paths(stage.owned_paths)
        edited.validate_evidence()
        self._elicitation_store.put_pending_patch(session_id, edited.model_dump(mode="json"))
        self._elicitation_store.put_pending_preview(session_id, None)
        try:
            return self.preview_elicitation_stage(session_id)
        except Exception:
            self._elicitation_store.update(
                session_id, lambda stored: stored.__dict__.update(before.__dict__)
            )
            raise

    def reopen_elicitation_stage(self, session_id: str, stage_id: str) -> dict[str, Any]:
        session = self._elicitation_engine.require_session(session_id)
        workflow = self._workflow_registry.get(session.workflow_id)
        stage = workflow.stage(stage_id)
        progress = session.stages.get(stage_id)
        if progress is None or progress.status != "approved":
            raise ValueError(f"ELICITATION_STATE_CONFLICT: stage '{stage_id}' is not approved")
        stage_ids = [item.id for item in workflow.stages]
        reopened_index = stage_ids.index(stage_id)
        configured_dependants = set(workflow.dependants_of(stage_id))
        invalidations: list[dict[str, Any]] = []
        for downstream in workflow.stages[reopened_index + 1 :]:
            if downstream.id not in configured_dependants:
                continue
            progress = session.stages.get(downstream.id)
            if progress is None or progress.status in {"not_started", "needs_review"}:
                continue
            invalidations.append(
                {
                    "stage": downstream.id,
                    "reason": f"Upstream stage '{stage_id}' was reopened for revision",
                    "affected_paths": list(workflow.invalidation),
                }
            )
        if invalidations:
            self._elicitation_store.mark_stages_needs_review(
                session_id, [(item["stage"], item["reason"]) for item in invalidations]
            )

        def _mutate(stored: Any) -> None:
            stored.current_stage = stage_id
            stored.status = "awaiting_answer"
            stored.current_question = stage.opening_question
            stored.current_suggestions = ()
            stored.pending_patch = None
            stored.pending_preview = None
            stored.invalidations = invalidations
            progress = stored.stages[stage_id]
            progress.status = "clarifying"
            progress.turn_count = 0
            progress.limit_reached = False
            progress.review_mode = False
            progress.asked_questions = []
            progress.decision_coverage = {
                decision.id: "unresolved" for decision in stage.critical_decisions
            }

        self._elicitation_store.update(session_id, _mutate)
        return self.get_elicitation(session_id)

    def preview_elicitation_stage(self, session_id: str) -> dict[str, Any]:
        """Deterministic YAML preview of the pending stage patch (IEL-024)."""
        session = self._elicitation_engine.require_session(session_id)
        if session.pending_patch is None:
            raise ValueError("ELICITATION_STATE_CONFLICT: no pending patch; request a draft first")
        workflow, stage = self._elicitation_engine.stage_for(session)
        try:
            patch = ElicitationSpecificationPatch.model_validate(session.pending_patch)
        except ValidationError as exc:
            raise ValueError(f"ASSISTANT_OUTPUT_INVALID: {exc}") from exc
        current_form = self._current_form_payload(
            self._specification_dir(session.specification_id), session.specification_id
        )
        preview = self._patch_preview.preview(
            patch=patch,
            stage=stage,
            workflow=workflow,
            current_form=current_form,
            source_turn_ids=session.turn_ids,
            package_hash=self._package_content_hash(
                self._specification_dir(session.specification_id)
            ),
            live_directory=self._specification_dir(session.specification_id),
        )
        preview["invalidations"] = self._predicted_invalidations(workflow, patch)
        self._elicitation_store.put_pending_preview(session_id, preview)
        return self.get_elicitation(session_id)

    def cancel_elicitation(self, session_id: str) -> dict[str, Any]:
        self._elicitation_engine.cancel(session_id)
        # Discard the unfinished session (IEL-012 in-memory contract).
        self._elicitation_store.delete(session_id)
        return {"session_id": session_id, "status": "cancelled"}

    def list_elicitation_workflows(self) -> list[dict[str, Any]]:
        return [self._workflow_projection(workflow) for workflow in self._workflow_registry.list()]

    def get_elicitation_workflow(self, workflow_id: str) -> dict[str, Any]:
        return self._workflow_projection(self._workflow_registry.get(workflow_id))

    @staticmethod
    def _workflow_projection(workflow: Any) -> dict[str, Any]:
        return {
            "id": workflow.id,
            "version": workflow.version,
            "title": workflow.title,
            "session_persistence": workflow.session_persistence,
            "stages": [
                {
                    "id": stage.id,
                    "title": stage.title,
                    "opening_question": stage.opening_question,
                    "owned_paths": list(stage.owned_paths),
                    "checklist_items": list(stage.checklist_items),
                    "ambiguity_topics": list(stage.ambiguity_topics),
                    "depends_on": list(stage.depends_on),
                    "completion_rules": [rule.model_dump() for rule in stage.completion_rules],
                    "approval_required": stage.approval_required,
                }
                for stage in workflow.stages
            ],
            "invalidation": {key: list(value) for key, value in workflow.invalidation.items()},
            "theory_templates": sorted(workflow.theory_templates),
        }

    def _provider_fail_safe(self, builder: Any, profile_id: str) -> tuple[str, Any]:
        provider = builder(profile_id)
        return getattr(provider, "provider", "unknown"), provider

    def retention_enforce(self, run_id: str) -> dict[str, Any]:
        """Enforce retention policies against durable storage (AW-20).

        Runs whose processes declare a purge retention remove raw-response
        artifact payloads from the object-linked artifacts table.
        """
        run = self.get_run(run_id)
        build_ref = run.get("build") or run.get("build_path")
        if not build_ref:
            return {"run_id": run_id, "purged_rows": 0, "policy": "none"}
        build_path = self.resolve_path(build_ref)
        processes = json.loads((build_path / "processes.json").read_text())
        if not self._retention_purges_raw(processes):
            return {"run_id": run_id, "purged_rows": 0, "policy": "retain"}
        purged = self.persistence.retention_purge(run_id)
        return {"run_id": run_id, "purged_rows": purged, "policy": "purge"}

    def export_preview(self, run_id: str) -> dict[str, Any]:
        """List exported content and enumerate sensitive classes before writing (AW-20)."""
        run = self.get_run(run_id)
        artifacts = self.artifacts_for_run(run_id)
        raw_responses = 0
        for artifact in artifacts:
            payload = artifact["payload"]
            if isinstance(payload, dict) and "response" in payload.get("outputs", {}):
                raw_responses += 1
        return {
            "run_id": run_id,
            "status": run["status"],
            "files": [
                "outcomes.json",
                "outcomes.csv",
                "outcomes.parquet",
                "run_manifest.json",
                "events.json",
                "events.parquet",
                "artifacts.json",
                "artifacts.parquet",
                "states.json",
                "states.parquet",
                "package/*.yaml",
            ],
            "sensitive": {
                "raw_responses": raw_responses,
                "credentials_stored": False,
            },
            "retention_policy": {
                "purge_raw_responses": raw_responses > 0,
            },
        }

    def export_run(
        self,
        run_id: str,
        output: str | Path,
        *,
        mode: str = ExportMode.EXPLORATION,
    ) -> list[Path]:
        """Export run evidence as a capability-labelled bundle (EVD-001/002).

        ``exploration`` (default) exports retained evidence and stored outcome
        snapshots. ``reproducibility`` requires the run-pinned package closure,
        build and execution manifest, and fails with a completeness report if
        any required input is absent; it never silently downgrades to explore.
        Package files come exclusively from the run-pinned closure, not the
        editable package. The bundle is written to a fresh staging directory,
        verify the manifest, then atomically published, refusing to overwrite
        an existing destination (spec §4.3).
        """
        destination = self.resolve_path(output).resolve()
        if destination.exists():
            raise ValueError(
                f"EXPORT_DESTINATION: refuses to overwrite existing destination {destination}"
            )
        staging = stage_bundle(destination)
        try:
            paths = self._write_bundle(run_id, staging, mode=mode)
            publish_bundle(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        # F14 fix: the staging directory was atomically renamed to the final
        # destination; remap the returned paths so they point at real files.
        return [
            destination / path.relative_to(staging)
            for path in paths
            if path.is_relative_to(staging)
        ]

    def _write_bundle(self, run_id: str, destination: Path, *, mode: str) -> list[Path]:
        """Write all bundle members for one run into a prepared directory."""
        run = self.get_run(run_id)
        outcomes = self.evaluate_outcomes(run_id)
        result_paths = AnalysisExporter().export_bundle(
            outcomes,
            destination,
            methods={"run_id": run_id, "engine": "genesis-local"},
            replay_lineage={"source_run_id": run_id},
        )
        output_paths = list(result_paths)
        extras: dict[str, str] = {
            "run_manifest.json": json.dumps(run.get("manifest") or {}, indent=2, sort_keys=True)
        }
        build_ref = run.get("build") or run.get("build_path")
        processes: list[Mapping[str, Any]] = []
        package_closure_digest = ""
        build_digest = ""
        has_closure = False
        has_build = False
        if build_ref:
            build_path = self.resolve_path(build_ref)
            has_build = True
            processes = json.loads((build_path / "processes.json").read_text())
            if mode == ExportMode.REPRODUCIBILITY:
                # F6: a reproducibility bundle must carry the verified
                # execution prerequisites, not just metadata. Capabilities
                # that claim reexecution/replay are only true when these
                # files are actually present in the bundle.
                for name in (
                    "processes.json",
                    "process_graph.json",
                    "context_policies.json",
                    "state_model.json",
                    "artifact_catalog.json",
                    "outcome_plan.json",
                    "theory_execution_plan.json",
                ):
                    source_file = build_path / name
                    if source_file.is_file():
                        extras[name] = source_file.read_text()
                # The live executors are rebuilt from these files, so a
                # reproducibility bundle must carry them too — otherwise the
                # restored build cannot re-execute generative/evaluator
                # processes (partial/branch replay on an import would fail).
                for name in (
                    "model_profiles.json",
                    "prompt_templates.json",
                    "initialization.json",
                    "schemas.json",
                ):
                    source_file = build_path / name
                    if source_file.is_file():
                        extras[name] = source_file.read_text()
            for name in (
                "build_manifest.json",
                "validation_report.json",
                "protocol.json",
                "data_manifest.json",
            ):
                source_file = build_path / name
                if source_file.is_file():
                    extras[name] = source_file.read_text()
            build_manifest = json.loads((build_path / "build_manifest.json").read_text())
            build_digest = str(build_manifest.get("build_hash", ""))
            package_closure_digest = str(build_manifest.get("package_closure_digest", ""))
            # Run-pinned package closure (EVD-002): copy original bytes from the
            # build's immutable closure directory, never from the editable package.
            closure_path = build_path / "package_closure.json"
            closure_dir = build_path / "closure"
            if closure_path.is_file() and closure_dir.is_dir():
                has_closure = True
                extras["package_closure.json"] = closure_path.read_text()
                closure_manifest = json.loads(closure_path.read_text())
                for asset in closure_manifest["assets"]:
                    relative = Path(str(asset["path"]))
                    source_asset = closure_dir / relative
                    if not source_asset.is_file():
                        raise ValueError(
                            f"EXPORT_CLOSURE: run-pinned closure asset missing: {relative}"
                        )
                    target = destination / "package" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_asset, target)
            event_rows = self.trace_run(run_id)
            artifact_rows = self.artifacts_for_run(run_id)
            if self._retention_purges_raw(processes):
                event_rows = GenesisService._redact_raw_responses(event_rows)
                artifact_rows = GenesisService._redact_raw_responses(artifact_rows)
            extras["events.json"] = json.dumps(event_rows, indent=2, default=str)
            output_paths.append(
                AnalysisExporter.rows_to_parquet(
                    _parquet_safe(event_rows), destination / "events.parquet"
                )
            )
            extras["artifacts.json"] = json.dumps(artifact_rows, indent=2, default=str)
            output_paths.append(
                AnalysisExporter.rows_to_parquet(
                    _parquet_safe(
                        [
                            {key: value for key, value in row.items() if key != "payload"}
                            for row in artifact_rows
                        ]
                    ),
                    destination / "artifacts.parquet",
                )
            )
            # NOTE: raw cumulative state snapshots are intentionally NOT exported
            # as JSON (hundreds of MB); the compact parquet projection is kept.
            state_history = self.persistence.list_state_history(run_id)
            state_rows_for_export = [
                {**snapshot, "state_version": version} for version, snapshot in state_history
            ]
            output_paths.append(
                AnalysisExporter.rows_to_parquet(
                    _parquet_safe(state_rows_for_export), destination / "states.parquet"
                )
            )
        else:
            # Exploration imports have retained traces but no executable build.
            extras["events.json"] = json.dumps(self.trace_run(run_id), indent=2, default=str)
            extras["artifacts.json"] = json.dumps(
                self.artifacts_for_run(run_id), indent=2, default=str
            )
        # Reproducibility requires the full pinned inputs (EVD-T04); never
        # silently downgrade a requested full export.
        if mode == ExportMode.REPRODUCIBILITY:
            missing = []
            if not has_build:
                missing.append("build")
            if not has_closure:
                missing.append("package_closure")
            run_manifest = run.get("manifest") or {}
            if not (run_manifest.get("execution") or {}).get("package_digest"):
                missing.append("execution_manifest")
            if missing:
                raise ValueError(
                    "REPRODUCIBILITY_EXPORT_INCOMPLETE: missing required inputs: "
                    + ", ".join(missing)
                    + "; choose exploration export instead"
                )
        for name, content in extras.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            output_paths.append(target)
        # Legacy integrity manifest preserved for backward-compatible importers.
        files = [
            path
            for path in destination.rglob("*")
            if path.is_file() and path.name not in {"integrity.json", "bundle_manifest.json"}
        ]
        integrity = {
            path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
        integrity_path = destination / "integrity.json"
        integrity_path.write_text(json.dumps(integrity, indent=2, sort_keys=True) + "\n")
        output_paths.append(integrity_path)
        # Machine-readable capability evaluation + bundle manifest (EVD-001).
        manifest = run.get("manifest") or {}
        execution = manifest.get("execution") or {}
        scientific_digest = str(manifest.get("scientific_config_digest", ""))
        # F6: capabilities must reflect what this *bundle actually contains*.
        # Executable build files are only bundled in reproducibility mode;
        # recorded-output availability depends on retained artifacts, not on
        # whether processes were declared.
        has_executable_build = bool(
            extras.get("processes.json")
            and extras.get("context_policies.json")
            and extras.get("state_model.json")
            and extras.get("artifact_catalog.json")
        )
        retained_records = bool(self._recorded_process_outputs(run_id, ()) if build_ref else False)
        capabilities = evaluate_capabilities(
            has_build=has_executable_build,
            has_closure=has_closure,
            has_recorded_outputs=retained_records,
            has_checkpoint_evidence=False,
            has_outcomes=bool(outcomes),
        )
        source_run_id = str(manifest.get("run_id", run_id))
        bundle_manifest_path = write_bundle_manifest(
            destination,
            export_mode=mode,
            run_id=run_id,
            source_run_id=source_run_id,
            local_import_id=None if run_id == source_run_id else run_id,
            package_digest=str(execution.get("package_digest", package_closure_digest)),
            build_digest=str(execution.get("build_digest", build_digest)),
            scientific_config_digest=scientific_digest,
            capabilities=capabilities,
            omissions=[
                "raw cumulative state snapshots",
                "provider response bodies when retention purges them",
            ],
            retention_policy=(
                "purge_raw_responses"
                if self._retention_purges_raw(processes)
                else "retain_raw_responses"
            ),
        )
        output_paths.append(bundle_manifest_path)
        return output_paths

    def list_builds(self) -> list[dict[str, Any]]:
        recorded = self.persistence.list_study_builds()
        if recorded:
            return [
                {
                    "build_hash": build["build_hash"],
                    "study_id": build["study_id"],
                    "compiler_version": build["compiler_version"],
                    "path": str(build.get("path", "")),
                }
                for build in recorded
            ]
        root = self.workspace / "builds"
        builds = []
        for directory in sorted(root.glob("*")):
            if not directory.is_dir():
                continue
            manifest_path = directory / "build_manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                continue
            builds.append(
                {
                    "build_hash": manifest.get("build_hash"),
                    "study_id": manifest.get("study_id"),
                    "compiler_version": manifest.get("compiler_version"),
                    "path": str(directory),
                }
            )
        return builds

    def list_traces(self, run_id: str) -> list[dict[str, Any]]:
        """The traces a run's build declares."""
        plan = self._run_outcome_plan(run_id)
        return [
            {
                "id": str(trace.get("id", "")),
                "title": trace.get("title"),
                "seed_dataset": str((trace.get("seed") or {}).get("dataset", "")),
            }
            for trace in plan.get("traces", [])
            if isinstance(trace, Mapping) and trace.get("id")
        ]

    def _run_outcome_plan(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        build_ref = run.get("build") or run.get("build_path")
        if not build_ref:
            return {"datasets": [], "outcomes": [], "traces": []}
        return compile_outcome_plan(self.resolve_path(build_ref))

    def natural_trace(
        self,
        run_id: str,
        trace: str | None = None,
        event: str | None = None,
        actor: str | None = None,
        phase: int | None = None,
        depth: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """One causal chain from a run, as declared or as asked for.

        Traversal follows the causal parents and artifact lineage every run
        records, so it needs no knowledge of a study's processes or state
        fields. Only the starting point differs:

        * ``event`` walks from any recorded event — no declaration required;
        * ``actor`` walks from that actor's earliest recorded invocation
          (optionally within ``phase``);
        * ``trace`` uses a trace the package declares, whose seed is a row of a
          declared outcome dataset;
        * with none of them, the package's single declared trace is used, and a
          package declaring none (or several) is told what to pass.
        """
        events = self.trace_run(run_id)
        artifacts = self.artifacts_for_run(run_id)
        labels: dict[str, str] = {}
        bounded = depth
        cap = max_steps
        if event:
            seed, rule = str(event), f"event (event={event})"
        elif actor:
            candidates = [
                item
                for item in events
                if actor in [str(a) for a in (item.get("actors") or ())]
                and (phase is None or int(item.get("phase", 0)) == int(phase))
            ]
            if not candidates:
                where = "" if phase is None else f" at phase {phase}"
                raise ValueError(
                    f"ACTOR_TRACE_NOT_FOUND: no recorded invocation for actor '{actor}'{where}"
                )
            candidates.sort(key=lambda item: (item.get("phase", 0), item.get("commit_order", 0)))
            seed = str(candidates[0]["event_id"])
            rule = f"actor (actor={actor}" + ("" if phase is None else f", phase={phase}") + ")"
        else:
            declared = [
                item
                for item in self._run_outcome_plan(run_id).get("traces", [])
                if isinstance(item, Mapping) and item.get("id")
            ]
            available = [str(item["id"]) for item in declared]
            if trace:
                chosen = next((item for item in declared if str(item["id"]) == trace), None)
                if chosen is None:
                    raise ValueError(
                        f"TRACE_NOT_DECLARED: run '{run_id}' declares no trace '{trace}'; "
                        f"declared traces are {available}"
                    )
            elif len(declared) == 1:
                chosen = declared[0]
            else:
                detail = (
                    "its package declares no trace"
                    if not declared
                    else f"its package declares several traces {available}"
                )
                raise ValueError(
                    f"TRACE_SELECTION_REQUIRED: {detail}; pass trace=, event= or actor= "
                    "to choose a starting point"
                )
            seed_spec = chosen.get("seed") or {}
            rows = self._dataset_rows(run_id, str(seed_spec.get("dataset", "")))
            seed = resolve_seed(
                rows,
                order_by=tuple(seed_spec.get("order_by") or ("phase", "commit_order")),
                select=str(seed_spec.get("select", "first")),
            )
            labels = {str(k): str(v) for k, v in (chosen.get("labels") or {}).items()}
            bounded = depth if depth is not None else int(chosen.get("depth", DEFAULT_DEPTH))
            if cap is None:
                cap = int(chosen.get("max_steps", DEFAULT_MAX_STEPS))
            rule = f"declared trace ({chosen['id']})"
        steps, coverage = build_chain(
            events,
            artifacts,
            seed,
            labels=labels,
            depth=bounded if bounded is not None else DEFAULT_DEPTH,
            max_steps=cap if cap is not None else DEFAULT_MAX_STEPS,
        )
        seed_step = next((item for item in steps if item["relation"] == "seed"), {})
        return {
            "run_id": run_id,
            "selection_rule": rule,
            "seed_event": seed,
            "coverage": coverage,
            "illustration": {
                "actors": seed_step.get("actors", []),
                "phase": seed_step.get("phase"),
                "steps": steps,
            },
        }

    def _round_annotated_state(
        self,
        run_id: str,
        run: Mapping[str, Any],
        event_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Committed state snapshots annotated with the round that produced them.

        One preparation shared by every consumer of state evidence. Outcome
        evaluation and trace seeding must see the same rows: preparing them
        separately let a ``each_completed_round`` dataset return one row per
        state commit on one path and one row per completed round on the other.

        F8: each snapshot carries the protocol phase of the event that committed
        it, so ``each_completed_round`` selects the final snapshot of a round
        rather than one per process invocation. F4: a phase counts as a
        COMPLETED round only when nothing in it failed or was left active, and
        an interrupted frontier is not proof of completion.
        """
        version_phase: dict[int, int] = {}
        for event in event_rows:
            version = event.get("state_version")
            phase = event.get("phase")
            if isinstance(version, int) and isinstance(phase, int):
                version_phase[version] = phase
        latest_attempts: dict[str, dict[str, Any]] = {}
        for event in sorted(event_rows, key=lambda row: row.get("commit_order", 0)):
            invocation = str(event.get("invocation_id") or event.get("event_id"))
            latest_attempts[invocation] = event
        failed_or_active_phases = {
            int(event.get("phase", 0))
            for event in latest_attempts.values()
            if event.get("kind") in {"process_failed", "process_active"}
        }
        if run.get("status") != "completed" and version_phase:
            failed_or_active_phases.add(max(version_phase.values()))
        state_rows: list[dict[str, Any]] = []
        for version, snapshot in self.persistence.list_state_history(run_id):
            row = dict(snapshot)
            row["state_version"] = version
            phase = version_phase.get(version)
            if phase is not None and phase in failed_or_active_phases:
                # Mark for exclusion rather than an arbitrary "own round".
                row["_round"] = _INCOMPLETE_ROUND
            else:
                row["_round"] = phase
            state_rows.append(row)
        return state_rows

    def _dataset_rows(self, run_id: str, dataset_id: str) -> list[dict[str, Any]]:
        """Materialize one declared dataset from a run's retained evidence."""
        if not dataset_id:
            raise ValueError("TRACE_SEED_UNKNOWN: the declared trace names no seed dataset")
        plan = self._run_outcome_plan(run_id)
        if not any(
            isinstance(item, Mapping) and str(item.get("id", "")) == dataset_id
            for item in plan.get("datasets", [])
        ):
            raise ValueError(f"TRACE_SEED_UNKNOWN: no declared dataset '{dataset_id}'")
        event_rows = [dict(item) for item in self.trace_run(run_id)]
        sources = {
            "events": event_rows,
            "artifacts": list(self.artifacts_for_run(run_id)),
            "state": self._round_annotated_state(run_id, self.get_run(run_id), event_rows),
        }
        return materialize_datasets(plan, sources).get(dataset_id, [])

    def import_run(
        self,
        source: str | Path,
        run_id: str | None = None,
        *,
        size_limit_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> dict[str, Any]:
        # All registry rows commit together; a rejected bundle leaves no run.
        with self.persistence._lock:
            connection = self.persistence.connection
            connection.execute("SAVEPOINT import_run")
            try:
                result = self._import_run(source, run_id, size_limit_bytes=size_limit_bytes)
                connection.execute("RELEASE SAVEPOINT import_run")
                return result
            except Exception:
                connection.execute("ROLLBACK TO SAVEPOINT import_run")
                connection.execute("RELEASE SAVEPOINT import_run")
                raise

    def _import_run(
        self,
        source: str | Path,
        run_id: str | None = None,
        *,
        size_limit_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Import an exported run bundle (see export_run) for exploration.

        Restores the run row, events, and artifacts from the bundle's
        run_manifest.json / events.json / artifacts.json (integrity-verified)
        so trace explorer, outcome evaluation, and export views can inspect
        runs executed elsewhere. Imported runs are exploration copies: replay
        machinery is not reconstructed.
        """
        import hashlib
        import sqlite3

        source_path = self.resolve_path(source)
        if not source_path.is_dir():
            raise ValueError("IMPORT_SOURCE: source must be a bundle directory")
        for required in ("run_manifest.json", "events.json", "artifacts.json"):
            if not (source_path / required).is_file():
                raise ValueError(f"IMPORT_RUN: bundle missing {required}")
        # EVD preflight: verify the bundle manifest member digests and reject
        # unsafe/duplicate normalized paths before publishing anything.
        bundle_manifest = verify_bundle_manifest_and_size(
            source_path, size_limit_bytes=size_limit_bytes
        )
        integrity_path = source_path / "integrity.json"
        if integrity_path.is_file() and bundle_manifest is None:
            try:
                manifest = json.loads(integrity_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError("IMPORT_INTEGRITY: bundle integrity manifest is invalid") from exc
            # F6: the legacy path must apply the same protections as the modern
            # bundle manifest — required-file coverage, contained relative
            # paths, actual member sizes and the size limit. An empty or
            # incomplete integrity manifest must not bypass these checks.
            if not isinstance(manifest, dict):
                raise ValueError("IMPORT_INTEGRITY: bundle integrity manifest is invalid")
            total = 0
            seen: set[str] = set()
            for relative, digest in manifest.items():
                if not _is_contained_relative(str(relative)):
                    raise ValueError(f"IMPORT_INTEGRITY: unsafe member path '{relative}'")
                if relative in seen:
                    raise ValueError(f"IMPORT_INTEGRITY: duplicate member path '{relative}'")
                seen.add(relative)
                asset = source_path / relative
                if not asset.is_file() or asset.is_symlink():
                    raise ValueError(f"IMPORT_INTEGRITY: bundle member '{relative}' is missing")
                total += asset.stat().st_size
                actual = hashlib.sha256(asset.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(
                        f"IMPORT_INTEGRITY: bundle member '{relative}' fails its digest"
                    )
            present = {
                path.relative_to(source_path).as_posix()
                for path in source_path.rglob("*")
                if path.is_file() and path.name != "integrity.json"
            }
            unlisted = sorted(present - seen)
            if unlisted:
                raise ValueError(
                    "IMPORT_INTEGRITY: incomplete member coverage; unlisted files: "
                    + ", ".join(unlisted[:10])
                )
            if total > size_limit_bytes:
                raise ValueError(f"IMPORT_SIZE: bundle exceeds {size_limit_bytes} bytes")
        # F3: a bundle with NEITHER a modern bundle_manifest.json NOR the
        # legacy integrity.json offers no supported membership/integrity
        # contract to verify. Such a bundle is not a product of this exporter;
        # requiring a supported manifest closes the gap where removing both
        # manifests bypassed all integrity and size enforcement.
        if bundle_manifest is None and not integrity_path.is_file():
            raise ValueError(
                "IMPORT_MANIFEST_REQUIRED: bundle has neither bundle_manifest.json "
                "nor integrity.json; refusing to import an unsupported unverified "
                "bundle (no member coverage, containment, digest or size checks "
                "could run)"
            )
        run_manifest = json.loads((source_path / "run_manifest.json").read_text())
        # Digests prove the bundle was not corrupted in transit; they cannot
        # show that the manifest is internally consistent. A manifest whose
        # seeds do not derive from the identity it states cannot reconstruct
        # its own run, so it is rejected rather than imported as evidence.
        self.verify_manifest_seeds(run_manifest)
        target_id = run_id or str(run_manifest.get("run_id", ""))
        if not target_id:
            raise ValueError("IMPORT_RUN: run_manifest has no run_id")
        try:
            self.persistence.get_run(target_id)
            raise ValueError(f"ALREADY_EXISTS: run '{target_id}' already imported")
        except KeyError:
            pass
        events = json.loads((source_path / "events.json").read_text())
        artifacts = json.loads((source_path / "artifacts.json").read_text())
        outcomes_path = source_path / "outcomes.json"
        outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else []
        if not isinstance(outcomes, list) or any(not isinstance(row, dict) for row in outcomes):
            raise ValueError("IMPORT_RUN: outcomes must be a list of records")
        # EVD-003: preserve the original run identity separately from the local
        # import identity; imported outcomes are visibly labelled as snapshots.
        origin = {}
        if bundle_manifest is not None:
            source_run_id = str(bundle_manifest.get("source_run_id", ""))
            if source_run_id:
                origin["source_run_id"] = source_run_id
        imported_manifest = dict(run_manifest)
        imported_manifest.setdefault("origin", {}).update(origin)
        imported_manifest["imported_outcomes"] = "snapshot"
        # F3: restore and register the verified executable build when the
        # bundle carries one, so replay/reexecution capabilities are real.
        restored_build, build_restored = self._reconstruct_imported_build(
            source_path, bundle_manifest, target_id
        )
        if build_restored:
            imported_manifest["build_restored"] = True
        run_payload = {
            "id": target_id,
            "manifest": imported_manifest,
            "build": restored_build,
            "imported_outcomes": outcomes,
            "study_id": run_manifest.get("study_id", ""),
            "status": str(run_manifest.get("status", "completed")),
            # F1: replay reads these from the run row, not the manifest. The
            # pinned experimental configuration (condition, factors,
            # replication) is lifted here so a replayed import inherits the
            # source treatment instead of rebasing to base/1.
            "condition_id": str(run_manifest.get("condition_id", "base")),
            "replication": int(run_manifest.get("replication", 1)),
            "condition": dict(run_manifest.get("condition") or {}),
        }
        # Direct insert: an imported run reflects a COMPLETED foreign run and
        # does not pass through the created->running lifecycle of local runs.
        encoded = self.persistence._encode_record(run_payload)
        try:
            self.persistence.connection.execute(
                "INSERT INTO runs(run_id, status, version, experiment_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (target_id, "completed", 1, run_manifest.get("experiment_id"), encoded),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"ALREADY_EXISTS: run '{target_id}' already imported") from exc
        object_root = self.workspace / ".genesis" / "objects"

        def _store(payload_json: str, media_type: str = "application/json") -> str:
            digest = hashlib.sha256(payload_json.encode()).hexdigest()
            (object_root / digest[:2]).mkdir(parents=True, exist_ok=True)
            (object_root / digest[:2] / digest[2:]).write_bytes(payload_json.encode())
            self.persistence._record_object(
                ObjectRef(digest=digest, media_type=media_type, size=len(payload_json.encode()))
            )
            return digest

        # Event and artifact ids are globally unique by design: if any id from
        # this bundle already exists, the run (or an overlapping run) is already
        # present. Import is therefore strictly idempotent and refuses duplicates
        # instead of silently producing an empty trace under a fresh run id.
        burst = [str(e.get("event_id", "")) for e in events[:50] if e.get("event_id")]
        if burst:
            placeholders = ",".join(["?"] * len(burst))
            existing = self.persistence.connection.execute(
                f"SELECT event_id FROM events WHERE event_id IN ({placeholders}) LIMIT 1",
                burst,
            ).fetchone()
            if existing is not None:
                raise ValueError(
                    "ALREADY_EXISTS: event ids from this bundle are already present "
                    "(the run was already imported)"
                )
        for event in events:
            payload_json = json.dumps(event, sort_keys=True, default=str)
            digest = _store(payload_json)
            event_hash = hashlib.sha256(
                f"{event.get('event_id', '')}:{digest}".encode()
            ).hexdigest()
            self.persistence.connection.execute(
                """INSERT INTO events(
                       event_id, run_id, kind, payload_ref, event_hash, commit_hash
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.get("event_id"),
                    target_id,
                    event.get("kind", "process_completed"),
                    digest,
                    event_hash,
                    event_hash,
                ),
            )
        for artifact in artifacts:
            payload_json = json.dumps(artifact["payload"], sort_keys=True, default=str)
            digest = _store(payload_json, str(artifact.get("media_type", "application/json")))
            self.persistence.connection.execute(
                """INSERT INTO artifacts(artifact_id, run_id, payload_ref)
                   VALUES (?, ?, ?)""",
                (artifact.get("artifact_id"), target_id, digest),
            )
        return {
            "run_id": target_id,
            "status": "imported",
            "events": len(events),
            "artifacts": len(artifacts),
        }

    def import_package(
        self,
        source: str | Path,
        specification_id: str | None = None,
        *,
        size_limit_bytes: int = 100 * 1024 * 1024,
    ) -> dict[str, Any]:
        """Import a canonical package directory as a new draft (ACC-001).

        Validates the canonical YAML, verifies an integrity manifest when one is
        present, enforces a total size limit, copies all asset directories
        (prompts, schemas, data, extensions), and reconstructs the full editable
        form so guided edits never erase imported detail (finding 10).
        """
        source_path = self.resolve_path(source)
        if not source_path.is_dir():
            raise ValueError("IMPORT_SOURCE: source must be a package directory")
        loaded = StudyCompiler(source_path)._load()  # schema validation, raises on invalid
        study = loaded["study"]
        spec_id = self._validate_specification_id(specification_id or study.study_id)
        target = self._specification_dir(spec_id)
        if target.exists():
            raise ValueError(f"ALREADY_EXISTS: specification '{spec_id}' already exists")
        integrity_path = source_path / "integrity.json"
        if integrity_path.is_file():
            try:
                manifest = json.loads(integrity_path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError("IMPORT_INTEGRITY: package integrity manifest is invalid") from exc
            for relative, digest in manifest.items():
                asset = source_path / relative
                actual = hashlib.sha256(asset.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(
                        f"IMPORT_INTEGRITY: package member '{relative}' fails its digest"
                    )
        total = 0
        for path in source_path.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        if total > size_limit_bytes:
            raise ValueError(f"IMPORT_LIMIT: package size {total} exceeds limit {size_limit_bytes}")
        target.mkdir(parents=True, exist_ok=True)
        for name in ("study", "openness", "theory", "domain", "protocol", "outcomes", "models"):
            shutil.copy(source_path / f"{name}.yaml", target / f"{name}.yaml")
        for relative in ("prompts", "schemas", "data", "extensions"):
            child = source_path / relative
            if child.is_dir():
                shutil.copytree(child, target / relative)
        form = self._form_from_package(loaded, source_path, spec_id)
        metadata = {
            "id": spec_id,
            "title": study.title,
            "description": study.description,
            "version": 1,
            "status": "draft",
            "form": form,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        persist_checklist(target, checklist_record(target))
        metadata["snapshot_digest"] = self._snapshot_package(target)
        self.persistence.record_package_version(
            spec_id, 1, self._package_content_hash(target), None, "draft", metadata
        )
        return metadata

    @staticmethod
    def _form_from_package(
        loaded: dict[str, Any], source_path: Path, spec_id: str
    ) -> dict[str, Any]:
        """Reconstruct the editable form from the canonical package (finding 10)."""
        study = loaded["study"]
        openness = loaded["openness"]
        theory = loaded["theory"]
        domain = loaded["domain"]
        protocol = loaded["protocol"]
        outcomes = loaded["outcomes"]
        models = loaded["models"]

        def block_fields(value: Any, allowed: frozenset[str]) -> dict[str, Any]:
            dumped = value.model_dump(mode="json")
            return {key: dumped[key] for key in allowed if key in dumped}

        prompts: dict[str, str] = {}
        prompt_dir = source_path / "prompts"
        if prompt_dir.is_dir():
            for path in sorted(prompt_dir.glob("*.txt")):
                prompts[path.stem] = path.read_text()
        form: dict[str, Any] = {
            "id": spec_id,
            "title": study.title,
            "description": study.description,
            "owners": study.owners,
            "source_citations": study.source_citations,
            "artifact_refs": study.artifact_refs,
            "processes": [p.model_dump(mode="json") for p in openness.processes],
            "theory": block_fields(theory, _THEORY_BLOCK_FIELDS),
            "domain": block_fields(domain, _DOMAIN_BLOCK_FIELDS),
            "protocol": block_fields(protocol, _PROTOCOL_BLOCK_FIELDS),
            "outcomes": [o.model_dump(mode="json") for o in outcomes.outcomes],
            "models": [m.model_dump(mode="json") for m in models.models],
            "prompts": prompts,
            "schemas": GenesisService._read_schema_files(source_path),
        }
        return {
            key: value
            for key, value in form.items()
            if key in {"schemas", "prompts"} or value not in ({}, [])
        }

    @staticmethod
    def _read_schema_files(source_path: Path) -> dict[str, Any]:
        schemas: dict[str, Any] = {}
        for suffix in ("*.yaml", "*.yml", "*.json"):
            for path in sorted((source_path / "schemas").glob(suffix)):
                schemas[path.stem] = yaml.safe_load(path.read_text())
        return schemas

    def doctor(self) -> dict[str, Any]:
        """Operational diagnostics: versions, schema, WAL, object integrity (AW-20)."""
        checks: dict[str, Any] = {}
        database = self.workspace / ".genesis/genesis.db"
        checks["database_present"] = database.is_file()
        checks["schema_current"] = (
            self.persistence.schema_version() == self.persistence.current_schema_version
        )
        checks["wal_mode"] = (
            self.persistence.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        )
        integrity_failures = 0
        inspected = 0
        for path in self.persistence.object_store.root.glob("??/*"):
            if not path.is_file():
                continue
            inspected += 1
            digest = path.parent.name + path.name
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    integrity_failures += 1
            except OSError:
                integrity_failures += 1
        checks["objects_inspected"] = inspected
        checks["object_integrity_failures"] = integrity_failures
        checks["credential_secrets_stored"] = False  # env-only credential policy
        try:
            usage = shutil.disk_usage(self.workspace)
            checks["disk_free_bytes"] = usage.free
            checks["disk_ok"] = usage.free > 100 * 1024 * 1024
        except OSError as exc:
            checks["disk_ok"] = False
            checks["disk_error"] = str(exc)
        checks["workspace_writable"] = os.access(self.workspace, os.W_OK)
        objects_dir = self.workspace / ".genesis" / "objects"
        checks["objects_readable"] = os.access(objects_dir, os.R_OK)
        credentials: dict[str, bool] = {
            str(profile_id): bool(os.environ.get(str(profile.get("api_key_env", ""))))
            for profile_id, profile in self._load_model_profiles().items()
        }
        checks["credential_presence"] = credentials
        healthy = all(
            [
                checks["database_present"],
                checks["schema_current"],
                checks["wal_mode"],
                integrity_failures == 0,
                checks.get("disk_ok", True),
                checks.get("workspace_writable", True),
                checks.get("objects_readable", True),
            ]
        )
        return {
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "workspace": str(self.workspace),
            "checks": checks,
        }

    def backup_run(self, run_id: str, destination: str | Path) -> dict[str, Any]:
        """Back up the entire local database (SQLite safe backup)."""
        self.get_run(run_id)
        target = self.resolve_path(destination)
        path = self.persistence.backup_to(target)
        return {"path": str(path), "status": "backed-up"}

    def close(self) -> None:
        self.persistence.close()

    def __enter__(self) -> GenesisService:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

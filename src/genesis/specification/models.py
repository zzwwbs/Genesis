"""Pydantic representation of the seven canonical GENESIS YAML artifacts.

The models deliberately reject unrecognised fields. Extension data must be put in the
explicit ``extensions`` namespace so that typos cannot silently alter a study.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

StableId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")]
SemVer = Annotated[
    str,
    StringConstraints(pattern=r"^(?:0|[1-9][0-9]*)\.[0-9]+(?:\.[0-9]+)?(?:-[0-9A-Za-z.-]+)?$"),
]
Origin = Literal[
    "researcher", "imported", "assistant_proposed", "synthetic", "endogenous", "derived"
]
ApprovalStatus = Literal["unresolved", "proposed", "confirmed", "rejected"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_identifier_fields(self) -> StrictModel:
        import re

        pattern = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*"
        for name, value in self.__dict__.items():
            if name == "id" or name.endswith("_id") or name.endswith("_ref"):
                values = value if isinstance(value, list) else [value]
                if any(
                    item is not None
                    and (not isinstance(item, str) or not re.fullmatch(pattern, item))
                    for item in values
                ):
                    raise ValueError(f"{name} must contain stable lowercase kebab-case identifiers")
            if name.endswith("_refs") and isinstance(value, list):
                if any(
                    not isinstance(item, str) or not re.fullmatch(pattern, item) for item in value
                ):
                    raise ValueError(f"{name} must contain stable lowercase kebab-case identifiers")
        return self


class OriginMetadata(StrictModel):
    origin: Origin
    evidence_refs: list[StableId] = Field(default_factory=list)
    rationale: str | None = None


class ApprovalMetadata(StrictModel):
    # Datetimes round-trip through YAML as ISO strings; strict mode is relaxed
    # for this metadata model only so canonical packages can be re-parsed.
    model_config = ConfigDict(strict=False)

    status: ApprovalStatus = "unresolved"
    evidence_refs: list[StableId] = Field(default_factory=list)
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None

    @model_validator(mode="after")
    def confirmation_consistency(self) -> ApprovalMetadata:
        if self.status == "confirmed" and (not self.confirmed_by or self.confirmed_at is None):
            raise ValueError("confirmed approval requires confirmed_by and confirmed_at")
        if self.status != "confirmed" and (
            self.confirmed_by is not None or self.confirmed_at is not None
        ):
            raise ValueError("only confirmed approval may carry confirmation fields")
        return self


class PackageCompatibility(StrictModel):
    min_genesis_version: str
    max_genesis_version: str | None = None
    schema_versions: dict[str, str] = Field(default_factory=dict)


class CanonicalArtifact(StrictModel):
    schema_version: SemVer
    study_id: StableId
    extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("extensions")
    @classmethod
    def extension_namespaces(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        import re

        if any(not re.fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", name) for name in value):
            raise ValueError("extension names must be lowercase namespaced identifiers")
        return value

    @field_validator("schema_version")
    @classmethod
    def schema_version_format(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]+(?:\.[0-9]+)?(?:-[0-9A-Za-z.-]+)?", value):
            raise ValueError("schema_version must be semantic major.minor[.patch]")
        return value

    @field_validator("study_id")
    @classmethod
    def study_identifier(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value):
            raise ValueError("study_id must be a stable lowercase kebab-case identifier")
        return value


class StudySpec(CanonicalArtifact):
    title: str
    description: str | None = None
    owners: list[str] = Field(default_factory=list)
    source_citations: list[str] = Field(default_factory=list)
    artifact_refs: list[StableId] = Field(default_factory=list)
    package_compatibility: PackageCompatibility | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class ExecutorBinding(StrictModel):
    mode: Literal[
        "deterministic",
        "stochastic",
        "generative",
        "semantic-evaluator",
        "rule",
        "state-transition",
        "computational",
        "extension",
        "recorded_artifact",
    ] = "deterministic"
    model_profile: StableId | None = None
    extension_ref: StableId | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def required_binding(self) -> ExecutorBinding:
        if self.mode in {"generative", "semantic-evaluator"} and not self.model_profile:
            raise ValueError(f"{self.mode} executor requires model_profile")
        if self.mode == "extension" and not self.extension_ref:
            raise ValueError("extension executor requires extension_ref")
        if self.mode == "rule" and not isinstance(self.parameters.get("rules"), list):
            raise ValueError("rule executor requires parameters.rules")
        if self.mode == "state-transition" and not isinstance(
            self.parameters.get("operations"), list
        ):
            raise ValueError("state-transition executor requires parameters.operations")
        return self


class OutputSpec(StrictModel):
    artifact_type: StableId
    schema_ref: StableId


class TracePolicy(StrictModel):
    record_context: bool = True
    record_raw_response: bool = True
    retention: str | None = None


class ActorSelector(StrictModel):
    """Data-driven selection of actor instances for one process occurrence."""

    ids: list[StableId] | None = None
    source: str | None = None
    id_field: str = "id"
    fan_out: bool = True

    @model_validator(mode="after")
    def exactly_one_source(self) -> ActorSelector:
        if (self.ids is None) == (self.source is None):
            raise ValueError("actor selector requires exactly one of ids or source")
        if self.source is not None:
            import re

            if not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", self.source):
                raise ValueError("actor selector source must be a safe dotted path")
        if not self.id_field or "." in self.id_field:
            raise ValueError("actor selector id_field must be one field name")
        return self


class ProcessSpec(StrictModel):
    id: StableId
    name: str | None = None
    actors: ActorSelector | list[StableId] | None = None
    trigger: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    openness_rationale: str | None = None
    closure_rationale: str | None = None
    executor: ExecutorBinding
    context_policy: StableId
    prompt_ref: StableId | None = None
    inputs: list[StableId] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    state_effects: list[dict[str, Any]] = Field(default_factory=list)
    trace_policy: TracePolicy = Field(default_factory=TracePolicy)
    retry_policy: dict[str, Any] = Field(default_factory=dict)
    terminal_skip: bool = False
    # A measurement process observes the simulated world to produce a research
    # observable. Its outputs must not re-enter the behaviour being measured;
    # declaring it here lets that isolation be checked for any package instead
    # of by naming a particular study's processes.
    measurement: bool = False
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None

    @field_validator("context_policy")
    @classmethod
    def context_identifier(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", value):
            raise ValueError("context_policy must be a stable lowercase kebab-case identifier")
        return value

    @model_validator(mode="after")
    def validate_generative_contract(self) -> ProcessSpec:
        if self.executor.mode == "generative":
            missing = [
                name
                for name, value in {
                    "openness_rationale": self.openness_rationale,
                    "prompt_ref": self.prompt_ref,
                }.items()
                if not value
            ]
            if not self.outputs:
                missing.append("outputs")
            if missing:
                raise ValueError("generative process requires: " + ", ".join(missing))
        return self


class OpennessSpec(CanonicalArtifact):
    processes: list[ProcessSpec] = Field(default_factory=list)


class TheoryConstruct(StrictModel):
    id: StableId
    theory_role: str | None = None
    description: str | None = None


class TheoryProcessMapping(StrictModel):
    process: StableId
    theory_function: str


class TheoryExecutionBinding(StrictModel):
    """An explicit researcher-approved operational binding (spec §6.1).

    Every executable declaration must name one supported kind; ``annotation``
    is the only nonexecutable classification and is reported in the theory
    coverage report without changing the runtime. ``precedence`` maps a
    producer/consumer process pair; ``feedback_context`` binds a retained
    source to a consumer's context slot (with lag and an explicit initial
    policy); ``mechanism_binding`` verifies that one declared mechanism/
    transition already realizes the relation, adding no second transition.
    """

    kind: Literal["precedence", "feedback_context", "mechanism_binding", "annotation"]
    reason: str | None = None
    producer_process: StableId | None = None
    consumer_process: StableId | None = None
    lag_rounds: int = 0
    source: dict[str, Any] | None = None
    context_slot: str | None = None
    initial: dict[str, Any] | None = None
    mechanism: StableId | None = None


class TheoryRelation(StrictModel):
    """A between-object relation; ``source``/``target`` serialize as ``from``/``to``."""

    model_config = ConfigDict(populate_by_name=True)

    id: StableId | None = None
    source: StableId = Field(alias="from")
    target: StableId = Field(alias="to")
    relation: str
    execution: TheoryExecutionBinding | None = None


class TheoryFeedback(StrictModel):
    """A feedback edge from a retained domain object back to a process."""

    model_config = ConfigDict(populate_by_name=True)

    id: StableId | None = None
    source: StableId = Field(alias="from")
    target: StableId = Field(alias="to")
    relation: str
    execution: TheoryExecutionBinding | None = None


class TheoryDelay(StrictModel):
    """A process delay; executable delays must name an affected binding."""

    id: StableId | None = None
    process: StableId
    rounds: int = 0
    execution: TheoryExecutionBinding | None = None


class ActorSpec(StrictModel):
    id: StableId
    theory_role: str | None = None
    description: str | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class StateSpec(StrictModel):
    id: StableId
    value_type: str = "object"
    initial: Any | None = None
    persistence: str | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class MechanismSpec(StrictModel):
    id: StableId
    implements: str | None = None
    description: str | None = None


class InstitutionSpec(StrictModel):
    id: StableId
    type: str | None = None


class VisibilitySpec(StrictModel):
    id: StableId
    allow: list[str] = Field(default_factory=list)
    redact: list[str] = Field(default_factory=list)
    cardinality: dict[str, Any] = Field(default_factory=dict)
    aggregate: dict[str, Any] = Field(default_factory=dict)
    available_when: dict[str, Any] = Field(default_factory=dict)


class AvailabilitySpec(StrictModel):
    path: str
    available_when: dict[str, Any] = Field(default_factory=dict)


class UpdateRule(StrictModel):
    state: StableId
    op: str
    declared_by: StableId | None = None


class PersistenceRule(StrictModel):
    object: StableId
    policy: str | None = None


class DomainInitialization(StrictModel):
    mode: str = "researcher"
    data_source: str | None = None
    state_field: str | None = None
    origin: str | None = None


class TheoryObservable(StrictModel):
    id: StableId
    definition: str | None = None


class TheorySpec(CanonicalArtifact):
    theory_family: str
    constructs: list[TheoryConstruct] = Field(default_factory=list)
    process_mappings: list[TheoryProcessMapping] = Field(default_factory=list)
    relations: list[TheoryRelation] = Field(default_factory=list)
    feedback: list[TheoryFeedback] = Field(default_factory=list)
    delays: list[TheoryDelay] = Field(default_factory=list)
    observables: list[TheoryObservable] = Field(default_factory=list)


class AttributeSpec(StrictModel):
    id: StableId
    value_type: str = "string"
    description: str | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class ArtifactSpec(StrictModel):
    id: StableId
    artifact_type: str | None = None
    schema_ref: StableId | None = None
    owner: StableId | None = None
    # Study-defined vocabulary describing who the artifact is FOR (e.g.
    # "private", "platform"). Access is decided by the context policy that
    # admits it, not by this label; it documents intent and is what the
    # measurement-isolation check reads. Do not read it as an enforced boundary.
    visibility: str | None = None
    # How long an instance stays resolvable as an input. Round-equivalent
    # scopes ("round", "phase", "event", "invocation") are enforced at input
    # resolution; anything else persists for the run.
    lifecycle_scope: str | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class DomainSpec(CanonicalArtifact):
    actors: list[ActorSpec] = Field(default_factory=list)
    attributes: list[AttributeSpec] = Field(default_factory=list)
    states: list[StateSpec] = Field(default_factory=list)
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    mechanisms: list[MechanismSpec] = Field(default_factory=list)
    institutions: list[InstitutionSpec] = Field(default_factory=list)
    initialization: DomainInitialization = Field(default_factory=DomainInitialization)
    visibility: list[VisibilitySpec] = Field(default_factory=list)
    availability: list[AvailabilitySpec] = Field(default_factory=list)
    updates: list[UpdateRule] = Field(default_factory=list)
    persistence: list[PersistenceRule] = Field(default_factory=list)


class TimeModel(StrictModel):
    type: Literal["rounds", "continuous", "event"]
    start: int | float | str | None = None
    end: int | float | str | None = None
    step: int | float | None = None


class ProtocolFactor(StrictModel):
    id: StableId
    levels: list[str | int | float | bool] = Field(min_length=1)
    branchable: bool = Field(default=False)


class ProtocolPhase(StrictModel):
    id: StableId
    start: int | float
    end: int | float

    @model_validator(mode="after")
    def ordered_bounds(self) -> ProtocolPhase:
        if self.end < self.start:
            raise ValueError("protocol phase end must not precede start")
        return self


class ProtocolSpec(CanonicalArtifact):
    time_model: TimeModel
    termination: list[dict[str, Any]] = Field(default_factory=list)
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    factors: list[ProtocolFactor] = Field(default_factory=list)
    phases: list[ProtocolPhase] = Field(default_factory=list)
    replications: int = Field(default=1, ge=1)
    matching: dict[str, Any] = Field(default_factory=dict)
    random_streams: list[dict[str, Any]] = Field(default_factory=list)
    model_freezing: bool = True
    budgets: dict[str, Any] = Field(default_factory=dict)
    checkpoints: dict[str, Any] = Field(default_factory=dict)
    replay_retention: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_design(self) -> ProtocolSpec:
        factor_ids = [factor.id for factor in self.factors]
        if len(factor_ids) != len(set(factor_ids)):
            raise ValueError("protocol factor ids must be unique")
        phase_ids = [phase.id for phase in self.phases]
        if len(phase_ids) != len(set(phase_ids)):
            raise ValueError("protocol phase ids must be unique")
        ordered = sorted(self.phases, key=lambda phase: (phase.start, phase.end, phase.id))
        if any(
            current.start <= previous.end
            for previous, current in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("protocol phases must not overlap")
        return self


class OutcomeDatasetSource(StrictModel):
    """Typed source of one named outcome dataset (spec §7.1).

    ``events`` reads a nested record list from a dotted path on committed
    events (e.g. ``state_delta.analytics``); ``artifacts`` reads retained
    artifacts of a declared artifact type or process; ``state`` reads state
    snapshots (``final`` or ``each_completed_round``). Cumulative snapshots
    must not be mistaken for independent delta events.
    """

    kind: Literal["events", "artifacts", "state"]
    path: str | None = None
    artifact_type: StableId | None = None
    process: StableId | None = None
    state: StableId | None = None
    snapshot: Literal["final", "each_completed_round"] | None = None


class OutcomeDatasetField(StrictModel):
    """One derived field via a fixed operation-registry step (no code execution)."""

    name: str
    op: Literal["copy", "literal", "arithmetic", "comparison", "conditional"]
    value: Any | None = None
    field: str | None = None
    with_field: str | None = None
    operator: str | None = None
    condition: dict[str, Any] | None = None
    else_value: Any | None = None


class OutcomeDatasetFilter(StrictModel):
    """One equality/comparison predicate narrowing a dataset's rows."""

    field: str
    op: Literal["eq", "ne", "lt", "lte", "gt", "gte", "truthy", "falsy"] = "eq"
    value: Any | None = None


class OutcomeDataset(StrictModel):
    """A named, versioned row relation feeding one or more outcomes."""

    id: StableId
    source: OutcomeDatasetSource
    fields: list[OutcomeDatasetField] = Field(default_factory=list)
    # Narrowing belongs to the relation, not only to the outcomes reading it:
    # a trace seed selects rows without defining an outcome over them.
    where: list[OutcomeDatasetFilter] = Field(default_factory=list)
    deduplicate_on: list[str] = Field(default_factory=list)
    missing: Literal["retain_null", "drop"] = "retain_null"


class TraceSeed(StrictModel):
    """Which recorded invocation a declared trace starts from."""

    dataset: StableId
    order_by: list[str] = Field(default_factory=lambda: ["phase", "commit_order"])
    select: Literal["first", "last"] = "first"


class TraceSpec(StrictModel):
    """A declared trajectory trace (spec: inspectable traceability).

    Traversal is generic — it follows the causal parents and artifact lineage
    every run records — so a study declares only where a chain starts and what
    to call its steps.
    """

    id: StableId
    title: str | None = None
    seed: TraceSeed
    labels: dict[str, str] = Field(default_factory=dict)
    depth: int = 12
    # How many steps the chain returns before keeping only the nearest
    # relations. A caller may override it per request.
    max_steps: int = 40


class OutcomeSpec(StrictModel):
    id: StableId
    source: list[StableId] | StableId
    filters: list[dict[str, Any]] = Field(default_factory=list)
    grouping: list[str] = Field(default_factory=list)
    aggregation: dict[str, Any] = Field(default_factory=dict)
    missingness: dict[str, Any] = Field(default_factory=dict)
    window: dict[str, Any] | None = None
    join: dict[str, Any] | None = None
    output_schema: StableId | None = None
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class OutcomesSpec(CanonicalArtifact):
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    datasets: list[OutcomeDataset] = Field(default_factory=list)
    traces: list[TraceSpec] = Field(default_factory=list)


class ModelProfile(StrictModel):
    id: StableId
    provider: str
    model: str
    capabilities: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    endpoint_ref: str | None = None
    # Credentials are intentionally not representable in this canonical model.
    origin: OriginMetadata | None = None
    approval: ApprovalMetadata | None = None


class ModelsSpec(CanonicalArtifact):
    models: list[ModelProfile] = Field(default_factory=list)

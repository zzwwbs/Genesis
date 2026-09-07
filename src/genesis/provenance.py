"""Immutable in-memory provenance ledger and checkpoint primitives."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ProvenanceEvent:
    event_id: str
    run_id: str
    invocation_id: str
    process_id: str
    outputs: Any
    parent_events: tuple[str, ...]
    integrity_hash: str
    context_hash: str | None = None
    input_refs: tuple[str, ...] = ()
    state_delta: Any = None
    attempt_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    invocation_id: str
    attempt: int
    status: str
    provider_request_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    run_id: str
    source_event_id: str | None
    state: Any
    scheduler_frontier: tuple[str, ...]
    random_stream_states: Any
    integrity_hash: str


class ProvenanceLedger:
    def __init__(self) -> None:
        self.events: dict[str, ProvenanceEvent] = {}
        self.checkpoints: dict[str, Checkpoint] = {}
        self.attempts: dict[str, AttemptRecord] = {}

    def append_attempt(
        self,
        *,
        invocation_id: str,
        attempt: int,
        status: str,
        provider_request_id: str | None = None,
        error: str | None = None,
    ) -> AttemptRecord:
        record = AttemptRecord(
            str(uuid.uuid4()), invocation_id, attempt, status, provider_request_id, error
        )
        self.attempts[record.attempt_id] = record
        return record

    def append_event(
        self,
        *,
        run_id: str,
        invocation_id: str,
        process_id: str,
        outputs: Any,
        parent_events: tuple[str, ...] = (),
        context_hash: str | None = None,
        input_refs: tuple[str, ...] = (),
        state_delta: Any = None,
        attempt_ids: tuple[str, ...] = (),
    ) -> ProvenanceEvent:
        event_id = str(uuid.uuid4())
        payload = {
            "event_id": event_id,
            "run_id": run_id,
            "invocation_id": invocation_id,
            "process_id": process_id,
            "outputs": outputs,
            "parent_events": parent_events,
            "context_hash": context_hash,
            "input_refs": input_refs,
            "state_delta": state_delta,
            "attempt_ids": attempt_ids,
        }
        event = ProvenanceEvent(
            event_id,
            run_id,
            invocation_id,
            process_id,
            _freeze(outputs),
            tuple(parent_events),
            _digest(payload),
            context_hash,
            tuple(input_refs),
            _freeze(state_delta),
            tuple(attempt_ids),
        )
        self.events[event_id] = event
        return event

    def checkpoint(
        self,
        *,
        run_id: str,
        state: Any,
        scheduler_frontier: list[str] | tuple[str, ...],
        source_event_id: str | None = None,
        random_stream_states: Any = None,
    ) -> Checkpoint:
        checkpoint_id = str(uuid.uuid4())
        payload = {
            "checkpoint_id": checkpoint_id,
            "run_id": run_id,
            "source_event_id": source_event_id,
            "state": state,
            "scheduler_frontier": list(scheduler_frontier),
            "random_stream_states": random_stream_states,
        }
        cp = Checkpoint(
            checkpoint_id,
            run_id,
            source_event_id,
            _freeze(state),
            tuple(scheduler_frontier),
            _freeze(random_stream_states),
            _digest(payload),
        )
        self.checkpoints[checkpoint_id] = cp
        return cp

    def verify(self, event_id: str) -> bool:
        if event_id not in self.events:
            raise ValueError("unknown event")
        event = self.events[event_id]
        payload = {
            "event_id": event.event_id,
            "run_id": event.run_id,
            "invocation_id": event.invocation_id,
            "process_id": event.process_id,
            "outputs": event.outputs,
            "parent_events": event.parent_events,
            "context_hash": event.context_hash,
            "input_refs": event.input_refs,
            "state_delta": event.state_delta,
            "attempt_ids": event.attempt_ids,
        }
        if _digest(payload) != event.integrity_hash:
            raise ValueError("event integrity failure")
        return True

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        cp = self.checkpoints[checkpoint_id]
        payload = {
            "checkpoint_id": cp.checkpoint_id,
            "run_id": cp.run_id,
            "source_event_id": cp.source_event_id,
            "state": cp.state,
            "scheduler_frontier": list(cp.scheduler_frontier),
            "random_stream_states": cp.random_stream_states,
        }
        if _digest(payload) != cp.integrity_hash:
            raise ValueError("checkpoint integrity failure")
        return {
            "state": _plain(cp.state),
            "scheduler_frontier": list(cp.scheduler_frontier),
            "random_stream_states": _plain(cp.random_stream_states),
        }

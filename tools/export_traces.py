"""Trace exploration exporter: dump any event trace from the run database.

Usage:
    python tools/export_traces.py --run RUN_ID [--process PID] [--phase N]
                                  [--prefix ACTOR] [--limit N] [--out PATH]
                                  [--values FULL|KEYS|NONE]

Examples:
    # every user-act event of phase 1 (5 samples), with resolved inputs
    python tools/export_traces.py --run pilot-large-gate-3 --process user-act \
        --phase 1 --limit 5 --out docs/demos/evidence/traces/user-act-p1.json

    # all update-follow-relation events across phases
    python tools/export_traces.py --run pilot-large-gate-3 --process update-follow-relation \
        --out docs/demos/evidence/traces/update-follow.json

    # u3's interpretation decisions across all rounds
    python tools/export_traces.py --run pilot-large-gate-3 --process user-interpret \
        --prefix u3 --out docs/demos/evidence/traces/interpret-u3.json

    # settle records of another run (e.g. the pilot full-chain smoke)
    python tools/export_traces.py --run pilot-full-smoke-4 --process settle-article-outcome \
        --phase 1 --limit 3 --out docs/demos/evidence/traces/settle-smoke4.json

No model calls are made; only the sqlite run database is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = ROOT / "genesis-workspace"
DB = WORKSPACE / ".genesis" / "genesis.db"
OBJECTS = WORKSPACE / ".genesis" / "objects"


def take_matching_events(
    events,
    *,
    process: str | None,
    phase: int | None,
    prefix: str | None,
    limit: int | None,
    values: str,
) -> tuple[list[dict], set[str]]:
    """Events that match the filters, stopping once ``limit`` of them are in hand.

    ``--limit`` used to apply only while the output list was built, after every
    matching event had already been reconstructed.
    """
    matched: list[dict] = []
    needed: set[str] = set()
    for payload in events:
        if not payload:
            continue
        if process and payload.get("process_id") != process:
            continue
        if phase is not None and payload.get("phase") != phase:
            continue
        actors = payload.get("actors") or []
        if prefix and not any(str(actor).startswith(prefix) for actor in actors):
            continue
        matched.append(payload)
        if values != "NONE":
            needed.update(str(ref) for ref in (payload.get("input_refs") or ()))
        if limit and len(matched) >= limit:
            break
    return matched, needed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="pilot-large-gate-3")
    parser.add_argument("--process", default=None, help="process id filter (default: all)")
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--prefix", default=None, help="actor id prefix, e.g. u3 or w1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--values",
        choices=("FULL", "KEYS", "NONE"),
        default="FULL",
        help="how to include resolved input artifact values",
    )
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    from genesis.persistence import PersistenceCoordinator

    store = PersistenceCoordinator(DB, OBJECTS, read_only=True)
    try:
        run_rows = store.connection.execute("SELECT run_id, status FROM runs").fetchall()
        if not any(r[0] == args.run for r in run_rows):
            print("available runs:", [r[0] for r in run_rows])
            return 1
        # Reconstructed, not decoded row by row: an event is stored whole or as a
        # patch against its predecessor (STH-009), and a patch row decoded alone
        # yields {'fields': ...} -- most events of a long run came out with no
        # process id, so the filters below silently dropped them.
        # Filtered as they are reconstructed. Listing the run first held every
        # event at once (about 1.9 GB on the largest run) before any filter ran.
        matched, needed = take_matching_events(
            store.iter_events(args.run),
            process=args.process,
            phase=args.phase,
            prefix=args.prefix,
            limit=args.limit,
            values=args.values,
        )
        artifacts: dict[str, dict] = {}
        if needed:
            for row in store.iter_artifacts(args.run):
                artifact_id = str(row["artifact_id"])
                if artifact_id not in needed:
                    continue
                payload_bytes = row["payload"]
                try:
                    value = json.loads(payload_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    value = {}
                artifacts[artifact_id] = value if isinstance(value, dict) else {}
    finally:
        store.close()

    records = []
    for payload in matched:
        event_id = payload.get("event_id")
        kind = payload.get("kind")
        if not payload:
            continue
        if args.process and payload.get("process_id") != args.process:
            continue
        if args.phase is not None and payload.get("phase") != args.phase:
            continue
        actors = payload.get("actors") or []
        if args.prefix and not any(str(a).startswith(args.prefix) for a in actors):
            continue
        input_refs = list(payload.get("input_refs") or ())
        input_values = {}
        if args.values != "NONE":
            for ref in input_refs:
                meta = artifacts.get(ref) or {}
                value = meta.get("value")
                if args.values == "KEYS":
                    value = list(value.keys()) if isinstance(value, dict) else value
                input_values[ref] = value
        metadata = payload.get("metadata") or {}
        records.append(
            {
                "event_id": event_id,
                "kind": kind,
                "process_id": payload.get("process_id"),
                "phase": payload.get("phase"),
                "actors": actors,
                "invocation_id": payload.get("invocation_id"),
                "context_hash": payload.get("context_hash"),
                "input_refs": input_refs,
                "input_values": input_values,
                "parent_events": payload.get("parent_events"),
                "state_delta_keys": sorted((payload.get("state_delta") or {}).keys()),
                "state_delta": payload.get("state_delta"),
                "metadata": {
                    k: metadata.get(k)
                    for k in ("schema_valid", "latency_ms", "provider_attempts", "usage")
                    if k in metadata
                },
            }
        )
        if args.limit and len(records) >= args.limit:
            break

    out = {
        "run_id": args.run,
        "filters": {
            "process": args.process,
            "phase": args.phase,
            "prefix": args.prefix,
            "limit": args.limit,
            "values": args.values,
        },
        "count": len(records),
        "records": records,
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
        print(f"exported {len(records)} events -> {path}")
    else:
        print(json.dumps(out, indent=2, sort_keys=True, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())

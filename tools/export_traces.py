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
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = ROOT / "genesis-workspace"
DB = WORKSPACE / ".genesis" / "genesis.db"


def objs(digest: str):
    for cand in (
        WORKSPACE / ".genesis" / "objects" / digest[:2] / digest[2:],
        WORKSPACE / ".genesis" / "objects" / digest,
    ):
        try:
            return json.loads(cand.read_text())
        except Exception:
            continue
    return None


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

    con = sqlite3.connect(DB)
    try:
        run_rows = con.execute("SELECT run_id, status FROM runs").fetchall()
        if not any(r[0] == args.run for r in run_rows):
            print("available runs:", [r[0] for r in run_rows])
            return 1
        # events store payloads in the object store; process/phase/actor filters are
        # applied after decoding (bounded by the run's event count).
        events = con.execute(
            "SELECT event_id, payload_ref, kind FROM events WHERE run_id=?", (args.run,)
        ).fetchall()
    finally:
        con.close()

    artifacts: dict[str, dict] = {}
    con = sqlite3.connect(DB)
    try:
        for aid, pref in con.execute(
            "SELECT artifact_id, payload_ref FROM artifacts WHERE run_id=?", (args.run,)
        ).fetchall():
            value = objs(str(pref))
            artifacts[aid] = value or {}
    finally:
        con.close()

    records = []
    for event_id, pref, kind in events:
        payload = objs(str(pref))
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

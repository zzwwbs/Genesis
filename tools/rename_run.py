"""Re-keyed run copy: duplicate a run under a new id (all associated names).

For presentation/illustration: the run row, every event payload, artifact
payload, input reference, parent link, and the manifest are rewritten so the
old run id no longer appears anywhere; the original run is untouched.

    python tools/rename_run.py --run pilot-large-gate-3 --new-id baseline-realization-1

The copy is an exploration-grade duplicate (same evidence, new identity);
replay-grade provenance belongs to the original run row.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--new-id", default="baseline-realization-1")
    args = parser.parse_args()

    con = sqlite3.connect(DB)
    try:
        row = con.execute("SELECT payload_json FROM runs WHERE run_id=?", (args.run,)).fetchone()
        if not row:
            print(f"run {args.run} not found")
            return 1
        run_payload = json.loads(row[0])
        manifest = run_payload.get("manifest") or {}
        event_ids = [
            r[0]
            for r in con.execute(
                "SELECT event_id FROM events WHERE run_id=?", (args.run,)
            ).fetchall()
        ]
        artifact_ids = [
            r[0]
            for r in con.execute(
                "SELECT artifact_id FROM artifacts WHERE run_id=?", (args.run,)
            ).fetchall()
        ]
    finally:
        con.close()

    events = []
    con = sqlite3.connect(DB)
    try:
        for pref in con.execute(
            "SELECT payload_ref FROM events WHERE run_id=? ORDER BY rowid", (args.run,)
        ).fetchall():
            events.append(objs(str(pref[0])))
        artifacts = []
        for aid, pref in con.execute(
            "SELECT artifact_id, payload_ref FROM artifacts WHERE run_id=? ORDER BY artifact_id",
            (args.run,),
        ).fetchall():
            artifacts.append({"artifact_id": aid, "payload": objs(str(pref[0]))})
    finally:
        con.close()

    old, new = args.run, args.new_id

    def rekey(value: object) -> object:
        if isinstance(value, dict):
            return {rekey(k): rekey(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rekey(v) for v in value]
        if isinstance(value, str):
            return value.replace(old, new)
        return value

    events = rekey(events)
    artifacts = rekey(artifacts)
    manifest = rekey(manifest)
    manifest["run_id"] = new
    source_build = run_payload.get("build") or run_payload.get("build_path") or ""
    if source_build:
        manifest["build"] = source_build
    run_payload["manifest"] = manifest
    run_payload["id"] = new

    # write a bundle for service.import_run
    bundle = WORKSPACE / "imports" / f"renamed-{new}"
    if bundle.is_dir():
        import shutil

        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    (bundle / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (bundle / "events.json").write_text(json.dumps(events, default=str))
    (bundle / "artifacts.json").write_text(json.dumps(artifacts, default=str))
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "integrity.json")
    integrity = {
        p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    (bundle / "integrity.json").write_text(json.dumps(integrity, indent=2, sort_keys=True))

    from genesis.service import GenesisService

    service = GenesisService(WORKSPACE)
    try:
        result = service.import_run(bundle, run_id=new)
        leftovers = (
            sum(str(e).count(old) for e in events)
            + sum(str(e).count(old) for e in artifacts)
            + sum(str(e).count(old) for e in manifest)
        )
        result["old_id_occurrences_remaining"] = leftovers
        print(json.dumps(result, default=str))
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

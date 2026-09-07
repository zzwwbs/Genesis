# GENESIS

GENESIS is a local-first generative agent-based modelling system for researchers studying complex social dynamics. Release 1 is a Python modular monolith: study specifications are authored as versioned YAML, compiled into immutable builds, executed locally, and persisted in SQLite with content-addressed objects. Analytical exports can be queried with the lightweight analysis layer and extended later with DuckDB/Parquet.

This README doubles as a run recipe for an AI agent: it explains how to author, compile, execute, and inspect a study end to end using only this repository — no pre-built experiment package is required.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -q        # optional: run the test suite
genesis version
genesis serve --workspace ./genesis-workspace
```

The API binds to `127.0.0.1` by default. Open http://127.0.0.1:8000/ui for the browser workflow or http://127.0.0.1:8000/docs for the API console.

## Configure an OpenAI-compatible model

Generative processes call a model through a **model profile**; deterministic-only studies need none. In `/ui` → Models: enter a profile ID, compatible base URL, model name, and the name of an environment variable holding the API key (for example `OPENAI_API_KEY`). Set it before starting the server:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace --port 8765
```

Save the profile, use **Check key** and **Test connection**, then select the profile for generative processes. The key is read by the server at request time and is never stored in the profile, study YAML, SQLite database, build, or export. Compilation makes no provider calls; model calls occur only during an explicit connection test or execution of an approved run.

## The browser workflow (approval-gated)

The `/ui` workflow is deliberately approval-gated:

1. **Models** — register and test a model profile (above).
2. **Specification** — author the study through the three-layer elicitation (staged workflow declared in `workflows/three-layer-study/`): study foundation → openness → theory → domain → experiment design. Each stage: the assistant asks clarifying questions (three editable suggestions per question) → request a draft → review the preview (YAML diff, files, evidence, assumptions, warnings) → approve explicitly; approving writes a new immutable package version. Revisions mark dependent downstream stages `needs_review`. After the final approval the package is approved and versioned.
3. **Run** — compile the approved specification (15 JSON artifacts under `builds/<study>/`, immutable build hash), then create and execute a run. Every run freezes a manifest: build/source/compiler hashes, model and prompt versions, seeds, and data provenance.
4. **Outputs** — inspect the run: dashboard counts, outcome evaluation, full run trace, phase-grouped trace explorer, natural trace (frozen rule), per-user traces, and the exportable evidence bundle. Runs executed elsewhere can be imported from an exported bundle.

Alternatively, author the canonical package directly as YAML and compile headlessly (below), or use the chat elicitation API (`POST /elicitation/sessions` with `workflow_id: three-layer-study`) for scripted authoring.

## The canonical study package

A study is 7 YAML files plus optional `prompts/`, `schemas/`, `data/` directories:

- `study.yaml` — identity and summary
- `theory.yaml` — constructs, theory functions, relations, delays
- `openness.yaml` — processes: generative (model-backed, schema-closed) or computational (Python entry points), with context policies
- `domain.yaml` — states, actors, context-visibility allow-lists, artifact catalog
- `protocol.yaml` — time model (rounds), conditions, replications, random streams
- `outcomes.yaml` — outcome measures over events/artifacts/states
- `models.yaml` — model profiles used by generative processes

## Headless run (scripted agent)

```python
from genesis.service import GenesisService
svc = GenesisService("./genesis-workspace")
draft = svc.create_specification({...})            # or import a YAML package
approved = svc.approve_specification("<id>", draft["version"], "agent")
compiled = svc.compile_study(None, "builds/<id>", specification_id="<id>")
svc.create_run({"id": "run-1", "study_id": "<id>", "build": compiled["path"]})
svc.execute_run("run-1")                            # deterministic-only? pass executor_overrides
events = svc.trace_run("run-1")
svc.export_run("run-1", "exports/run-1")
```

## Implementing processes

- **Computational** processes declare `executor.parameters.entry_point = "module:function"`. The function receives a `ProcessInvocation` with `actor_ids`, `phase`, a read-only `context` scoped by the process's context policy (allow-listed state; mutable collections are frozen — use `collections.abc.Mapping`, never `isinstance(dict)`), and `inputs` (resolved artifact instances). Return a `ProcessResult` with `outputs` and declared `state_effects`. Reference implementations live in `tests/executor_functions.py`; contracts are exercised by `tests/test_actor_instance_execution.py` and `tests/test_artifact_instance_routing.py`.
- **Generative** processes declare `prompt_ref` (a `prompts/*.txt` template supporting `{context}`, `{actor_ids}`, `{phase}`), an `outputs` artifact with a JSON schema, and `openness_rationale`/`closure_rationale`. The model must answer with schema-valid JSON only; schemas use single JSON types (union `type` arrays are not supported by validation).

## Key concepts for agents

- **Prior-phase dependency semantics** — `dependencies.after` gates a process on the dependency's last completed phase; use it for one-round causal latency (e.g., performance → reflection → next-round strategy).
- **Terminal settlement** — add a settlement-only phase after the final content round: content processes declare `terminal_skip: true` (default) and only settlement runs there. Set `GENESIS_TERMINAL_PHASE=<end>` before launching for run-scale control.
- **Measurement isolation** — processes that label or score generated content must stay out of causal context: their artifacts must not appear in any creator/user-facing context policy, and the processes must not declare `state_effects`. `tools/check_measurement_isolation.py` verifies this on a package.
- **Immutability & provenance** — builds, events, artifacts, and state snapshots are content-hashed; replays (full/artifact/partial/branch) reconstruct traces from the same records; exported run bundles carry `run_manifest.json` + `events.json` + `artifacts.json` plus an `integrity.json` digest manifest and can be imported into any workspace (`tools/export_traces.py`, `tools/rename_run.py` for trace slicing and re-keyed copies).
- **Trace selection for evidence** — pick ONE trace per run under a rule frozen *before* execution (e.g., the first naturally occurring target event, else a documented fallback), then render it deterministically with `GET /runs/{id}/natural-trace`. Post-hoc browsing (`?user=&phase=`) is for inspection, not evidence.

## Package layout

- `genesis.specification` — canonical Pydantic schemas and validation contracts.
- `genesis.compiler` — package loading, cross-reference validation, deterministic builds, integrity manifests.
- `genesis.persistence` — SQLite WAL coordinator and content-addressed object storage.
- `genesis.runtime` — immutable invocation/result contracts, context/state/artifact services, scheduler, executors, run controller.
- `genesis.providers` — provider-neutral model adapters, deterministic mock, recorded-artifact provider.
- `genesis.provenance` / `genesis.replay` — immutable lineage and replay modes.
- `genesis.analysis` — declarative outcome evaluation and JSON/CSV export helpers.

## Useful endpoints

| Purpose | Endpoint |
|---|---|
| List runs / studies / builds | `GET /runs` · `GET /studies` · `GET /builds` |
| Trace events | `GET /runs/{id}/events` |
| Outcome evaluation | `GET /runs/{id}/outcomes` |
| Natural trace / per-user trace | `GET /runs/{id}/natural-trace[?user=&phase=]` |
| Compiled process graph | `GET /builds/{ref}/processes` |
| Export / import run bundles | `POST /exports` · `POST /runs/import` |
| Replays | `POST /runs/{id}/replays` (`full`/`artifact`/`partial`/`branch`) |
| Elicitation | `POST /elicitation/sessions` (chat authoring) |

## Development

```bash
ruff check src tests tools
ruff format --check src tests tools
python -m pytest -q
python tools/check_measurement_isolation.py
python tools/check_traceability.py
```

External commands remain disabled in the first local release. Network model providers require an explicitly configured OpenAI-compatible profile and researcher-supplied credentials. Workspace extensions must be explicitly enabled and should be treated as trusted code.

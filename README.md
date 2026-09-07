# GENESIS

GENESIS is a local-first generative agent-based modelling system for researchers studying complex social dynamics. Release 1 is a Python modular monolith: study specifications are authored as versioned YAML, compiled into immutable builds, executed locally, and persisted in SQLite with content-addressed objects. Analytical exports can be queried with the lightweight analysis layer and extended later with DuckDB/Parquet.

This README provides two instruction sets:

- **[Part 1 — Instructions for an AI agent](#part-1--instructions-for-an-ai-agent)**: how to author, compile, execute, and inspect a study headlessly using only this repository — no pre-built experiment package required.
- **[Part 2 — Instructions for a human researcher](#part-2--instructions-for-a-human-researcher)**: the interactive, approval-gated web workflow (specification elicitation, running, and inspecting studies in the browser).

## Quick start (both)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -q        # optional: run the test suite
genesis version
genesis serve --workspace ./genesis-workspace
```

Open http://127.0.0.1:8000/ui for the browser workflow or http://127.0.0.1:8000/docs for the API console.

---

# Part 1 — Instructions for an AI agent

This part is written for an agent driving GENESIS programmatically. Workflows that require a human in the loop are described in Part 2.

## 1. Configure an OpenAI-compatible model

Generative processes call a model through a **model profile**; deterministic-only studies need none. A profile needs a compatible base URL, model name, and an API key read from an environment variable:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace
```

```python
from genesis.service import GenesisService
svc = GenesisService("./genesis-workspace")
svc.create_model_profile({
    "id": "openai-default", "provider": "openai-compatible",
    "model": "<model>", "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
})
svc.test_model_profile("openai-default")
```

The key is read at request time and never stored in the profile, study YAML, SQLite database, build, or export. Compilation makes no provider calls.

## 2. The canonical study package

A study is 7 YAML files plus optional `prompts/`, `schemas/`, `data/` directories:

- `study.yaml` — identity and summary
- `theory.yaml` — constructs, theory functions, relations, delays
- `openness.yaml` — processes: generative (model-backed, schema-closed) or computational (Python entry points), with context policies
- `domain.yaml` — states, actors, context-visibility allow-lists, artifact catalog
- `protocol.yaml` — time model (rounds), conditions, replications, random streams
- `outcomes.yaml` — outcome measures over events/artifacts/states
- `models.yaml` — model profiles used by generative processes

## 3. Headless run

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

## 4. Implementing processes

- **Computational** processes declare `executor.parameters.entry_point = "module:function"`. The function receives a `ProcessInvocation` with `actor_ids`, `phase`, a read-only `context` scoped by the process's context policy (allow-listed state; mutable collections are frozen — use `collections.abc.Mapping`, never `isinstance(dict)`), and `inputs` (resolved artifact instances). Return a `ProcessResult` with `outputs` and declared `state_effects`. Reference implementations live in `tests/executor_functions.py`; contracts are exercised by `tests/test_actor_instance_execution.py` and `tests/test_artifact_instance_routing.py`.
- **Generative** processes declare `prompt_ref` (a `prompts/*.txt` template supporting `{context}`, `{actor_ids}`, `{phase}`), an `outputs` artifact with a JSON schema, and `openness_rationale`/`closure_rationale`. The model must answer with schema-valid JSON only; schemas use single JSON types (union `type` arrays are not supported by validation).

## 5. Key concepts for agents

- **Prior-phase dependency semantics** — `dependencies.after` gates a process on the dependency's last completed phase; use it for one-round causal latency (e.g., performance → reflection → next-round strategy).
- **Terminal settlement** — add a settlement-only phase after the final content round: content processes declare `terminal_skip: true` (default) and only settlement runs there. Set `GENESIS_TERMINAL_PHASE=<end>` before launching for run-scale control.
- **Measurement isolation** — processes that label or score generated content must stay out of causal context: their artifacts must not appear in any creator/user-facing context policy, and the processes must not declare `state_effects`. `tools/check_measurement_isolation.py` verifies this on a package.
- **Immutability & provenance** — builds, events, artifacts, and state snapshots are content-hashed; replays (full/artifact/partial/branch) reconstruct traces from the same records; exported run bundles carry `run_manifest.json` + `events.json` + `artifacts.json` plus an `integrity.json` digest manifest and can be imported into any workspace (`tools/export_traces.py`, `tools/rename_run.py` for trace slicing and re-keyed copies).
- **Trace selection for evidence** — pick ONE trace per run under a rule frozen *before* execution (e.g., the first naturally occurring target event, else a documented fallback), then render it deterministically with `GET /runs/{id}/natural-trace`. Post-hoc browsing (`?user=&phase=`) is for inspection, not evidence.

## 6. Useful endpoints

| Purpose | Endpoint |
|---|---|
| List runs / studies / builds | `GET /runs` · `GET /studies` · `GET /builds` |
| Trace events | `GET /runs/{id}/events` |
| Outcome evaluation | `GET /runs/{id}/outcomes` |
| Natural trace / per-user trace | `GET /runs/{id}/natural-trace[?user=&phase=]` |
| Compiled process graph | `GET /builds/{ref}/processes` |
| Export / import run bundles | `POST /exports` · `POST /runs/import` |
| Replays | `POST /runs/{id}/replays` (`full`/`artifact`/`partial`/`branch`) |
| Elicitation (scripted authoring) | `POST /elicitation/sessions` with `workflow_id: three-layer-study` |

---

# Part 2 — Instructions for a human researcher

This part walks a human through the interactive web workflow in `/ui`. Every state-changing step is approval-gated: drafts cannot be compiled until they are explicitly approved.

## 1. Configure a model in the UI

Open `/ui` → **Models**. Enter a profile ID, the OpenAI-compatible base URL, the model name, and the name of an environment variable containing the API key (for example `OPENAI_API_KEY`). Set that variable before starting the server:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace --port 8765
```

Save the profile, use **Check key** and **Test connection**, then select the profile for the generative process. Deterministic-only studies can skip this step.

## 2. Specification — chat-first three-layer elicitation

Open `/ui` → **Specification**. Authoring is a conversational, stage-by-stage workflow (the staged workflow content — stages, questions, templates, invalidation — lives under `workflows/three-layer-study/` and is configurable without code changes):

1. **Start a session** in the chat workspace (or `POST /elicitation/sessions` with `specification_id`, `workflow_id: three-layer-study`, `model_profile_id`, `researcher_id`).
2. **Answer the opening question.** The assistant asks one clarifying question at a time, each with exactly three editable suggestions — select, edit, or ignore them; free-form answers are first-class researcher evidence.
3. **Request a draft** and **review the preview**: YAML diff, full files, evidence, assumptions, warnings, and errors.
4. **Approve each stage explicitly.** Approval writes a new immutable package version. Use *revise* to return to clarification, or *cancel* to discard the in-memory session.
5. **Upstream revisions** mark dependent downstream stages `needs_review`; re-approve them to continue.
6. After the **final stage approval** the package is approved and versioned — ready to compile.

Notes: unfinished sessions are intentionally in-memory only — a service restart discards them, while every accepted specification version survives. Provider errors leave the session retryable at the same question. Alternatively, author the 7 canonical YAML files directly and import them (`POST /imports`).

## 3. Run — compile and execute

Open `/ui` → **Run**:

1. **Compile** the approved specification. Compilation is deterministic and makes no provider calls; it emits a build directory (`builds/<study>/`, 15 JSON artifacts) with an immutable build hash, plus validation and integrity manifests.
2. **Create and execute** the run. Every run freezes a manifest recording its configuration: build/source/compiler hashes, model and prompt versions, condition, random streams and resolved seeds, and data provenance.
3. Inspect the run status and review errors if any phase fails; retry policy and failure handling are recorded in the run log.

## 4. Outputs — inspect the evidence

Open `/ui` → **Outputs**. Choose the run from the **Explore run** selector (demo runs and imported bundles both appear):

- **Dashboard** — overview counts of studies, builds, runs, experiments.
- **Evaluate outcomes** — the Outcome Plan applied to the selected run (aggregates computed from the executed trace).
- **Show run trace / Trace explorer** — the run's events as an ordered list or grouped by phase.
- **Natural trace (frozen rule)** — the single rule-selected trace for the run (evidence-grade; deterministic).
- **Trace by user…** — inspect any individual user's chain (ad-hoc, labeled `user-specified`).
- **Process map** — the compiled process graph for the run's build.
- **Export evidence bundle** — writes the full evidence package (outcomes, events, artifacts, manifests, integrity digests) to `exports/…`.
- **Import run bundle…** — load a run exported elsewhere (integrity-verified) into this workspace for exploration.

Replays (`full`/`artifact`/`partial`/`branch`) are available through the Run view and `POST /runs/{id}/replays` and reconstruct a run's trace from its recorded boundaries.

## 5. Package layout (for maintainers)

- `genesis.specification` — canonical Pydantic schemas and validation contracts.
- `genesis.compiler` — package loading, cross-reference validation, deterministic builds, integrity manifests.
- `genesis.persistence` — SQLite WAL coordinator and content-addressed object storage.
- `genesis.runtime` — immutable invocation/result contracts, context/state/artifact services, scheduler, executors, run controller.
- `genesis.providers` — provider-neutral model adapters, deterministic mock, recorded-artifact provider.
- `genesis.provenance` / `genesis.replay` — immutable lineage and replay modes.
- `genesis.analysis` — declarative outcome evaluation and JSON/CSV export helpers.

## 6. Development

```bash
ruff check src tests tools
ruff format --check src tests tools
python -m pytest -q
python tools/check_measurement_isolation.py
python tools/check_traceability.py
```

External commands remain disabled in the first local release. Network model providers require an explicitly configured OpenAI-compatible profile and researcher-supplied credentials. Workspace extensions must be explicitly enabled and should be treated as trusted code.

# GENESIS — GENerative Environment for Simulation of Interactive Systems

GENESIS is research infrastructure for studying open-ended social processes with generative simulation. It is a local-first Python system: a study is authored as a versioned, approved study package, compiled into an immutable build, executed locally, and recorded in SQLite with content-addressed objects. Runs export as integrity-checked evidence bundles and as flat analysis tables (CSV and Parquet) for the researcher's own statistics.

## Background

Many social and sociotechnical processes are open-ended: people and organisations respond to rules, platforms and one another in ways that cannot be fully anticipated. Agent-based models can simulate how such processes unfold, but what agents can do is usually fixed in advance. Generative AI lets selected behaviours and content take shape during a simulation instead, which opens new questions for research — and makes it harder to keep the link between a study's theory, its model and its evidence.

## Objectives

GENESIS helps researchers use generative AI in simulation without losing control of the study:

- **Theory-driven design** — decide explicitly where generation is allowed, what role it plays, and in what context it operates.
- **Specification separate from execution** — a study is a versioned, reviewable package, independent of the software that runs it.
- **Traceable evidence** — every run records what each agent saw and produced, so results can be inspected, compared across runs and replayed.
- **Researchers in charge** — AI assists with specifying and running studies; decisions about theory, design and interpretation stay with the researcher.

This README provides two instruction sets:

- **[Part 1 — Instructions for an AI agent](#part-1--instructions-for-an-ai-agent)**: authoring, compiling, running and analysing a study headlessly.
- **[Part 2 — Instructions for a human researcher](#part-2--instructions-for-a-human-researcher)**: the approval-gated browser workflow.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'    # '.[dev]' adds pytest, ruff and mypy; 'pip install -e .' is enough to run
python -m pytest -q        # optional
genesis version
genesis serve --workspace ./genesis-workspace
```

Open http://127.0.0.1:8000/ui for the browser workflow or http://127.0.0.1:8000/docs for the API console.

---

# Part 1 — Instructions for an AI agent

## 1. Configure an OpenAI-compatible model

Generative processes call a model through a **model profile**; deterministic-only studies need none. Give the profile a base URL, a model name, and the name of the environment variable that holds the key:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace
```

```python
from genesis.service import GenesisService

svc = GenesisService("./genesis-workspace")
svc.create_model_profile(
    {
        "id": "openai-default",
        "provider": "openai-compatible",
        "model": "<model>",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    }
)
svc.test_model_profile("openai-default")
```

With `api_key_env` the key is read from the environment at request time and stored nowhere. A key pasted into a profile as `api_key` is kept in the workspace's local profile store and never returned by any read; it is never written into a study package, build, run record or export. Compilation makes no provider calls. A study's `models.yaml` names a profile through `endpoint_ref`; a run refuses a local profile whose model or provider has drifted from the package.

## 2. The canonical study package

A study is 7 YAML files plus optional `prompts/`, `schemas/` and `data/` directories:

- `study.yaml` — identity and summary
- `theory.yaml` — constructs, theory functions, relations (with executable precedence and lag bindings), feedback
- `openness.yaml` — processes: model-backed (`generative`, `semantic-evaluator`) or closed (`computational`, `deterministic`, `stochastic`, `rule`, `state-transition`, `recorded_artifact`, `extension`), each with a context policy
- `domain.yaml` — states, actors, artifacts, visibility policies (what each process may see), initialisation
- `protocol.yaml` — time model, factors or conditions, phases, random streams, matching
- `outcomes.yaml` — datasets (analysis tables), simple outcome summaries, declared traces
- `models.yaml` — the model profiles generative processes use

The package fixes **what the study is**. How much of it to realise — replications, which conditions, which data worlds, an event cap — is chosen per run (section 4), so the build hash identifies one study rather than one sample size. The guided workflow's grammar, with worked templates, lives under `workflows/three-layer-study/`.

## 3. The compiler refuses what would silently do nothing

A declaration that compiles but is never read looks like a working design. The compiler refuses, with a named error, among others:

- a condition gate, availability rule or visibility `allow` entry that resolves to nothing (for example an artifact name where a state is expected — artifacts reach a model under `inputs`);
- a model prompt with no `{context}` placeholder, so the model would answer blind;
- a trigger or dependency key the scheduler never reads, a non-integer round, an immediate cycle (including one formed only when theory edges merge with declared ones);
- a `termination` other than `{type: end_time, at: <time_model.end>}`, and an unknown `trace_policy.retention` value;
- an outcome whose source, aggregation, missingness, window or join the evaluator would ignore or cannot compute;
- an empirical `data_source` outside the package, and output actor/phase fields a schema cannot hold.

It warns about softer risks, such as a condition trigger listing several rounds without `repeat: true` (it fires once).

## 4. Headless run

```python
from genesis.service import GenesisService

svc = GenesisService("./genesis-workspace")
draft = svc.create_specification({...})       # or svc.import_package("imports/<dir>")
svc.approve_specification("<id>", draft["version"], "agent")
compiled = svc.compile_study(None, "builds/<id>", specification_id="<id>")
svc.create_run({"id": "exp-1", "study_id": "<id>", "build": compiled["path"]})

# Plan first (no calls), then realise conditions x replications as runs.
svc.execute_protocol("exp-1", replications=1, max_events=2000, plan_only=True)
svc.execute_protocol("exp-1", replications=1, max_events=2000,
                     only_conditions=["<condition-id>"], parallel=True, worker_kind="thread")

svc.trace_run("exp-1-<condition-id>-1")
svc.export_run("exp-1-<condition-id>-1", "exports/run")      # one run's evidence bundle
svc.export_experiment("exp-1", "exports/exp-1-tables")        # every cell, stacked tables
```

For a single run without a protocol, `svc.execute_run("<run-id>")` executes it (deterministic-only studies can pass `executor_overrides`). A run can be paused, resumed or cancelled (`POST /runs/{id}/pause|resume|cancel`); a paused run resumes where it stopped with `execute_run`. `max_events` is a spend guard: a run stopped by it is recorded as `stopped_by: max_events`, never as a finished run.

## 5. Implementing processes

- **Computational** processes declare `executor.parameters.entry_point = "module:function"`. The function receives a `ProcessInvocation` with `actor_ids`, `phase`, `condition`, a read-only `context` scoped by the process's context policy (collections are frozen — test with `collections.abc.Mapping`, never `isinstance(dict)`), and `inputs` (resolved artifact instances, keyed by instance id, value under `value`). Return a `ProcessResult` with `outputs` and declared `state_effects`. Reference implementations live in `tests/executor_functions.py`.
- **Generative** processes declare `prompt_ref` (a plain-text `prompts/<id>.txt` template; context reaches the model only through `{context}` or `{context.<path>}`, with `{phase}` and `{actor_ids}` also filled), an `outputs` artifact with a JSON Schema (Draft 2020-12), and `openness_rationale` / `closure_rationale`. The model must return schema-valid JSON. Condition labels and engine provenance are removed from what the model sees.
- **Engine-filled fields.** An output may declare `actor_fields: [<field>]` and `phase_fields: [<field>]`; the engine writes the acting actor's id and the round there, over whatever the model returned. Use them for any field a later step joins on — a model asked to repeat an id back gets it wrong.

## 6. Key concepts for agents

- **Timing.** `dependencies: {after: [<process>]}` orders processes **within a round**. A lag is declared explicitly: `delay: {rounds: 1}` for every edge, or `delay: {per_dependency: {<process>: 1}}`; a theory precedence relation can carry `lag_rounds`. A `trigger` is `{phase: N, repeat: true}`, `{type: condition, predicate: {...}, repeat: true}` or `{type: event, event: <id>}`.
- **Condition gates** read factor levels at `condition.factors.<factor_id>` and the round at `protocol.phase`.
- **Terminal round.** A process with `terminal_skip: true` does not run in the final round (`time_model.end`), which leaves that round for settlement-type processes. The default is `false`.
- **Measurement isolation.** A process that scores or labels generated content declares `measurement: true`. Its outputs never reach behaviour unless the design declares a `measurement_use` (a `source`, a `when` predicate over `condition` and `protocol.phase`, and a `rationale`) — for example a detector score that sets revenue only under a sanction condition. `tools/check_measurement_isolation.py` checks a source package.
- **Immutability and provenance.** Builds, events, artifacts and state snapshots are content-hashed. Replays (`full` / `artifact` / `partial` / `branch`) reconstruct traces from the same records. Run bundles carry the run manifest, events, artifacts, state history, the package closure and integrity digests, and import into any workspace; `reproducibility` mode also carries the executable build.
- **Trace selection for evidence.** Declare the traces a study reports in `outcomes.yaml` before running, and render them with `GET /runs/{id}/natural-trace?trace=<id>`. Browsing by actor (`?actor=&phase=`) is for inspection, not evidence.

## 7. Analysis tables and outcomes

Analysis is data-first: GENESIS supplies the tables, the researcher computes the statistics.

- **Datasets** in `outcomes.yaml` define flat tables — one row per article, detection, response, and so on — from `artifacts` (by `artifact_type`), `events` (a `path` to a list on each event) or `state` (`final` or `each_completed_round` snapshots). `explode: <path>` gives one row per element of a list or entry of a mapping (a feed's items, a settlement keyed by id); `fields` derive columns with `copy`, `literal`, `arithmetic`, `comparison` or `conditional`; `where` filters rows.
- **Every row carries its cell**: `run_id`, `experiment_id`, `condition_id`, `replication`, and a `factor_<id>` column per factor, written by the engine.
- **Exports.** A run bundle contains `datasets/<id>.csv`, `datasets/<id>.parquet` and `datasets/dictionary.json`. `export_experiment` (API `POST /experiments/{id}/export`, CLI `genesis export --experiment <id> --output <dir>`) stacks every cell into one table per dataset with `experiment.json`, which lists each run and whether it completed.
- **Revised analyses.** What a study measures may change after its runs; what it ran may not. `export_experiment(..., analysis_build="builds/<newer>")` applies a newer build's datasets and outcomes, and is refused unless every file that determined execution is byte-identical.
- **Outcomes** are simple summaries over a dataset or a built-in source: `count`, `sum`, `mean` (a mean of a true/false column is a proportion), `distribution` or `trajectory`, with `grouping`, equality `filters`, `missingness: {policy: exclude|zero}` and `window: {time_field, start, end}`. Differences between cells, difference-in-differences and similar estimates are computed from the exported tables.

## 8. Useful endpoints

| Purpose | Endpoint |
|---|---|
| List runs / studies / builds / experiments | `GET /runs` · `GET /studies` · `GET /builds` · `GET /experiments` |
| Model profiles | `GET/POST /llm/profiles` · `POST /llm/profiles/{id}/test` |
| Import a package | `POST /imports` (`source`, optional `specification_id`) |
| Plan or realise a protocol | `POST /runs/{id}/protocol` (`replications`, `only_conditions`, `initializations`, `max_events`, `plan`, `parallel`) |
| Execute, pause, resume, cancel a run | `POST /runs/{id}/execute` · `POST /runs/{id}/pause` · `/resume` · `/cancel` |
| Trace events | `GET /runs/{id}/events` |
| Declared / per-actor trace | `GET /runs/{id}/natural-trace[?trace=&actor=&phase=]` |
| Outcomes | `GET /runs/{id}/outcomes` |
| Analysis tables | `GET /runs/{id}/datasets` · `GET /runs/{id}/datasets/{name}?offset=&limit=` |
| Export a run / an experiment | `POST /exports` · `POST /experiments/{id}/export` |
| Import a run bundle | `POST /runs/import` |
| Compiled process graph / read a build back | `GET /builds/{ref}/processes` · `POST /builds/{ref}/readback` |
| Replays | `POST /runs/{id}/replays` (`full` / `artifact` / `partial` / `branch`) |
| Elicitation (scripted authoring) | `POST /elicitation/sessions` with `workflow_id: three-layer-study` |

---

# Part 2 — Instructions for a human researcher

This part walks through the browser workflow at `/ui`. Every state-changing step is approval-gated: a draft cannot be compiled until it is approved.

## 1. Models

Open `/ui` → **Models**. Enter a profile ID, the OpenAI-compatible base URL, the model name, and the name of an environment variable that holds the API key (for example `OPENAI_API_KEY`). Set that variable before starting the server:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace
```

**Save model profile**, then **Check key** and **Test connection**. Deterministic-only studies can skip this step.

## 2. Specification — guided three-layer elicitation

Open `/ui` → **Specification** and **Start guided specification**. Authoring is conversational and stage by stage; the stages, questions, templates and invalidation rules live under `workflows/three-layer-study/` and can be changed without code.

1. **Answer the question.** The assistant asks one question at a time, with three editable suggestions; select, edit or ignore them — a free-form answer is first-class evidence.
2. **Generate stage draft** and **Review generated changes**: YAML diff, full files, evidence, validation.
   The **Intent** check reads the draft against what you said and lists anything that differs. It is advisory: acknowledge a finding or revise the stage.
3. **Approve this stage.** Approval writes a new immutable package version; **Request revision** returns to clarification.
4. **Reopening an upstream stage** marks the stages that depend on it for review; re-approve them to continue.
5. After the final stage the package is approved and versioned, ready to compile.

Unfinished sessions live in memory only: a restart discards them, while every accepted version survives. Alternatively, write the seven YAML files yourself and use **Import package**; **Inspect package** and **Show checklist** report what is missing.

## 3. Run

Open `/ui` → **Run**:

1. **Compile approved draft.** Compilation is deterministic, makes no provider calls, and produces an immutable build (`builds/<study>/`) with validation and integrity manifests — or the refusals described in Part 1, section 3.
2. **Read this build back** (optional): a model restates the compiled study in plain language, blind to your stated intent, so you can check it says what you meant.
3. **Plan — costs nothing** shows how many runs and events a choice of replications, conditions, worlds and event cap implies. **Execute protocol** then runs exactly the planned settings.
4. Every run freezes a manifest: build and compiler hashes, model and prompt versions, condition, random streams and seeds, data provenance, and the event cap.

## 4. Outputs

Open `/ui` → **Outputs** and choose a run under **Explore run** (completed runs are preferred; imported bundles appear too):

- **Dashboard**, **Evaluate outcomes**, **Show run trace**, **Trace explorer**, **Process map**.
- **Declared traces** / **Declared trace** — the traces the study declared, rendered deterministically (evidence-grade). **Trace by actor…** follows any actor's chain for inspection.
- **Analysis tables** — **List datasets for run**, then page through any dataset. An optional analysis build applies a newer outcome plan to the same runs.
- **Export an experiment** — stacks every cell of an experiment into one CSV/Parquet table per dataset.
- **Export evidence bundle** and **Import run bundle…** — the full, integrity-checked evidence for one run.

Replays of recorded runs are available through `POST /runs/{id}/replays` and the Run tab's replay controls.

## 5. Package layout (for maintainers)

- `genesis.specification` — canonical Pydantic models and validation contracts.
- `genesis.compiler` — package loading, cross-layer validation, deterministic builds, integrity manifests.
- `genesis.theory_execution` — executable bindings of theory relations and feedback.
- `genesis.runtime` — invocation/result contracts, context views, state and artifact stores, scheduler, executors, run controller.
- `genesis.information_timing` / `genesis.measurement` — simultaneous-batch timing and measurement isolation.
- `genesis.providers` — provider adapters, model blinding, deterministic and recorded providers.
- `genesis.persistence` / `genesis.state_encoding` — SQLite WAL coordinator, content-addressed objects, compact state history.
- `genesis.service` / `genesis.app` / `genesis.cli` — the application service, HTTP API and command line.
- `genesis.elicitation` / `genesis.assistant` / `genesis.checklists` / `genesis.readback` / `genesis.intent_check` — guided authoring and its checks.
- `genesis.outcome_plan` / `genesis.analysis` — datasets, outcome evaluation and table export.
- `genesis.evidence` / `genesis.execution_manifest` / `genesis.provenance` / `genesis.replay` / `genesis.tracing` — bundles, manifests, lineage, replay and traces.
- `genesis.extensions` — explicitly enabled workspace extensions.

## 6. Development

```bash
ruff check src tests tools
ruff format --check src tests tools
mypy src
python -m pytest -q
python tools/smoke_check.py    # the CLI end to end: init, validate, compile, run, outcomes
```

CI runs the lint, format and type checks, the test suite and `tools/smoke_check.py` on a clean checkout. `tools/check_measurement_isolation.py` and `tools/check_traceability.py` validate study assets and the requirements-traceability map for development; they are not gates in CI.

External commands are disabled. Network model providers require an explicitly configured OpenAI-compatible profile and researcher-supplied credentials. Workspace extensions must be explicitly enabled and are trusted code.

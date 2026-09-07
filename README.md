# GENESIS

GENESIS is a local-first generative agent-based modelling system for researchers studying complex social dynamics. Release 1 is a Python modular monolith: study specifications are authored as versioned YAML, compiled into immutable builds, executed locally, and persisted in SQLite with content-addressed objects. Analytical exports can be queried with the lightweight analysis layer and extended later with DuckDB/Parquet.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
genesis version
genesis serve --workspace ./genesis-workspace
```

The API binds to `127.0.0.1` by default. Open [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui) for the browser workflow or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API console.

The browser workflow is deliberately approval-gated:

1. Enter study metadata and generate a draft.
2. Inspect the assistant validation report.
3. Review the generated YAML and explicitly approve the current version.
4. Compile the approved specification.
5. Create and execute the experiment.
6. Inspect the run and export its results.

Drafts are stored under `.genesis/specifications/` and cannot be compiled while they remain in `draft` status. The canonical system design is in [GENESIS_Complete_System_Design_Specification.md](GENESIS_Complete_System_Design_Specification.md), with a traceability map in [docs/requirements-traceability.yaml](docs/requirements-traceability.yaml). Work still outstanding against that design is tracked in [docs/GENESIS_Additional_Works_Specification.md](docs/GENESIS_Additional_Works_Specification.md).
The current system architecture (modules, data flows, persistence, and design decisions) is documented in [docs/GENESIS_Architecture.md](docs/GENESIS_Architecture.md).

### Configure an OpenAI-compatible model

The `/ui` page includes a model configuration section. Enter a profile ID, compatible base URL, model name, and the name of an environment variable containing the API key (for example, `OPENAI_API_KEY`). Set that variable before starting the server:

```bash
export OPENAI_API_KEY="your-key"
genesis serve --workspace ./genesis-workspace --port 8765
```

Save the profile, use **Check key** and **Test connection**, then select the profile for the generative process. The key is read by the server at request time and is never stored in the profile, study YAML, SQLite database, build, or export. Compilation itself makes no provider calls; model calls occur only during an explicit connection test or execution of an approved run.

## Package layout

- `genesis.specification`: canonical Pydantic schemas and validation contracts.
- `genesis.compiler`: package loading, cross-reference validation, deterministic builds, and integrity manifests.
- `genesis.persistence`: SQLite WAL coordinator and content-addressed object/checkpoint storage.
- `genesis.runtime`: immutable invocation/result contracts, context/state/artifact services, scheduler, executors, and run controller.
- `genesis.providers`: provider-neutral model adapters, deterministic mock, and recorded-artifact provider.
- `genesis.provenance` / `genesis.replay`: immutable event/checkpoint lineage and replay modes.
- `genesis.analysis`: declarative outcome evaluation and JSON/CSV export helpers.

## Development

```bash
ruff check src tests tools/check_traceability.py
ruff format --check src tests tools/check_traceability.py
PYTHONPATH=src pytest -q
```

External commands remain disabled in the first local release. Network model providers require an explicitly configured OpenAI-compatible profile and researcher-supplied credentials. Workspace extensions must be explicitly enabled and should be treated as trusted code.


## Interactive elicitation (chat-first study authoring)

The local web app supports a conversational five-stage workflow that elicits a complete study package with explicit researcher approval at every stage (see `docs/plans/2026-08-30-interactive-three-layer-elicitation-design.md`):

1. **Configure an assistant model profile** (web UI or `POST /llm/profiles`).
2. **Start a session** from the `/ui` chat workspace or `POST /elicitation/sessions` (`specification_id`, `workflow_id: three-layer-study`, `model_profile_id`, `researcher_id`).
3. **Answer the opening question**; the assistant asks one clarifying question at a time with exactly three editable suggestions (`POST /elicitation/sessions/{id}/messages`). Select, edit, or ignore suggestions; free-form answers are first-class researcher evidence.
4. **Request a draft** (`POST .../draft`) and **review the preview** (`GET .../preview`): YAML diff, full files, evidence, assumptions, warnings, and errors.
5. **Approve each stage explicitly** (`POST .../approve` with `approved_by`); approving writes a new immutable package version. `.../revise` returns to clarification; `.../cancel` discards the in-memory session.
6. **Upstream revisions** (`reopen_elicitation_stage`) mark dependent downstream stages `needs_review`; re-approve them to continue.
7. After the fifth approval the package is approved; compile and run through the existing `/compile` and run APIs.

Notes: unfinished sessions are intentionally in-memory only — a service restart discards them, while every accepted specification version survives. Provider errors leave the session retryable at the same question. The workflow content (stages, questions, templates, invalidation) lives under `workflows/three-layer-study/` and is configurable without code changes.

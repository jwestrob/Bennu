# AGENTS.md — Project Conventions for Agents

Scope: Entire repository. These instructions constrain planner behavior, operator choices, and code changes made by agents working in this repo.

## Do Not Hard‑Code Biology

- NEVER inject biological identifiers (e.g., KO, PFAM, CAZy, gene symbols) directly into plans, prompts, code paths, or templates.
  - Examples of disallowed suggestions/changes: “Add TonB/ExbB/ExbD (K03832/K03550/K03551) to the KO list,” “Hard‑wire FeoA/B KOs into the pipeline,” “Manually append PF05031 to the query.”
- All identifier selection MUST be data‑driven:
  - Use catalog operators (e.g., SearchPfamCatalogFuzzy, SearchKoCatalogFuzzy) or explicit IDs provided by the user or dataset.
  - If curation is desired (allow/deny), it must come from data files under `data/reference/` (e.g., TSV/JSON), not from constants in code. These files are optional and should be consulted only when present.
  - Planners and operators MUST NOT invent, expand, or “helpfully” add identifiers that are not returned by a catalog search or user input.

## Planner & Composite Discipline

- Prefer composites that derive outputs deterministically from available data. Avoid heuristics that rely on implicit biological knowledge.
- When comparing features across genomes, use FeatureProfile (keyword → PFAM/KO catalogs → exact counts) or PathwayProfile (KO presence/completeness) as appropriate; do not augment identifier sets beyond catalog results.
- Avoid global queries by default. Use DatasetContext genome sampling unless the user explicitly requests full scope.

## UX / Reporting

- Favor compact matrices and labeled summaries over raw row dumps. If labels are needed, load them from `data/reference` (e.g., `pfam_id_desc.tsv`, `ko_list`). Do not embed labels in code.

## Pre‑Compaction Note — 2025‑09‑17

- Recent changes added FeatureProfile (per‑genome PFAM/KO counts), improved planner scoping, and portable Neo4j export. CRISPR arrays are integrated E2E. Future agents must adhere to the “No hard‑coded biology” rule above and keep identifier selection data‑driven.

## Architecture & Finalizer (must adhere to AGENT_STRUCTURE.md)

- The authoritative operating model lives in `AGENT_STRUCTURE.md`. All agents MUST follow it:
  - Region‑first, minimal‑call design: Retrieval and Compute+Viz are in‑process. Only the Plan/Report LLM calls and the optional Code‑Interpreter (CI) call are external APIs.
  - Artifact policy: write packs/plots/reports to session folders; only display‑safe paths are printed to the console (see AGENT_STRUCTURE.md “Artifact Store Policy”).

### Finalizer (Code Interpreter) — current contract

- Triggering: controlled by `CI_MODE` (auto|always|never) and user intent. In “auto”, CI runs when matrices exist or the prompt asks to plot/visualize/heatmap/bar/cluster/figure.
- Inputs: the finalizer consumes `analysis_payload.json` and `matrices/*.csv` written under `data/session_notes/<session_id>/` after plan execution.
- Generator: the finalizer builds analysis code programmatically (no prompt‑specific script), using only:
  - matplotlib (Agg), pandas, numpy, pathlib, and builtins.open (seaborn and os are disallowed in the sandbox).
  - Heatmaps from `PerGenomeTopMatrix` and optional bar charts from `FeatureProfileSummary`.
- Auditability: the exact code is saved to `data/session_notes/<sid>/ci_code.py` every run.
- Outputs: the CI service returns `files_created`; the caller downloads them to `data/session_notes/<sid>/plots/` via `GET /sessions/{sid}/files/{name}` and prints a one‑line summary to the console.
- Reporting: `report_v2.md` includes stdout/stderr, the service‑side file list, and the downloaded host file paths.

### CLI & Config (agent‑facing)

- `--ci {auto|always|never}` controls whether finalizer runs (default: auto). Also available as `CI_MODE` env.
- `--ci-model <label>` prints an engine label (e.g., `matplotlib`) when the CI call is made. Also available as `CI_MODEL` env.
- `CODE_INTERPRETER_URL` points to the service (default `http://localhost:8000`).

### Guardrails for agents

- Do NOT add prompt‑specific or hard‑coded analysis scripts. Add capabilities by extending the programmatic generator in `src/llm/rag_system/finalizer.py` (new recipes over packs), not by embedding static snippets.
- Keep retrieval/compute pack schemas stable (see AGENT_STRUCTURE.md Contracts & Data Shapes). If you change `FeatureProfile` or add new packs, update the generator accordingly.
- Never introduce seaborn or os usage in CI‑side code; the sandbox runs with restricted globals. Stick to matplotlib + pathlib + builtins.
- Console signals are part of UX: printing the CI call (model + URL) and the download summary is required; do not remove without a replacement.

### Container notes (CI service)

- The CI Docker image is intentionally slim for Python 3.11 compatibility. Heavy/legacy packages (e.g., pyvcf, circos, graph‑tool, rdkit, etc.) are excluded; do not re‑add unless strictly necessary.
- The service exposes:
  - `POST /execute` — runs code
  - `GET /health` — readiness check
  - `GET /sessions/{sid}/files/{relpath}` — artifact download (with traversal protection)

Follow‑ups planned: once validated, we will compact documentation and keep only the summary + authoritative contracts in `AGENT_STRUCTURE.md`, linking from here.

# Spec‑First Genomic RAG — Operating Structure (Regions + Calls)

This document is the source of truth for how the system should run. It replaces legacy descriptions and aligns everyone on the minimal‑call, deterministic architecture we agreed on.

Sections:
- A) Region‑first diagram with the only external API calls.
- B) Implementation plan (phased) with progress checkboxes.
- C) Artifact store policy and end‑of‑run display output (paths only; packs and non‑display reports are not printed).

---

## A) Regions And Calls (ASCII)

Each big box is an in‑process region. Small inner boxes are the only external API calls on the hot path. Everything else runs locally to economize on calls.

```
Spec‑First Genomic RAG — Regions (outer) with External API Calls (inner)
────────────────────────────────────────────────────────────────────────

┌───────────────────────────────────────────────────────────────────────────────┐
│ INTENT (in‑process planner/orchestrator)                                      │
│   ┌───────────────────────────────┐                                           │
│   │  API CALL: LLM PLAN /plan     │  (single model call to generate PlanSpec) │
│   └───────────────────────────────┘                                           │
│   Policy evaluation & budgets → local (no API)                                │
└───────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ RETRIEVAL (in‑process QueryEngine)                                            │
│   • Neo4j driver session/transaction (named templates, UNWIND batches)        │
│   • Local catalogs: PFAM/KO TSV caches                                        │
│   — no external API calls in this region —                                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ COMPUTE + VISUALIZATION (in‑process AnalysisEngine)                            │
│   • Matrix transforms, stats, clustering, top‑k feature selection              │
│   • Plot rendering (matplotlib/seaborn)                                        │
│   — no external API calls in this region —                                     │
└───────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ FINALIZER (optional analysis backend)                                          │
│   ┌──────────────────────────────────────┐                                     │
│   │  API CALL: CODE INTERPRETER /execute │  (only if enabled/healthy)         │
│   └──────────────────────────────────────┘                                     │
│   • On failure: 1 repair attempt (local codegen), else skip                    │
└───────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ SYNTHESIS (in‑process report orchestrator)                                     │
│   ┌────────────────────────────────┐      ┌────────────────────────────────┐   │
│   │  API CALL: LLM REPORT /report  │ ───▶ │  API CALL: LLM AMEND /amend   │   │
│   └────────────────────────────────┘      └────────────────────────────────┘   │
│   • Initial table‑first report              • Optional amend with diffs         │
└───────────────────────────────────────────────────────────────────────────────┘


                    ┌─────────────────────────────────────────────┐
                    │ ARTIFACT STORE (local FS/manifests)         │
                    │   • Packs (matrices, completeness)          │
                    │   • Plots (PNG/SVG + thumbs)                │
                    │   • Reports (v1/v2 + diffs)                 │
                    │   — filesystem writes only; no API —        │
                    └─────────────────────────────────────────────┘

Legend:
- Outer boxes = execution regions/modules that run in‑process to economize calls.
- Inner small boxes = the only external API calls on the hot path:
  (1) LLM PLAN (once), (2) optional CODE INTERPRETER, (3) LLM REPORT, (4) optional LLM AMEND.
- Retrieval and Compute+Viz remain local (driver + libraries), not network APIs.
```

---

## B) Implementation Plan (Trackable)

We will implement the architecture in small, verifiable increments. Keep these boxes up to date (change `[ ]` → `[x]`).

### Phase 1 — Baseline Flow (Plan → Retrieve → Compute/Viz → Report v1)
- [ ] Intent: add `--ci {auto|always|never}` flag (default `auto`) and surface in PolicyEngine.
- [ ] Retrieval: ensure batched, single‑session Neo4j execution for FeatureProfile/FunctionalProfile (UNWIND, server‑side aggregates).
- [ ] Compute+Viz: render plots in‑process (matplotlib/seaborn) from matrices; clamp figure count and sizes.
- [ ] Synthesis: reporter renders tables first, then (if present) plots, with compact captions.
- [ ] Artifact writes: persist packs, plots, and report v1 under `data/session_notes/<session_id>/`.
- [ ] End‑of‑run summary: print only display artifacts (plots, report v1 path); do not print packs or raw/non‑display artifacts.

### Phase 2 — Finalizer + Amended Report (Optional)
- [ ] Health gate: check code interpreter `/health`; skip if unavailable.
- [ ] Payload writer: create `analysis_payload.json` + CSVs (matrices) in `session_notes`.
- [ ] Code Interpreter call: single `/execute` with templated code; 1 repair attempt on failure.
- [ ] Reporter amend: generate report v2 with diffs; embed thumbnails; preserve provenance.
- [ ] End‑of‑run summary: add v2 path and plot locations to printed output.

### Phase 3 — Contracts & Policies
- [ ] Define PlanSpec (LLM plan output) and enforce minimal fields.
- [ ] Define AnalysisPayload schema (see below) and validate before CI.
- [ ] Define PlotPolicy (max_figs, max_pixels, top_k features) and enforce in compute.
- [ ] Telemetry: record latencies/rows per region; persist `session_manifest.json` with versions/hashes.

### Phase 4 — QA & Resilience
- [ ] PlanLint: verify at least one retrieval step; no unused binds; gates respected.
- [ ] ResultLint: flag all‑zero/degenerate outputs and attach hints (non‑fatal).
- [ ] Golden runs: canned sessions for CI with snapshot plots and minimal diffs.

---

## Contracts & Data Shapes (authoritative)

### PlanSpec (LLM → Intent)
- `question`: string
- `operators`: ordered list of composites (e.g., FeatureProfile, FunctionalProfile)
- `params`: composite‑level params (e.g., pfam_top_k, ko_top_k, include)
- `policy_hints`: optional limits (e.g., prefer local only, ci=auto)

### AnalysisPayload (Finalizer input)
Written to: `data/session_notes/<sid>/analysis_payload.json`
- `question`, `dataset_context` (genome sample, counts)
- `plan` (operators + params actually executed)
- `feature_profile`: { `summary`, `per_genome_counts`, `per_genome_top_matrix`, `feature_order`, `label_maps`, `warnings` }
- `functional_profile`: { `present_kos_by_genome`, `completeness_matrix`, `completeness_summary`, `cazy_rows`, `cazy_counts`, `bgcs` }
- `omitted`: [{ name, reason, approx_size }]
- `provenance`: [{ operator, template, slots, row_count }]
- `limits`: { row_caps, timeouts }
- `versions`: { app, graph_snapshot, image_tag }

### Plot artifacts (Compute+Viz)
- Saved under `data/session_notes/<sid>/plots/`
- Filenames include dataset id + short descriptor (e.g., `feature_heatmap_top20.png`).
- Thumbnails (optional): `*.thumb.png` for fast embedding.

### Report artifacts
- Report v1 (initial): `data/session_notes/<sid>/report_v1.md`
- Report v2 (amended): `data/session_notes/<sid>/report_v2.md` (adds plots/diffs)
- Diff metadata (optional JSON): `data/session_notes/<sid>/report_diff.json`

Policy reminder: Do not hard‑code biology (see AGENTS.md). All identifiers come from data or user input.

---

## Runtime Flow (concise)
1) Intent: call LLM PLAN once → PlanSpec.
2) Retrieval: execute named templates in one driver session; read catalogs from local files.
3) Compute+Viz: produce matrices and plots in‑process; clamp outputs by policy.
4a) Synthesis v1: generate initial report from matrices/plots.
4b) Finalizer (optional): if `--ci!=never` and matrices exist, write AnalysisPayload + CSVs; call code interpreter once (repair once on error).
5) Synthesis v2 (optional): amend report with diffs and embedded plot thumbnails.
6) End‑of‑run: print a display‑safe summary with file paths (see below).

---

## C) Artifact Store Policy & End‑of‑Run Output

We store everything for reproducibility, but we only print user‑facing artifact locations. Packs and non‑display report variants are NOT printed in the end summary.

Directory layout per session (`data/session_notes/<sid>/`):
- `plots/`                — display images (PNG/SVG) and thumbnails
- `report_v1.md`         — initial display report
- `report_v2.md`         — amended display report (if CI ran)
- `analysis_payload.json` — structured input for CI (kept quiet unless `--ci=debug`)
- `matrices/`            — CSVs (wide/long) for compute; not printed in summary
- `packs/`               — internal JSON packs; not printed
- `report_raw/`          — non‑display report assets (e.g., tokens, prompts); not printed

End‑of‑run summary (example output):

```
Session: 08c5d4ab-01d9-4749-833b-21f961b1d899
Artifacts (display‑safe):
- Report (initial): data/session_notes/08c5.../report_v1.md
- Plots directory:  data/session_notes/08c5.../plots/  (e.g., feature_heatmap_top20.png, pathway_heatmap.png)
- Report (amended): data/session_notes/08c5.../report_v2.md   [present when CI ran]
Other generated files written to session folder. See manifest for full list.
```

Display rules:
- Print only the report_v1/v2 paths and the plots directory.
- For any other generated artifacts (e.g., CI code, payloads), write them to files under the session folder and print only the directory path (not file contents).

---

## Notes For Implementers
- Keep module boundaries, not network boundaries: Retrieval and Compute+Viz stay in‑process.
- One LLM for plan; one for report; optional amend. Everything else is deterministic and local.
- The optional code interpreter is a finalizer—not a planner tool—and only runs when policy and data warrant it.
- Provenance is mandatory: every pack/plot/report has a manifest entry and hashes for reproducibility.


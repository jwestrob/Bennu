# Agent Refactor Plan — Macro Fast Path (MFP)

Purpose: Implement a deterministic, batched, low-API-call controller for locus discovery with strict compiled-Cypher DB access and structure-first synthesis, while preserving existing planner paths behind feature flags.

Role and style
- Role: GPT-5 Pro surgical code transformation engine for a Python project.
- Style: Precise, minimal, reproducible patches. Add feature flags and docs. Preserve public APIs unless specified. Prefer composition over inheritance. Keep functions pure unless side effects are required.

Goals
- G1: Two-Mode Controller — Macro Fast Path (MFP) by default; Reactive Plan Path (RPP) only on escalation triggers.
- G2: Batched, template-only Cypher for seed retrieval and neighborhood extraction with deterministic EVI gates.
- G3: Deterministic EVI surrogate and early-exit logic to avoid useless expansions; no mini-model involvement.
- G4: Skeptic auditor to enforce invariants post costly steps; optional mini-model adjudication only on flagged anomalies.
- G5: Structure-first synthesis and persistence of Locus entities for reuse; a single heavy final synthesis call.

Constraints
- No dummy data in code or tests. Tests are dataset-agnostic and assert contracts/parameters/gates.
- Existing FSM and typed tool calls remain; fast path bypasses them via feature flag (FAST_PATH_ENABLED).
- All DB interactions use precompiled Cypher templates; no LLM-generated Cypher.
- API budget per fast-path run: heavy ≤1 (final synthesis); mini ≈0; Skeptic can trigger ≤1 mini on rare anomalies.

Feature flags
- FAST_PATH_ENABLED (default true): Enable deterministic macro options and bypass per-step LLM routing when safe.
- SKEPTIC_ENABLED (default true): Run auditor after costly batched ops; may request mini-model adjudication on anomaly.

Implementation plan (edits)
1) Deterministic router and option runner
   - Add `src/llm/options/router.py` with `parse_macro_intent` to detect queries like “Find N loci with <marker> … ±k … then nn closest”.
   - Add `src/llm/options/locus_discovery.py` implementing MFP LocusDiscoveryOption: seeds → EVI gate → single batched neighborhoods → optional batched kNN → persist `:Locus`.
   - Add `src/llm/options/skeptic.py` auditor skeleton for post-batch checks.

2) Deterministic EVI and invariants
   - Add deterministic EVI gate and invariants helpers (embedding parity check, schema assertions, dedup of nearby seeds).
   - Place under a dedicated deterministic namespace to avoid import conflicts.

3) Compiled Cypher templates (resources/cypher)
   - `seeds_by_marker.cypher`: union-PFAM/KOFAM marker lookup, return seed candidates with contig/genome context.
   - `batched_neighborhoods_gated.cypher`: single UNWIND batch to fetch neighbors by contig order for gated seeds.
   - `locus_schema_migration.cypher`: merge/persist Locus nodes and relationships.

4) Wiring (deferred safely)
   - Integrate fast path into `UnifiedAgentExecutor` behind `FAST_PATH_ENABLED`, with clean escalation back to planner on guard failure.
   - Provide structure-first finalization helper that can render deterministic summaries or make one heavy synthesis call.

5) Tests and docs
   - Contract tests for router intent parsing, option runner’s escalation behavior, and EVI determinism.
   - Architecture doc `docs/architecture/fast_path.md` summarizing controller, APIs, and persistence model.

Acceptance criteria
- Fast path executes for “Find N loci with <marker> … then kNN … tell me about them” without per-step mini-model calls.
- DB round-trips ≤2 before synthesis (seeds + batched neighborhoods). LanceDB ≤1 batched kNN when requested.
- Heavy model calls per run ≤1 in fast path.
- Insufficient gated seeds escalate back to planner gracefully without duplicating work.
- Logging clearly shows seeds count, gated count, and any escalation.

Compaction scope (added)
- Router: Replace ad‑hoc regex with a single compact Lark grammar and a typed Intent (+Obligations). Keep it tolerant to filler phrasing (e.g., “perform a …”, “by cosine similarity”), while remaining deterministic and LLM‑free.
- Scheduling: Build steps strictly from the obligation ledger (seeds → neighborhoods → optional LanceDB) and enforce a single batched LanceDB kNN call. Finalization is gated until all required obligations are done.
- Tools: Promote LanceDB into a first‑class tool with typed IO, manifest parity checks, and one batched call shared across fast path and FSM. Apply a single KG filter join (Pfam/Kofam) — no dummy biology.
- Templates: Keep compiled Cypher only; reuse the same templates across fast path and tools. Avoid duplicate queries via per‑executor dedup and deterministic seeds.
- Logging: Emit concise, high‑signal logs for parse state (GRAMMAR_*), fast‑path intent/results (MFP_*), DB seed summaries, and obligation gates.

Current status & recent debugging
- Grammar compatibility: Removed unsupported rule shorthands and alias operators; lower‑cased rule names; added optional “perform/a” and filler tokens to accept natural phrasing around the LanceDB stage.
- Obligations: Ledger created from Intent; finalization gate blocks synthesis if LanceDB is required and unmet; FSM restricts allowed tools accordingly when obligations exist.
- LanceDB: First‑class tool added; one batched call with a single KG‑filter join. Fast path can call it once; FSM fall‑back will be restricted until it runs.
- Instrumentation: Added GRAMMAR_COMPILE_FAIL/OK/PARSE_* logs, MFP_INTENT/MFP_RESULT, DB_SEED_SUMMARY, TOOL_INVOCATION neighborhood_extractor, and finalization‑gate logs.

Near‑term actions (follow‑ups)
- FSM bridge: pass seed_ids from the most recent DB results into lancedb_knn/neighborhood_extractor when fast path is skipped, so obligations can be satisfied without LLM parameter repair.
- Neighborhood strictness (optional): switch to exact ±k (upstream/downstream) rather than nearest‑by‑order window where required.
- Grammar coverage: extend number words and filler tokens as needed; add explicit tolerance for “tell me about them”, “cosine similarity”, emphasis markup.

Risk management and adapters
- No scientific logic or schema semantics changed; queries target existing labels/relations via templates.
- Deterministic helpers live in a new `deterministic/` namespace to avoid import conflicts with existing `utils.py` modules.
- Executor wiring will be feature-guarded and isolated to avoid breaking legacy flows.

Rollout steps
1. Land deterministic modules, templates, tests, and docs (no executor changes).
2. Add feature flags and guarded executor wiring; verify no regressions.
3. Enable Skeptic and final synthesis integration; verify API budgets via logs.
4. Iterate thresholds for EVI gate via config (no code changes to logic).

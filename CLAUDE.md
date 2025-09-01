Supersedes: previous CLAUDE note (2025-08-26) and pre‑compaction notes

Context: KEGG Pathway Completeness (Fast Path + Code Interpreter)

TL;DR
- Fast path for pathway completeness is live (canonicalizer-first → deterministic Cypher).
- Fixed schema direction for Protein→ENCODEDBY→Gene→BELONGSTOGENOME→Genome.
- Native totals recompute is implemented (bennu-native): reads data/reference/ko_pathway.list in-process and uses present KOs per genome; no CI required.
- Optional CI can still pretty-print, but totals come from native mapping by default.

Key Flags (export before running)
- USE_NATIVE_TOTALS_FOR_PATHWAYS=1      # default; compute totals from ko_pathway.list natively
- USE_CODE_INTERPRETER_IN_FAST_PATH=0   # optional pretty-print only; leave off for pure native
- CODE_INTERPRETER_URL=http://localhost:8000
- USE_CI_TOTALS_FOR_PATHWAYS=0          # legacy CI totals mode (not needed with native)

How to Run (Fast Path)
1) Ensure Code Interpreter is running on port 8000 and healthy.
2) In shell:
   export USE_NATIVE_TOTALS_FOR_PATHWAYS=1
   # optional
   # export USE_CODE_INTERPRETER_IN_FAST_PATH=1
   # export CODE_INTERPRETER_URL=http://localhost:8000
   python -m src.cli ask "Which KEGG pathways are complete in this metagenome?"

What To Expect
- Logs:
  - DB_TEMPLATE_EXECUTE: pathway_completeness_by_genome.cypher
  - FASTPATH_PC: rows=N use_ci=True
  - FASTPATH_PC: invoking code_interpreter (base_url=...)
  - FASTPATH_PC: invoking CI totals recompute (embedded ko_list)   # when USE_CI_TOTALS_FOR_PATHWAYS=1
- Answer: compact, per-genome listing of fully complete pathways computed using the full ko_pathway.list.

Why You Saw Many 100% Pathways Before
- Neo4j currently has KO→Pathway edges only for present KOs, so totals per pathway are computed from the present-only subgraph (total==present → 1.00).
- Native totals use ko_pathway.list to get true pathway KO sets and intersect with present KOs per genome.

Diagnostics / Sanity Checks
- Verify KO→Pathway presence count (present-only):
  MATCH (:KEGGOrtholog)-[:PARTICIPATESIN]->(:Pathway) RETURN count(*)
- Verify Protein→KO links exist:
  MATCH (p:Protein)-[:HASFUNCTION]->(:KEGGOrtholog) RETURN count(p)
- CI totals recompute is active when you see: “FASTPATH_PC: invoking CI totals recompute (embedded ko_list)”.

Files Changed
- resources/cypher/pathway_completeness_by_genome.cypher
  - Fixed ENCODEDBY/BELONGSTOGENOME direction; computes present_kos and joins to global totals.
- resources/cypher/present_kos_by_genome.cypher
  - Filters by optional genome_ids.
- src/llm/kegg/pathway_mapping.py (new)
  - Native loader for ko_pathway.list with cached maps; pathway filter helpers.
- src/llm/options/pathway_completeness.py
  - Added native mode (default) using present_kos_by_genome + ko_pathway.list; supports pathway filters.
- src/llm/rag_system/agent_executor.py
  - PATHWAY_COMPLETENESS branch: passes pathways and uses native totals flag; cleaner formatting.
- src/llm/intent/models.py, src/llm/intent/canonicalizer.py
  - Canonical intent extended with optional 'pathways' filter.
- src/llm/config.py
  - Added USE_NATIVE_TOTALS_FOR_PATHWAYS (default true); env parsing.

Open Issues / Next Steps
1) Validate native totals output and consider filtering meta pathways (map01100/01110/01120) by default.
2) Auto-set min_completeness=1.0 when the question asks for “complete”.
3) Optionally load full KO→Pathway mapping into Neo4j to make DB-only totals accurate (native path remains authoritative).
4) Improve presentation: add pathway name lookup to native output.

Quick Toggle Cheatsheet
- Enable CI formatting only:
  export USE_CODE_INTERPRETER_IN_FAST_PATH=1; unset USE_CI_TOTALS_FOR_PATHWAYS
- Enable CI totals recompute:
  export USE_CODE_INTERPRETER_IN_FAST_PATH=1; export USE_CI_TOTALS_FOR_PATHWAYS=1

Notes
- If CI is enabled but you still see DB-only output, check logs:
  - CI result failures or missing flags will leave the Neo4j summary intact.
- If CI embedded fallback says “genomes=0 pathways=…”, present_kos_by_genome returned no rows; verify Protein→KO links.


---

Current Status (2025-08-27)

- MacroPlanner is the primary path; no SM-specific fallback or augmentation. Plans are 100% model-driven from the operator catalog.
- Planner context:
  - operator_catalog (with PFAM, KO, union, BGC/CAZyme, completeness operators)
  - ko_reference (compact KO knum: definition list)
  - constraints: empty (retry adds allow_keyword_discovery=1)
  - PFAM reminder added to MacroPlannerSignature docstring: encourages breadth-first exploration across PFAM and KO, optional completeness, and corroboration.
- Retry logic: If first plan yields no evidence, perform one replan/retry, then always synthesize.
- Task Graph: Final answers prepend a TASK GRAPH section listing executed steps.
- Keyword search: PFAM and KO keyword operators and a PFAM+KO union operator are available; queries are tokenized internally for broader matches.
- Completeness: No nudges (prefer_native_totals removed). Planner may use completeness, but it’s optional and requires valid map IDs for filtering.

Planned/Next
- Add KEGG pathway map ID→name list to planner context for reliable completeness filters.
- Consider light diversity cues (e.g., diversify_probes, try_both_pfam_and_ko) to encourage PFAM usage on first pass without prescribing content.
- Integrate SQLite-backed sequence retrieval for motif checks/alignment.

---

Tooling Roadmap + Planner/Policy (2025-08-27)

Overview
- This section captures current tool integration status, upcoming additions, defaults, and toggles that influence planner behavior and synthesis.

Integrated Tools (status)
- Literature search (PubMed via Biopython): Integrated and gated by heuristics in core + FSM; requires `Bio.Entrez` and `config.email` (API key optional). Graceful degradation when unavailable.
- Code interpreter (HTTP microservice): Integrated with health check; used for optional analysis/formatting; controlled by `CODE_INTERPRETER_URL`.
- DB templates (Cypher via FileCypherRunner): Core path for KG queries; used by MacroPlanner operators.
- LanceDB kNN: First-class tool (`lancedb_knn`); used for neighborhood enrichment and locus summaries.
- Neighborhood extractor: DB-backed adjacency/window extractors; available to the agent.
- Annotation discovery + Concept discovery: Keyword → PFAM/KOFAM → proteins → neighborhoods flow; agent-triggerable.
- MacroPlanner operators for keyword search: FindProteinsByPfamKeyword, FindProteinsByKoKeyword, and AnnotationDiscovery (PFAM+KO union) exposed to the planner for free‑text functional queries (e.g., “hydrogenase”, “rubisco”).
- Report synthesis: Tool wrapper to request final synthesis over session data.
- Sequence retrieval (planned): Add a sequence fetcher backed by the existing SQLite database to retrieve protein/CDS/contig sequences by `protein_id`/`gene_id`/`genome_id`. Expose as a planner-visible tool to enable motif checks, boundary validation, and downstream alignment without leaving the agent.

Known Gaps / Fixups
- genome_selector tool: Update integration to call `UnifiedGenomeSelector.analyze_genome_intent(...)` (current call to `select_genome` will fail). Map fields to the tool envelope on success.
- Whole-genome reader: Integrated with caching; add tests and guardrails (size/timeouts). Not used by MacroPlanner yet.
- External tools in MFP: MacroPlanner path doesn’t automatically run literature/code tools. Optionally add a post-plan hook to incorporate them before final synthesis when heuristics say they help.

Defaults and Behaviors (agent-visible)
- ComputePathwayCompleteness: `min_completeness` now defaults to `1e-6` to filter empty pathways; set `0.0` to include empties. The operator catalog documents this default so the planner “knows”.
- BGC/CAZyme queries: Operators accept `genome_id` (optional) and `genome_ids` (optional). Empty/null filters → global mode.
- CAZyme family summary: We append a small, global family-counts step for SM queries to aid summarization. Can be gated by a flag (see below).
- Macro results handoff: MacroPlanner results are passed as structured raw items to the synthesizer (not a single giant blob), avoiding context loss in compression.

Planner vs. Fallback (clarity)
- Primary: `MacroPlannerSignature` (DSPy) returns a STRICT plan JSON from the operator catalog; we execute it deterministically.
- Fallback: Disabled. We no longer inject any hard-coded SM plan when the planner returns no plan. The system proceeds to other execution paths (e.g., Macro Fast Path or FSM) without pre-made task graphs.
- Augmentation: Disabled by default. We do not append CAZyme evidence or family-count summaries to planner-produced plans unless explicitly requested.

Proposed Config Flags
- `USE_MFP_PLANNER=1` (default): Enable MacroPlanner path.
- `INCLUDE_CAZYME_SUMMARY` (no default; off unless explicitly set): If set to `1`, allow augmenting planner plans with a CAZyme family-counts step.
- `DEFAULT_MIN_PC=1e-6`: Optional env to override the `ComputePathwayCompleteness` default minimum completeness (set `0.0` to include empty pathways by default).
- `CODE_INTERPRETER_URL=http://localhost:8000`: Endpoint for code interpreter microservice.

Planned Tool Additions (high value)
- MIBiG lookup: Given BGCs, fetch nearest MIBiG clusters (local JSON index preferred, API optional) and return concise similarity + product summaries for quick chemical context.
- KEGG map name resolver: Tiny helper to map `mapxxxxx` → human-readable names for reporting (load once; cache).
- FASTA/TSV exporter: Export protein or BGC gene sets as FASTA/TSV for downstream tools; return a signed path or chunked payload.
- Coverage summarizer: Aggregate coverage/depth per BGC (when fields exist) to prioritize clusters in metagenomes; degrade gracefully if absent.
- Compound linker: Map predicted BGC product classes to representative compounds (PubChem/ChEBI) for background context (gated by network policy).

Testing Plan (quick wins)
- Smoke tests for: neighborhood_extractor, annotation_discovery, concept_discovery, lancedb_knn, report_synthesis.
- Literature search: network-skipped integration test; assert graceful fallback when Biopython missing.
- Genome selector tool: unit-test with a mocked selector for `analyze_genome_intent`.

---

Context Compaction Policy (2025-08-29) — supersedes pre‑compaction notes

Goal
- Keep early testing and typical analyses out of ProgressiveSynthesizer (Map‑Reduce) by default while preserving strong evidence. Allow full JSON detail when the task is small and intentionally requests it.

Defaults (now live)
- Evidence format: discovered_proteins are deduplicated globally (by genome_id, protein_id) and rendered compactly as counts + up to 10 examples. Catalog hits (PFAM/KO) are not fed to synthesis.
- True totals: we preserve full row counts during pre‑compaction (total_rows) and display correct totals in synthesis headers. The example list is capped (10 by default) but counts reflect the full set.
- Compact by default: macro_result lists are pre‑compacted before token counting to avoid triggering Map‑Reduce solely due to large JSON payloads.
- Planner rubric: Two‑stage search (catalog → IDs → exact retrieval) remains. Prefer compact evidence unless explicitly requested (see below).

Optional full JSON (small, targeted queries)
- Operator flag: AnnotationDiscovery accepts `return_full_rows=true`. When set, bindings are marked with `_format: 'full'` and the formatter will include full JSON rows for reasonably small results (≤ 2,000 rows).
- Guidance for planner: Use `return_full_rows=true` only for compact targets (e.g., “find methyltransferases in genome GX…”). For broader, multi‑marker searches, keep compact summaries.
- Caution: Full rows can be very large on bigger datasets; prefer compact unless you truly need row‑level detail.

Evidence matrix (recommended presentation)
- For synthesis, include a small summary table per marker (marker → total rows, optional per‑genome counts, and 3 examples). Full lists are still available via tool cache.

Debugging & Guardrails
- Context debug log: We emit a single line summarizing how many lists were collected and the top largest lists with row counts. This quickly pinpoints bloat sources.
- Trim messages: We log when duplicate discovered_proteins rows were dropped during collection.
- Tool cache: Large raw lists can be cached on disk (session tool cache) and referenced by ID in notes when needed, keeping the model context small.

Planner Language (concise)
- “Prefer compact evidence (counts + examples). If full row detail is essential and the target is small, set `return_full_rows=true` on AnnotationDiscovery. For broad probes, keep compact to prevent excessive context.”

Next tighten‑ups (optional; off by default)
- Example cap: Soft‑cap displayed examples to 10 per marker; store full lists in cache and reference by ID.
- Aggregation: For very large runs, aggregate across overlapping markers (counts + union examples) while keeping full detailed lists in the cache.

Scaling Note
- Final synthesis can target large-context models (e.g., Sonnet 4 with 1M context). Macro results are already structured to enable efficient summarization without overwhelming the context window.

## New Model Integration (Internal Guide)

Follow these steps when adding support for a new model or provider. These patterns avoid routing and logging pitfalls we’ve seen.

1) Centralize routing in `src/llm/lm_factory.py`
- Add a friendly alias in `_resolve_alias` (e.g., `gpt-5-high`, `4.1-mini`, `sonnet-4`).
- Preserve explicit provider prefixes; if user passes `openrouter/...`, keep it so we route via OpenRouter explicitly.

2) GPT‑5 (OpenAI) and chat models
- Use `dspy.LM(model="openai/<id>")` with minimal args. Do not set `model_type="responses"` for planner/retrieval — LiteLLM may downroute to `/v1/completions` and 404 for chat models.
- Do not pass `max_tokens` or `response_format`. If the adapter injects them, set `drop_params=True` with `additional_drop_params=["max_tokens","max_output_tokens","max_completion_tokens","response_format"]` and strip them from `lm.kwargs` defensively.

3) OpenRouter (e.g., Anthropic Sonnet 4 via OpenAI‑compatible endpoint)
- Inside the `openrouter/...` branch: set `OPENAI_API_BASE=https://openrouter.ai/api/v1` and `OPENAI_API_KEY` from `OPENROUTER_API_KEY`.
- Force OpenAI‑compatible routing by using `dspy.LM(model="openai/<vendor>/<model>")` (e.g., `openai/anthropic/claude-sonnet-4`). This prevents LiteLLM from selecting the native Anthropic adapter even though `anthropic/` is present.
- Do not use `dspy.OpenAI(...)` — it’s not guaranteed across DSPy versions.

4) Native Anthropic
- Only use `dspy.LM(model="anthropic/<id>")` when the user asked for native Anthropic and `ANTHROPIC_API_KEY` exists.

5) Per‑step defaults and flags
- Defaults: planner = `gpt-5-high`, irb = `gpt-4.1-mini`, reporter = `gpt-5-high`.
- Flags override per run: `-planner`, `-irb`, `-reporter`.

6) IRB bypass and fast‑path during testing
- Small contexts bypass IRB. To exercise reporter/IRB reliably, export: `IRB_BYPASS_TOKENS=0` and `FAST_PATH_ENABLED=0` prior to running.

7) LiteLLM logging controls
- Suppress heavy logging/cold storage to avoid atexit errors:
  - Loggers: set `LiteLLM`, `litellm`, `litellm.proxy` to CRITICAL.
  - Env: `LITELLM_LOGGING=False`, `LITELLM_DISABLE_COLD_STORAGE=1`, `LITELLM_DISABLE_STANDARD_LOGGING=1`.

8) Planner plan stability
- Plans may reference inputs before they exist. The executor now tolerates missing inputs and lets operators handle empty lists. Prefer operators that treat missing `ko_ids`/`pfam_ids` as empty.

9) Quick validation checklist
- Add alias + provider routing in `lm_factory.py`.
- Verify planner uses chat semantics (no responses) and no max tokens.
- Verify OpenRouter path uses `openai/<vendor>/<model>` and honors `OPENROUTER_API_KEY` + `OPENAI_API_BASE`.
- Run: `FAST_PATH_ENABLED=0 IRB_BYPASS_TOKENS=0` with `-planner`, `-irb`, `-reporter` pointing to the new model. Confirm no Anthropic key errors and no `/v1/completions` 404s.

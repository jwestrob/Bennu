Supersedes: previous CLAUDE note (2025-08-26)

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

Scaling Note
- Final synthesis can target large-context models (e.g., Sonnet 4 with 1M context). Macro results are already structured to enable efficient summarization without overwhelming the context window.

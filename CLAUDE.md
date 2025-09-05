Supersedes: previous CLAUDE note (2025-08-26) and pre‑compaction notes

Compaction Prep — 2025‑09‑04 (current focus)

What we’re working on
- Neighborhood analysis around RuBisCO/PRK/nif anchors (Rubisco PF00016/PF00101) with fast, agent‑controlled breadth/latency.
- Ensuring the agent (planner) produces a valid task graph that wires seeds into NeighborhoodContext deterministically (no fix‑ups).
- Guaranteeing Neo4j adjacency substrate ([:NEXT]) is present and annotated neighbors (PFAM/KO) are returned.

What’s in place (works now)
- NeighborhoodContext operator (strict): requires explicit seeds via `inputs.discovered_proteins` (bound rowset) or params (`protein_ids` / `seed_pfam_ids` / `seed_ko_ids`). No fallbacks.
- AnnotationDiscovery: accepts `inputs.pfam_ids`/`inputs.ko_ids` or params (`pfam_ids`/`ko_ids`) for anchored rowsets; always includes PFAM IDs internally for downstream seed filtering.
- Planner rubric: Includes hard constraints + example that binds rowset and passes it to NeighborhoodContext. Example uses `k=5`; agent may choose any k or omit k for flanking (±5).
- Neo4j Stage 07 post‑load: constraints/indexes + precompute [:NEXT] edges (runs automatically when NEO4J creds are set). Clear logging and pairs≈ count.
- Diagnostics script: `scripts/diagnostics/neo4j_check_next.py` verifies [:NEXT], k‑step adjacency, and flanking (±N) with PFAM/KO neighbor summaries.

Known issues / observations
- Some RuBisCO contigs are single‑gene; adjacency and flanking both return 0 neighbors by definition. Others have exactly two genes (one RuBisCO + one neighbor), which is expected for these short contigs.
- gpt‑5‑low sometimes sets `k=1` by habit (adjacency ±1). On single‑gene contigs this yields empty neighborhoods; omit `k` (flanking) or set a wider `k` only when adjacency is required.
- Reporter may not surface neighbor details if only macro summaries are used. Use `output_profile='rowset'` in NeighborhoodContext to inject full neighbor rows into the environment when needed.

Next steps (agent‑first, no fix‑ups)
- Keep planner example copy‑friendly and seed‑wiring explicit: SearchPfamCatalogFuzzy → AnnotationDiscovery (anchored rowset via inputs/params) → bind → NeighborhoodContext with `inputs.discovered_proteins` and explicit `seeds_limit`.
- Prefer omitting `k` by default (flanking ±5). Set `k` only when adjacency semantics are specifically requested.
- For better surfacing, have NeighborhoodContext use `output_profile='rowset'` on small seed sets (≤ 12) so neighbor rows (protein_id, pfam_desc, ko_desc) are visible to the reporter.
- If adjacency is mandatory and a contig returns 0 over [:NEXT] despite multiple genes, re‑run post‑load tuning and confirm contig ID normalization in gene loading.

Important references
- Local Neo4j (dev): URI `bolt://localhost:7687`; user `neo4j`; password `your_new_password` (test instance).
- Post‑load (manual): `python -m src.build_kg.postload_tuning --create-indexes` or `--neighbors-only` with NEO4J env vars set.
- Diagnostics: `python scripts/diagnostics/neo4j_check_next.py --uri ... --user neo4j --password your_new_password --k 5 --flank_n 5 --limit 6`.


Agent Status — 2025‑09‑01 (compact)

Where we are
- Per‑step model overrides stable; IRB uses 4.1‑mini and now sets `max_tokens=30000` (prevents dspy 4k truncation).
- IRB Option A shipped: multi‑anchor returns minimal JSON; patches built locally → 1 API call/pack, no 60+ call explosions.
- IRB artifacts persisted: `data/session_notes/<session_id>/synthesis_notes/{irb_report.md, report_context.md, irb_report.json}`.
- Prompt cleanup: dataset‑specific IDs removed from LLM prompts (placeholders used) to avoid contaminating outputs.
- Planner rubric updated with PFAM policy: avoid exact equality on PFxxxxx; use accession‑prefix, short‑name, and description matching via flexible templates.

Working on
- AnnotationDiscovery fallback: if exact PFAM IDs return zero, automatically run flexible PFAM retrieval (prefix/name/description) and merge results (prevents RubisCO misses like PF00016.26/RuBisCO_large).
- IRB performance knobs (pack size/grouped multi‑anchor) and guardrails; optional “multi‑only + deterministic” mode for predictable latency.
- Optional: save planner JSON plan and compact macro items alongside IRB outputs for inspection.

Next steps / TODOs
- Implement flexible PFAM fallback in AnnotationDiscovery (accession prefix OR id prefix OR description contains) with token normalization (PF00016 ↔ PF00016.*, underscore→space).
- Expose IRB pack/grouping flags and a per‑pack time/call cap; add smoke tests for new IRB path and PFAM fallback (RubisCO present).
- Optional: small helper to dump planner plan and raw macro items into the session folder.

Quick refs
- RubisCO verified in current DB: PF00016.26 (RuBisCO_large) ~5 proteins; PF00101.25 (RuBisCO_small) ~1 protein; PRK K00855 present; rbcL/rbcS KOs absent in this snapshot.
- Script: `scripts/search_rubisco.py` queries domains/KO presence (neo4j/your_new_password).

Local Neo4j Credentials (dev)
- URI: `bolt://localhost:7687`
- User: `neo4j`
- Password: `your_new_password`

Neo4j Post‑Load Tuning & Neighborhood Diagnostics (required for neighborhoods)
- Purpose: Create `[:NEXT]` gene adjacency edges and validate neighborhood queries before running agent prompts.

Automatic (Stage 07)
- After Stage 07 KG build, the CLI runs a post‑load step if `NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD` are set:
  - Ensures constraints/indexes (incl. `Domain.pfamAccession`).
  - Precomputes `[:NEXT]` per contig in coordinate order.
  - Logs: “Running Neo4j post-load tuning…”, “✓ Constraints and indexes ensured”, “✓ NEXT edges processed (pairs≈N)”.

Manual (explicit)
```bash
# Full pass (indexes + NEXT)
NEO4J_URI=bolt://localhost:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=your_new_password \
python -m src.build_kg.postload_tuning --create-indexes

# NEXT only
NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=your_new_password \
python -m src.build_kg.postload_tuning --neighbors-only
```

Diagnostics script (quick sanity)
```bash
python scripts/diagnostics/neo4j_check_next.py \
  --uri bolt://localhost:7687 --user neo4j --password your_new_password \
  --k 5 --flank_n 5 --limit 6
```
- Prints global `[:NEXT]` count; per-seed NEXT degree; adjacency (k) and flanking (±flank_n) neighbor counts; sample neighbor annotations `{protein_id, pfams, kos, contig, start, end}`.
- If adjacency=0 but flanking>0 for a seed, the contig likely lacked NEXT at load (re-run post-load) or is single-gene.

Planner/Operator reminders
- NeighborhoodContext is strict: provide seeds via `inputs.discovered_proteins` (bound rowset) or `params.protein_ids/seed_pfam_ids/seed_ko_ids`.
- Adjacency breadth: set `k` (e.g., 1/5/10). Omit `k` to use flanking (±5 by contig order).
Problem Summary — 2025‑09‑02

Observed
- The database contains hallmark evidence for gas fixation in SRR6231169:
  - PRK K00855: 6 proteins
  - Nitrogenase nifH/D/K (K02588/K02586/K02591): 4 / 3 / 4 proteins
  - RuBisCO PFAMs: PF00016.26 (RuBisCO_large; ~5 proteins), PF00101.25 (RuBisCO_small)
- Final reports often claim “no PRK, no nitrogenase, no RuBisCO Pfams”. PFAM discovery now returns many families (substring), but specific Calvin/wl anchors don’t reflect PRK/RuBisCO/nif.

Root Cause (dataflow)
- MacroPlanner → IRB collector whitelisting (src/llm/rag_system/core.py, _collect_macro_raw_items):
  - Only passes whitelisted list-shaped bindings into IRB: {discovered_proteins, pathway_completeness, bgcs, cazymes, cazyme_family_counts}.
  - Drops FetchPresentKOs output (“present”), which is precisely where K00855 and nifHDK appear. IRB never sees these hallmarks.
  - Global dedup across anchors removes repeated proteins. Evidence found under broad anchors gets removed from specific anchors (e.g., Calvin), starving those anchors of relevant rows.
- PFAM display adjustments:
  - We switched to names-only in discovered_proteins and hid pfam_ids (PFxxxxx) by default. Domain.name is NULL in this DB, so pfam_name falls back to description/id. The recognizer doesn’t see PF00016/PF00101 tokens it relies on, and the small RuBisCO signal gets lost in large substring result sets.

Quick DB Proof (Cypher)
- PRK: MATCH (p)-[:HASFUNCTION]->(:KEGGOrtholog {id:'K00855'})-[:<-]-() MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(:Genome {id:'SRR6231169'}) RETURN count(DISTINCT p) → 6
- nifH/D/K: K02588=4, K02586=3, K02591=4 (scoped to SRR6231169)
- RuBisCO PFAM: MATCH (d:Domain) WHERE toLower(d.pfamAccession) STARTS WITH 'pf00016' RETURN d.id,d.pfamAccession → RuBisCO_large PF00016.26; proteins with PF00016* = 5

Consequence
- IRB synthesizes from incomplete context (no “present” KOs, and anchor-specific evidence trimmed), leading to false negatives for gas-fixation capabilities despite their presence in the KG.

Proposed Plan — Robust, Extensible Collector + Facet‑First IRB

Phase 0 — Instrumentation (no behavior change)
- Save full MacroPlanner environment before collection: synthesis_notes/all_env.json (keys, example rows).
- Save per‑anchor summaries to synthesis_notes/anchors_debug.json: rows count; top 10 unique_pfams (names); top 10 unique_pfam_accessions (PFxxxxx); top 10 unique_kos; up to 5 example (genome_id, protein_id).
- Add DEBUG_ANN_DISCOVERY=1 flag to include pfam_ids in discovered_proteins JSON (not printed), so IRB sees hallmark PF tokens.

Phase 1 — Replace whitelist with Auto‑Adapter + Budget‑Aware Aggregator
- Auto‑adapter (collector): For any binding value, infer a lightweight “evidence envelope”:
  {
    type: 'rowset' | 'enriched' | 'summary',
    name: '<binding>',
    rows: [...], // optional if summarized
    facets: { proteins, pfams, pfam_ids, kos, pathways },
    schema_version: 'v1',
    provenance: [...] 
  }
- Facet extraction:
  - proteins → [{genome_id, protein_id}]
  - pfams → readable names (name > desc > id)
  - pfam_ids → normalized PFxxxxx for internal hallmark use
  - kos → ['Kxxxxx']
  - pathways → KEGG map ids when present
- Budget-aware aggregator:
  - MAX_BINDINGS_PER_ANCHOR (e.g., 8); MAX_ROWS_PER_BINDING (e.g., 500)
  - Summarize large sets into facets (unique_pfams, unique_kos, counts, examples). Keep rows optional (sampled) to stay within token budgets.
- Dedup scoped per-binding/per-anchor only; never global. Preserve proteins across anchors when they are evidence for multiple capabilities.

Phase 2 — IRB Facet‑First Summarization (names-only display)
- IRB consumes facets rather than opaque bindings.
- Names-only: present PFAM names in text; keep pfam_ids internally for recognition (never printed unless debug).
- Integrate FetchPresentKOs by including its facets (kos) so PRK/nifHDK are always visible to the editor.
- Add small hallmark detectors that read facets:
  - rubisco: any pfam_ids startswith PF00016 or PF00101
  - prk: 'K00855' in kos
  - nif: 'K02588'/'K02586'/'K02591' in kos
  - mcrA: 'K00399' in kos; rTCA: 'K15230','K15231' in kos

Phase 3 — Query and Display Hygiene
- Accession filters: For `PFxxxxx` tokens, prefer STARTS WITH on pfamAccession (index‑backed, version‑tolerant). Use CONTAINS for name/desc only.
- Domain.name backfill (optional): load from data/reference/pfam_id_desc.tsv (pfamAccession STARTS WITH). Improves name rendering without altering logic.

Phase 4 — Validation & Acceptance
- DB sanity checks:
  - PRK (K00855) and nifHDK present in SRR6231169; RuBisCO PFAMs present.
- One ask run with instrumentation on; inspect:
  - all_env.json contains 'present' KOs and reasonable size.
  - anchors_debug.json for Calvin/N fixation anchors shows K00855 and PF00016/PF00101 in top facets; proteins ≥ expected counts.
- Final report mentions PRK, nif, and RuBisCO when present, using names; no PFxxxxx printed.

Rationale for Removing Whitelist
- Original intent: control tokens, stabilize IRB input, reduce duplication.
- Replacement (auto-adapter + caps + facets) keeps those benefits while eliminating brittle, manual allowlists and global dedup.
- New tools “just work”: any row-shaped output gets summarized to facets; no IRB schema edits needed per tool.

Flags & Rollback
- DEBUG_ANN_DISCOVERY (include pfam_ids in JSON only).
- NEW_COLLECTOR=0 (fallback to old collector if needed).
- IRB_FACET_MODE=1 (enable facet-first summarization; switch off to revert).

Action Items
1) Add instrumentation (Phase 0) and run a single ask to capture combined_env + anchors_debug.
2) Implement collector auto-adapter + budget caps; per-binding/per-anchor dedup (Phase 1).
3) Switch IRB to facets and add hallmark detectors; keep names-only in display (Phase 2).
4) Validate against SRR6231169 (Phase 4). Consider optional Domain.name backfill.

Expected Outcome
- Reports correctly reflect PRK/nif/RuBisCO when present in SRR6231169.
- Adding new tools won’t require updating whitelists or IRB schemas; they’ll be summarized automatically with controlled token usage.

Database Schema — Update (2025‑09‑02)

- Domain nodes now carry accession metadata:
  - Properties: `id` (family identifier; prefer canonical PFxxxxx), `pfamAccession` (PFxxxxx; indexed), `name` (short), `description`.
  - Index: `CREATE INDEX domain_pfamAccession IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession)`.
  - Query guidance: For accession-based retrieval, prefer `d.pfamAccession = 'PFxxxxx'`; use short-name/description for keyword discovery.

PFAM Accession Field — Plan (2025‑09‑02)

Summary
- We need a dedicated, queryable Domain.pfamAccession property in Neo4j. Current graphs don’t consistently carry it, which causes misses for versioned PFAM families (e.g., RuBisCO PF00016.26) when matching by unversioned “PFxxxxx” tokens.
- This plan adds pfamAccession end-to-end: Astra output → annotation processors → RDF → CSV → Neo4j, plus indexes and Cypher templates. No behavioral code changes are committed until approved; this section records the integration plan.

Sources to inspect/adjust
- Stage 04 (Astra output): `src/ingest/04_astra_scan.py` and `data/stage04_astra/pfam_results/PFAM_hits_df.tsv` columns.
- Annotation processors: `src/build_kg/annotation_processors.py` (PfamProcessor).
- RDF builder: `src/build_kg/rdf_builder.py` (PFAM triples; KG.pfamAccession).
- RDF→CSV converter: `src/build_kg/rdf_to_csv_converter.py` (property passthrough).
- Neo4j bulk load + tuning: `src/build_kg/neo4j_bulk_loader.py`, `src/build_kg/postload_tuning.py` (indexes/constraints).
- Cypher templates: `resources/cypher/*.cypher`, `src/llm/kg/cypher_templates/*.cypher` that reference pfamAccession.

Data model (proposed)
- Domain node properties:
  - id: PFAM family identifier used in URI (prefer canonical unversioned PFxxxxx; legacy graphs may contain versioned IDs like PFxxxxx.yy).
  - pfamAccession: Unversioned accession “PFxxxxx” (string, indexed). Primary field for accession matching.
  - pfamVersion: Optional version integer (e.g., 26) when known.
  - name: Short name (e.g., RuBisCO_large), when available.
  - description: Family description, when available.
- DomainAnnotation nodes remain unchanged; they link Protein → Domain via DOMAINFAMILY.

Pipeline changes (end-to-end)
1) Astra output (Stage 04)
   - Goal: ensure each PFAM hit carries both short name and accession (and version if available).
   - Action: verify `PFAM_hits_df.tsv` columns. Target columns: `sequence_id`, `hmm_name` (short), `hmm_acc` (PFxxxxx[.yy]).
   - If `hmm_acc` missing: update `astra search` flags (or postprocess) to include accession; otherwise join against `data/reference/pfam_id_desc.tsv` to map `hmm_name → PFxxxxx` (unversioned). Record uncertainties when names map to multiple accessions.

2) Annotation processors
   - File: `src/build_kg/annotation_processors.py` (PfamProcessor.create_domain_entities)
   - Changes:
     - Capture both `pfam_acc` and `pfam_name` fields from the PFAM hits. Prefer `hmm_acc` when present; else map from reference TSV.
     - Normalize accession into `pfamAccession` (unversioned PFxxxxx) and `pfamVersion` (if the hit reports version, otherwise null).
     - Keep `domain_id` stable and unique; continue emitting `start_pos`, `end_pos`, `bitscore`, `evalue` as-is.

3) RDF builder
   - File: `src/build_kg/rdf_builder.py`
   - Changes:
     - Domain family URI: use PFAM namespace + canonical unversioned accession (PFxxxxx) for stability.
  - Add triples on the family node: `KG.pfamAccession` (PFxxxxx), and optionally `KG.name` (short name) and `KG.description` when available from reference.
     - Continue linking DomainAnnotation → Domain via `KG.domainFamily` and Protein via `KG.hasDomain`.

4) RDF→CSV converter
   - File: `src/build_kg/rdf_to_csv_converter.py`
  - No code change expected; it already preserves arbitrary node properties as columns. Confirm Domain CSVs include `pfamAccession`.

5) Neo4j load + indexes
   - Files: `src/build_kg/neo4j_bulk_loader.py`, `src/build_kg/postload_tuning.py`
   - Add index: `CREATE INDEX domain_pfamAccession IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession)`.
   - Keep existing uniqueness on `d.id`. Fulltext index remains on `[d.id, d.name, d.description]`.

6) Cypher templates (GenomicRAG and legacy)
   - Files: `resources/cypher/proteins_by_pfam_ids.cypher`, `resources/cypher/pfam_ids_by_query.cypher`, `resources/cypher/proteins_by_pfam_keyword.cypher`, and `src/llm/kg/cypher_templates/*`.
   - Matching policy with pfamAccession:
     - For accession tokens (`PFxxxxx`), match `d.pfamAccession = $pf` (exact) for best precision; use `STARTS WITH` only if you must tolerate partials.
     - For short-name terms, prefer `LOWER(d.name) = term` or `d.description CONTAINS term`; keep `LOWER(d.id)` for legacy graphs.
   - Backward compatibility: retain id/name/description search in templates until all environments are rebuilt with pfamAccession.

7) Backfill strategy for existing Neo4j databases
   - Option A (rebuild): Reconstruct RDF with updated builder and bulk-import anew (fastest path, lowest risk).
   - Option B (in-place backfill):
     - Load `pfam_id_desc.tsv` into Neo4j (LOAD CSV) and set `d.pfamAccession` by:
     - If `d.id` matches `^PF\d{5}(?:\.\d+)?$`, set `pfamAccession = substringBefore(d.id, '.')`.
       - Else join on short name and set `pfamAccession` accordingly; log conflicts for manual review.
     - Create the `domain_pfamAccession` index.

8) Validation
   - Script: `scripts/search_rubisco.py` already checks RuBisCO presence across Domain and KO annotations; extend it to verify that `pfamAccession` exists and that PF00016*/PF00101* prefix/exact queries return counts > 0.
   - Spot-check a few other families (e.g., PF00106, PF00389) and confirm round-trip from PFAM accession → proteins works.

Rollout plan
- Phase 1: Implement processors + RDF builder updates behind a guard; generate a small test KG and CSV; verify with `search_rubisco.py`.
- Phase 2: Add indexes and adjust Cypher templates to prefer `d.pfamAccession` when present; keep backward-compatible name/id matching.
  - Update DSPy signatures (schema + PFAM rubric) to reflect the new `pfamAccession` field and recommend matching on it.
- Phase 3: Rebuild production KG from RDF and bulk-import; then enable `pfamAccession` exact/prefix matching policies.
- Phase 4: Remove temporary fallbacks after confidence is high.

Notes
- No code changes are committed without approval. This plan documents the intended edits so we can review and stage them safely.

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
Policy Update — 2025‑09‑03 (Facet controls; no hidden caps)

- No hidden caps in PFAM facet counts:
  - Removed per-token “best match” cap (LIMIT 1) and pre-clamping of catalog tokens.
  - Remaining internal cap: candidate PFAM families per token (default 200); now exposed to the agent as pfam_candidate_cap.
- Agent‑controlled knobs for facet breadth/latency:
  - pfam_tokens_top_n, ko_tokens_top_n: how many catalog tokens to carry into counts.
  - pfam_candidate_cap: cap per token inside PFAM counts (default 200).
  - pfam_top_k, ko_top_k: final facet sizes.
- Facet schema: rows now standardize to {id, name, count}. “score” is not computed.
- Keyword discipline (planner rubric reminder):
  - Avoid generic enzyme classes (aldolase, epimerase, dehydrogenase, carboxylase) in PFAM/KO probes unless explicitly required.
  - Prefer hallmark accessions/names with up to 1–2 concise synonyms.
  - Keep per‑theme probes tight and separate.
- Completeness semantics (current): KO‑based coverage; union scope for single‑genome runs. Multi‑genome semantics to be documented when test set is in use.

Observability
- PFAM count logs include tokens considered, total candidate PFAM rows returned (post‑cap), and elapsed time. This helps identify noisy tokens for planner tightening.

Implementation Notes — 2025‑09‑03 (Progress + Latest Changes)

Overview
- Goal: Restore recall for hallmark markers (RuBisCO/PRK/nifHDK) while keeping the pipeline fast, static, and agent‑controllable (no hidden caps; no env flags; no theme‑specific hard‑coding).
- Status: Facet discovery stabilized. Planner rubric updated. PFAM/KO facets now carry names. Completeness fixed. Indexes and Cypher made more index‑friendly.

Key Changes (surgical, agent‑controllable)
- AnnotationDiscovery (facet‑first)
  - group_by controls PFAM vs KO (no mixed internal work). Rowset mode unchanged.
  - New knobs (planner‑set): pfam_tokens_top_n, ko_tokens_top_n, pfam_candidate_cap (per‑token PFAM candidate cap; default 200), pfam_top_k, ko_top_k.
  - Facet rows standardized to {id, name, count}. No "score" field.
  - PFAM token provenance surfaced in selection_metadata when counts path is used (debugging aid only).

- PFAM counts (resources/cypher/count_proteins_by_pfam_tokens.cypher)
  - No hidden per‑token LIMIT 1 (removed); planner governs breadth via pfam_tokens_top_n and candidate cap.
  - Parameterized $candidate_cap (default 200) for candidate domains per token; uses index‑friendly STARTS WITH for accessions and CONTAINS for names.

- KO counts (resources/cypher/count_proteins_by_ko_ids.cypher)
  - Deterministic UNWIND + equality on ko.id. Facet rows now include name (from description).

- Planner rubric (MacroPlannerSignature)
  - Keyword discipline: avoid generic enzyme classes in PFAM/KO probes (aldolase, epimerase, dehydrogenase, carboxylase); prefer hallmark accessions + 1–2 concise synonyms; keep probes per theme tight.
  - Explicit caps: always set pfam_tokens_top_n, ko_tokens_top_n, pfam_candidate_cap, pfam_top_k, ko_top_k. Standardize fields to ['id','name','count'].

- ComputePathwayCompleteness (operator)
  - Defensively unwraps common envelopes (e.g., {present:{...}, present_summary:[...]}) for present/totals inputs. Restores non‑empty completeness when plans bind envelopes.

- Logs & Observability
  - Planner: latency logs (model id + reasoning_effort) for initial and retry calls.
  - PFAM counts: logs include token_count, candidate_count (post‑cap), cap value, elapsed_ms.

- Indexes + Cypher hygiene
  - Added Domain.pfamAccession index; PFAM filters use accession STARTS WITH (version‑tolerant) and id STARTS WITH; keyword discovery remains in catalog stage.
  - pfam_search/kofam_search moved to full‑text indexes (discovery step only). Count queries remain deterministic and index‑friendly.

Reporter Guidance (no code path changes)
- Consume facets as {id, name, count}. Render key callouts as "Kxxxxx (name)" / "PFxxxxx (name)"; avoid inferring names from prompts or theme text.
- When PF00016/PF00101 appear but K01601/K01602 are absent, note RuBisCO‑like (Form IV) ambiguity explicitly — no pinning or whitelists.

No Hidden Caps / No Env Flags / No Theme Lists
- Hidden caps removed (best‑per‑token LIMIT 1, pre‑clamp). The only internal cap is the per‑token candidate cap (default 200) which is agent‑controllable.
- Do not use env flags to steer behavior; all behavior is via static operators and explicit planner parameters.
- No theme‑specific whitelists or hard‑coded IDs in prompts or code; all specificity comes from catalog‑first discovery plus accession‑priority and planner‑set caps.

Files of Interest
- src/llm/mfp/operators/builtin.py — AnnotationDiscovery (facet controls, field standardization, PFAM provenance), ComputePathwayCompleteness (unwrap).
- resources/cypher/count_proteins_by_pfam_tokens.cypher — Parameterized candidate cap; no per‑token LIMIT 1.
- resources/cypher/count_proteins_by_ko_ids.cypher — KO counts with label returned.
- src/llm/kg/cypher_templates/*.cypher — Index‑friendly PFAM/KO templates; pfam_search/kofam_search use full‑text discovery.
- src/llm/rag_system/dspy_signatures.py — Planner rubric updated (keyword discipline, explicit caps, field schema).
- src/llm/rag_system/core.py — Planner latency logs; IRB bypass logic unchanged.
- scripts/neo4j/indices.cypher — Domain.pfamAccession index added (and created at bulk load Step 6).

Practical Defaults to Recommend (agent‑controlled)
- pfam_tokens_top_n: 20–30 per theme probe.
- ko_tokens_top_n: 20–30 per theme probe.
- pfam_candidate_cap: 200 (raise only if tokens routinely hit the cap and hallmarks are still missed).
- pfam_top_k / ko_top_k: 30–50 depending on desired summary breadth.

Known Semantics & Caveats
- Completeness: KO‑based union for single‑genome runs; not organism‑complete by itself. Multi‑genome semantics (per‑genome vs union vs mean) to be documented with multi‑genome test set.
- PFAM generic families can still surface for broad tokens; planner should avoid generic tokens and keep per‑theme probes tight.

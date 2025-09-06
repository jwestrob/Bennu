Supersedes: all prior compaction notes (2025‑08‑26, 2025‑09‑04). This is the current single source of truth.

Compaction Note — 2025‑09‑05

Context (what’s now solid)
- Neighborhoods: deterministic and auditable
  - NeighborhoodContext is strict about seeds; accepts `inputs.discovered_proteins` or explicit IDs; no implicit fallbacks.
  - Degree‑aware seed filter: excludes contig‑isolated seeds (`Gene.nextDegree = 0`) by default; override with `include_degree_zero_seeds=true`.
  - Enriched neighbors (PFAM/KO) without APOC; flanking query fixed.
  - Tool calls (op, params, inputs, outputs preview) are persisted to `data/session_notes/<sid>/synthesis_notes/tool_calls.json`.
- Planner / plan validation
  - AnnotationDiscovery MUST set `params.keyword` (or `q`). Plans omitting this are rejected at validation time.
  - Wiring rule: chain IDs via `inputs` (e.g., `inputs:{"pfam_ids":"pfam_ids"}`) rather than hard‑coding.
  - Keyword hygiene: for gene/subunit context, prefer direct subunit terms; avoid broad class or “‑like” analog tokens unless explicitly exploring; keep synonyms ≤ 2.
  - Rubric example uses placeholders (`<KEYWORD>`) only; numeric defaults documented separately to avoid biasing examples.
- Stage 07 (default, no creds needed)
  - RDF→CSV conversion precomputes `[:NEXT]` edges (`next_relationships.csv`) and writes `Gene.nextDegree` and `Gene.genesOnContig` directly into Gene CSVs.
  - Bulk import with `neo4j-admin` loads everything in one shot; no Neo4j auth required; no post‑load fixes needed.
  - Optional constraints/indexes can be applied post import if env creds exist, but are not required for neighborhoods.
- Diagnostics
  - `scripts/diagnostics/neo4j_check_next.py` prints a degree histogram and per‑seed computed vs stored degree, plus adjacency/flanking neighbors and PFAM/KO annotations.

How to run (Stage 07)
- Build just stage 7 (creates TTL/NT, CSVs, NEXT, nextDegree, then bulk import):
  - `python -m src.cli build -f 7 -t 7 --force`
  - Outputs: `data/stage07_kg/knowledge_graph.ttl` and `data/stage07_kg/csv/*`, then imports via `neo4j-admin`.

What to expect in agent logs
- `NeighborhoodContext: filtered X degree-zero seeds; using Y seeds` (degree filter summary).
- No Neo4j UnknownPropertyKey warnings for `nextDegree` (property exists on Gene from import).
- Tool‑call capture file at `synthesis_notes/tool_calls.json` for full parameter audit.

Planner guidance (summary)
- Always set AnnotationDiscovery.keyword and wire IDs via inputs. Do not call AnnotationDiscovery with only formatting params (output_profile / group_by / fields) — this yields empty results.
- For gene/subunit context, keep keywords tight (direct subunit terms; ≤ 2 concise synonyms). Avoid “‑like” analogs unless exploring broadly.
- Numeric defaults (documented, not enforced in examples): `pfam_tokens_top_n=30`, `ko_tokens_top_n=30`, `pfam_candidate_cap=200`, `pfam_top_k=20`, `ko_top_k=20`. For targeted rowsets, use a small budget (≈ 50–200) to control latency.

Operator details (NeighborhoodContext)
- Params of note: `seeds_limit` (default 10), `k` (omit for flanking ±5), `include_degree_zero_seeds` (default false), `output_profile` (`summary` or `rowset`).
- Degree filter batching Cypher:
  - `UNWIND $pids AS pid MATCH (p:Protein {id: pid})-[:ENCODEDBY]->(g:Gene)`
  - `OPTIONAL MATCH (g)-[:NEXT]-(:Gene) WITH pid, g, count(*) AS c` → `WITH pid, coalesce(g.nextDegree, c) AS deg` → `RETURN pid, toInteger(deg)`
  - This uses stored nextDegree if present, else falls back to live count.

Diagnostics (quick)
- `python scripts/diagnostics/neo4j_check_next.py --k 5 --flank_n 5 --limit 6`
  - Prints global `[:NEXT]` count; degree histogram; and per‑seed `NEXT degree=K (prop=D) | genes_on_contig=N`.
  - Shows adjacency (k) and flanking (±N) neighbors with PFAM/KO summaries.

Open items / test & complete
- Planner tightening (non‑breaking): when intent is “context around specific genes/subunits”, recommend smaller `pfam_tokens_top_n` (≈ 8–12) to reduce noise; keep wording general (no hard‑coding biology).
- Reporter visibility: ensure neighborhoods_json is leveraged to summarize seed‑level adjacency and loci examples succinctly; prefer `output_profile='rowset'` when seed set is small (≤ 12).
- Indexes (default): post‑import constraints/indexes are created by default (no‑auth supported; docker engine runs with `NEO4J_AUTH=none`). Includes unique IDs, composite `:Gene(contig,startCoordinate)` (and contig,start,end), and helpful full‑text indexes.
- E2E checks:
  - Stage 07 default path produces Gene.nextDegree on import; agent runs with no nextDegree warnings.
  - AnnotationDiscovery validation rejects missing keyword; planner respects wiring rules.
  - Degree filter behavior: defaults to excluding degree‑0; include when explicitly requested.

Known pitfalls
- Extremely fragmented assemblies produce many degree‑0 seeds; degree filter helps, but expect fewer neighborhoods.
- If someone bypasses Stage 07’s CSV import and attaches to an older DB, nextDegree may be missing — the operator will still work (falls back to live count) but Neo4j will warn about the unfamiliar property key until nextDegree is set.

Key files
- Planner constraints/signatures: `src/llm/rag_system/core.py`, `src/llm/rag_system/dspy_signatures.py`
- NeighborhoodContext: `src/llm/mfp/operators/builtin.py`
- Tool call capture: `src/llm/mfp/executor.py`
- Stage 07 CSV import path: `src/build_kg/rdf_to_csv_converter.py`, `src/build_kg/neo4j_bulk_loader.py`, `src/cli.py`

Cleanup TODO
- Remove functional enrichment from Stage 07 (PFAM/KO label additions from reference files). It is not required for neighborhoods and adds noisy logs; the pipeline should remain focused on annotations produced in Stage 04 and core graph structure.

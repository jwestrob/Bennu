Supersedes: all prior compaction notes (2025‑08‑26, 2025‑09‑04, 2025‑09‑05). This is the current single source of truth.

Compaction Note — 2025‑09‑06

Progress checkpoint (post‑cleanups)
- Functional enrichment removed from Stage 07
  - Dropped PFAM/KO/CAZy description adders and tests; Stage 07 focuses on Stage 04 outputs (PFAM, KO) + core graph (Genome→Gene→Protein), BGC/CAZy when available, NEXT edges, and KEGG pathways.
  - Result: cleaner logs, no empty enrichment fields; deterministic import.
- Stage 07 build path validated
  - TTL created with ≈10.48M triples, CSVs emitted including `next_relationships.csv`, and bulk import completes via `neo4j-admin` (docker engine, auth=none) with no enrichment warnings.
  - Diagnostics confirm `[:NEXT]` present, `Gene.nextDegree`/`Gene.genesOnContig` populated from CSV load; neighborhood operators see consistent degrees.
- Post‑import indexes
  - Constraints and indexes are created by default after import. Docker path uses unauth bolt by default; when creds are provided, they are used.
  - Composite indexes on `:Gene(contig,startCoordinate,endCoordinate)` and `:Gene(contig,startCoordinate)` accelerate locus/flanking scans.

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
  - Post‑import constraints/indexes are applied by default (no‑auth supported). Optional additional indexes can still be added manually.
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

Accessing the Augmented Neo4j Schema
- Connection (docker default)
  - URI: `bolt://localhost:7687`; Auth: none (container runs with `NEO4J_AUTH=none`).
  - With creds: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars.
- Core labels
  - `Genome`, `Gene`, `Protein`, `Domain`, `DomainAnnotation`, `FunctionalAnnotation`, `KEGGOrtholog`, `Pathway`, `Bgc`, `QualityMetrics`, `Dataset`.
- Key properties
  - `Gene`: `id`, `contig`, `startCoordinate`, `endCoordinate`, `strand`, `nextDegree`, `genesOnContig`.
  - `Protein`: `id` (optional: `name`, `description` when present).
  - `Domain`: `id`, `pfamAccession`, `name` (description may be empty by design post‑cleanup).
  - `KEGGOrtholog`: `id`, `description`.
- Relationships (subset)
  - `(:Protein)-[:ENCODEDBY]->(:Gene)`
  - `(:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(:Domain)`
  - `(:Protein)-[:HASFUNCTION]->(:KEGGOrtholog)`
  - `(:KEGGOrtholog)-[:PARTICIPATESIN]->(:Pathway)`
  - `(:Gene)-[:NEXT]->(:Gene)` (directed; treat as undirected for degree)
  - `(:Gene)-[:BELONGSTOGENOME]->(:Genome)` and provenance edges (e.g., quality metrics)
- Helpful queries
  - Global NEXT count: `MATCH ()-[:NEXT]->() RETURN count(*) AS c`.
  - Stored vs live degree for a seed: `MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene) OPTIONAL MATCH (g)-[:NEXT]-() WITH g, count(*) AS c RETURN toInteger(coalesce(g.nextDegree,c)) AS degree, toInteger(g.genesOnContig) AS onContig`.
  - Flanking neighbors (±N by contig order): see `scripts/diagnostics/neo4j_check_next.py` for a compact, index‑aware pattern.
  - PFAM to proteins: `MATCH (d:Domain {pfamAccession:$pf})<-[:DOMAINFAMILY]-(:DomainAnnotation)<-[:HASDOMAIN]-(p:Protein) RETURN p.id LIMIT 25`.
  - KO to pathways: `MATCH (ko:KEGGOrtholog {id:$ko})-[:PARTICIPATESIN]->(pw:Pathway) RETURN pw.id, pw.name LIMIT 25`.
  - Index/constraint visibility (Neo4j 5): `SHOW INDEXES`, `SHOW CONSTRAINTS`.

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

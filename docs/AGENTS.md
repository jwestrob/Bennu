### Future: Final Single-Pass “Publish” Call

After incremental synthesis converges, we plan a final API call where gpt-5-high (or a cost-efficient alternative, e.g., gpt-5-medium or 4.1-mini for templated prose) renders the polished final report from the IRB AST. This step is not implemented now; it should be configurable and budget-aware. Keep the IRB validators and claims ledger as the source of truth; the publish step must not introduce new claims—formatting and phrasing only.

## Project Policy: No Environment Flags

- Do not introduce, rely on, or require environment flags to alter runtime behavior. This project does not use env flags for feature gating or configuration.
- All behavior toggles must be explicit in code or configuration files that are versioned and reviewed.
- Rationale: env flags cause hidden state between runs and make experiments hard to reproduce. Keep behavior deterministic and visible in the repo.

## Operator Upgrades — Facet-First, Static Controls (2025‑09‑02)

Summary
- We added facet-first capabilities to AnnotationDiscovery and a per-KO present summary to FetchPresentKOs to reduce post-hoc filtering and pass only relevant, compact evidence to IRB.
- No paging; the model controls quantity and fields statically via operator params. Tools always echo totals and clamping — no invisible thresholds.

AnnotationDiscovery
- Outputs: `facet_summary`, `selection_metadata`, `discovered_proteins` (rowset optional)
- Params (all optional; defaults shown):
  - `output_profile`: `facet_summary` | `rowset` (default `facet_summary`)
  - `return_mode`: `top_k` | `all` (default `top_k`)
  - `ko_top_k`: int (default 30)
  - `pfam_top_k`: int (default 20)
  - `fields`: List[str] (requested fields for summaries)
  - `group_by`: `ko` | `pfam` | `both` (default `both`)
  - `include_examples`: `none` | `counts` | `ids` (ids only valid for contig/locus anchors)
  - `limit`: int (row budget for legacy rowset mode)
  - `genome_ids`: List[str] | null
- Behavior:
  - Performs keyword→(PFAM/KO) mapping → exact retrieval; aggregates KO/PFAM counts into `facet_summary` with score order.
  - Applies `return_mode` with explicit `top_k` or returns `all` up to a documented `max_server_cap` (echoed).
  - `selection_metadata` reports requested vs applied, totals, cap, clamped, estimated_tokens.
  - `discovered_proteins` rowset is omitted unless `output_profile='rowset'` or `return_full_rows=True`.

FetchPresentKOs
- Outputs: `present` (genome→KO ids), `present_summary` (List[{ko_id, present_genome_count}])
- Use `present_summary` for compact KO evidence across genomes; avoid per-genome expansion.

Design Notes
- IRB consumes compact facets and counts; row-level details are a separate, explicit step if needed.
- `top_k` is global over the chosen `group_by` (e.g., total KOs), not per-entity; for “top N per KO” use a dedicated rowset operator with `per_entity_top_k` (future work).
- Contig/locus IDs are allowed only for contig/locus anchors; elsewhere we use counts/examples_count (no raw IDs).

## Planner Rubric — Facet-First Plans (2025‑09‑02)

- Separate KO and PFAM summary steps (do not mix via `group_by='both'`).
- Use AnnotationDiscovery with `output_profile='facet_summary'`, explicit `return_mode` ('top_k' or 'all') and per-group caps (`ko_top_k`, `pfam_top_k`), plus `fields=['id','count','score']`.
- For breadth, break by theme: CBB, WL, rTCA, 3‑HP family, methanogenesis, methanotrophy, nitrogenase (and others as needed). Plan KO and PFAM summaries per theme.
- Run FetchPresentKOs early and leverage `present_summary` to bias KO facet steps or define filtered KO target sets. Avoid per-genome expansions.
- Only request rowsets when needed (per-entity detail). Do not inflate summary top_k for per-entity lists — add a dedicated detail step instead.
- Prefer accession tokens (Kxxxxx, PFxxxxx) where available; use flexible matching for PFAM (accession STARTS WITH + name/desc CONTAINS) per policy.

## Neo4j Index & Cypher Plan (2025‑09‑02)

Goal
- Make summary queries fast and index‑friendly; avoid label scans and post‑hoc filtering.

Indexes to create (server‑side)
- CREATE INDEX IF NOT EXISTS FOR (k:KEGGOrtholog) ON (k.id)
- CREATE INDEX IF NOT EXISTS FOR (d:Domain) ON (d.pfamAccession)
- CREATE INDEX IF NOT EXISTS FOR (d:Domain) ON (d.id)
- CREATE INDEX IF NOT EXISTS FOR (g:Genome) ON (g.id)

Cypher changes (compatible; no behavior change in outputs)
- KO retrieval: use exact equality on k.id (index seek) and anchor on Genome id before relationship expansion.
  - UNWIND $ko_ids AS kid
    MATCH (k:KEGGOrtholog {id:kid})
    MATCH (p:Protein)-[:HASFUNCTION]->(k)
    MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
    WHERE $genome_ids IS NULL OR size($genome_ids)=0 OR g.id IN $genome_ids
    RETURN …
- PFAM retrieval: split accession vs name/desc into two branches and UNION ALL; never call functions on properties in WHERE.
  - Accession branch (index seek): d.pfamAccession STARTS WITH $acc OR d.id STARTS WITH $acc
  - Name/desc branch (limited terms): coalesce(d.name,'') CONTAINS $term OR coalesce(d.description,'') CONTAINS $term OR d.id CONTAINS $term

Facet summary fast path
- Add count_proteins_by_ko_ids.cypher and count_proteins_by_pfam_ids.cypher.
- AnnotationDiscovery with output_profile='facet_summary' uses count_* templates only (no row materialization), returning {id,count} (+ optional score) shaped facets.
- Rowset (detail) still uses existing proteins_by_* templates; unchanged.

Agent impact
- No new flags or APIs needed. The planner already sets output_profile/return_mode/top_k/fields; tools will hit the indexes automatically with the new Cypher. Behavior remains consistent; only faster.

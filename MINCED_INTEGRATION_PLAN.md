Title: CRISPR (MinCED) Integration Plan — progress tracker

Updated: 2025-09-09

Scope
- Add MinCED-based CRISPR array detection as Stage 05 and integrate arrays into the Stage 07 KG build and Neo4j.
- Keep [:NEXT] edges intact; annotate with crisprBetween/crisprCountBetween and add flank edges Gene–[:FLANKS_CRISPR]→CrisprArray.

Status (this session)
- Environment: added `minced` to `env/environment.yml` (bioconda) and docs.
- Quick run: executed MinCED on the SRR6231169 metagenome (data/stage00_prepared/SRR6231169.fasta).
  - Outputs in data/stage05_crispr/: `SRR6231169.gff` (~3.1 MB), `SRR6231169_spacers.fa` (~138 KB), `SRR6231169.crisprs` (~184 KB).
  - Timing (usr/bin/time -p): real ≈ 9.0 s, user ≈ 7.0 s, sys ≈ 2.6 s.
  - Parsed JSON: 212 arrays, 1,566 spacers → data/stage05_crispr/SRR6231169_crispr_arrays.json
- Stage 07 integration implemented:
  - rdf_to_csv_converter now reads `data/stage05_crispr/*_crispr_arrays.json`, emits:
    - `crispr_arrays.csv` (nodes), `belongstogenome_relationships.csv` (+CRISPR rows), `flanks_crispr_relationships.csv`.
    - Rewrites `next_relationships.csv` with properties: `contig, delta:int, same_strand:boolean, crisprBetween:boolean, crisprCountBetween:int`.
    - Updates Gene CSV rows with `genesOnContig`, `nextDegree`, `isCrisprFlankLeft`, `isCrisprFlankRight`, `nearestCrisprDistanceGenes`.
  - neo4j_bulk_loader label override for `crispr_arrays.csv` → `:CrisprArray`.
  - Indexes added: `:CrisprArray(contig,startCoordinate,endCoordinate)` and `(contig,startCoordinate)`.
  - Diagnostics script added: `scripts/diagnostics/neo4j_check_crispr.py`.
  - CLI integration: Stage 05 now also runs MinCED over `data/stage00_prepared` and writes `data/stage05_crispr/*`. Running build from `-f 5 -t 7` will include CRISPR arrays.
- Parser: implementing `src/ingest/minced_crispr.py` to convert GFF + spacers into JSON artifacts:
  - `<genome>_crispr_arrays.json` with arrays and attributes.
  - `crispr_summary.json` and `processing_manifest.json` for Stage 07 consumption.

Data model (initial)
- Node: `CrisprArray` with `id`, `genomeId`, `contig`, `startCoordinate`, `endCoordinate`, `repeatConsensus`, `repeatLength`, `repeatsCount`, `spacerCount`, `evidence='minced'`, `toolVersion`.
- Relationships:
  - `(:CrisprArray)-[:BELONGSTOGENOME]->(:Genome)`
  - `(:Gene)-[:FLANKS_CRISPR {side:'left'|'right', distanceBp}]->(:CrisprArray)` (computed Stage 07)
  - Decorate `(:Gene)-[:NEXT]->(:Gene)` with `{crisprBetween: bool, crisprCountBetween: int}` (computed Stage 07)

Planned changes
1) Stage 05 runner (optional): thin wrapper to run MinCED and place outputs under `data/stage05_crispr/`. (optional)
2) Parser: `src/ingest/minced_crispr.py` to emit JSON artifacts listed above. (done)
3) Stage 07 builder updates: CSV emission + NEXT decoration + gene props. (done)
4) Diagnostics: `scripts/diagnostics/neo4j_check_crispr.py`. (done)
5) Operator wiring: expose `include_crispr` in `NeighborhoodContext`; optional `require_crispr_in_flank` in `AnnotationDiscovery`. (pending)

Verification checklist
- Stage 05: JSON present (arrays, summary, manifest) for SRR6231169.
- Stage 07: import completes; `MATCH (ca:CrisprArray) RETURN count(ca)` > 0; decorated NEXT edges present.
- Neighborhood summaries show arrays in loci.

Open items
- Decide whether to persist individual spacers/repeats as nodes (default: no; JSON keeps sequences for audit only).
- Finalize array `id` format (deterministic: `${genomeId}|${contig}|CRISPR${n}` with URL-safe hashing fallback for very long contig IDs).

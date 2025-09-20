Title: Portable Neo4j Knowledge Graph Bundles — Export, Share, Serve

Status: v1 (initial), scope-approved for implementation

Overview
- Goal: Package a fully-built Neo4j database from data/neo4j into a portable bundle you can copy, share, archive, and restore elsewhere (Docker or system install) without rebuilding from RDF/CSV.
- Primary artifact: Neo4j .dump (fast, faithful restore). Optional reproducible path: CSV + post_import.cypher + TTL for provenance.

Bundle Layout (proposed)
- bundle_root/
  - dumps/neo4j-5.x/neo4j.dump            # primary portable artifact
  - csv/…                                  # optional: data/stage07_kg/csv copy
  - ttl/knowledge_graph.ttl                # optional: provenance
  - scripts/
    - restore_docker.sh                    # load + serve (docker)
    - restore_system.sh                    # load + start (system install)
    - post_import.cypher                   # used only for CSV restore
  - manifest.json                          # metadata, checksums, counts
  - README.md                              # quickstart

Why two paths?
- .dump: best for fidelity, performance, and simple restore; tied to Neo4j major version (5.x) which we pin in the manifest.
- CSV: best for transparency, long-term archiving, and rebuilds across versions; slower to import and re-create indexes.

Manifest (manifest.json)
- Required
  - spec_version: "kg-bundle/1"
  - dataset_id: string (e.g., SRR6231169)
  - created_at: ISO 8601
  - neo4j: { major: 5, image: "neo4j:5" }
  - database: { name: "neo4j" }
  - artifacts:
    - dump: { path: dumps/neo4j-5.x/neo4j.dump, sha256, size_bytes }
    - csv_dir?: { path: csv/, node_csv_count, rel_csv_count, sha256_manifest }
    - ttl?: { path: ttl/knowledge_graph.ttl, sha256, size_bytes }
  - counts: { nodes, relationships }      # from CSV or a quick query if DB available
  - git: { commit?: short_sha, dirty?: bool }
  - notes?: string

CLI Additions
- genome-kg export db
  - Args: --format dump|csv|both (default dump), --out PATH, --engine docker|system (default docker)
  - dump (docker):
    - docker run --rm -v $PWD/data/neo4j:/data -v $OUT/dumps:/out neo4j:5 \
      neo4j-admin database dump neo4j --to-path=/out
  - dump (system):
    - neo4j-admin database dump neo4j --to-path $OUT/dumps
  - csv: copy data/stage07_kg/csv → $OUT/csv, write scripts/post_import.cypher (from our index/constraint set) and optionally $OUT/ttl/knowledge_graph.ttl
  - Write $OUT/manifest.json with hashes/sizes and basic counts.

- genome-kg serve
  - Args: --bundle PATH, --engine docker|system (default docker), --auth none|user:pass (default none), --db-name neo4j
  - docker:
    - If dump present: load via neo4j-admin database load neo4j --from-path=/import (one-shot), then start neo4j:5 with data volume; expose 7474/7687; set NEO4J_AUTH accordingly.
    - If only CSV present: import via our bulk loader + run post_import.cypher, then start server.
  - system: neo4j-admin database load + neo4j start.

- genome-kg validate-bundle
  - Verify manifest keys and file existence; recompute SHA256 for listed artifacts; optional: quick docker verify of the dump header (no full load).

Constraints/Indexes (post_import.cypher)
- Reuse the statements we already apply post-import (in src/build_kg/neo4j_bulk_loader.py). We expose a helper to emit the exact statements for reproducibility.

Counts & Checksums
- Counts (when DB not running): sum CSV line counts minus headers.
- SHA256 for .dump and a compact CSV manifest (filename → hash) to avoid hashing huge directories repeatedly.

Serving Recipes
- Docker one-liner (load and run):
  - docker run --rm -v $(pwd)/dumps:/import neo4j:5 neo4j-admin database load neo4j --from-path=/import
  - docker run -d --name kg-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=none -v $(pwd)/data:/data neo4j:5

Minimum Viable Implementation (this iteration)
1) Expose post-import statements in the loader (pure-Python function returning the list).
2) Add export command supporting dump|csv|both (docker + system backends).
3) Write manifest.json with hashes and counts; emit restore scripts.
4) Basic validate-bundle: file presence + checksum.
5) Docs (this file + README blurb).

Future Enhancements
- Compose file that auto-loads dump on first start.
- Deep validate: spin a short-lived docker neo4j:5, load dump into /data/tmp, run 2–3 template queries.
- Bundle naming: kg_<dataset>_<yyyymmdd>.<ext> with symlink latest.
- Multi-database support (if we split datasets later).

Risks & Mitigations
- Version skew: record Neo4j major.minor in manifest and warn on mismatch.
- Large bundles: prefer dump (compact); CSV bundle optional.
- Permissions on restore scripts: mark +x during export, document if a git checkout strips executable.

Quick Usage (anticipated)
- Export dump + csv bundle:
  - python -m src.cli export --format both --out bundle/SRR6231169
- Serve via docker:
  - python -m src.cli serve --bundle bundle/SRR6231169 --engine docker
- Validate a bundle:
  - python -m src.cli validate-bundle --bundle bundle/SRR6231169


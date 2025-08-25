# Macro Fast Path (MFP)

- Deterministic macro-options for common queries (LocusDiscovery).
- Batched Cypher templates, deterministic EVI gate, early exit.
- Optional Skeptic only after costly batches.
- Escalation back to Reactive Plan Path when guards fail.
API budget: heavy=1, mini≈0; DB=2; LanceDB=0/1.

## Locus entities
- Persist (:Locus {seed_pid, contig_id, verdict, created_at}) with (:INDEXES)->(:Protein), (:ON_CONTIG)->(:Contig).
- Enables reuse across runs, templated synthesis, and audit.


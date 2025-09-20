# Diagnostics

Use the provided script to validate that `[:NEXT]` edges and degrees are consistent and that PFAM/KO annotations are accessible.

## Quick Check

- `python scripts/diagnostics/neo4j_check_next.py --k 5 --flank_n 5 --limit 6`

Prints:
- Global `[:NEXT]` count
- Degree histogram
- Per‑seed: `NEXT degree=K (prop=D) | genes_on_contig=N`
- Adjacency (k) and flanking (±N) neighbors with PFAM/KO summaries


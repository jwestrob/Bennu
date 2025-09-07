# Performance Notes

- Bulk import with `neo4j-admin` is ~100× faster than MERGE‑based loaders.
- Precomputing `[:NEXT]` and embedding `nextDegree`/`genesOnContig` into CSVs eliminates post‑load compute.
- Composite indexes on `:Gene(contig,startCoordinate,endCoordinate)` and `:Gene(contig,startCoordinate)` accelerate flanking scans.
- Plan small, tight seeds (≤ 12) for neighborhoods to minimize I/O and synthesis time.


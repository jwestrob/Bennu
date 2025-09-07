# Troubleshooting

## UnknownPropertyKey: `nextDegree`
- Cause: Connecting to an older DB without `Gene.nextDegree` property schema.
- Fix: Rebuild Stage 07 (preferred) or run Post‑Load Tuning to set `nextDegree` and `genesOnContig`.

## Many degree‑0 seeds in neighborhoods
- Cause: Highly fragmented assemblies (short contigs).
- Fix: Default filter excludes them; to include, set `include_degree_zero_seeds=true` in `NeighborhoodContext`.

## No CAZy nodes/edges present
- Cause: Missing Stage 06 JSON artifacts.
- Fix: Re‑run Stage 06 via CLI (which now saves JSON), or run synthesis fallback (dbCAN ran externally) then Stage 07 again.

## Bulk import succeeded, but queries feel slow
- Check that composite indexes exist:
  - `SHOW INDEXES` should list `:Gene(contig,startCoordinate)` and `(contig,startCoordinate,endCoordinate)`.
- If absent, re‑apply via Post‑Load Tuning.


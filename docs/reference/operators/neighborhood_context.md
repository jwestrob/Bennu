# Operator: NeighborhoodContext

Computes neighborhoods around seed proteins via adjacency (k-step on `[:NEXT]`) or flanking windows (±N by contig order), without APOC.

## Parameters

- `seeds_limit` (default 10): Maximum seeds to evaluate.
- `k`: Steps for adjacency mode. Omit to use flanking mode (±5 by default).
- `include_degree_zero_seeds` (default false): Include seeds whose genes have `nextDegree=0`.
- `output_profile`: `summary` or `rowset`.

## Degree-Aware Seed Filter

Cypher batch used to filter seeds (uses stored degree when available, falls back to live count):

```
UNWIND $pids AS pid
MATCH (p:Protein {id: pid})-[:ENCODEDBY]->(g:Gene)
OPTIONAL MATCH (g)-[:NEXT]-(:Gene)
WITH pid, g, count(*) AS c
WITH pid, coalesce(g.nextDegree, c) AS deg
RETURN pid, toInteger(deg)
```

## Notes

- Treat `[:NEXT]` as undirected for degree computation.
- Flanking queries are index-aware; composite `:Gene(contig,startCoordinate,endCoordinate)` accelerates scans.

## Outputs

- `neighborhoods`: when `output_profile='rowset'`, list of per‑seed neighbor rows with seed/neighbor gene/protein IDs and summaries.
- `neighborhood_summary`: summary table with per‑seed counts and example neighbors.
- `neighborhood_macro_result`: compact macro result envelope for synthesis.
- `seeds_used`: list of seed protein IDs that passed the degree filter.

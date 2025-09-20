// CRISPR context around a seed protein: arrays flanking genes within ±flank_n positions
// Required slots: protein_id
// Optional slots: flank_n (default 5), limit

WITH toInteger(coalesce($flank_n, 5)) AS N
MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(seed:Gene)
MATCH (g:Gene {contig: seed.contig})
WITH seed, g ORDER BY toInteger(g.startCoordinate)
WITH seed, collect(g) AS gs,
     [i IN range(0, size(gs)-1) WHERE gs[i].id = seed.id][0] AS idx, N
WITH seed, gs, idx, N, range(-N, N) AS off
UNWIND off AS d
WITH seed, gs, idx, d WHERE d <> 0 AND (idx + d) >= 0 AND (idx + d) < size(gs)
WITH seed, gs[(idx + d)] AS ng
OPTIONAL MATCH (ng)-[f:FLANKS_CRISPR]->(ca:CrisprArray)
WITH ca WHERE ca IS NOT NULL
RETURN DISTINCT ca.id AS crispr_id,
       ca.contig      AS contig,
       toInteger(ca.startCoordinate) AS start,
       toInteger(ca.endCoordinate)   AS end,
       toInteger(ca.spacerCount)     AS spacers
ORDER BY start
LIMIT coalesce($limit, 100)


// Return CRISPR arrays on a contig, optionally constrained to a coordinate window
// Required slots: contig
// Optional slots: start, end, limit

MATCH (ca:CrisprArray {contig: $contig})
WHERE ($start IS NULL OR toInteger(ca.startCoordinate) >= toInteger($start))
  AND ($end   IS NULL OR toInteger(ca.endCoordinate) <= toInteger($end))
RETURN ca.id          AS crispr_id,
       ca.contig      AS contig,
       toInteger(ca.startCoordinate) AS start,
       toInteger(ca.endCoordinate)   AS end,
       toInteger(ca.repeatsCount)    AS repeats,
       toInteger(ca.spacerCount)     AS spacers
ORDER BY start
LIMIT coalesce($limit, 100)


// List CRISPR arrays for a specific genome
// Required: genome_id
// Optional: limit

MATCH (g:Genome {id: $genome_id})
MATCH (ca:CrisprArray)-[:BELONGSTOGENOME]->(g)
RETURN ca.id AS crispr_id,
       ca.contig AS contig,
       toInteger(ca.startCoordinate) AS start,
       toInteger(ca.endCoordinate)   AS end,
       toInteger(ca.repeatsCount)    AS repeats,
       toInteger(ca.spacerCount)     AS spacers
ORDER BY contig, start
LIMIT coalesce($limit, 100)


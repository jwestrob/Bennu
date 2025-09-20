// List CRISPR arrays across all genomes (lightweight, capped)
// Optional: limit

MATCH (ca:CrisprArray)
OPTIONAL MATCH (ca)-[:BELONGSTOGENOME]->(g:Genome)
RETURN ca.id AS crispr_id,
       ca.contig AS contig,
       toInteger(ca.startCoordinate) AS start,
       toInteger(ca.endCoordinate)   AS end,
       toInteger(ca.repeatsCount)    AS repeats,
       toInteger(ca.spacerCount)     AS spacers,
       g.id AS genome_id
ORDER BY genome_id, contig, start
LIMIT coalesce($limit, 100)


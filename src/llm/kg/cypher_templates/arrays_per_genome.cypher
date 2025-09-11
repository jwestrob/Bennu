// Count CRISPR arrays per genome
MATCH (g:Genome)<-[:BELONGSTOGENOME]-(ca:CrisprArray)
RETURN g.id AS genome_id, count(ca) AS arrays
ORDER BY arrays DESC, genome_id


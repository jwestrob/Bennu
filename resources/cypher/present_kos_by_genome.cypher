// Parameters:
//   $genome_ids : [string] (optional) — if empty or not provided, compute for all genomes

// Present KO IDs per genome (optionally filtered by genome_ids)
MATCH (p:Protein)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
RETURN g.id AS genome_id, collect(DISTINCT ko.id) AS present_ko_ids
ORDER BY genome_id

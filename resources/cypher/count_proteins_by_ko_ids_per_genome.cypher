// Count proteins per genome for each exact KO id
// Params:
//   $ko_ids    : [string]
//   $genome_ids: [string] optional; filter by genome id when provided

UNWIND $ko_ids AS kid
WITH kid, $genome_ids AS gids
MATCH (ko:KEGGOrtholog {id: kid})
OPTIONAL MATCH (p:Protein)-[:HASFUNCTION]->(ko)
OPTIONAL MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE gids IS NULL OR size(gids)=0 OR g.id IN gids
RETURN g.id AS genome_id, kid AS ko_id, count(DISTINCT p) AS count
ORDER BY genome_id, ko_id;


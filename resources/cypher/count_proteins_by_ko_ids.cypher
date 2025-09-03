// Count proteins per exact KO id and return optional description
// Params:
//   $ko_ids     : [string]
//   $genome_ids : [string] optional; filter by genome id when provided

UNWIND $ko_ids AS kid
WITH kid, $genome_ids AS gids
MATCH (ko:KEGGOrtholog {id: kid})
OPTIONAL MATCH (p:Protein)-[:HASFUNCTION]->(ko)
OPTIONAL MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WITH kid, ko, (CASE WHEN gids IS NULL OR size(gids)=0 THEN true ELSE g.id IN gids END) AS pass, p
WHERE pass
RETURN kid AS ko_id, coalesce(ko.description, ko.id) AS label, count(DISTINCT p) AS count;


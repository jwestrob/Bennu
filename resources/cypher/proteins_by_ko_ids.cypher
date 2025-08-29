// Proteins annotated with specific KO IDs (exact match on KEGGOrtholog.id)
// Params:
//   $ko_ids    : [string] (Kxxxxx; case-insensitive)
//   $genome_ids: [string] (optional; empty/null means global)
//   $limit     : integer (max rows)

WITH [x IN $ko_ids WHERE x IS NOT NULL] AS kos
WITH [x IN kos | toLower(x)] AS lowers
UNWIND lowers AS kid
MATCH (ko:KEGGOrtholog)
WHERE toLower(ko.id) = kid
WITH DISTINCT ko
MATCH (p:Protein)-[:HASFUNCTION]->(ko)
MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
RETURN
  g.id AS genome_id,
  p.id AS protein_id,
  ko.id AS ko_id,
  ko.description AS ko_desc
ORDER BY ko_id, protein_id
LIMIT toInteger($limit)

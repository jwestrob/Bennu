// Proteins annotated with KOs matching a keyword (global or filtered by genomes)
// Params:
//   $q          : string (keyword; matches KEGGOrtholog.id or description)
//   $limit      : integer (max rows)
//   $genome_ids : [string] (optional; empty/null means global)

MATCH (ko:KEGGOrtholog)
WHERE toLower(ko.id) CONTAINS toLower($q)
   OR toLower(coalesce(ko.description, '')) CONTAINS toLower($q)
WITH collect(DISTINCT ko) AS kos
UNWIND kos AS ko
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


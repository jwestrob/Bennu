// All CAZyme-annotated proteins (global or filtered by genome) via CAZy nodes
// Schema used in Stage 07:
//   (Protein)-[:HASCAZYME]->(CAZymeAnnotation)-[:CAZYMEFAMILY]->(CAZymeFamily)
//   Optional genome filter via (Protein)-[:ENCODEDBY]->(Gene)-[:BELONGSTOGENOME]->(Genome)
// Parameters:
//   $genome_id  : string (optional) — single genome filter
//   $genome_ids : [string] (optional) — multiple genome filter; empty or null means all
MATCH (gen:Genome)
WHERE ($genome_id IS NULL OR gen.id = $genome_id)
  AND ($genome_ids IS NULL OR size($genome_ids) = 0 OR gen.id IN $genome_ids)
MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen)
MATCH (p)-[:HASCAZYME]->(a:Cazymeannotation)-[:CAZYMEFAMILY]->(f:Cazymefamily)
RETURN
  gen.id AS genome_id,
  p.id AS protein_id,
  f.familyId AS cazyme_family,
  a.evalue AS evalue,
  a.coverage AS coverage,
  a.startPosition AS domain_start,
  a.endPosition AS domain_end
ORDER BY cazyme_family, protein_id

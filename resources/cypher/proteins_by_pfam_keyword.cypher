// Proteins annotated with PFAM domains matching a keyword (global or filtered by genomes)
// Params:
//   $q          : string (keyword; matches Domain.id, pfamAccession, or description)
//   $limit      : integer (max rows)
//   $genome_ids : [string] (optional; empty/null means global)

MATCH (d:Domain)
WHERE toLower(d.id) CONTAINS toLower($q)
   OR (d.pfamAccession IS NOT NULL AND toLower(d.pfamAccession) CONTAINS toLower($q))
   OR toLower(coalesce(d.description, '')) CONTAINS toLower($q)
WITH collect(DISTINCT d) AS domains
UNWIND domains AS d
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d)
MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
RETURN
  g.id AS genome_id,
  p.id AS protein_id,
  coalesce(d.pfamAccession, d.id) AS pfam_id,
  d.id AS domain_id,
  da.evalue AS evalue,
  da.bitscore AS bitscore,
  da.domainStart AS domain_start,
  da.domainEnd AS domain_end
ORDER BY pfam_id, protein_id
LIMIT toInteger($limit)


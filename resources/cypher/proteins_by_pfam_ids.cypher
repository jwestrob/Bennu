// Proteins annotated with specific PFAM IDs (exact match on Domain.id or pfamAccession)
// Params:
//   $pfam_ids  : [string] (PFxxxxx or family IDs; case-insensitive)
//   $genome_ids: [string] (optional; empty/null means global)
//   $limit     : integer (max rows)

WITH [x IN $pfam_ids WHERE x IS NOT NULL] AS pfams
WITH [x IN pfams | toLower(x)] AS lowers
UNWIND lowers AS pf
MATCH (d:Domain)
WHERE toLower(d.id) = pf OR (d.pfamAccession IS NOT NULL AND toLower(d.pfamAccession) = pf)
WITH DISTINCT d
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d)
MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
RETURN
  g.id AS genome_id,
  p.id AS protein_id,
  coalesce(d.pfamAccession, d.id) AS pfam_id,
  d.id AS domain_id,
  d.description AS domain_desc,
  da.evalue AS evalue,
  da.bitscore AS bitscore,
  da.domainStart AS domain_start,
  da.domainEnd AS domain_end
ORDER BY pfam_id, protein_id
LIMIT toInteger($limit)

// Proteins annotated with specific PFAM IDs
// Matching policy (robust, accession-aware):
// - Normalize PFxxxxx[.yy] tokens to base PFxxxxx.
// - Accession tokens (PFxxxxx): prefer indexed STARTS WITH matching on pfamAccession (version tolerant) or id.
// - Non-accession tokens: substring matching on id/name/description (case-insensitive).
// Params:
//   $pfam_ids  : [string] (PFxxxxx or family IDs; case-insensitive)
//   $genome_ids: [string] (optional; empty/null means global)
//   $limit     : integer (max rows)

WITH [x IN $pfam_ids WHERE x IS NOT NULL] AS pfams
WITH [x IN pfams | toLower(x)] AS lowers
// Normalize accessions: PFxxxxx[.yy] -> PFxxxxx base
WITH [x IN lowers | CASE WHEN x =~ '^pf\\d{5}(?:\\.\\d+)?$' THEN split(x, '.')[0] ELSE x END] AS tokens
WITH tokens
MATCH (d:Domain)
WHERE (
  ANY(t IN tokens WHERE t =~ '^pf\\d{5}$' AND (
    toLower(coalesce(d.pfamAccession, '')) STARTS WITH t OR
    toLower(d.id) STARTS WITH t
  ))
  OR
  ANY(t IN tokens WHERE NOT t =~ '^pf\\d{5}$' AND (
    toLower(d.id) CONTAINS t OR
    toLower(coalesce(d.name, '')) CONTAINS t OR
    toLower(coalesce(d.description, '')) CONTAINS t
  ))
)
WITH DISTINCT d
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d)
MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
RETURN
  g.id AS genome_id,
  p.id AS protein_id,
  coalesce(d.name, d.description, d.id) AS pfam_name,
  // keep ids available for internal use; UI layers may omit
  coalesce(d.pfamAccession, d.id) AS pfam_id,
  d.id AS domain_id,
  d.description AS domain_desc,
  da.evalue AS evalue,
  da.bitscore AS bitscore,
  da.domainStart AS domain_start,
  da.domainEnd AS domain_end
ORDER BY pfam_id, protein_id
LIMIT toInteger($limit)

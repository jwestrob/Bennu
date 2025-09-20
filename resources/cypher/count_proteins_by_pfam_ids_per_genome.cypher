// Count proteins per genome for each PFAM id (accession or keyword token)
// Behavior:
// - Normalize PFxxxxx[.yy] tokens to base PFxxxxx and match pfamAccession/id STARTS WITH (index-friendly)
// - For non-accession tokens, match id/name/description CONTAINS (case-insensitive)
// Params:
//   $pfam_ids  : [string]
//   $genome_ids: [string] optional; filter by genome id when provided

UNWIND $pfam_ids AS raw
WITH toLower(raw) AS tkn, $genome_ids AS gids
WITH (CASE WHEN tkn =~ '^pf\\d{5}(?:\\.\\d+)?$' THEN split(tkn, '.')[0] ELSE tkn END) AS norm, gids
CALL (norm, gids) {
  MATCH (d:Domain)
  WHERE (
    norm =~ '^pf\\d{5}$' AND (
      toLower(coalesce(d.pfamAccession,'')) STARTS WITH norm OR
      toLower(d.id) STARTS WITH norm
    )
  ) OR (
    NOT norm =~ '^pf\\d{5}$' AND (
      toLower(d.id) CONTAINS norm OR
      toLower(coalesce(d.name,'')) CONTAINS norm OR
      toLower(coalesce(d.description,'')) CONTAINS norm
    )
  )
  WITH DISTINCT d, gids, norm
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  OPTIONAL MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
  WHERE gids IS NULL OR size(gids)=0 OR g.id IN gids
  RETURN norm AS pfam_id, g.id AS genome_id, count(DISTINCT p) AS count
}
RETURN genome_id, pfam_id,
       coalesce(count, 0) AS count
ORDER BY genome_id, pfam_id;


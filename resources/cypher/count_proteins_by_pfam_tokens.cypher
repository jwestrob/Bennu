// Count proteins per PFAM token (accession or keyword) and return a representative label
// Behavior:
// - For accession-like tokens (PFxxxxx[.yy]), normalize to base PFxxxxx and match pfamAccession/id STARTS WITH (index-friendly)
// - For other tokens, match id/name/description CONTAINS (case-insensitive)
// Returns one row per input token with {token, label, count}
// Params:
//   $tokens     : [string]
//   $genome_ids : [string] optional; filter by genome id when provided

UNWIND $tokens AS raw
WITH toLower(raw) AS tkn, $genome_ids AS gids
WITH (CASE WHEN tkn =~ '^pf\\d{5}(?:\\.\\d+)?$' THEN split(tkn, '.')[0] ELSE tkn END) AS norm, gids
CALL (norm, gids) {
  // Find the best-matching PFAM family for this token and count proteins
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
  WITH d, gids LIMIT 200
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  OPTIONAL MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
  WHERE gids IS NULL OR size(gids)=0 OR g.id IN gids
  WITH d, count(DISTINCT p) AS c
  RETURN coalesce(d.name, d.description, d.id) AS label, c
  ORDER BY c DESC
  LIMIT 1
}
RETURN norm AS token, coalesce(label, norm) AS label, coalesce(c, 0) AS count;

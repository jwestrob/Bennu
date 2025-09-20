// Count proteins per PFAM token and return a representative label
// This is the id-based counterpart to count_proteins_by_pfam_tokens, accepting
// PFxxxxx (with optional version) or short-name tokens.
// Params:
//   $pfams     : [string]
//   $genome_ids: [string] optional; filter by genome id when provided

UNWIND $pfams AS raw
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
  WITH d, gids LIMIT 1
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  OPTIONAL MATCH (p)-[:ENCODEDBY]->(:Gene)-[:BELONGSTOGENOME]->(g:Genome)
  WHERE gids IS NULL OR size(gids)=0 OR g.id IN gids
  WITH d, count(DISTINCT p) AS c
  RETURN coalesce(d.name, d.description, d.id) AS label,
         coalesce(d.pfamAccession, d.id) AS acc,
         c
}
WITH norm,
     label,
     CASE
       WHEN acc =~ '(?i)^pf\\d{5}(?:\\.\\d+)?$' THEN toUpper(substring(acc,0,7))
       ELSE acc
     END AS pfam_id,
     c
RETURN pfam_id, coalesce(label, norm) AS label, coalesce(c, 0) AS count;


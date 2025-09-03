// Proteins annotated with any of the provided PFAM domain families (exact, prefix, or substring)
MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE (
  d.id IN $pfams OR d.pfamAccession IN $pfams OR
  ANY(pf IN $pfams WHERE (
    // accession-like tokens → prefer STARTS WITH on pfamAccession/id
    toLower(pf) =~ '^pf\\d{5}$' AND (
      toLower(coalesce(d.pfamAccession,'')) STARTS WITH toLower(pf) OR
      toLower(d.id) STARTS WITH toLower(pf)
    )
  ) OR ANY(pf IN $pfams WHERE (
    // other tokens → substring on id/name/description
    NOT toLower(pf) =~ '^pf\\d{5}$' AND (
      toLower(d.id) CONTAINS toLower(pf) OR
      toLower(coalesce(d.name,'')) CONTAINS toLower(pf) OR
      toLower(coalesce(d.description,'')) CONTAINS toLower(pf)
    )
  ))
)
RETURN DISTINCT p
LIMIT $limit;

// Index-friendly PFAM matching with optional exact mode and full-text fallback for keywords
// Branches:
//  - exact=true         → equality on d.pfamAccession or d.id (index-backed)
//  - accession token    → STARTS WITH on d.pfamAccession or d.id (index-backed)
//  - keyword (default)  → full-text search on domainText, then join

// Exact branch
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH toUpper(q) AS q, ex
  WHERE ex
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE d.pfamAccession = q OR d.id = q
  RETURN DISTINCT p
}
UNION
// Accession/prefix branch
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH toUpper(q) AS q, ex
  WHERE NOT ex AND q =~ '^PF\\d{5}$'
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE d.pfamAccession STARTS WITH q OR d.id STARTS WITH q
  RETURN DISTINCT p
}
UNION
// Keyword branch via full-text index
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH q, ex
  WHERE NOT ex AND NOT toUpper(q) =~ '^PF\\d{5}$'
  CALL db.index.fulltext.queryNodes('domainText', q) YIELD node AS d
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  RETURN DISTINCT p
}
RETURN DISTINCT p;

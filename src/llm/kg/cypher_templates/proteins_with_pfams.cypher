// Proteins annotated with any of the provided PFAM tokens (accession or keyword)
// Accession tokens use STARTS WITH on indexed fields; keywords use full-text lookup

WITH [pf IN $pfams | toUpper(pf)] AS toks
// Accession/prefix branch
CALL {
  WITH toks
  UNWIND toks AS q
  WITH DISTINCT q WHERE q =~ '^PF\\d{5}$'
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE d.pfamAccession STARTS WITH q OR d.id STARTS WITH q
  RETURN DISTINCT p
}
UNION
// Keyword branch via full-text index
CALL {
  WITH toks
  UNWIND toks AS q
  WITH DISTINCT q WHERE NOT q =~ '^PF\\d{5}$'
  CALL db.index.fulltext.queryNodes('domainText', q) YIELD node AS d
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  RETURN DISTINCT p
}
RETURN DISTINCT p
LIMIT $limit;

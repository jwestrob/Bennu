// Count proteins by PFAM family with index-friendly branches and full-text for keywords
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH toUpper(q) AS q, ex
  WHERE ex
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE d.pfamAccession = q OR d.id = q
  RETURN count(DISTINCT p) AS c
}
UNION
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH toUpper(q) AS q, ex
  WHERE NOT ex AND q =~ '^PF\\d{5}$'
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE d.pfamAccession STARTS WITH q OR d.id STARTS WITH q
  RETURN count(DISTINCT p) AS c
}
UNION
CALL {
  WITH $pfam AS q, coalesce($exact,false) AS ex
  WITH q, ex
  WHERE NOT ex AND NOT toUpper(q) =~ '^PF\\d{5}$'
  CALL db.index.fulltext.queryNodes('domainText', q) YIELD node AS d
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d)
  RETURN count(DISTINCT p) AS c
}
RETURN sum(c) AS count;

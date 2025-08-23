// Flexible PFAM/domain matching: exact or prefix/name match
// If $exact is true, use strict equality; otherwise allow versioned/accession prefixes and name contains
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE CASE WHEN coalesce($exact,false)
  THEN (d.id = $pfam OR d.pfamAccession = $pfam)
  ELSE (
    (d.pfamAccession STARTS WITH $pfam) OR
    (d.id STARTS WITH $pfam) OR
    (toLower(d.description) CONTAINS toLower($pfam))
  )
END
RETURN p;

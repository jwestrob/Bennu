// Flexible PFAM/domain matching: exact or prefix/name match
// If $exact is true, use strict equality; otherwise allow versioned/accession prefixes and name contains
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE CASE WHEN coalesce($exact,false)
  THEN (d.id = $pfam OR d.pfamAccession = $pfam)
  ELSE (
    // If $pfam is a canonical accession, prefer STARTS WITH on pfamAccession/id
    (toLower($pfam) =~ '^pf\\d{5}$' AND (
      toLower(coalesce(d.pfamAccession,'')) STARTS WITH toLower($pfam) OR
      toLower(d.id) STARTS WITH toLower($pfam)
    )) OR
    // Otherwise, substring on name/desc/id
    (NOT toLower($pfam) =~ '^pf\\d{5}$' AND (
      toLower(d.id) CONTAINS toLower($pfam) OR
      toLower(coalesce(d.name,'')) CONTAINS toLower($pfam) OR
      toLower(coalesce(d.description,'')) CONTAINS toLower($pfam)
    ))
  )
END
RETURN p;

// Count proteins by PFAM family with flexible matching
// If $exact is true, use strict equality; otherwise allow versioned/accession prefixes and description matches
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE CASE WHEN coalesce($exact,false)
  THEN (d.id = $pfam OR d.pfamAccession = $pfam)
  ELSE (
    (toLower($pfam) =~ '^pf\\d{5}$' AND (
      toLower(coalesce(d.pfamAccession,'')) STARTS WITH toLower($pfam) OR
      toLower(d.id) STARTS WITH toLower($pfam)
    )) OR
    (NOT toLower($pfam) =~ '^pf\\d{5}$' AND (
      toLower(d.id) CONTAINS toLower($pfam) OR
      toLower(coalesce(d.name,'')) CONTAINS toLower($pfam) OR
      toLower(coalesce(d.description,'')) CONTAINS toLower($pfam)
    ))
  )
END
RETURN count(p) AS count;

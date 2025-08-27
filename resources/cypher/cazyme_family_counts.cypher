// Count CAZyme-annotated proteins by family (dbCAN via Domain/DomainAnnotation)
MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE d.id =~ '^(GH|GT|PL|CE|CBM)[0-9].*' OR d.id =~ '^AA[0-9].*'
WITH d.id AS family, count(DISTINCT p) AS proteins
RETURN family, proteins
ORDER BY proteins DESC, family

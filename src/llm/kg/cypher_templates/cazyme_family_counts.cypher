// Count CAZyme-annotated proteins by family
MATCH (p:Protein)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WITH cf.familyId AS family, count(DISTINCT p) AS proteins
RETURN family, proteins
ORDER BY proteins DESC, family

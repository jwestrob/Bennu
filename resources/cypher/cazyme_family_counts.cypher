// Count CAZy families using Stage 07 CAZy nodes
//   (Protein)-[:HASCAZYME]->(CAZymeAnnotation)-[:CAZYMEFAMILY]->(CAZymeFamily)
MATCH (p:Protein)-[:HASCAZYME]->(:Cazymeannotation)-[:CAZYMEFAMILY]->(f)
WHERE f.familyId IS NOT NULL
RETURN f.familyId AS family, count(DISTINCT p) AS proteins
ORDER BY proteins DESC, family

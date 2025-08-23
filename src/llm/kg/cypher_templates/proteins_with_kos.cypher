// Proteins annotated with any of the provided KEGG Ortholog IDs
MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
WHERE ko.id IN $kos
RETURN DISTINCT p
LIMIT $limit;


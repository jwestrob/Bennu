// Proteins annotated with any of the provided PFAM domain families
MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE d.id IN $pfams OR d.pfamAccession IN $pfams
RETURN DISTINCT p
LIMIT $limit;


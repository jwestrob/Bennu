MATCH (p:Protein)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE d.id = $pfam OR d.pfamAccession = $pfam
RETURN count(p) AS count;


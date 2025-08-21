MATCH (p:Protein)-[:HAS_CAZY]->(c:CAZY {family:$family})
RETURN p;


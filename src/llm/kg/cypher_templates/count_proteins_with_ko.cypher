MATCH (p:Protein)-[:HAS_KO]->(k:KO {id:$ko})
RETURN count(p) AS count;

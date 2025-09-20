MATCH (p:Protein)-[:HASFUNCTION]->(k:KEGGOrtholog {id:$ko})
RETURN count(p) AS count;

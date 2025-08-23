MATCH (p:Protein)-[:HASFUNCTION]->(k:KEGGOrtholog {id:$ko})
RETURN p;

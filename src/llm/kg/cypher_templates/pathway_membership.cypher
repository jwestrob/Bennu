MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway {id:$pathway})
RETURN DISTINCT p;

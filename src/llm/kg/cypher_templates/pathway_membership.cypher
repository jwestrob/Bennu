MATCH (p:Protein)-[:IN_PATHWAY]->(pw:Pathway {id:$pathway})
RETURN p;


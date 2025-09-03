// Count proteins per KO id
UNWIND $kos AS ko
WITH DISTINCT ko AS kid
OPTIONAL MATCH (p:Protein)-[:HASFUNCTION]->(:KEGGOrtholog {id: kid})
RETURN kid AS ko, count(DISTINCT p) AS count;


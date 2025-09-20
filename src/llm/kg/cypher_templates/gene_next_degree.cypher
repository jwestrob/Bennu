// Count the degree of NEXT relationships (undirected) for a given gene
MATCH (g:Gene {id:$gene_id})-[:NEXT]-(:Gene)
RETURN count(*) AS next_degree;


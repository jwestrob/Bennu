MATCH (g:Genome {id:$genome_id})<-[:BELONGSTOGENOME]-(p:Protein)
RETURN p;


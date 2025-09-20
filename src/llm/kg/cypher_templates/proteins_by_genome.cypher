MATCH (gen:Genome {genomeId:$genome_id})<-[:BELONGSTOGENOME]-(gene:Gene)
MATCH (gene)<-[:ENCODEDBY]-(p:Protein)
RETURN DISTINCT p;

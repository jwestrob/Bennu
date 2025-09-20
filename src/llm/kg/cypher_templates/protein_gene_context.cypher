// Resolve a protein's encoding gene and return its coordinates
MATCH (p:Protein {id:$protein_id})<-[:ENCODEDBY]-(g:Gene)
RETURN g.id AS gene_id,
       g.contig AS contig,
       toInteger(g.startCoordinate) AS start,
       toInteger(g.endCoordinate) AS end,
       g.strand AS strand
LIMIT 1;


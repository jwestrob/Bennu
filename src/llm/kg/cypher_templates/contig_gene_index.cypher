// Return gene index (0-based) and total gene count for a contig by coordinate order
WITH $contig AS contig, $gene_id AS seed_id
MATCH (g:Gene {contig: contig})
WITH g ORDER BY toInteger(g.startCoordinate)
WITH collect(g.id) AS ids
UNWIND range(0, size(ids)-1) AS i
WITH ids, i WHERE ids[i] = seed_id
RETURN size(ids) AS contig_gene_count, i AS gene_index;


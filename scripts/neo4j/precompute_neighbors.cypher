// Precompute bidirectional genomic neighbor relationships within each contig
// Creates :NEXT and :PREV between adjacent genes (by startCoordinate), plus strand-aware variants.
// Assumes Gene nodes have: contig, startCoordinate, endCoordinate, strand (string or int)

// Optional cleanup (uncomment if you need a full rebuild of neighbor edges)
// MATCH ()-[r:NEXT|PREV|NEXT_SAME_STRAND|PREV_SAME_STRAND]->() DELETE r;

// 1) Bidirectional adjacency across contigs (ignores strand, pure genomic order)
MATCH (g:Gene)
WHERE g.contig IS NOT NULL AND g.startCoordinate IS NOT NULL AND g.endCoordinate IS NOT NULL
WITH g.contig AS contig, g
ORDER BY contig, toInteger(g.startCoordinate)
WITH contig, collect(g) AS genes
UNWIND range(0, size(genes)-2) AS i
WITH contig, genes[i] AS a, genes[i+1] AS b
MERGE (a)-[r:NEXT]->(b)
SET r.contig = contig,
    r.delta = toInteger(b.startCoordinate) - toInteger(a.endCoordinate),
    r.same_strand = toString(b.strand) = toString(a.strand)
MERGE (b)-[p:PREV]->(a)
SET p.contig = contig,
    p.delta = toInteger(a.startCoordinate) - toInteger(b.endCoordinate),
    p.same_strand = toString(b.strand) = toString(a.strand);

// 2) Strand-specific adjacency (same-strand doubly linked list)
MATCH (g:Gene)
WHERE g.contig IS NOT NULL AND g.startCoordinate IS NOT NULL AND g.endCoordinate IS NOT NULL AND g.strand IS NOT NULL
WITH g.contig AS contig, toString(g.strand) AS strand, g
ORDER BY contig, toInteger(g.startCoordinate)
WITH contig, strand, collect(g) AS genes
UNWIND range(0, size(genes)-2) AS i
WITH contig, strand, genes[i] AS a, genes[i+1] AS b
MERGE (a)-[r:NEXT_SAME_STRAND]->(b)
SET r.contig = contig,
    r.strand = strand,
    r.delta = toInteger(b.startCoordinate) - toInteger(a.endCoordinate)
MERGE (b)-[p:PREV_SAME_STRAND]->(a)
SET p.contig = contig,
    p.strand = strand,
    p.delta = toInteger(a.startCoordinate) - toInteger(b.endCoordinate);

// Notes:
// - delta > 0 implies intergenic distance; delta < 0 implies overlap by |delta| bp
// - same_strand on NEXT/PREV allows quick filtering without following *_SAME_STRAND edges
// - Composite index on (Gene.contig, Gene.startCoordinate, Gene.endCoordinate) is recommended


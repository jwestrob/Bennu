// NEXT edges marked as crossing CRISPR arrays
// Optional slots: contig, limit

MATCH (a:Gene)-[r:NEXT]->(b:Gene)
WHERE coalesce(r.crisprBetween,false) = true
  AND ($contig IS NULL OR r.contig = $contig)
RETURN a.id AS gene_a,
       b.id AS gene_b,
       r.contig AS contig,
       toInteger(r.crisprCountBetween) AS crispr_between,
       toInteger(r.delta) AS delta
ORDER BY crispr_between DESC, contig, gene_a
LIMIT coalesce($limit, 200)


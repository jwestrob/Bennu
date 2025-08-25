/* Input: $loci: [{seed_protein_id:string, contig_id:string, verdict:string}] */
UNWIND $loci AS L
MATCH (p:Protein {id: L.seed_protein_id})
OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)
MERGE (loc:Locus {seed_protein_id:L.seed_protein_id, contig_id:L.contig_id})
  ON CREATE SET loc.verdict = L.verdict, loc.created_at = timestamp()
  ON MATCH SET  loc.verdict = L.verdict, loc.updated_at = timestamp()
MERGE (loc)-[:INDEXES]->(p)
// Anchor to the gene if available; there is no Contig node in the schema
WITH loc, g
FOREACH (_ IN CASE WHEN g IS NULL THEN [] ELSE [1] END |
  MERGE (loc)-[:ANCHORS_TO_GENE]->(g)
)
RETURN count(loc) AS upserted

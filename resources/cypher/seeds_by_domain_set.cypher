/* Input: $pfam_ids: [string], $ko_ids: [string], $limit: int */
// Seeds by explicit PFAM/KO ID sets (no description matching)
// Returns same shape as seeds_by_marker.cypher for compatibility

// PFAM side
WITH $pfam_ids AS pfams
WITH [x IN pfams WHERE x IS NOT NULL] AS pfams
WITH [x IN pfams | toLower(x)] AS lowers
CALL (lowers) {
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE toLower(d.id) IN lowers OR (d.pfamAccession IS NOT NULL AND toLower(d.pfamAccession) IN lowers)
  RETURN DISTINCT p
  LIMIT $limit
}
WITH p
MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen:Genome)
CALL (g) {
  MATCH (h:Gene {contig: g.contig})
  RETURN max(toInteger(h.endCoordinate)) AS max_end,
         min(toInteger(h.startCoordinate)) AS min_start
}
OPTIONAL MATCH (:Gene {contig: g.contig})<-[:ENCODEDBY]-(q:Protein)
WITH p, g, gen, max_end, min_start, count(q) AS orf_count
RETURN gen.genomeId AS genome_id,
       g.contig AS contig_id,
       p.id AS seed_protein_id,
       toInteger(p.length) AS aa,
       orf_count,
       (max_end - min_start + 1) AS contig_len
LIMIT $limit

UNION

// KO side
WITH $ko_ids AS kos
WITH [x IN kos WHERE x IS NOT NULL] AS kos
WITH [x IN kos | toLower(x)] AS lowers
CALL (lowers) {
  MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
  WHERE toLower(ko.id) IN lowers
  RETURN DISTINCT p
  LIMIT $limit
}
WITH p
MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen:Genome)
CALL (g) {
  MATCH (h:Gene {contig: g.contig})
  RETURN max(toInteger(h.endCoordinate)) AS max_end,
         min(toInteger(h.startCoordinate)) AS min_start
}
OPTIONAL MATCH (:Gene {contig: g.contig})<-[:ENCODEDBY]-(q:Protein)
WITH p, g, gen, max_end, min_start, count(q) AS orf_count
RETURN gen.genomeId AS genome_id,
       g.contig AS contig_id,
       p.id AS seed_protein_id,
       toInteger(p.length) AS aa,
       orf_count,
       (max_end - min_start + 1) AS contig_len
LIMIT $limit

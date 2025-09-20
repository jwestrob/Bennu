/* Input: $markers: [string], $limit: int */
// Find seed proteins by PFAM Domain or KEGG KO terms (id/accession/description)
WITH [x IN $markers | toLower(x)] AS lowers

// PFAM/Domain side (bounded early to avoid heavy scans)
CALL (lowers) {
  MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
  WHERE toLower(d.id) IN lowers
     OR (d.pfamAccession IS NOT NULL AND toLower(d.pfamAccession) IN lowers)
     OR any(m IN lowers WHERE toLower(coalesce(d.description, '')) CONTAINS m)
  RETURN DISTINCT p
  LIMIT $limit
}
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

// KO side (bounded early to avoid heavy scans)
WITH [x IN $markers | toLower(x)] AS lowers
CALL (lowers) {
  MATCH (p:Protein)-[:HASFUNCTION]->(ko:KEGGOrtholog)
  WHERE toLower(ko.id) IN lowers
     OR any(m IN lowers WHERE toLower(coalesce(ko.description, '')) CONTAINS m)
  RETURN DISTINCT p
  LIMIT $limit
}
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

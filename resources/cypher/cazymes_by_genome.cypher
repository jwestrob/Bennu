// All CAZyme-annotated proteins (global or filtered by genome) via Domain/DomainAnnotation
// Schema observed:
//   (Protein)-[:HASDOMAIN]->(DomainAnnotation)-[:DOMAINFAMILY]->(Domain)
//   Domain.id contains CAZy families: GHxx, GTxx, PLxx, CExx, CBMxx, AAxx
// Parameters:
//   $genome_id  : string (optional) — single genome filter
//   $genome_ids : [string] (optional) — multiple genome filter; empty or null means all
MATCH (gen:Genome)
WHERE ($genome_id IS NULL OR gen.id = $genome_id)
  AND ($genome_ids IS NULL OR size($genome_ids) = 0 OR gen.id IN $genome_ids)
MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen)
MATCH (p)-[:HASDOMAIN]->(da:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE d.id =~ '^(GH|GT|PL|CE|CBM)[0-9].*' OR d.id =~ '^AA[0-9].*'
RETURN
  gen.id AS genome_id,
  p.id AS protein_id,
  d.id AS cazyme_family,
  da.evalue AS evalue,
  da.bitscore AS bitscore,
  da.domainStart AS domain_start,
  da.domainEnd AS domain_end
ORDER BY cazyme_family, protein_id

// Genes and proteins that are part of a specific BGC (with optional annotations)
MATCH (bgc:Bgc {bgcId:$bgc_id})
MATCH (bgc)<-[:PARTOFBGC]-(gene:Gene)
OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(gene)
// Annotations (PFAM / KO / CAZy)
OPTIONAL MATCH (p)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
OPTIONAL MATCH (p)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WITH bgc, gene, p,
     collect(DISTINCT coalesce(d.pfamAccession, d.id)) AS pfam_ids,
     collect(DISTINCT coalesce(d.name, d.description)) AS pfam_names,
     collect(DISTINCT ko.id) AS ko_ids,
     collect(DISTINCT ko.description) AS ko_desc,
     collect(DISTINCT cf.familyId) AS cazy_families
RETURN 
  bgc.bgcId AS bgc_id,
  gene.id AS gene_id,
  p.id AS protein_id,
  toInteger(gene.startCoordinate) AS start,
  toInteger(gene.endCoordinate) AS end,
  gene.strand AS strand,
  pfam_ids AS pfam_ids,
  pfam_names AS pfam_names,
  ko_ids AS ko_ids,
  ko_desc AS ko_desc,
  cazy_families AS cazy_families
ORDER BY start

// Genes and proteins that are part of a specific BGC
MATCH (bgc:Bgc {bgcId:$bgc_id})
MATCH (bgc)<-[:PARTOFBGC]-(gene:Gene)
OPTIONAL MATCH (p:Protein)-[:ENCODEDBY]->(gene)
RETURN 
  bgc.bgcId AS bgc_id,
  gene.id AS gene_id,
  p.id AS protein_id,
  gene.startCoordinate AS start,
  gene.endCoordinate AS end,
  gene.strand AS strand
ORDER BY toInteger(start)

// GECCO BGCs filtered by predicted product
MATCH (bgc:Bgc)
WHERE toLower(bgc.bgcProduct) CONTAINS toLower($product)
OPTIONAL MATCH (gen:Genome)-[:HASBGC]->(bgc)
RETURN 
  bgc.bgcId AS bgc_id,
  coalesce(bgc.bgcProduct, 'Unknown') AS bgc_product,
  gen.id AS genome_id,
  bgc.averageProbability AS averageProbability,
  bgc.maxProbability AS maxProbability,
  bgc.contig AS contig,
  bgc.startCoordinate AS startCoordinate,
  bgc.endCoordinate AS endCoordinate
ORDER BY maxProbability DESC, bgc_id

// GECCO-predicted BGCs for a given genome
MATCH (gen:Genome {id:$genome_id})-[:HASBGC]->(bgc:Bgc)
RETURN 
  gen.id AS genome_id,
  bgc.bgcId AS bgc_id,
  bgc.bgcProduct AS bgc_product,
  bgc.contig AS contig,
  bgc.startCoordinate AS startCoordinate,
  bgc.endCoordinate AS endCoordinate,
  bgc.lengthNt AS lengthNt,
  bgc.proteinCount AS proteinCount,
  bgc.averageProbability AS averageProbability,
  bgc.maxProbability AS maxProbability
ORDER BY maxProbability DESC, bgc_id

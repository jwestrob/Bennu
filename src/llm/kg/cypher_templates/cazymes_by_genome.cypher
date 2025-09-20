// All CAZyme-annotated proteins for a given genome
MATCH (gen:Genome {id:$genome_id})
MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen)
MATCH (p)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
RETURN 
  gen.id AS genome_id,
  p.id AS protein_id,
  ca.familyId AS cazyme_family,
  ca.cazymeType AS cazyme_type,
  ca.substrateSpecificity AS substrate,
  ca.evalue AS evalue,
  ca.coverage AS coverage
ORDER BY cazyme_family, protein_id

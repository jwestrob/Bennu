// Proteins annotated by dbCAN (CAZyme) in a specific family
// Schema: (Protein)-[:HASCAZYME]->(Cazymeannotation)-[:CAZYMEFAMILY]->(Cazymefamily)
MATCH (p:Protein)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(gen:Genome)
MATCH (p)-[:HASCAZYME]->(ca:Cazymeannotation)-[:CAZYMEFAMILY]->(cf:Cazymefamily)
WHERE cf.familyId = $family OR ca.familyId = $family
RETURN 
  p.id AS protein_id,
  gen.id AS genome_id,
  ca.familyId AS cazyme_family,
  ca.cazymeType AS cazyme_type,
  ca.substrateSpecificity AS substrate,
  ca.evalue AS evalue,
  ca.coverage AS coverage
ORDER BY genome_id, cazyme_family, protein_id

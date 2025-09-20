// Fetch PFAM/KO annotations for a set of protein IDs
UNWIND $protein_ids AS pid
MATCH (p:Protein {id: pid})
OPTIONAL MATCH (p)-[:ENCODEDBY]->(g:Gene)-[:BELONGSTOGENOME]->(genome:Genome)
OPTIONAL MATCH (p)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(dom:Domain)
OPTIONAL MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)
WITH pid, genome, g,
     collect(DISTINCT dom) AS doms,
     collect(DISTINCT ko) AS kos
RETURN pid AS protein_id,
       genome.id AS genome_id,
       g.id AS gene_id,
       [d IN doms WHERE d IS NOT NULL | coalesce(d.pfamAccession, d.id)] AS pfam_ids,
       [d IN doms WHERE d IS NOT NULL | coalesce(d.name, d.description)] AS pfam_desc,
       [k IN kos WHERE k IS NOT NULL | k.id] AS ko_ids,
       [k IN kos WHERE k IS NOT NULL | k.description] AS ko_desc
ORDER BY protein_id

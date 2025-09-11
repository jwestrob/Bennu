// Return exactly five upstream and five downstream genes by contig order relative to the seed protein's gene
// Upstream/downstream defined by coordinate order on the contig (not transcription orientation)

MATCH (p:Protein {id:$protein_id})-[:ENCODEDBY]->(seed:Gene)
WITH seed
MATCH (g:Gene {contig: seed.contig})
WITH seed, g
ORDER BY toInteger(g.startCoordinate)
WITH seed, collect(g) AS gs
WITH seed, gs, [i IN range(0, size(gs)-1) WHERE gs[i].id = seed.id][0] AS idx
WITH seed, gs, idx, range(-5, 5) AS offsets
UNWIND offsets AS off
WITH seed, gs, idx, off WHERE off <> 0
WITH seed, gs, idx, off, (idx + off) AS ni
WHERE ni >= 0 AND ni < size(gs)
WITH off AS relative_position, gs[ni] AS ng
OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng)
OPTIONAL MATCH (ng)-[f:FLANKS_CRISPR]->(ca:CrisprArray)
RETURN ng.id AS gene_id,
       ng.contig AS contig,
       toInteger(ng.startCoordinate) AS start,
       toInteger(ng.endCoordinate) AS end,
       ng.strand AS strand,
       np.id AS protein_id,
       relative_position,
       ca.id AS crispr_id,
       toInteger(f.distanceBp) AS crispr_distance_bp
ORDER BY relative_position, start;

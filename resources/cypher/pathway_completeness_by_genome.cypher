// Parameters:
//   $genome_ids : [string] (optional) — if empty or not provided, compute for all genomes
//   $min_completeness : float (optional) — if provided, filter pathways by completeness >= threshold

// Present KO hits per (genome, pathway)
MATCH (p:Protein)-[:ENCODEDBY]->(gene:Gene)-[:BELONGSTOGENOME]->(g:Genome)
MATCH (p)-[:HASFUNCTION]->(ko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw:Pathway)
WHERE $genome_ids IS NULL OR size($genome_ids) = 0 OR g.id IN $genome_ids
WITH g, pw, collect(DISTINCT ko.id) AS present_kos

// All KOs defined for each pathway (global, independent of presence)
OPTIONAL MATCH (allko:KEGGOrtholog)-[:PARTICIPATESIN]->(pw)
WITH g, pw, present_kos, collect(DISTINCT allko.id) AS all_kos
WITH g, pw, present_kos, all_kos,
     size(all_kos) AS total_kos,
     size(present_kos) AS present_count,
     [x IN all_kos WHERE NOT x IN present_kos] AS missing_kos
WITH g, pw, present_kos, all_kos, total_kos, present_count, missing_kos,
     (CASE WHEN total_kos > 0 THEN toFloat(present_count) / toFloat(total_kos) ELSE 0.0 END) AS completeness
WHERE $min_completeness IS NULL OR completeness >= $min_completeness
RETURN g.id AS genome_id,
       pw.id AS pathway_id,
       coalesce(pw.name, pw.id) AS pathway_name,
       present_count AS present_kos,
       total_kos AS total_kos,
       completeness AS completeness,
       missing_kos AS missing_ko_ids,
       present_kos AS present_ko_ids
ORDER BY genome_id, completeness DESC, present_kos DESC, pathway_id

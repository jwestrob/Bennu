/* Input: $seeds: [{seed_pid, seed_protein_id, contig_len, orf_count, genome_id, contig_id}],
          $min_contig_len:int, $min_orf:int, $k_window:int */
UNWIND $seeds AS s
WITH s
WHERE toInteger(s.contig_len) >= toInteger($min_contig_len)
  AND toInteger(s.orf_count)  >= toInteger($min_orf)
// Resolve seed gene via ENCODEDBY; also capture seed PFAM/KO annotations
MATCH (sp:Protein {id: s.seed_protein_id})-[:ENCODEDBY]->(sg:Gene)
OPTIONAL MATCH (sp)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(sd:Domain)
OPTIONAL MATCH (sp)-[:HASFUNCTION]->(sko:KEGGOrtholog)
WITH s, sp, sg,
     [x IN collect(DISTINCT sd.id) WHERE x IS NOT NULL]  AS seed_pfams,
     [x IN collect(DISTINCT sko.id) WHERE x IS NOT NULL] AS seed_kos
MATCH (g:Gene {contig: sg.contig})
WITH s, sp, sg, seed_pfams, seed_kos, g
ORDER BY toInteger(g.startCoordinate)
WITH s, sp, sg, seed_pfams, seed_kos, collect(g) AS gs
WITH s, sp, sg, seed_pfams, seed_kos, gs, [i IN range(0, size(gs)-1) WHERE gs[i].id = sg.id][0] AS idx
WITH s, sp, sg, seed_pfams, seed_kos, gs, idx,
     range( case when idx - $k_window < 0 then 0 else idx - $k_window end,
            case when idx + $k_window >= size(gs) then size(gs)-1 else idx + $k_window end ) AS win
UNWIND win AS wi
WITH s, sp, sg, seed_pfams, seed_kos, gs[wi] AS ng WHERE ng.id <> sg.id
OPTIONAL MATCH (np:Protein)-[:ENCODEDBY]->(ng)
// OPTIONAL PFAM and KO annotations per neighbor protein
OPTIONAL MATCH (np)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
OPTIONAL MATCH (np)-[:HASFUNCTION]->(ko:KEGGOrtholog)
WITH s, sp, sg, seed_pfams, seed_kos, ng, np,
     [x IN collect(DISTINCT d.id) WHERE x IS NOT NULL]  AS pfams,
     [x IN collect(DISTINCT ko.id) WHERE x IS NOT NULL] AS kos
WITH s, sp, sg, seed_pfams, seed_kos,
     collect({
       protein_id: coalesce(np.id, ''),
       strand: ng.strand,
       name: coalesce(np.id, ''),
       order: toInteger(ng.startCoordinate),
       pfams: pfams,
       kos: kos
     }) AS neigh
RETURN coalesce(s.seed_protein_id, sp.id) AS seed_protein_id,
        sg.contig AS contig_id,
        s.genome_id AS genome_id,
        s.contig_len AS contig_len,
        seed_pfams AS seed_pfams,
        seed_kos AS seed_kos,
        neigh AS neighbors

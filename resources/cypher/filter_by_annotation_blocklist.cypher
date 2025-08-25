/* Input: $candidate_ids: [string], $namespace: string, $markers: [string] (lowercased) */
WITH [x IN $markers | toLower(x)] AS blocked, toLower($namespace) AS ns
UNWIND $candidate_ids AS pid
MATCH (p:Protein {id: pid})
OPTIONAL MATCH (p)-[:HAS_DOMAIN]->(d)
OPTIONAL MATCH (p)-[:HAS_KO]->(k)
WITH p, ns,
     [x IN collect(DISTINCT d) WHERE x IS NOT NULL] AS dlist,
     [x IN collect(DISTINCT k) WHERE x IS NOT NULL] AS klist,
     blocked
WITH p,
     CASE WHEN ns='pfam' THEN dlist
          WHEN ns='kofam' THEN klist
          ELSE dlist + klist END AS mlist,
     blocked
WITH p, [m IN mlist WHERE toLower(m.id) IN blocked OR toLower(m.name) IN blocked] AS hits
WHERE size(hits) = 0
RETURN p.id AS protein_id


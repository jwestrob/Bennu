// Proteins annotated with any of the provided PFAM tokens (accession or keyword)
// Accession tokens use STARTS WITH on indexed fields; keywords use full-text lookup

WITH [pf IN $pfams | toUpper(pf)] AS toks
UNWIND toks AS q
// Keyword candidates via full-text (empty when q is accession)
OPTIONAL CALL db.index.fulltext.queryNodes('domainText', q) YIELD node AS dkw
WITH q, collect(dkw) AS dkw_nodes
MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE (q =~ '^PF\\d{5}$' AND (d.pfamAccession STARTS WITH q OR d.id STARTS WITH q))
   OR (NOT q =~ '^PF\\d{5}$' AND d IN dkw_nodes)
RETURN DISTINCT p
LIMIT $limit;

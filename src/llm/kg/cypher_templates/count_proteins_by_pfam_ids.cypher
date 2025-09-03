// Count proteins per PFAM accession token (PFxxxxx), index-friendly
UNWIND $pfams AS raw
WITH DISTINCT toUpper(raw) AS pf
OPTIONAL MATCH (p:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(d:Domain)
WHERE pf =~ '^PF\\d{5}$' AND (d.pfamAccession STARTS WITH pf OR d.id STARTS WITH pf)
RETURN pf AS pfam, count(DISTINCT p) AS count;


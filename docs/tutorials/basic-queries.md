# Basic Queries

## Counting and Statistics

- Count `[:NEXT]` edges:
  - `MATCH ()-[:NEXT]->() RETURN count(*) AS edges`.

- Degree vs stored property:
  - `MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene)`
  - `OPTIONAL MATCH (g)-[:NEXT]-()`
  - `RETURN count(*) AS live, g.nextDegree AS prop`.

## Functional Searches

- Proteins with PFAM ID:
  - `MATCH (d:Domain {pfamAccession:$pf})<-[:DOMAINFAMILY]-(:DomainAnnotation)<-[:HASDOMAIN]-(p:Protein)`
  - `RETURN p.id LIMIT 25`.

- KO to pathways:
  - `MATCH (ko:KEGGOrtholog {id:$ko})-[:PARTICIPATESIN]->(pw:Pathway) RETURN pw.id, pw.name LIMIT 25`.

## Structural Queries

- Flanking by contig order (see Diagnostics for an index‑aware version):
  - Use the Neighborhood operator via the agent layer for convenience and batching.


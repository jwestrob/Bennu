# Augmented Neo4j Schema

## Connection (Docker default)

- URI: `bolt://localhost:7687` (container runs with `NEO4J_AUTH=none`)
- With creds: set `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` env vars

## Core Labels

- `Genome`, `Gene`, `Protein`, `Domain`, `DomainAnnotation`, `FunctionalAnnotation`, `KEGGOrtholog`, `Pathway`, `Bgc`, `QualityMetrics`, `Dataset`

## Key Properties

- `Gene`: `id`, `contig`, `startCoordinate`, `endCoordinate`, `strand`, `nextDegree`, `genesOnContig`
- `Protein`: `id` (optional: `name`, `description` when present)
- `Domain`: `id`, `pfamAccession`, `name` (description may be empty by design post‑cleanup)
- `KEGGOrtholog`: `id`, `description`

## Relationships (subset)

- `(:Protein)-[:ENCODEDBY]->(:Gene)`
- `(:Protein)-[:HASDOMAIN]->(:DomainAnnotation)-[:DOMAINFAMILY]->(:Domain)`
- `(:Protein)-[:HASFUNCTION]->(:KEGGOrtholog)`
- `(:KEGGOrtholog)-[:PARTICIPATESIN]->(:Pathway)`
- `(:Gene)-[:NEXT]->(:Gene)` (directed; treat as undirected for degree)
- `(:Gene)-[:BELONGSTOGENOME]->(:Genome)` and provenance edges

## Helpful Queries

- Global NEXT count:
  - `MATCH ()-[:NEXT]->() RETURN count(*) AS c`.
- Stored vs live degree for a seed:
  - `MATCH (p:Protein {id:$pid})-[:ENCODEDBY]->(g:Gene)`
  - `OPTIONAL MATCH (g)-[:NEXT]-()`
  - `WITH g, count(*) AS c` 
  - `RETURN toInteger(coalesce(g.nextDegree,c)) AS degree, toInteger(g.genesOnContig) AS onContig`.
- PFAM → proteins:
  - `MATCH (d:Domain {pfamAccession:$pf})<-[:DOMAINFAMILY]-(:DomainAnnotation)<-[:HASDOMAIN]-(p:Protein) RETURN p.id LIMIT 25`.
- KO → pathways:
  - `MATCH (ko:KEGGOrtholog {id:$ko})-[:PARTICIPATESIN]->(:Pathway) RETURN count(*)`.
- Index visibility (Neo4j 5): `SHOW INDEXES`, `SHOW CONSTRAINTS`.

## Indexes

- Unique IDs and composite `:Gene(contig,startCoordinate)` and `(contig,startCoordinate,endCoordinate)` are applied after import.
- Additional full‑text indexes may be present for convenience.

